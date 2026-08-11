/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// ONNX LSTM decomposition.
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/Transforms/DecomposeLSTM.hpp"

#include <cassert>

#include "mlir/IR/PatternMatch.h"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace onnx_mlir {
namespace {

// Decompose LSTM into basic ONNX operations in four stages:
//   1. Validate static inputs and normalize layout-1 tensors to
//      [sequence, batch, feature].
//   2. Materialize missing optional inputs and normalize the initial states.
//   3. Lower each direction through a shared step builder. A single timestep is
//      emitted directly; longer static sequences use onnx.Loop to carry the
//      hidden and cell states and scan the hidden state.
//   4. Reassemble the directional results and restore layout-1 outputs.
//
// The four gate projections are  fused in the width dimension.
// Rather than emitting separate input and recurrent MatMuls for I, O, F, and
// C, the decomposition computes X * W^T as one [sequence * batch, 4 * hidden]
// MatMul outside the recurrence (including the combined Wb + Rb bias). Each
// iteration then performs one H * R^T MatMul of the same packed width, adds the
// two packed projections, and slices the four gates.
// Eight per-gate projections per timestep become one hoisted input MatMul and
// one recurrent MatMul per iteration.
class DecomposeLSTM : public OpRewritePattern<ONNXLSTMOp> {
public:
  DecomposeLSTM(MLIRContext *context, PatternBenefit benefit,
      LSTMDecompositionPredicate predicate = {})
      : OpRewritePattern(context, benefit), predicate(std::move(predicate)) {}

private:
  LSTMDecompositionPredicate predicate;

  struct DirectionResult {
    Value y;
    Value h;
    Value c;
  };

  struct StepResult {
    Value h;
    Value c;
    Value y;
  };

  struct ActivationSpec {
    StringRef name;
    double alpha;
    double beta;
  };

  // ONNX packs W and R gates as IOFC. Its peephole input omits Cell and uses
  // the corresponding IOF prefix.
  enum class Gate : int64_t { Input, Output, Forget, Cell };

  struct LSTMConfig {
    Value x;
    Value w;
    Value r;
    Value bias;
    Value peepholes;
    Value initialH;
    Value initialC;
    Value sequenceLens;
    int64_t sequence;
    int64_t batch;
    int64_t inputSize;
    int64_t hiddenSize;
    Type elementType;
    bool inputForget;
    FloatAttr clipAttr;
  };

  struct DirectionConfig {
    int64_t index;
    bool reverse;
    ActivationSpec fActivation;
    ActivationSpec gActivation;
    ActivationSpec hActivation;
  };

  struct ClipBounds {
    Value min;
    Value max;

    explicit operator bool() const { return min && max; }
  };

  static Value scalarI64(
      OnnxBuilder &onnx, PatternRewriter &rewriter, int64_t value) {
    const auto type = RankedTensorType::get({}, rewriter.getI64Type());
    return onnx.constant(DenseElementsAttr::get(
        type, rewriter.getIntegerAttr(rewriter.getI64Type(), value)));
  }

  static Value scalarI32(
      OnnxBuilder &onnx, PatternRewriter &rewriter, int32_t value) {
    const auto type = RankedTensorType::get({}, rewriter.getI32Type());
    return onnx.constant(DenseElementsAttr::get(
        type, rewriter.getIntegerAttr(rewriter.getI32Type(), value)));
  }

  static Value zeroTensor(OnnxBuilder &onnx, PatternRewriter &rewriter,
      ArrayRef<int64_t> shape, Type elementType) {
    const auto type = RankedTensorType::get(shape, elementType);
    return onnx.constant(
        DenseElementsAttr::get(type, rewriter.getZeroAttr(elementType)));
  }

  static Value getDirectionSlice(OnnxBuilder &onnx, Value value,
      int64_t direction, ArrayRef<int64_t> resultShape) {
    const auto inputType = cast<RankedTensorType>(value.getType());
    auto slicedShape = llvm::to_vector(inputType.getShape());
    slicedShape[0] = 1;
    Value sliced = onnx.slice(
        RankedTensorType::get(slicedShape, inputType.getElementType()), value,
        direction, direction + 1);
    return onnx.squeeze(
        RankedTensorType::get(resultShape, inputType.getElementType()), sliced,
        onnx.constantInt64({0}));
  }

  static Value sliceGate(OnnxBuilder &onnx, Value packed,
      RankedTensorType gateType, Gate gate, int64_t hiddenSize) {
    const int64_t start = static_cast<int64_t>(gate) * hiddenSize;
    return onnx.slice(gateType, packed, onnx.constantInt64({start}),
        onnx.constantInt64({start + hiddenSize}), onnx.constantInt64({1}),
        onnx.constantInt64({1}));
  }

  static Value slicePeephole(OnnxBuilder &onnx, Value peepholes,
      RankedTensorType peepholeType, Gate gate, int64_t hiddenSize) {
    assert(gate != Gate::Cell && "LSTM has no cell peephole");
    const int64_t start = static_cast<int64_t>(gate) * hiddenSize;
    return onnx.slice(peepholeType, peepholes, start, start + hiddenSize);
  }

  static Value castToI32(
      OnnxBuilder &onnx, PatternRewriter &rewriter, Value input) {
    const auto type = RankedTensorType::get({}, rewriter.getI32Type());
    return onnx.cast(type, input, nullptr, TypeAttr::get(rewriter.getI32Type()),
        /*inferShape=*/false);
  }

  static bool hasStaticShape(Value value) {
    if (isNoneValue(value))
      return true;
    const auto type = dyn_cast<RankedTensorType>(value.getType());
    return type && type.hasStaticShape();
  }

  static void assertStandardShape(
      Value value, [[maybe_unused]] ArrayRef<int64_t> expectedShape) {
    if (isNoneValue(value))
      return;
    [[maybe_unused]] const auto type = cast<RankedTensorType>(value.getType());
    assert(type.getShape() == expectedShape &&
           "LSTM operand must have its specification shape");
  }

  static double activationParameter(
      ArrayAttr values, unsigned index, double defaultValue) {
    if (!values || index >= values.size())
      return defaultValue;
    return cast<FloatAttr>(values[index]).getValueAsDouble();
  }

  static double defaultAlpha(StringRef name) {
    if (name == "LeakyRelu")
      return 0.01;
    if (name == "ThresholdedRelu" || name == "Elu")
      return 1.0;
    if (name == "HardSigmoid")
      return 0.2;
    if (name == "Selu")
      return 1.6732631921768188;
    return 1.0;
  }

  static double defaultBeta(StringRef name) {
    if (name == "HardSigmoid")
      return 0.5;
    if (name == "Selu")
      return 1.0507010221481323;
    return name == "Affine" ? 0.0 : 1.0;
  }

  static ActivationSpec getActivationSpec(ArrayAttr names, ArrayAttr alphas,
      ArrayAttr betas, unsigned index, StringRef defaultName) {
    const StringRef name = names && index < names.size()
                               ? cast<StringAttr>(names[index]).getValue()
                               : defaultName;
    return {name, activationParameter(alphas, index, defaultAlpha(name)),
        activationParameter(betas, index, defaultBeta(name))};
  }

  static Value scalarFloat(OnnxBuilder &onnx, PatternRewriter &rewriter,
      Type elementType, double value) {
    const auto type = RankedTensorType::get({}, elementType);
    return onnx.constant(DenseElementsAttr::get(
        type, rewriter.getFloatAttr(elementType, value)));
  }

  static Value activate(OnnxBuilder &onnx, PatternRewriter &rewriter,
      Location loc, Value input, RankedTensorType type,
      const ActivationSpec &activation) {
    if (activation.name == "Affine") {
      Value a =
          scalarFloat(onnx, rewriter, type.getElementType(), activation.alpha);
      Value b =
          scalarFloat(onnx, rewriter, type.getElementType(), activation.beta);
      return onnx.add(onnx.mul(type, input, a), b);
    }
    if (activation.name == "ScaledTanh") {
      Value a =
          scalarFloat(onnx, rewriter, type.getElementType(), activation.alpha);
      Value b =
          scalarFloat(onnx, rewriter, type.getElementType(), activation.beta);
      Value scaled = onnx.mul(type, input, b);
      Value tanh = rewriter.create<ONNXTanhOp>(loc, type, scaled);
      return onnx.mul(type, tanh, a);
    }
    if (activation.name == "Relu")
      return rewriter.create<ONNXReluOp>(loc, type, input);
    if (activation.name == "Tanh")
      return rewriter.create<ONNXTanhOp>(loc, type, input);
    if (activation.name == "Sigmoid")
      return rewriter.create<ONNXSigmoidOp>(loc, type, input);
    if (activation.name == "LeakyRelu") {
      SmallVector<NamedAttribute> attrs{rewriter.getNamedAttr(
          "alpha", rewriter.getF32FloatAttr(activation.alpha))};
      return rewriter.create<ONNXLeakyReluOp>(loc, type, input, attrs);
    }
    if (activation.name == "ThresholdedRelu") {
      SmallVector<NamedAttribute> attrs{rewriter.getNamedAttr(
          "alpha", rewriter.getF32FloatAttr(activation.alpha))};
      return rewriter.create<ONNXThresholdedReluOp>(loc, type, input, attrs);
    }
    if (activation.name == "HardSigmoid") {
      SmallVector<NamedAttribute> attrs{
          rewriter.getNamedAttr(
              "alpha", rewriter.getF32FloatAttr(activation.alpha)),
          rewriter.getNamedAttr(
              "beta", rewriter.getF32FloatAttr(activation.beta))};
      return rewriter.create<ONNXHardSigmoidOp>(loc, type, input, attrs);
    }
    if (activation.name == "Elu") {
      SmallVector<NamedAttribute> attrs{rewriter.getNamedAttr(
          "alpha", rewriter.getF32FloatAttr(activation.alpha))};
      return rewriter.create<ONNXEluOp>(loc, type, input, attrs);
    }
    if (activation.name == "Selu") {
      SmallVector<NamedAttribute> attrs{
          rewriter.getNamedAttr(
              "alpha", rewriter.getF32FloatAttr(activation.alpha)),
          rewriter.getNamedAttr(
              "gamma", rewriter.getF32FloatAttr(activation.beta))};
      return rewriter.create<ONNXSeluOp>(loc, type, input, attrs);
    }
    if (activation.name == "Softsign")
      return rewriter.create<ONNXSoftsignOp>(loc, type, input);
    if (activation.name == "Softplus")
      return rewriter.create<ONNXSoftplusOp>(loc, type, input);
    return input;
  }

  static ClipBounds getClipBounds(OnnxBuilder &onnx, PatternRewriter &rewriter,
      Type elementType, FloatAttr clipAttr) {
    if (!clipAttr)
      return {};
    const double clip = clipAttr.getValueAsDouble();
    return {scalarFloat(onnx, rewriter, elementType, -clip),
        scalarFloat(onnx, rewriter, elementType, clip)};
  }

  static Value clip(OnnxBuilder &onnx, Value input, const ClipBounds &bounds) {
    return bounds
               ? onnx.clip(input, bounds.min, bounds.max, /*scalarType=*/true)
               : input;
  }

  static Value projectInput(OnnxBuilder &onnx, Value x, Value transposedW,
      Value combinedBias, int64_t sequence, int64_t batch, int64_t inputSize,
      int64_t hiddenSize, Type elementType) {
    // Flatten sequence and batch to fuse all timestep input projections into
    // one MatMul, then restore the timestep-major packed-gate tensor.
    const auto flatXType =
        RankedTensorType::get({sequence * batch, inputSize}, elementType);
    Value flatX =
        onnx.reshape(flatXType, x, onnx.constantInt64(flatXType.getShape()));
    const auto flatProjectedType =
        RankedTensorType::get({sequence * batch, 4 * hiddenSize}, elementType);
    Value inputProjection = onnx.matmul(flatProjectedType, flatX, transposedW);
    Value biasedProjection = onnx.add(inputProjection, combinedBias);
    const auto projectedType =
        RankedTensorType::get({sequence, batch, 4 * hiddenSize}, elementType);
    return onnx.reshape(projectedType, biasedProjection,
        onnx.constantInt64(projectedType.getShape()));
  }

  static Value selectProjectedTimestep(OnnxBuilder &onnx,
      PatternRewriter &rewriter, Location loc, Value projected,
      Value projectedBatchMajor, Value iter, Value sequenceLens, bool reverse,
      int64_t sequence, int64_t batch, int64_t hiddenSize, Type elementType) {
    const auto timestepType =
        RankedTensorType::get({batch, 4 * hiddenSize}, elementType);
    auto gather = [&](Value data, Value index) -> Value {
      return rewriter.create<ONNXGatherOp>(loc, timestepType, data, index,
          rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), 0));
    };
    if (!reverse)
      return gather(projected, iter);

    if (isNoneValue(sequenceLens)) {
      Value reverseIndex =
          onnx.sub(scalarI64(onnx, rewriter, sequence - 1), iter);
      return gather(projected, reverseIndex);
    }

    Value iterI32 = castToI32(onnx, rewriter, iter);
    Value last = onnx.sub(sequenceLens, scalarI32(onnx, rewriter, 1));
    Value unclampedIndex = onnx.sub(last, iterI32);
    Value batchIndex = onnx.max({unclampedIndex, scalarI32(onnx, rewriter, 0)});
    const auto index2DType =
        RankedTensorType::get({batch, 1}, rewriter.getI32Type());
    Value index2D =
        onnx.unsqueeze(index2DType, batchIndex, onnx.constantInt64({1}));
    const auto index3DType =
        RankedTensorType::get({batch, 1, 1}, rewriter.getI32Type());
    Value index3D =
        onnx.unsqueeze(index3DType, index2D, onnx.constantInt64({2}));
    const auto expandedIndexType = RankedTensorType::get(
        {batch, 1, 4 * hiddenSize}, rewriter.getI32Type());
    Value expandedIndex = onnx.expand(expandedIndexType, index3D,
        onnx.constantInt64(expandedIndexType.getShape()));
    const auto selectedType =
        RankedTensorType::get({batch, 1, 4 * hiddenSize}, elementType);
    Value selected = rewriter.create<ONNXGatherElementsOp>(loc, selectedType,
        projectedBatchMajor, expandedIndex,
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), 1));
    return onnx.squeeze(timestepType, selected, onnx.constantInt64({1}));
  }

  static DirectionResult decomposeDirection(OnnxBuilder &onnx,
      PatternRewriter &rewriter, Location loc, const LSTMConfig &config,
      const DirectionConfig &direction) {
    Value wd = getDirectionSlice(onnx, config.w, direction.index,
        {4 * config.hiddenSize, config.inputSize});
    Value rd = getDirectionSlice(onnx, config.r, direction.index,
        {4 * config.hiddenSize, config.hiddenSize});
    Value bd = getDirectionSlice(
        onnx, config.bias, direction.index, {8 * config.hiddenSize});
    Value pd = getDirectionSlice(
        onnx, config.peepholes, direction.index, {3 * config.hiddenSize});
    Value wt = onnx.transpose(
        RankedTensorType::get(
            {config.inputSize, 4 * config.hiddenSize}, config.elementType),
        wd, rewriter.getI64ArrayAttr({1, 0}));
    Value rt = onnx.transpose(
        RankedTensorType::get(
            {config.hiddenSize, 4 * config.hiddenSize}, config.elementType),
        rd, rewriter.getI64ArrayAttr({1, 0}));
    const auto biasType =
        RankedTensorType::get({4 * config.hiddenSize}, config.elementType);
    Value wb = onnx.slice(biasType, bd, 0, 4 * config.hiddenSize);
    Value rb =
        onnx.slice(biasType, bd, 4 * config.hiddenSize, 8 * config.hiddenSize);
    // Both bias halves are invariant across timesteps and are additive, so
    // combine them before the hoisted input projection.
    Value combinedBias = onnx.add(wb, rb);

    // The input projection has no recurrent dependency. Compute it once for
    // every timestep instead of placing it in the recurrence.
    Value projected = projectInput(onnx, config.x, wt, combinedBias,
        config.sequence, config.batch, config.inputSize, config.hiddenSize,
        config.elementType);
    const bool hasSequenceLens = !isNoneValue(config.sequenceLens);
    // A reverse LSTM with per-batch sequence lengths reads a different
    // timestep for each batch element. Keep a batch-major view for the
    // GatherElements selection in that case.
    Value projectedBatchMajor =
        direction.reverse && hasSequenceLens
            ? onnx.transpose(
                  RankedTensorType::get(
                      {config.batch, config.sequence, 4 * config.hiddenSize},
                      config.elementType),
                  projected, rewriter.getI64ArrayAttr({1, 0, 2}))
            : Value{};

    const auto peepholeType =
        RankedTensorType::get({config.hiddenSize}, config.elementType);
    Value pi =
        slicePeephole(onnx, pd, peepholeType, Gate::Input, config.hiddenSize);
    Value po =
        slicePeephole(onnx, pd, peepholeType, Gate::Output, config.hiddenSize);
    Value pf =
        slicePeephole(onnx, pd, peepholeType, Gate::Forget, config.hiddenSize);
    Value h0 = getDirectionSlice(onnx, config.initialH, direction.index,
        {config.batch, config.hiddenSize});
    Value c0 = getDirectionSlice(onnx, config.initialC, direction.index,
        {config.batch, config.hiddenSize});
    const auto stateType = RankedTensorType::get(
        {config.batch, config.hiddenSize}, config.elementType);
    Value paddingValue = hasSequenceLens
                             ? zeroTensor(onnx, rewriter, stateType.getShape(),
                                   config.elementType)
                             : Value{};
    if (hasSequenceLens) {
      const auto activeMaskType =
          RankedTensorType::get({config.batch}, rewriter.getI1Type());
      Value activeMask = rewriter.create<ONNXLessOp>(loc, activeMaskType,
          scalarI32(onnx, rewriter, 0), config.sequenceLens);
      const auto activeMask2DType =
          RankedTensorType::get({config.batch, 1}, rewriter.getI1Type());
      Value activeMask2D =
          onnx.unsqueeze(activeMask2DType, activeMask, onnx.constantInt64({1}));
      h0 = onnx.where(stateType, activeMask2D, h0, paddingValue);
      c0 = onnx.where(stateType, activeMask2D, c0, paddingValue);
    }

    auto emitStep = [&](Value iter, Value hPrev, Value cPrev) {
      Value projectedT = selectProjectedTimestep(onnx, rewriter, loc, projected,
          projectedBatchMajor, iter, config.sequenceLens, direction.reverse,
          config.sequence, config.batch, config.hiddenSize, config.elementType);
      const auto packedType = RankedTensorType::get(
          {config.batch, 4 * config.hiddenSize}, config.elementType);
      Value recurrentProjection = onnx.matmul(packedType, hPrev, rt);
      Value packed = onnx.add(projectedT, recurrentProjection);
      const auto gateType = RankedTensorType::get(
          {config.batch, config.hiddenSize}, config.elementType);
      Value rawI =
          sliceGate(onnx, packed, gateType, Gate::Input, config.hiddenSize);
      Value rawO =
          sliceGate(onnx, packed, gateType, Gate::Output, config.hiddenSize);
      Value rawF =
          sliceGate(onnx, packed, gateType, Gate::Forget, config.hiddenSize);
      Value rawC =
          sliceGate(onnx, packed, gateType, Gate::Cell, config.hiddenSize);

      Value inputPeephole = onnx.mul(gateType, pi, cPrev);
      Value forgetPeephole = onnx.mul(gateType, pf, cPrev);
      Value inputPreactivation = onnx.add(rawI, inputPeephole);
      Value forgetPreactivation = onnx.add(rawF, forgetPeephole);
      const ClipBounds clipBounds =
          getClipBounds(onnx, rewriter, config.elementType, config.clipAttr);

      Value inputGate = activate(onnx, rewriter, loc,
          clip(onnx, inputPreactivation, clipBounds), gateType,
          direction.fActivation);
      Value forgetGate =
          config.inputForget
              ? onnx.sub(scalarFloat(onnx, rewriter, config.elementType, 1.0),
                    inputGate)
              : activate(onnx, rewriter, loc,
                    clip(onnx, forgetPreactivation, clipBounds), gateType,
                    direction.fActivation);
      Value cellGate = activate(onnx, rewriter, loc,
          clip(onnx, rawC, clipBounds), gateType, direction.gActivation);
      Value retainedCell = onnx.mul(gateType, forgetGate, cPrev);
      Value inputCell = onnx.mul(gateType, inputGate, cellGate);
      Value cellState = onnx.add(retainedCell, inputCell);

      Value outputPeephole = onnx.mul(gateType, po, cellState);
      Value outputPreactivation = onnx.add(rawO, outputPeephole);
      // The output peephole is part of the output-gate preactivation, so it
      // must be included before clipping and f activation.
      Value outputGate = activate(onnx, rewriter, loc,
          clip(onnx, outputPreactivation, clipBounds), gateType,
          direction.fActivation);
      Value activatedCell = activate(
          onnx, rewriter, loc, cellState, gateType, direction.hActivation);
      Value hiddenState = onnx.mul(gateType, outputGate, activatedCell);

      if (!hasSequenceLens)
        return StepResult{hiddenState, cellState, hiddenState};

      const auto maskType =
          RankedTensorType::get({config.batch}, rewriter.getI1Type());
      Value mask = rewriter.create<ONNXLessOp>(
          loc, maskType, castToI32(onnx, rewriter, iter), config.sequenceLens);
      const auto mask2DType =
          RankedTensorType::get({config.batch, 1}, rewriter.getI1Type());
      Value mask2D = onnx.unsqueeze(mask2DType, mask, onnx.constantInt64({1}));
      Value maskedHidden = onnx.where(gateType, mask2D, hiddenState, hPrev);
      Value maskedCell = onnx.where(gateType, mask2D, cellState, cPrev);
      Value maskedOutput =
          onnx.where(gateType, mask2D, hiddenState, paddingValue);
      return StepResult{maskedHidden, maskedCell, maskedOutput};
    };

    auto restoreOriginalTimeOrder = [&](Value scan) -> Value {
      if (!direction.reverse || config.sequence == 1)
        return scan;

      // Loop scans follow recurrence order, but ONNX Y is indexed by the
      // original input timestep.
      const auto scanType = RankedTensorType::get(
          {config.sequence, config.batch, config.hiddenSize},
          config.elementType);
      Value sequenceLens;
      if (hasSequenceLens) {
        const auto lensType =
            RankedTensorType::get({config.batch}, rewriter.getI64Type());
        sequenceLens = onnx.cast(lensType, config.sequenceLens, nullptr,
            TypeAttr::get(rewriter.getI64Type()), /*inferShape=*/false);
      } else {
        sequenceLens = onnx.constantInt64(
            SmallVector<int64_t>(config.batch, config.sequence));
      }
      return onnx.reverseSequence(
          scanType, scan, sequenceLens, /*batchAxis=*/1, /*timeAxis=*/0);
    };

    if (config.sequence == 1) {
      const auto state = emitStep(scalarI64(onnx, rewriter, 0), h0, c0);
      Value scan = onnx.unsqueeze(
          RankedTensorType::get(
              {1, config.batch, config.hiddenSize}, config.elementType),
          state.y, onnx.constantInt64({0}));
      return {restoreOriginalTimeOrder(scan), state.h, state.c};
    }

    const auto scanType = RankedTensorType::get(
        {config.sequence, config.batch, config.hiddenSize}, config.elementType);
    Value loopH, loopC, loopScan;
    {
      auto loop = rewriter.create<ONNXLoopOp>(loc,
          TypeRange{stateType, stateType, scanType},
          scalarI64(onnx, rewriter, config.sequence), onnx.none(),
          ValueRange{h0, c0});
      OpBuilder::InsertionGuard guard(rewriter);
      Region &body = loop.getBody();
      Block *block = rewriter.createBlock(&body, body.end(),
          {RankedTensorType::get({}, rewriter.getI64Type()),
              RankedTensorType::get({}, rewriter.getI1Type()), stateType,
              stateType},
          {loc, loc, loc, loc});
      rewriter.setInsertionPointToStart(block);
      const auto state = emitStep(
          block->getArgument(0), block->getArgument(2), block->getArgument(3));
      const auto boolType = RankedTensorType::get({}, rewriter.getI1Type());
      Value trueValue = onnx.constant(
          DenseElementsAttr::get(boolType, rewriter.getBoolAttr(true)));
      rewriter.create<ONNXYieldOp>(
          loc, ValueRange{trueValue, state.h, state.c, state.y});
      loopH = loop.getResult(0);
      loopC = loop.getResult(1);
      loopScan = loop.getResult(2);
    }
    return {restoreOriginalTimeOrder(loopScan), loopH, loopC};
  }

  LogicalResult matchAndRewrite(
      ONNXLSTMOp op, PatternRewriter &rewriter) const override {
    if (predicate && !predicate(op.getOperation()))
      return rewriter.notifyMatchFailure(op, "caller excluded this LSTM");
    Location loc = op.getLoc();
    OnnxBuilder onnx(rewriter, loc);
    const auto xType = dyn_cast<RankedTensorType>(op.getX().getType());
    const auto wType = dyn_cast<RankedTensorType>(op.getW().getType());
    const auto rType = dyn_cast<RankedTensorType>(op.getR().getType());
    if (!xType || !wType || !rType || !xType.hasStaticShape() ||
        !wType.hasStaticShape() || !rType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "static ranked X, W and R required");
    assert(xType.getRank() == 3 && wType.getRank() == 3 &&
           rType.getRank() == 3 && "LSTM X, W and R must be rank-3 tensors");

    const StringRef direction = op.getDirection();
    assert((direction == "forward" || direction == "reverse" ||
               direction == "bidirectional") &&
           "LSTM direction must be valid");
    assert((op.getLayout() == 0 || op.getLayout() == 1) &&
           "LSTM layout must be valid");
    assert((op.getInputForget() == 0 || op.getInputForget() == 1) &&
           "LSTM input_forget must be valid");
    const bool layoutOne = op.getLayout() == 1;
    const int64_t sequence = xType.getDimSize(layoutOne ? 1 : 0);
    const int64_t batch = xType.getDimSize(layoutOne ? 0 : 1);
    const int64_t input = xType.getDimSize(2);
    const int64_t hidden = rType.getDimSize(2);
    assert((!op.getHiddenSizeAttr() || op.getHiddenSize() == hidden) &&
           "LSTM hidden_size must agree with R");
    const int64_t directions = direction == "bidirectional" ? 2 : 1;
    if (sequence < 1)
      return rewriter.notifyMatchFailure(op, "unsupported static LSTM shape");
    assert(hidden > 0 && wType.getDimSize(0) == directions &&
           wType.getDimSize(1) == 4 * hidden && wType.getDimSize(2) == input &&
           rType.getDimSize(0) == directions &&
           rType.getDimSize(1) == 4 * hidden && rType.getDimSize(2) == hidden &&
           "LSTM W and R must have their specification shapes");

    const SmallVector<int64_t> stateShape =
        layoutOne ? SmallVector<int64_t>{batch, directions, hidden}
                  : SmallVector<int64_t>{directions, batch, hidden};
    if (!hasStaticShape(op.getB()) || !hasStaticShape(op.getP()) ||
        !hasStaticShape(op.getInitialH()) ||
        !hasStaticShape(op.getInitialC()) ||
        !hasStaticShape(op.getSequenceLens()))
      return rewriter.notifyMatchFailure(
          op, "optional LSTM operands must have static standard shapes");
    assertStandardShape(op.getB(), {directions, 8 * hidden});
    assertStandardShape(op.getP(), {directions, 3 * hidden});
    assertStandardShape(op.getInitialH(), stateShape);
    assertStandardShape(op.getInitialC(), stateShape);
    if (!isNoneValue(op.getSequenceLens())) {
      [[maybe_unused]] const auto lensType =
          cast<RankedTensorType>(op.getSequenceLens().getType());
      assert(lensType.getRank() == 1 && lensType.getDimSize(0) == batch &&
             lensType.getElementType().isInteger(32) &&
             "LSTM sequence_lens must be tensor<batchxi32>");
    }
    assert((!op.getClipAttr() || op.getClipAttr().getValueAsDouble() >= 0.0) &&
           "LSTM clip must be non-negative");
    const Type elementType = xType.getElementType();
    const auto xCanonicalType =
        RankedTensorType::get({sequence, batch, input}, elementType);
    Value x = layoutOne ? onnx.transpose(xCanonicalType, op.getX(),
                              rewriter.getI64ArrayAttr({1, 0, 2}))
                        : op.getX();

    const auto allStateType =
        RankedTensorType::get({directions, batch, hidden}, elementType);
    auto normalizeState = [&](Value value) {
      if (isNoneValue(value))
        return zeroTensor(onnx, rewriter, allStateType.getShape(), elementType);
      if (!layoutOne)
        return value;
      return onnx.transpose(
          allStateType, value, rewriter.getI64ArrayAttr({1, 0, 2}));
    };
    Value initialH = normalizeState(op.getInitialH());
    Value initialC = normalizeState(op.getInitialC());

    const auto biasAllType =
        RankedTensorType::get({directions, 8 * hidden}, elementType);
    Value bias =
        isNoneValue(op.getB())
            ? zeroTensor(onnx, rewriter, biasAllType.getShape(), elementType)
            : op.getB();
    const auto peepholeAllType =
        RankedTensorType::get({directions, 3 * hidden}, elementType);
    Value peepholes = isNoneValue(op.getP())
                          ? zeroTensor(onnx, rewriter,
                                peepholeAllType.getShape(), elementType)
                          : op.getP();

    const LSTMConfig config{x, op.getW(), op.getR(), bias, peepholes, initialH,
        initialC, op.getSequenceLens(), sequence, batch, input, hidden,
        elementType, op.getInputForget() != 0, op.getClipAttr()};
    const ArrayAttr activationNames = op.getActivationsAttr();
    const ArrayAttr activationAlpha = op.getActivationAlphaAttr();
    const ArrayAttr activationBeta = op.getActivationBetaAttr();
    SmallVector<DirectionResult> results;
    for (int64_t d = 0; d < directions; ++d) {
      const auto activationBase = static_cast<unsigned>(3 * d);
      const DirectionConfig directionConfig{d, direction == "reverse" || d == 1,
          getActivationSpec(activationNames, activationAlpha, activationBeta,
              activationBase, "Sigmoid"),
          getActivationSpec(activationNames, activationAlpha, activationBeta,
              activationBase + 1, "Tanh"),
          getActivationSpec(activationNames, activationAlpha, activationBeta,
              activationBase + 2, "Tanh")};
      results.push_back(
          decomposeDirection(onnx, rewriter, loc, config, directionConfig));
    }

    SmallVector<Value> directionY, directionH, directionC;
    for (const DirectionResult &result : results) {
      directionY.push_back(onnx.unsqueeze(
          RankedTensorType::get({sequence, 1, batch, hidden}, elementType),
          result.y, onnx.constantInt64({1})));
      directionH.push_back(
          onnx.unsqueeze(RankedTensorType::get({1, batch, hidden}, elementType),
              result.h, onnx.constantInt64({0})));
      directionC.push_back(
          onnx.unsqueeze(RankedTensorType::get({1, batch, hidden}, elementType),
              result.c, onnx.constantInt64({0})));
    }
    const auto canonicalYType = RankedTensorType::get(
        {sequence, directions, batch, hidden}, elementType);
    Value canonicalY = directions == 1
                           ? directionY.front()
                           : onnx.concat(canonicalYType, directionY, 1);
    Value canonicalYH = directions == 1
                            ? directionH.front()
                            : onnx.concat(allStateType, directionH, 0);
    Value canonicalYC = directions == 1
                            ? directionC.front()
                            : onnx.concat(allStateType, directionC, 0);
    const auto layoutStateType =
        RankedTensorType::get({batch, directions, hidden}, elementType);
    Value y = layoutOne
                  ? onnx.transpose(
                        RankedTensorType::get(
                            {batch, sequence, directions, hidden}, elementType),
                        canonicalY, rewriter.getI64ArrayAttr({2, 0, 1, 3}))
                  : canonicalY;
    Value yh = layoutOne ? onnx.transpose(layoutStateType, canonicalYH,
                               rewriter.getI64ArrayAttr({1, 0, 2}))
                         : canonicalYH;
    Value yc = layoutOne ? onnx.transpose(layoutStateType, canonicalYC,
                               rewriter.getI64ArrayAttr({1, 0, 2}))
                         : canonicalYC;
    rewriter.replaceOp(
        op, {isa<NoneType>(op.getY().getType()) ? onnx.none() : y,
                isa<NoneType>(op.getYH().getType()) ? onnx.none() : yh,
                isa<NoneType>(op.getYC().getType()) ? onnx.none() : yc});
    return success();
  }
};

} // namespace

void populateDecomposeLSTMPatterns(RewritePatternSet &patterns,
    PatternBenefit benefit, LSTMDecompositionPredicate predicate) {
  patterns.insert<DecomposeLSTM>(
      patterns.getContext(), benefit, std::move(predicate));
}

} // namespace onnx_mlir
