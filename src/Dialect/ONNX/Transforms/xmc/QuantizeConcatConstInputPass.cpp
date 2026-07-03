// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//
// QuantizeConcatConstInputPass
//
// XMC pass that removes the floating-point "island" left around a Concat whose
// output (or a sibling input) is per-tensor quantized but which still has one
// or more plain f32 constant inputs (e.g. the coordinate constants fed into the
// `coord` concats of face_parsing).
//
// At this stage (after QuantTypesPass) such a Concat looks like:
//
//   %c   = onnx.Constant dense<...f32...> : tensor<1x112x112x2xf32>
//   %y   = onnx.Concat %x_q, %c
//            : (tensor<...x!quant.uniform<u8:f32, s:zp>>,
//               tensor<1x112x112x2xf32>)
//            -> tensor<...x!quant.uniform<u8:f32, s:zp>>
//
// The f32 constant is baked into a quantized constant using the concat's own
// quantization parameters, so every concat input shares the same quantized
// element type:
//
//   q = clamp(round_to_even(x / scale) + zero_point, qmin, qmax)
//
// The quantization parameters are taken from the Concat OUTPUT when it is
// per-tensor quantized, otherwise from the first per-tensor quantized sibling
// INPUT. Only per-tensor UniformQuantizedType targets are handled; per-axis
// targets and non-float constants are skipped.
//===----------------------------------------------------------------------===//

#include <cfenv>
#include <cmath>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/// Return the per-tensor UniformQuantizedType of `v`'s element type, or null.
quant::UniformQuantizedType getPerTensorQuantType(Value v) {
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType)
    return nullptr;
  return dyn_cast<quant::UniformQuantizedType>(tensorType.getElementType());
}

/// Return true if `v` is an ONNXConstantOp holding a plain floating-point
/// tensor (i.e. not already quantized).
bool isFloatConstant(Value v) {
  auto constOp = v.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return false;
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType || !tensorType.hasStaticShape())
    return false;
  return isa<FloatType>(tensorType.getElementType());
}

/// Quantize a single real value with the given quant type, rounding to the
/// nearest even integer and saturating to the storage range.
int64_t quantizeValue(double x, quant::UniformQuantizedType qType) {
  double scaled = std::nearbyint(x / qType.getScale()) +
                  static_cast<double>(qType.getZeroPoint());
  int64_t qmin = qType.getStorageTypeMin();
  int64_t qmax = qType.getStorageTypeMax();
  if (scaled <= static_cast<double>(qmin))
    return qmin;
  if (scaled >= static_cast<double>(qmax))
    return qmax;
  return static_cast<int64_t>(scaled);
}

/// Pattern: bake plain-float constant inputs of a quantized Concat into
/// quantized constants that share the concat's quantization parameters.
class QuantizeConcatConstInputPattern : public OpRewritePattern<ONNXConcatOp> {
public:
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConcatOp concatOp, PatternRewriter &rewriter) const override {
    // Determine the quantization target: prefer the concat output, then fall
    // back to the first per-tensor quantized sibling input.
    quant::UniformQuantizedType targetQType =
        getPerTensorQuantType(concatOp.getResult());
    if (!targetQType) {
      for (Value input : concatOp.getInputs()) {
        if (auto q = getPerTensorQuantType(input)) {
          targetQType = q;
          break;
        }
      }
    }
    if (!targetQType)
      return rewriter.notifyMatchFailure(
          concatOp, "No per-tensor quantized output or input to borrow from");

    auto storageType = dyn_cast<IntegerType>(targetQType.getStorageType());
    if (!storageType)
      return rewriter.notifyMatchFailure(
          concatOp, "Quant storage type is not an integer type");
    unsigned width = storageType.getWidth();

    // Collect the float-constant operands that still need quantizing.
    bool changed = false;
    SmallVector<Value> newInputs;
    newInputs.reserve(concatOp.getInputs().size());

    for (Value input : concatOp.getInputs()) {
      if (!isFloatConstant(input)) {
        newInputs.push_back(input);
        continue;
      }

      ElementsAttr elements = onnx_mlir::getElementAttributeFromONNXValue(input);
      if (!elements) {
        newInputs.push_back(input);
        continue;
      }

      auto inputType = cast<RankedTensorType>(input.getType());

      // Quantize every element with round-to-even + saturation.
      SmallVector<APInt> quantized;
      quantized.reserve(inputType.getNumElements());
      for (APFloat f : elements.getValues<APFloat>()) {
        int64_t q = quantizeValue(f.convertToDouble(), targetQType);
        quantized.emplace_back(width, static_cast<uint64_t>(q));
      }

      // The dense value uses the integer storage type; the constant's result
      // type carries the quantized element type (mirrors the QuantTypesPass
      // constant form: `dense<i8> : tensor<...x!quant.uniform<...>>`).
      auto storageTensorType =
          RankedTensorType::get(inputType.getShape(), storageType);
      auto denseAttr = DenseElementsAttr::get(storageTensorType, quantized);
      auto quantResultType =
          RankedTensorType::get(inputType.getShape(), targetQType);

      auto newConst = rewriter.create<ONNXConstantOp>(input.getLoc(),
          quantResultType, /*sparse_value=*/Attribute(), /*value=*/denseAttr,
          /*value_floats=*/nullptr, /*value_float=*/nullptr,
          /*value_ints=*/nullptr, /*value_int=*/nullptr,
          /*value_strings=*/nullptr, /*value_string=*/nullptr);
      newInputs.push_back(newConst.getResult());
      changed = true;
    }

    if (!changed)
      return rewriter.notifyMatchFailure(
          concatOp, "No plain-float constant inputs to quantize");

    rewriter.modifyOpInPlace(
        concatOp, [&]() { concatOp.getInputsMutable().assign(newInputs); });
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct QuantizeConcatConstInputPass
    : public PassWrapper<QuantizeConcatConstInputPass,
          OperationPass<func::FuncOp>> {
  [[nodiscard]] StringRef getArgument() const override {
    return "quantize-concat-const-input";
  }
  [[nodiscard]] StringRef getDescription() const override {
    return "Bake plain-float constant inputs of a per-tensor quantized Concat "
           "into quantized constants sharing the concat's quantization "
           "parameters (output QDQ preferred, else a sibling input's).";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<quant::QuantDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<QuantizeConcatConstInputPattern>(ctx);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createQuantizeConcatConstInputPass() {
  return std::make_unique<QuantizeConcatConstInputPass>();
}

} // namespace onnx_mlir
