// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// HoistGatherAboveLayerNormPass
//
// Matches either:
//   (A) dq0 -> LayerNorm -> q0 -> dq1 -> Gather -> q1 ...
//   (B) LayerNorm -> Gather ...
//
// Rewrites to hoist Gather before LayerNorm, e.g. for (A) with full Q/DQ tail:
//   Gather -> dq0 -> LayerNorm -> q0 -> dq1 -> q1 -> ...
// or for (B) with no Q/DQ around:
//   Gather -> LayerNorm -> ...
// LayerNorm normalized axis must be greater than Gather normalized axis.
//
// Implemented as a dedicated pass (not a canonicalizer pattern) because the
// rewrite updates several ops in place, runs shape inference on each, and
// relies on ResultNamesUpdater during greedy rewriting.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using onnx_mlir::isPerTensorDequant;
using onnx_mlir::isPerTensorQuant;

namespace {

int64_t normalizedAxis(int64_t axis, int64_t rank) {
  return onnx_mlir::getAxisInRange(axis, rank, /*includeRank=*/false);
}

bool layerNormAxisAboveGatherAxis(
    int64_t lnAxis, int64_t gatherAxis, int64_t rank) {
  return normalizedAxis(lnAxis, rank) > normalizedAxis(gatherAxis, rank);
}

SmallVector<int64_t> onnxGatherOutputShape(
    ArrayRef<int64_t> dataShape, ArrayRef<int64_t> indicesShape, int64_t axis) {
  int64_t rank = dataShape.size();
  axis = normalizedAxis(axis, rank);
  SmallVector<int64_t> out;
  out.reserve(rank - 1 + indicesShape.size());
  for (int64_t i = 0; i < rank; ++i) {
    if (i == axis) {
      for (int64_t d : indicesShape)
        out.push_back(d);
    } else {
      out.push_back(dataShape[i]);
    }
  }
  return out;
}

bool gatherReducesTensorSize(ONNXGatherOp gatherOp, RankedTensorType dataTy) {
  auto indicesTy = cast<RankedTensorType>(gatherOp.getIndices().getType());
  SmallVector<int64_t> outShape = onnxGatherOutputShape(
      dataTy.getShape(), indicesTy.getShape(), gatherOp.getAxis());
  int64_t inElems = ShapedType::getNumElements(dataTy.getShape());
  int64_t outElems = ShapedType::getNumElements(outShape);
  if (ShapedType::isDynamic(inElems) || ShapedType::isDynamic(outElems))
    return static_cast<int64_t>(outShape.size()) < dataTy.getRank();
  return outElems < inElems;
}

int64_t adjustLayerNormAxisAfterGather(
    int64_t lnAxis, int64_t gatherAxis, int64_t inputRank, int64_t outputRank) {
  int64_t normGAxis = normalizedAxis(gatherAxis, inputRank);
  int64_t normLNAxis = normalizedAxis(lnAxis, inputRank);
  int64_t rankDelta = inputRank - outputRank;
  if (rankDelta > 0 && normLNAxis > normGAxis)
    normLNAxis -= rankDelta;
  if (lnAxis < 0)
    return normLNAxis - outputRank;
  return normLNAxis;
}

RankedTensorType computeHoistedGatherOutputType(
    ONNXGatherOp gatherOp, RankedTensorType dataTy, Type outElemType) {
  auto indicesTy = cast<RankedTensorType>(gatherOp.getIndices().getType());
  SmallVector<int64_t> outShape = onnxGatherOutputShape(
      dataTy.getShape(), indicesTy.getShape(), gatherOp.getAxis());
  return RankedTensorType::get(outShape, outElemType);
}

SmallVector<Type, 3> computeLayerNormResultTypes(RankedTensorType lnInputTy,
    int64_t axis, Type meanTy, Type invStdTy, Type yElemType) {
  SmallVector<int64_t> yShape(lnInputTy.getShape());
  auto yOutTy = RankedTensorType::get(yShape, yElemType);

  Type meanOutTy = meanTy;
  Type invOutTy = invStdTy;
  if (auto meanRanked = dyn_cast<RankedTensorType>(meanTy)) {
    SmallVector<int64_t> auxShape(yShape);
    int64_t rank = auxShape.size();
    axis = normalizedAxis(axis, rank);
    for (int64_t r = axis; r < rank; ++r)
      auxShape[r] = 1;
    meanOutTy = RankedTensorType::get(auxShape, meanRanked.getElementType());
    if (auto invRanked = dyn_cast<RankedTensorType>(invStdTy))
      invOutTy = RankedTensorType::get(auxShape, invRanked.getElementType());
  }

  return {yOutTy, meanOutTy, invOutTy};
}

LogicalResult inferShapesForOp(Operation *op) {
  auto infer = [](auto typedOp) -> LogicalResult {
    return typedOp.inferShapes([](Region &) {});
  };
  if (auto g = dyn_cast<ONNXGatherOp>(op))
    return infer(g);
  if (auto ln = dyn_cast<ONNXLayerNormalizationOp>(op))
    return infer(ln);
  if (auto dq = dyn_cast<ONNXDequantizeLinearOp>(op))
    return infer(dq);
  if (auto q = dyn_cast<ONNXQuantizeLinearOp>(op))
    return infer(q);
  return failure();
}

bool auxiliaryOutputUnused(Value v) {
  return v.use_empty() || isa<NoneType>(v.getType());
}

void copyOnnxProvenance(Operation *from, Operation *to) {
  if (!from || !to)
    return;
  if (auto name = from->getAttrOfType<StringAttr>("onnx_node_name"))
    to->setAttr("onnx_node_name", name);
  if (auto resultNames = from->getAttrOfType<ArrayAttr>("ResultNames"))
    to->setAttr("ResultNames", resultNames);
  if (auto layout = from->getAttrOfType<StringAttr>("node_layout"))
    to->setAttr("node_layout", layout);
}

// Backward from Gather's data operand: optional dq1 -> q0 -> LayerNorm.Y.
struct LayerNormBeforeGather {
  ONNXLayerNormalizationOp layerNorm;
  ONNXQuantizeLinearOp q0;
  ONNXDequantizeLinearOp dq1;
};

std::optional<LayerNormBeforeGather> matchLayerNormBeforeGather(
    Value gatherData) {
  Value v = gatherData;
  ONNXDequantizeLinearOp dq1 = v.getDefiningOp<ONNXDequantizeLinearOp>();
  ONNXQuantizeLinearOp q0;

  if (dq1) {
    if (!dq1.getResult().hasOneUse())
      return std::nullopt;
    if (!isPerTensorDequant(dq1))
      return std::nullopt;
    v = dq1.getX();
    q0 = v.getDefiningOp<ONNXQuantizeLinearOp>();
    if (!q0)
      return std::nullopt;
    if (!q0.getResult().hasOneUse())
      return std::nullopt;
    if (!isPerTensorQuant(q0))
      return std::nullopt;
    v = q0.getX();
  } else if (auto q = v.getDefiningOp<ONNXQuantizeLinearOp>()) {
    q0 = q;
    if (!q0.getResult().hasOneUse())
      return std::nullopt;
    if (!isPerTensorQuant(q0))
      return std::nullopt;
    v = q0.getX();
  }

  auto layerNorm = v.getDefiningOp<ONNXLayerNormalizationOp>();
  if (!layerNorm || v != layerNorm.getY())
    return std::nullopt;

  if (!layerNorm.getY().hasOneUse())
    return std::nullopt;

  if (!auxiliaryOutputUnused(layerNorm.getMean()) ||
      !auxiliaryOutputUnused(layerNorm.getInvStdDev()))
    return std::nullopt;

  return LayerNormBeforeGather{layerNorm, q0, dq1};
}

ONNXQuantizeLinearOp matchQuantizeAfterGather(ONNXGatherOp gatherOp) {
  if (!gatherOp.getResult().hasOneUse())
    return nullptr;
  auto q1 = dyn_cast<ONNXQuantizeLinearOp>(*gatherOp.getResult().user_begin());
  if (q1 && !q1.getResult().hasOneUse())
    return nullptr;
  if (q1 && !isPerTensorQuant(q1))
    return nullptr;
  return q1;
}

// (A) dq0 feeds LayerNorm X; (B) no dq0, Gather runs on LN input tensor X.
ONNXDequantizeLinearOp matchDq0BeforeLayerNorm(
    ONNXLayerNormalizationOp layerNorm) {
  return layerNorm.getX().getDefiningOp<ONNXDequantizeLinearOp>();
}

// Element type for the hoisted Gather: dq0's input (quantized X) when dq0
// exists, otherwise LayerNormalization's X input type (typically f32).
Type gatherOutputElementType(
    ONNXDequantizeLinearOp dq0, ONNXLayerNormalizationOp layerNorm) {
  Value elemTypeSource = dq0 ? dq0.getX() : layerNorm.getX();
  return cast<RankedTensorType>(elemTypeSource.getType()).getElementType();
}

LogicalResult rewireDq0ToGatherOutput(PatternRewriter &rewriter,
    ONNXDequantizeLinearOp dq0, Value gatherResult,
    RankedTensorType f32ResultTy) {
  rewriter.modifyOpInPlace(dq0, [&]() {
    dq0.getOperation()->setOperand(0, gatherResult);
    dq0.getResult().setType(f32ResultTy);
  });
  return inferShapesForOp(dq0.getOperation());
}

LogicalResult reshapeLayerNormForGatheredInput(PatternRewriter &rewriter,
    ONNXLayerNormalizationOp layerNorm, Value lnInput, int64_t newAxis,
    ArrayRef<Type> resultTypes, bool rewireXOperand) {
  rewriter.modifyOpInPlace(layerNorm, [&]() {
    if (rewireXOperand)
      layerNorm.getOperation()->setOperand(0, lnInput);
    layerNorm.setAxisAttr(
        IntegerAttr::get(layerNorm.getAxisAttr().getType(), newAxis));
    layerNorm.getY().setType(resultTypes[0]);
    layerNorm.getMean().setType(resultTypes[1]);
    layerNorm.getInvStdDev().setType(resultTypes[2]);
  });
  return inferShapesForOp(layerNorm.getOperation());
}

bool allInSameBlock(Block *block, Operation *op) {
  return !op || op->getBlock() == block;
}

bool chainInSameBlockAsGather(ONNXGatherOp gatherOp,
    ONNXLayerNormalizationOp layerNormOp, ONNXDequantizeLinearOp dq0,
    ONNXQuantizeLinearOp q0, ONNXDequantizeLinearOp dq1,
    ONNXQuantizeLinearOp q1) {
  Block *block = gatherOp->getBlock();
  return allInSameBlock(block, layerNormOp) && allInSameBlock(block, dq0) &&
         allInSameBlock(block, q0) && allInSameBlock(block, dq1) &&
         allInSameBlock(block, q1);
}

struct HoistGatherAboveLayerNormPattern
    : public OpRewritePattern<ONNXGatherOp> {
  using OpRewritePattern<ONNXGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXGatherOp gatherOp, PatternRewriter &rewriter) const override {
    if (!gatherOp.getResult().hasOneUse())
      return failure();

    auto beforeOpt = matchLayerNormBeforeGather(gatherOp.getData());
    if (!beforeOpt)
      return failure();

    ONNXLayerNormalizationOp layerNormOp = beforeOpt->layerNorm;
    ONNXQuantizeLinearOp q0 = beforeOpt->q0;
    ONNXDequantizeLinearOp dq1 = beforeOpt->dq1;
    ONNXQuantizeLinearOp q1 = matchQuantizeAfterGather(gatherOp);

    ONNXDequantizeLinearOp dq0 = matchDq0BeforeLayerNorm(layerNormOp);
    if (dq0) {
      if (!dq0->hasOneUse())
        return failure();
      if (!isPerTensorDequant(dq0))
        return failure();
    }

    if (!chainInSameBlockAsGather(gatherOp, layerNormOp, dq0, q0, dq1, q1))
      return failure();

    auto lnInputType = dyn_cast<RankedTensorType>(layerNormOp.getX().getType());
    if (!lnInputType || !lnInputType.hasRank())
      return failure();
    int64_t lnRank = lnInputType.getRank();
    if (!layerNormAxisAboveGatherAxis(
            layerNormOp.getAxis(), gatherOp.getAxis(), lnRank))
      return failure();

    // Gather input: dq0's quantized tensor for (A), LayerNorm X for (B).
    Value preGather = dq0 ? dq0.getX() : layerNormOp.getX();
    auto preGatherTy = dyn_cast<RankedTensorType>(preGather.getType());
    if (!preGatherTy)
      return failure();

    if (!gatherReducesTensorSize(gatherOp, preGatherTy))
      return failure();

    Type f32Type = rewriter.getF32Type();
    Type gatherOutElemType = gatherOutputElementType(dq0, layerNormOp);
    RankedTensorType newGatherOutTy = computeHoistedGatherOutputType(
        gatherOp, preGatherTy, gatherOutElemType);

    Operation *insertAnchor =
        dq0 ? dq0.getOperation() : layerNormOp.getOperation();
    rewriter.setInsertionPoint(insertAnchor);

    // Gather -> [dq0] -> LayerNorm -> [q0 -> dq1] -> [q1]
    auto newGather =
        rewriter.create<ONNXGatherOp>(gatherOp.getLoc(), newGatherOutTy,
            preGather, gatherOp.getIndices(), gatherOp.getAxisAttr());
    if (failed(inferShapesForOp(newGather.getOperation())))
      return failure();
    // Preserve the original Gather's ONNX metadata on the hoisted Gather so
    // downstream mapping keeps the same node identity.
    copyOnnxProvenance(gatherOp.getOperation(), newGather.getOperation());
    newGatherOutTy = cast<RankedTensorType>(newGather.getType());
    RankedTensorType gatheredF32Ty =
        RankedTensorType::get(newGatherOutTy.getShape(), f32Type);

    int64_t adjustedLnAxis =
        adjustLayerNormAxisAfterGather(layerNormOp.getAxis(),
            gatherOp.getAxis(), lnRank, newGatherOutTy.getRank());

    Value lnInput;
    if (dq0) {
      if (failed(rewireDq0ToGatherOutput(
              rewriter, dq0, newGather.getResult(), gatheredF32Ty)))
        return failure();
      lnInput = dq0.getResult();
    } else {
      lnInput = newGather.getResult();
    }

    auto lnInputF32Ty = cast<RankedTensorType>(lnInput.getType());
    auto lnResultTypes = computeLayerNormResultTypes(lnInputF32Ty,
        adjustedLnAxis, layerNormOp.getMean().getType(),
        layerNormOp.getInvStdDev().getType(),
        cast<RankedTensorType>(layerNormOp.getY().getType()).getElementType());
    if (failed(reshapeLayerNormForGatheredInput(rewriter, layerNormOp, lnInput,
            adjustedLnAxis, lnResultTypes, /*rewireXOperand=*/!dq0)))
      return failure();

    Value chainHead = layerNormOp.getY();
    auto reducedShape =
        cast<RankedTensorType>(layerNormOp.getY().getType()).getShape();

    if (q0) {
      auto q0OutTy = RankedTensorType::get(
          reducedShape, cast<RankedTensorType>(q0.getType()).getElementType());
      rewriter.modifyOpInPlace(q0, [&]() { q0.getResult().setType(q0OutTy); });
      if (failed(inferShapesForOp(q0.getOperation())))
        return failure();
      chainHead = q0.getResult();

      if (dq1) {
        auto dq1OutTy = RankedTensorType::get(
            cast<RankedTensorType>(chainHead.getType()).getShape(), f32Type);
        rewriter.modifyOpInPlace(
            dq1, [&]() { dq1.getResult().setType(dq1OutTy); });
        if (failed(inferShapesForOp(dq1.getOperation())))
          return failure();
        chainHead = dq1.getResult();
      }
    }

    if (q1) {
      auto q1OutTy = RankedTensorType::get(
          cast<RankedTensorType>(chainHead.getType()).getShape(),
          cast<RankedTensorType>(q1.getType()).getElementType());
      rewriter.modifyOpInPlace(q1, [&]() { q1.getResult().setType(q1OutTy); });
      if (failed(inferShapesForOp(q1.getOperation())))
        return failure();
    }

    // Replace the old Gather with the hoisted chain head. Do not erase Gather
    // directly when q1 is present: replaceOp keeps the rewriter and listener
    // state consistent and updates q1's operand automatically.
    rewriter.replaceOp(gatherOp, chainHead);

    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct HoistGatherAboveLayerNormPass
    : public PassWrapper<HoistGatherAboveLayerNormPass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistGatherAboveLayerNormPass)

  HoistGatherAboveLayerNormPass() = default;
  HoistGatherAboveLayerNormPass(const HoistGatherAboveLayerNormPass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const override {
    return "hoist-gather-above-layernorm";
  }
  StringRef getDescription() const override {
    return "Hoist Gather above LayerNormalization (Q/DQ chains optional).";
  }

  Option<bool> enabled{*this, "enabled",
      llvm::cl::desc("Enable the hoist-gather-above-layernorm patterns"),
      llvm::cl::init(true)};

  void runOnOperation() override {
    if (!enabled)
      return;

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<HoistGatherAboveLayerNormPattern>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createHoistGatherAboveLayerNormPass() {
  return std::make_unique<HoistGatherAboveLayerNormPass>();
}

} // namespace onnx_mlir
