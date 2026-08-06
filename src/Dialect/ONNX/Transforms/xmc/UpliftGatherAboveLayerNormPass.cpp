// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// UpliftGatherAboveLayerNormPass
//
// Matches either:
//   (A) dq0 -> LayerNorm -> [q0 -> [dq1 ->]] Gather [-> q1 [-> dq2]]
//   (B) LayerNorm -> Gather ...
//
// Rewrites to uplift Gather before LayerNorm, e.g. for (A) with full Q/DQ tail:
//   Gather -> dq0 -> LayerNorm -> q0 -> dq1 -> q1 -> dq2
//
// LayerNorm normalized axis must be greater than Gather normalized axis.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <type_traits>

#include "llvm/ADT/SmallVector.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

int64_t normalizedAxis(int64_t axis, int64_t rank) {
  return onnx_mlir::getAxisInRange(axis, rank, /*includeRank=*/false);
}

bool layerNormAxisAboveGatherAxis(
    int64_t lnAxis, int64_t gatherAxis, int64_t rank) {
  return normalizedAxis(lnAxis, rank) > normalizedAxis(gatherAxis, rank);
}

SmallVector<int64_t> onnxGatherOutputShape(ArrayRef<int64_t> dataShape,
    ArrayRef<int64_t> indicesShape, int64_t axis) {
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

template <typename GatherOpTy>
RankedTensorType computeUpliftedGatherOutputType(
    GatherOpTy gatherOp, RankedTensorType dataTy, Type outElemType) {
  auto indicesTy = cast<RankedTensorType>(gatherOp.getIndices().getType());
  SmallVector<int64_t> outShape;
  if constexpr (std::is_same_v<GatherOpTy, ONNXGatherOp>) {
    outShape = onnxGatherOutputShape(
        dataTy.getShape(), indicesTy.getShape(), gatherOp.getAxis());
  } else {
    outShape = SmallVector<int64_t>(indicesTy.getShape());
  }
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
  if (auto g = dyn_cast<ONNXGatherElementsOp>(op))
    return infer(g);
  if (auto ln = dyn_cast<ONNXLayerNormalizationOp>(op))
    return infer(ln);
  if (auto dq = dyn_cast<ONNXDequantizeLinearOp>(op))
    return infer(dq);
  if (auto q = dyn_cast<ONNXQuantizeLinearOp>(op))
    return infer(q);
  return success();
}

// Keep origin-node resolution aligned with pre-uplift ONNX ops.
void copyOnnxProvenance(Operation *from, Operation *to) {
  if (!from || !to)
    return;
  to->setLoc(from->getLoc());
  if (auto name = from->getAttrOfType<StringAttr>("onnx_node_name"))
    to->setAttr("onnx_node_name", name);
  if (auto resultNames = from->getAttrOfType<ArrayAttr>("ResultNames"))
    to->setAttr("ResultNames", resultNames);
  if (auto layout = from->getAttrOfType<StringAttr>("node_layout"))
    to->setAttr("node_layout", layout);
}

struct LayerNormToGatherChain {
  ONNXLayerNormalizationOp layerNorm;
  ONNXQuantizeLinearOp q0;
  ONNXDequantizeLinearOp dq1;
};

struct GatherTailChain {
  ONNXQuantizeLinearOp q1;
  ONNXDequantizeLinearOp dq2;
};

std::optional<LayerNormToGatherChain> matchLayerNormToGather(Value gatherData) {
  Value v = gatherData;
  ONNXDequantizeLinearOp dq1 = v.getDefiningOp<ONNXDequantizeLinearOp>();
  ONNXQuantizeLinearOp q0;

  if (dq1) {
    v = dq1.getX();
    q0 = v.getDefiningOp<ONNXQuantizeLinearOp>();
    if (!q0)
      return std::nullopt;
    if (!q0.getResult().hasOneUse())
      return std::nullopt;
    v = q0.getX();
  } else if (auto q = v.getDefiningOp<ONNXQuantizeLinearOp>()) {
    q0 = q;
    if (!q0.getResult().hasOneUse())
      return std::nullopt;
    v = q0.getX();
  }

  auto layerNorm = v.getDefiningOp<ONNXLayerNormalizationOp>();
  if (!layerNorm || v != layerNorm.getY())
    return std::nullopt;

  if (q0 && !layerNorm.getY().hasOneUse())
    return std::nullopt;

  return LayerNormToGatherChain{layerNorm, q0, dq1};
}

template <typename GatherOpTy>
GatherTailChain matchOptionalGatherTail(GatherOpTy gatherOp) {
  if (!gatherOp.getResult().hasOneUse())
    return {nullptr, nullptr};

  auto q1 = dyn_cast<ONNXQuantizeLinearOp>(*gatherOp.getResult().user_begin());
  if (!q1)
    return {nullptr, nullptr};

  ONNXDequantizeLinearOp dq2 = nullptr;
  if (q1.getResult().hasOneUse())
    dq2 = dyn_cast<ONNXDequantizeLinearOp>(*q1.getResult().user_begin());
  return {q1, dq2};
}

template <typename GatherOpTy>
struct UpliftGatherAboveLayerNormPattern : public OpRewritePattern<GatherOpTy> {
  using OpRewritePattern<GatherOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      GatherOpTy gatherOp, PatternRewriter &rewriter) const override {
    auto chainOpt = matchLayerNormToGather(gatherOp.getData());
    if (!chainOpt)
      return failure();

    GatherTailChain tail = matchOptionalGatherTail(gatherOp);

    ONNXLayerNormalizationOp layerNormOp = chainOpt->layerNorm;
    ONNXQuantizeLinearOp q0 = chainOpt->q0;
    ONNXDequantizeLinearOp dq1 = chainOpt->dq1;
    ONNXQuantizeLinearOp q1 = tail.q1;
    ONNXDequantizeLinearOp dq2 = tail.dq2;

    auto dq0 = layerNormOp.getX().getDefiningOp<ONNXDequantizeLinearOp>();
    if (dq0 && !dq0->hasOneUse())
      return failure();

    auto lnInputType = dyn_cast<RankedTensorType>(layerNormOp.getX().getType());
    if (!lnInputType || !lnInputType.hasRank())
      return failure();
    int64_t lnRank = lnInputType.getRank();
    if (!layerNormAxisAboveGatherAxis(layerNormOp.getAxis(), gatherOp.getAxis(),
            lnRank))
      return failure();

    Location gatherLoc = gatherOp.getLoc();
    Type f32Type = rewriter.getF32Type();

    Value preGather = dq0 ? dq0.getX() : layerNormOp.getX();
    auto preGatherTy = dyn_cast<RankedTensorType>(preGather.getType());
    if (!preGatherTy)
      return failure();

    Type gatherOutElemType = f32Type;
    if (dq0 || q1) {
      if (q1)
        gatherOutElemType =
            cast<RankedTensorType>(q1.getType()).getElementType();
      else
        gatherOutElemType = preGatherTy.getElementType();
    } else {
      gatherOutElemType = cast<RankedTensorType>(gatherOp.getType()).getElementType();
    }

    RankedTensorType newGatherOutTy = computeUpliftedGatherOutputType(
        gatherOp, preGatherTy, gatherOutElemType);

    Operation *insertAnchor =
        dq0 ? dq0.getOperation() : layerNormOp.getOperation();
    rewriter.setInsertionPoint(insertAnchor);

    auto newGather = rewriter.create<GatherOpTy>(gatherLoc, newGatherOutTy,
        preGather, gatherOp.getIndices(), gatherOp.getAxisAttr());
    if (failed(inferShapesForOp(newGather.getOperation())))
      return failure();
    copyOnnxProvenance(gatherOp.getOperation(), newGather.getOperation());
    newGatherOutTy = cast<RankedTensorType>(newGather.getType());

    auto lnInputF32Ty =
        RankedTensorType::get(newGatherOutTy.getShape(), f32Type);
    Value lnInput;
    if (dq0) {
      auto newDq0 = rewriter.create<ONNXDequantizeLinearOp>(
          dq0.getLoc(), lnInputF32Ty, newGather.getResult(), dq0.getXScale(),
          dq0.getXZeroPoint(), dq0.getAxisAttr(), dq0.getBlockSizeAttr());
      if (failed(inferShapesForOp(newDq0.getOperation())))
        return failure();
      copyOnnxProvenance(dq0.getOperation(), newDq0.getOperation());
      lnInputF32Ty = cast<RankedTensorType>(newDq0.getType());
      lnInput = newDq0.getResult();
    } else {
      lnInput = newGather.getResult();
      lnInputF32Ty = cast<RankedTensorType>(lnInput.getType());
    }

    auto lnResultTypes = computeLayerNormResultTypes(lnInputF32Ty,
        layerNormOp.getAxis(), layerNormOp.getMean().getType(),
        layerNormOp.getInvStdDev().getType(),
        cast<RankedTensorType>(layerNormOp.getY().getType()).getElementType());
    auto newLayerNorm = rewriter.create<ONNXLayerNormalizationOp>(
        layerNormOp.getLoc(), lnResultTypes, lnInput, layerNormOp.getScale(),
        layerNormOp.getB(), layerNormOp.getAxisAttr(),
        layerNormOp.getEpsilonAttr(), layerNormOp.getStashTypeAttr());
    if (failed(inferShapesForOp(newLayerNorm.getOperation())))
      return failure();
    copyOnnxProvenance(layerNormOp.getOperation(), newLayerNorm.getOperation());

    Value afterLn = newLayerNorm.getY();
    auto reducedShape =
        cast<RankedTensorType>(newLayerNorm.getY().getType()).getShape();

    if (q0) {
      auto q0OutTy = RankedTensorType::get(reducedShape,
          cast<RankedTensorType>(q0.getType()).getElementType());
      auto newQ0 = rewriter.create<ONNXQuantizeLinearOp>(
          q0.getLoc(), q0OutTy, afterLn, q0.getYScale(), q0.getYZeroPoint(),
          q0.getAxisAttr(), q0.getBlockSizeAttr(), q0.getOutputDtypeAttr(),
          q0.getSaturateAttr());
      if (failed(inferShapesForOp(newQ0.getOperation())))
        return failure();
      copyOnnxProvenance(q0.getOperation(), newQ0.getOperation());
      afterLn = newQ0.getResult();

      if (dq1) {
        auto dq1OutTy = RankedTensorType::get(
            cast<RankedTensorType>(afterLn.getType()).getShape(), f32Type);
        rewriter.modifyOpInPlace(dq1, [&]() {
          dq1.getOperation()->setOperand(0, afterLn);
          dq1.getResult().setType(dq1OutTy);
        });
        if (failed(inferShapesForOp(dq1.getOperation())))
          return failure();
        afterLn = dq1.getResult();
      }
    }

    if (q1) {
      auto q1OutTy = RankedTensorType::get(
          cast<RankedTensorType>(afterLn.getType()).getShape(),
          cast<RankedTensorType>(q1.getType()).getElementType());
      rewriter.modifyOpInPlace(q1, [&]() {
        q1.getOperation()->setOperand(0, afterLn);
        q1.getResult().setType(q1OutTy);
      });
      if (failed(inferShapesForOp(q1.getOperation())))
        return failure();

      if (dq2) {
        auto dq2OutTy = RankedTensorType::get(
            cast<RankedTensorType>(q1.getType()).getShape(), f32Type);
        rewriter.modifyOpInPlace(dq2, [&]() {
          dq2.getOperation()->setOperand(0, q1.getResult());
          dq2.getResult().setType(dq2OutTy);
        });
        if (failed(inferShapesForOp(dq2.getOperation())))
          return failure();
      }
      rewriter.eraseOp(gatherOp);
    } else {
      rewriter.replaceOp(gatherOp, afterLn);
    }

    if (q0)
      rewriter.eraseOp(q0);
    rewriter.eraseOp(layerNormOp);
    if (dq0)
      rewriter.eraseOp(dq0);

    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct UpliftGatherAboveLayerNormPass
    : public PassWrapper<UpliftGatherAboveLayerNormPass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UpliftGatherAboveLayerNormPass)

  UpliftGatherAboveLayerNormPass() = default;
  UpliftGatherAboveLayerNormPass(const UpliftGatherAboveLayerNormPass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const override {
    return "uplift-gather-above-layernorm";
  }
  StringRef getDescription() const override {
    return "Uplift Gather/GatherElements above LayerNormalization (Q/DQ "
           "chains optional).";
  }

  Option<bool> enabled{*this, "enabled",
      llvm::cl::desc("Enable the uplift-gather-above-layernorm patterns"),
      llvm::cl::init(true)};

  void runOnOperation() override {
    if (!enabled)
      return;

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<UpliftGatherAboveLayerNormPattern<ONNXGatherOp>,
        UpliftGatherAboveLayerNormPattern<ONNXGatherElementsOp>>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createUpliftGatherAboveLayerNormPass() {
  return std::make_unique<UpliftGatherAboveLayerNormPass>();
}

} // namespace onnx_mlir
