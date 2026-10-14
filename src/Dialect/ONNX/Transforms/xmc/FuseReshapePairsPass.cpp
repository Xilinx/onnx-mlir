// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// FuseReshapePairsPass
//
// Drops a pair of consecutive reshape-like ops (Reshape, Squeeze, Unsqueeze)
// whose combined effect is the identity.
//
// Only an exactly cancelling pair is folded. Normalizing reshape-like ops to
// Reshape, or merging non-cancelling neighbours, would hide Squeeze/Unsqueeze
// from ONNXTransposeOptimizationPass, which needs them to rewrite the
// permutation of an adjacent transpose.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/// Data operand of a reshape-like op, or null if the op is not reshape-like.
Value getReshapeLikeInput(Operation *op) {
  if (auto reshape = dyn_cast_or_null<ONNXReshapeOp>(op))
    return reshape.getData();
  if (auto squeeze = dyn_cast_or_null<ONNXSqueezeOp>(op))
    return squeeze.getData();
  if (auto unsqueeze = dyn_cast_or_null<ONNXUnsqueezeOp>(op))
    return unsqueeze.getData();
  return nullptr;
}

template <typename OpT>
struct DropCancellingPair : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      OpT op, PatternRewriter &rewriter) const override {
    Value intermediate = getReshapeLikeInput(op.getOperation());
    if (!intermediate || !intermediate.hasOneUse())
      return failure();

    Value source = getReshapeLikeInput(intermediate.getDefiningOp());
    if (!source || source.getType() != op.getResult().getType())
      return failure();

    rewriter.replaceOp(op, source);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct FuseReshapePairsPass
    : public PassWrapper<FuseReshapePairsPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "fuse-reshape-pairs"; }

  StringRef getDescription() const override {
    return "Drop consecutive reshape-like ops that cancel out";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DropCancellingPair<ONNXReshapeOp>,
        DropCancellingPair<ONNXSqueezeOp>, DropCancellingPair<ONNXUnsqueezeOp>>(
        context);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.setListener(&rnUpdater);

    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createFuseReshapePairsPass() {
  return std::make_unique<FuseReshapePairsPass>();
}

} // namespace onnx_mlir
