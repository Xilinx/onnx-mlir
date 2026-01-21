// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <memory>

#include <llvm/Support/CommandLine.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

using namespace mlir;

namespace onnx_mlir {

struct FoldQDQPattern : public OpRewritePattern<ONNXQuantizeLinearOp> {
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qOp, PatternRewriter &rewriter) const override {

    auto dqOp = qOp.getX().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp)
      return failure();
    if (!isDequantQuantSame(dqOp, qOp))
      return failure();
    rewriter.replaceOp(qOp, dqOp.getX());
    return success();
  }
};

void getDQBinaryQPatterns(RewritePatternSet &patterns, MLIRContext *context);

class QDQCanonicalizePass
    : public PassWrapper<QDQCanonicalizePass, OperationPass<func::FuncOp>> {
public:
  Option<bool> removeBinary{*this, "remove-binary", llvm::cl::init(false)};

  StringRef getArgument() const override { return "qdq-canonicalize"; }

  QDQCanonicalizePass(bool removeBinary) {
    this->removeBinary = removeBinary;
  }

  QDQCanonicalizePass(const QDQCanonicalizePass &pass)
      : frozenPatterns(pass.frozenPatterns) {
    copyOptionValuesFrom(&pass);
  }

  LogicalResult initialize(MLIRContext *context) override {
    mlir::RewritePatternSet patterns(context);
    if (removeBinary)
      getDQBinaryQPatterns(patterns, context);
    patterns.add<FoldQDQPattern>(context);
    frozenPatterns = std::move(patterns);
    return success();
  }

  void runOnOperation() override {
    onnx_mlir::ResultNamesUpdater rnUpdater;
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns,
            GreedyRewriteConfig{.listener = &rnUpdater})))
      signalPassFailure();
  }

private:
  FrozenRewritePatternSet frozenPatterns;
};

std::unique_ptr<mlir::Pass> createQDQCanonicalizePass(
    bool removeBinary) {
  return std::make_unique<QDQCanonicalizePass>(
      removeBinary);
}

} // namespace onnx_mlir
