// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "duplicate-transpose-for-each-consumer"

using namespace mlir;

namespace {

/// Give a cloned Transpose a unique ResultNames so it does not collide with the
/// original. Mirrors the xmodel/VAIP naming ("/duplicated", "/duplicated_token_N")
/// for the per-consumer copies of a shared transpose. If the names cannot be
/// rewritten as plain strings, the attribute is dropped so it can be
/// regenerated downstream.
static void renameDuplicate(
    Operation *clone, ArrayAttr origNames, unsigned copyIdx, Builder &builder) {
  std::string suffix =
      copyIdx == 0 ? "/duplicated"
                   : ("/duplicated_token_" + std::to_string(copyIdx - 1));

  if (!origNames) {
    clone->removeAttr("ResultNames");
    return;
  }

  SmallVector<Attribute> newNames;
  for (Attribute a : origNames) {
    auto s = mlir::dyn_cast<StringAttr>(a);
    if (!s) {
      // Non-trivial (e.g. nested) ResultNames: drop to avoid corrupting meta.
      clone->removeAttr("ResultNames");
      return;
    }
    newNames.push_back(builder.getStringAttr(s.getValue().str() + suffix));
  }
  clone->setAttr("ResultNames", builder.getArrayAttr(newNames));
}

/// Pattern that duplicates a Transpose that fans out to more than one consumer,
/// giving every consumer its own private Transpose. This is the inverse of
/// CombineTransposePair and reproduces the xmodel behavior where a shared
/// QKV-split transpose feeding multiple strided_slice ops is materialized once
/// per consumer.
///
/// Example:
///   %t = onnx.Transpose(%x) {perm = [2,0,1,3]}
///   use_a(%t)
///   use_b(%t)
///   use_c(%t)
/// becomes:
///   %t  = onnx.Transpose(%x) {perm = [2,0,1,3]}            // keeps use_a
///   %t1 = onnx.Transpose(%x) {perm = [2,0,1,3]}            // -> use_b
///   %t2 = onnx.Transpose(%x) {perm = [2,0,1,3]}            // -> use_c
struct DuplicateTransposePattern : public OpRewritePattern<ONNXTransposeOp> {
  using OpRewritePattern<ONNXTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXTransposeOp transposeOp, PatternRewriter &rewriter) const override {
    Value result = transposeOp.getResult();

    // Collect all uses. Nothing to do for single- (or zero-) consumer ops.
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : result.getUses())
      uses.push_back(&use);
    if (uses.size() <= 1)
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "duplicate-transpose: fan-out of " << uses.size()
               << " for " << transposeOp << "\n");

    ArrayAttr origNames =
        transposeOp->getAttrOfType<ArrayAttr>("ResultNames");

    // Keep uses[0] on the original transpose; clone for every other consumer.
    for (size_t i = 1; i < uses.size(); ++i) {
      OpOperand *use = uses[i];
      Operation *owner = use->getOwner();
      unsigned operandIdx = use->getOperandNumber();

      rewriter.setInsertionPoint(transposeOp);
      Operation *clone = rewriter.clone(*transposeOp.getOperation());
      renameDuplicate(clone, origNames, /*copyIdx=*/i - 1, rewriter);

      rewriter.modifyOpInPlace(
          owner, [&]() { owner->setOperand(operandIdx, clone->getResult(0)); });
    }

    return success();
  }
};

} // namespace

namespace onnx_mlir {

/**
 * \brief Pass to duplicate fan-out Transpose operations per consumer.
 *
 * For every onnx.Transpose whose result is consumed by more than one operation,
 * this pass creates a dedicated copy of the transpose for each extra consumer so
 * that every consumer reads from its own single-use transpose. This reproduces
 * the xmodel/VAIP layout in which a shared transpose (e.g. the QKV-split
 * transpose feeding the Q/K/V strided_slice ops in multi-head attention) is
 * materialized once per branch.
 *
 * It is the inverse of CombineTransposePair, so it must run after that pass (and
 * after CSE) and must not be followed by a transpose-merging/CSE pass.
 */
struct DuplicateTransposeForEachConsumerPass
    : public PassWrapper<DuplicateTransposeForEachConsumerPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "duplicate-transpose-for-each-consumer";
  }
  StringRef getDescription() const override {
    return "Duplicate fan-out Transpose ops so each consumer has its own copy";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DuplicateTransposePattern>(context);
    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createDuplicateTransposeForEachConsumerPass() {
  return std::make_unique<DuplicateTransposeForEachConsumerPass>();
}

} // namespace onnx_mlir
