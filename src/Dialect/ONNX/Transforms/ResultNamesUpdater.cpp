// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <deque>
#include <iterator>
#include <memory>
#include <unordered_set>

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "src/Dialect/ONNX/TensorName.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

template <>
struct std::hash<Value> {
  size_t operator()(Value value) const { return hash_value(value); }
};

namespace onnx_mlir {

namespace {

void inferTensorNames(ValueRange replOperands) {
  // Collect the values that don't have TensorNames
  std::unordered_set<Value> workList;
  {
    std::deque<Value> stack(replOperands.begin(), replOperands.end());

    // Process stack entries, adding values to worklist
    while (!stack.empty()) {
      Value value = stack.front();
      stack.pop_front();
      if (!TensorName(value) && workList.insert(value).second) {
        if (Operation *defOp = value.getDefiningOp())
          for (Value operand : defOp->getOperands())
            stack.push_back(operand);
      }
    }
  }

  // Process worklist
  size_t wlen;
  do {
    wlen = workList.size();
    for (Value value : workList) {
      if (auto tname = TensorName::infer(value)) {
        workList.erase(value);
        break;
      }
    }
  } while (workList.size() > 0 && wlen > workList.size());
}

bool hasNameAndManyUses(Value value) {
  if (!TensorName(value))
    return false;
  // Count only live uses. A use whose owner is already trivially dead (but not
  // yet DCE'd by the greedy driver) is not a real consumer, so ignore it. This
  // mirrors the driver's own O(1) deadness check and avoids a transient dead
  // use blocking ResultNames propagation onto a producer.
  int liveUses = 0;
  for (OpOperand &use : value.getUses())
    if (!isOpTriviallyDead(use.getOwner()))
      ++liveUses;
  return liveUses > 1;
}

// True if `value` currently carries a MultiUseConflict transform in its name.
bool hasConflictMarker(Value value) {
  TensorName tn(value);
  return tn && llvm::any_of(tn.getTransforms(),
                   [](Transform *t) { return isa<MultiUseConflict>(t); });
}

} // namespace

void ResultNamesUpdater::notifyOperationReplaced(
    Operation *op, Operation *replacement) {
  if (!op->hasAttrOfType<ArrayAttr>("ResultNames"))
    return;

  // Don't overwrite the replacement's ResultNames if it already has a name with
  // many uses, or if it is a flagged conflict producer. A MultiUseConflict
  // marker means "this producer's name must be preserved"; it can be
  // momentarily single-use during greedy rewriting (e.g. right before a
  // transpose is folded onto it), so guard on the marker directly rather than
  // only on the live-use count.
  if (llvm::any_of(replacement->getResults(), hasNameAndManyUses) ||
      llvm::any_of(replacement->getResults(), hasConflictMarker))
    return;

  // First, copy the ResultNames attribute for the last value
  auto resultNamesArray = op->getAttrOfType<ArrayAttr>("ResultNames");
  replacement->setAttr("ResultNames", resultNamesArray);

  // Infer the TensorNames for defining values
  inferTensorNames(replacement->getOperands());
}

void ResultNamesUpdater::notifyOperationReplaced(
    Operation *op, ValueRange replacement) {
  if (!op->hasAttrOfType<ArrayAttr>("ResultNames"))
    return;

  // If the op is replaced by a single op, use the simpler method
  if (Operation *replSingleOp = replacement.front().getDefiningOp();
      replSingleOp && replSingleOp->getResults() == replacement)
    return notifyOperationReplaced(op, replSingleOp);

  // If replacements have existing name and many uses, or carry a conflict
  // marker, don't update ResultNames (see the single-op overload above).
  if (llvm::any_of(replacement, hasNameAndManyUses) ||
      llvm::any_of(replacement, hasConflictMarker))
    return;

  // First, copy the ResultNames attribute for the last value
  auto resultNamesArray = op->getAttrOfType<ArrayAttr>("ResultNames");
  MLIRContext *ctx = op->getContext();
  for (auto [name, value] : llvm::zip_equal(resultNamesArray, replacement)) {
    if (OpResult replResult = dyn_cast<OpResult>(value)) {
      Operation *replOp = replResult.getOwner();

      // Get new or existing ResultNames
      SmallVector<Attribute> replResultNames(
          replOp->getNumResults(), StringAttr::get(ctx));
      if (auto existing = replOp->getAttrOfType<ArrayAttr>("ResultNames"))
        replResultNames = SmallVector<Attribute>(existing.getValue());

      // Replace the ResultName of current result
      replResultNames[replResult.getResultNumber()] = name;
      replOp->setAttr("ResultNames", ArrayAttr::get(ctx, replResultNames));
    }
  }

  // Infer the TensorNames of defining values
  SmallVector<Value> inferenceVals;
  for (Value value : replacement) {
    if (Operation *defOp = value.getDefiningOp())
      inferenceVals.insert(
          inferenceVals.end(), defOp->operand_begin(), defOp->operand_end());
  }
  inferTensorNames(inferenceVals);
}

class InferTensorNamesPass
    : public PassWrapper<InferTensorNamesPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "onnx-infer-tensornames"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func->walk([](Operation *op) {
      for (auto result : op->getResults())
        TensorName::infer(result);
    });
  }
};

std::unique_ptr<mlir::Pass> createInferTensorNames() {
  return std::make_unique<InferTensorNamesPass>();
}

// Canonicalizer that attaches a ResultNamesUpdater listener so that
// ResultNames attributes survive op replacements during canonicalization.
struct CanonicalizeWithResultNamesPass
    : public mlir::PassWrapper<CanonicalizeWithResultNamesPass,
          mlir::OperationPass<func::FuncOp>> {
  llvm::StringRef getArgument() const override {
    return "canonicalize-with-rn";
  }

  llvm::StringRef getDescription() const override {
    return "Canonicalizer pass that preserves ResultNames attributes";
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    for (auto *dialect : ctx->getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    for (auto regOp : ctx->getRegisteredOperations())
      regOp.getCanonicalizationPatterns(patterns, ctx);

    if (isQDQDataMovementCanonicalizationEnabled())
      populateQDQDataMovementCanonicalizationPatterns(patterns);
    if (isPositiveAxisCanonicalizationEnabled())
      populateONNXPositiveAxisCanonicalizationPatterns(patterns);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.setListener(&rnUpdater);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> createCanonicalizeWithResultNamesPass() {
  return std::make_unique<CanonicalizeWithResultNamesPass>();
}

} // namespace onnx_mlir
