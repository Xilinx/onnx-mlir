/*
 * SPDX-License-Identifier: Apache-2.0
 * Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
 * affiliates
 */

// Build flattened deferred constant expressions without evaluating tensors.

#include "src/Dialect/ONNX/Transforms/DeferredFoldable.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Interface/DeferredConstantOpInterface.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
#define GEN_PASS_DEF_FORMDEFERREDFOLDABLEEXPRESSIONSPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"
} // namespace onnx_mlir

namespace {

// Non-tensor ConstantLike values are compiler metadata rather than replaceable
// initializer payloads. Keep their defining operation in a deferred expression
// so dialect verifier contracts, such as TOSA's resolvable shape operands,
// remain valid inside the isolated body.
bool shouldInlineConstant(Value value) {
  return isConstLikeValue(value) && !isa<TensorType, NoneType>(value.getType());
}

// Check only direct operands: formation proceeds bottom-up, so a previously
// formed expression already represents any deeper constant computation.
bool canFormDeferredFoldableExpression(Operation *op) {
  if (!op || op->getParentOfType<ONNXDeferredFoldableOp>() ||
      dyn_cast<DeferredConstantOpInterface>(op) ||
      isa<ONNXDeferredFoldableOp, ONNXYieldOp>(op) || !isPure(op))
    return false;
  // Leave zero-operand producers as leaves; resultless operations have no
  // value to retain for later evaluation.
  if (op->getNumOperands() == 0 || op->getNumResults() == 0)
    return false;
  // The body is a flat expression DAG, not a container for nested control flow.
  if (op->getNumRegions() != 0)
    return false;
  return llvm::all_of(op->getOperands(), [](Value operand) {
    return isDeferredConstant(operand) || isConstLikeValue(operand);
  });
}

// Construction mechanics for one operation whose direct operands satisfy the
// deferred-expression contract. Matching policy belongs to the fallback
// pattern (and the checked public single-operation entry point), rather than
// this builder.
class DeferredFoldableExpressionBuilder {
public:
  // Replace the operation with one deferred expression per result.
  static FailureOr<SmallVector<Value>> build(
      Operation *op, RewriterBase &rewriter) {
    assert(canFormDeferredFoldableExpression(op) &&
           "expected an eligible direct constant expression");
    if (op->getNumResults() == 1) {
      FailureOr<Value> result = buildSingleResult(op, rewriter);
      if (failed(result))
        return failure();
      return SmallVector<Value>{*result};
    }
    return buildMultiResult(op, rewriter);
  }

private:
  // Reuse a single unshared operand wrapper only when its new captures
  // dominate it; otherwise build a new wrapper at the consuming operation.
  static ONNXDeferredFoldableOp findExtensibleRecipe(Operation *op) {
    ONNXDeferredFoldableOp extensible;
    bool hasMultipleDeferredWrappers = false;
    for (Value operand : op->getOperands()) {
      auto nested =
          dyn_cast_or_null<ONNXDeferredFoldableOp>(operand.getDefiningOp());
      if (!nested)
        continue;
      if (!extensible)
        extensible = nested;
      else if (extensible != nested)
        hasMultipleDeferredWrappers = true;
    }
    bool capturesDominateExtensible = true;
    if (extensible) {
      DominanceInfo dominance(extensible->getParentOp());
      capturesDominateExtensible =
          llvm::all_of(op->getOperands(), [&](Value operand) {
            return operand == extensible.getResult() ||
                   shouldInlineConstant(operand) ||
                   dominance.dominates(operand, extensible.getOperation());
          });
    }
    if (extensible && !hasMultipleDeferredWrappers &&
        capturesDominateExtensible &&
        llvm::all_of(extensible.getResult().getUses(),
            [&](OpOperand &use) { return use.getOwner() == op; }))
      return extensible;
    return {};
  }

  // Append the consumer to a reusable body, adding captures and updating the
  // yielded value, result type and provenance before replacing the consumer.
  static Value extendRecipe(Operation *op, ONNXDeferredFoldableOp extensible,
      RewriterBase &rewriter) {
    Block &body = extensible.getBody().front();
    auto oldYield = cast<ONNXYieldOp>(body.getTerminator());
    Value prefix = oldYield.getOperand(0);
    IRMapping mapping;
    mapping.map(extensible.getResult(), prefix);

    OpBuilder bodyBuilder(rewriter.getContext());
    bodyBuilder.setInsertionPoint(oldYield);

    DenseMap<Value, BlockArgument> captureArguments;
    for (auto [capture, argument] :
        llvm::zip_equal(extensible.getCaptures(), body.getArguments()))
      captureArguments.try_emplace(capture, argument);
    for (Value operand : op->getOperands()) {
      if (operand == extensible.getResult())
        continue;
      if (mapping.contains(operand))
        continue;
      if (shouldInlineConstant(operand)) {
        Operation *clone = bodyBuilder.clone(*operand.getDefiningOp());
        mapping.map(operand, clone->getResult(0));
        continue;
      }
      BlockArgument argument = captureArguments.lookup(operand);
      if (!argument) {
        rewriter.modifyOpInPlace(extensible, [&]() {
          extensible.getCapturesMutable().append(operand);
          argument = body.addArgument(operand.getType(), op->getLoc());
        });
        captureArguments.try_emplace(operand, argument);
      }
      mapping.map(operand, argument);
    }

    Operation *terminal = bodyBuilder.clone(*op, mapping);
    bodyBuilder.create<ONNXYieldOp>(op->getLoc(), terminal->getResult(0));
    rewriter.eraseOp(oldYield);
    rewriter.modifyOpInPlace(extensible, [&]() {
      extensible.getResult().setType(op->getResult(0).getType());
      extensible->setLoc(op->getLoc());
      for (StringRef name : {"onnx_node_name", "ResultNames"}) {
        if (Attribute attr = op->getAttr(name))
          extensible->setAttr(name, attr);
        else
          extensible->removeAttr(name);
      }
    });
    rewriter.replaceOp(op, extensible.getResult());
    return extensible.getResult();
  }

  // Collect unique external inputs in operand order. Nested expressions
  // contribute their captures because their bodies will be inlined.
  static SmallVector<Value> collectCaptures(Operation *op) {
    SmallVector<Value> captures;
    DenseMap<Value, unsigned> captureIndices;
    auto addCapture = [&](Value capture) {
      auto [it, inserted] =
          captureIndices.try_emplace(capture, captures.size());
      if (inserted)
        captures.push_back(capture);
      return it->second;
    };
    for (Value operand : op->getOperands()) {
      if (auto nested = dyn_cast_or_null<ONNXDeferredFoldableOp>(
              operand.getDefiningOp())) {
        for (Value capture : nested.getCaptures())
          addCapture(capture);
      } else if (!shouldInlineConstant(operand)) {
        addCapture(operand);
      }
    }

    return captures;
  }

  // Inline a nested body using the new wrapper's capture arguments and return
  // the mapped value that the nested body yielded.
  static Value cloneNestedRecipe(ONNXDeferredFoldableOp nested,
      ArrayRef<BlockArgument> arguments,
      const DenseMap<Value, unsigned> &captureIndices, OpBuilder &builder) {
    IRMapping mapping;
    for (auto [capture, argument] :
        llvm::zip_equal(nested.getCaptures(), nested.getBody().getArguments()))
      mapping.map(argument, arguments[captureIndices.lookup(capture)]);
    for (Operation &operation : nested.getBody().front().without_terminator())
      builder.clone(operation, mapping);
    auto yield = cast<ONNXYieldOp>(nested.getBody().front().getTerminator());
    return mapping.lookup(yield.getOperand(0));
  }

  // Build an isolated body for one result, inlining nested deferred operands
  // and cloning non-tensor metadata constants needed by body verifiers.
  static Value buildRecipeForResult(Operation *op, OpResult result,
      ArrayRef<Value> captures, bool copyResultNames, RewriterBase &rewriter) {
    DenseMap<Value, unsigned> captureIndices;
    for (auto [index, capture] : llvm::enumerate(captures))
      captureIndices.try_emplace(capture, index);
    auto deferred = rewriter.create<ONNXDeferredFoldableOp>(
        op->getLoc(), result.getType(), captures);
    if (Attribute nodeName = op->getAttr("onnx_node_name"))
      deferred->setAttr("onnx_node_name", nodeName);

    if (copyResultNames)
      if (Attribute resultNames = op->getAttr("ResultNames"))
        deferred->setAttr("ResultNames", resultNames);

    Region &body = deferred.getBody();
    Block *block = rewriter.createBlock(&body);
    SmallVector<BlockArgument> arguments;
    arguments.reserve(captures.size());
    for (Value capture : captures)
      arguments.push_back(block->addArgument(capture.getType(), op->getLoc()));

    OpBuilder bodyBuilder(rewriter.getContext());
    bodyBuilder.setInsertionPointToStart(block);
    IRMapping expressionMapping;
    for (Value operand : op->getOperands()) {
      if (expressionMapping.contains(operand))
        continue;
      if (shouldInlineConstant(operand)) {
        Operation *clone = bodyBuilder.clone(*operand.getDefiningOp());
        expressionMapping.map(operand, clone->getResult(0));
        continue;
      }
      if (auto nested = dyn_cast_or_null<ONNXDeferredFoldableOp>(
              operand.getDefiningOp())) {
        expressionMapping.map(operand,
            cloneNestedRecipe(nested, arguments, captureIndices, bodyBuilder));
        continue;
      }
      expressionMapping.map(operand, arguments[captureIndices.lookup(operand)]);
    }

    Operation *terminal = bodyBuilder.clone(*op, expressionMapping);
    bodyBuilder.create<ONNXYieldOp>(
        op->getLoc(), terminal->getResult(result.getResultNumber()));
    return deferred.getResult();
  }

  // Prefer extending an unshared operand body over cloning its computation.
  static FailureOr<Value> buildSingleResult(
      Operation *op, RewriterBase &rewriter) {
    assert(op->getNumResults() == 1);
    if (auto extensible = findExtensibleRecipe(op))
      return extendRecipe(op, extensible, rewriter);
    SmallVector<Value> captures = collectCaptures(op);
    Value replacement = buildRecipeForResult(op, op->getResult(0), captures,
        /*copyResultNames=*/true, rewriter);
    rewriter.replaceOp(op, replacement);
    return replacement;
  }

  // Give each result its own single-result wrapper. Each body retains a
  // clone of the producer and yields the corresponding result.
  static FailureOr<SmallVector<Value>> buildMultiResult(
      Operation *op, RewriterBase &rewriter) {
    assert(op->getNumResults() > 1);
    SmallVector<Value> captures = collectCaptures(op);
    SmallVector<Value> replacements;
    replacements.reserve(op->getNumResults());
    for (OpResult result : op->getResults()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);
      replacements.push_back(buildRecipeForResult(
          op, result, captures, /*copyResultNames=*/false, rewriter));
    }
    rewriter.replaceOp(op, replacements);
    return replacements;
  }
};

// Retain a pure, regionless operation with at least one operand and result when
// every direct operand is ConstantLike or a DeferredConstant. The ordinary
// fallback additionally requires a DeferredConstant operand. Greedy requeueing
// grows chains bottom-up, while the builder flattens direct deferred expression
// operands. Give this pattern lower benefit than ordinary rewrites so they can
// consume the original operation first.
class FormDeferredFoldableExpression : public RewritePattern {
public:
  FormDeferredFoldableExpression(
      MLIRContext *context, PatternBenefit benefit, bool requireDeferredSource)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        requireDeferredSource(requireDeferredSource) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (!canFormDeferredFoldableExpression(op) ||
        (requireDeferredSource &&
            llvm::none_of(op->getOperands(), isDeferredConstant)))
      return failure();
    return success(
        succeeded(DeferredFoldableExpressionBuilder::build(op, rewriter)));
  }

private:
  bool requireDeferredSource;
};

// Keep an existing deferred expression flat when ordinary formation replaces
// one of its captured producers with another deferred expression. Formation
// cannot match the outer wrapper itself, so this canonicalization inlines a
// direct deferred expression capture and deduplicates its captures.
class FlattenDeferredFoldableCapture
    : public OpRewritePattern<ONNXDeferredFoldableOp> {
public:
  FlattenDeferredFoldableCapture(MLIRContext *context, PatternBenefit benefit)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(ONNXDeferredFoldableOp deferred,
      PatternRewriter &rewriter) const override {
    ONNXDeferredFoldableOp nested;
    SmallVector<unsigned> captureIndices;
    for (auto [index, capture] : llvm::enumerate(deferred.getCaptures())) {
      auto producer = capture.getDefiningOp<ONNXDeferredFoldableOp>();
      if (!producer || (nested && producer != nested))
        continue;
      nested = producer;
      captureIndices.push_back(index);
    }
    if (!nested)
      return failure();

    Block &body = deferred.getBody().front();
    auto getOrAddCaptureArgument = [&](Value capture) -> BlockArgument {
      for (auto [index, existing] : llvm::enumerate(deferred.getCaptures()))
        if (existing == capture)
          return body.getArgument(index);

      BlockArgument argument;
      rewriter.modifyOpInPlace(deferred, [&]() {
        deferred.getCapturesMutable().append(capture);
        argument = body.addArgument(capture.getType(), deferred.getLoc());
      });
      return argument;
    };

    SmallVector<BlockArgument> oldArguments;
    for (unsigned captureIndex : captureIndices)
      oldArguments.push_back(body.getArgument(captureIndex));
    IRMapping mapping;
    for (auto [nestedCapture, nestedArgument] :
        llvm::zip_equal(nested.getCaptures(), nested.getBody().getArguments()))
      mapping.map(nestedArgument, getOrAddCaptureArgument(nestedCapture));

    OpBuilder bodyBuilder(rewriter.getContext());
    bodyBuilder.setInsertionPointToStart(&body);
    for (Operation &nestedOp : nested.getBody().front().without_terminator())
      bodyBuilder.clone(nestedOp, mapping);
    auto nestedYield =
        cast<ONNXYieldOp>(nested.getBody().front().getTerminator());
    Value replacement = mapping.lookup(nestedYield.getOperand(0));

    for (BlockArgument oldArgument : oldArguments)
      oldArgument.replaceAllUsesWith(replacement);
    rewriter.modifyOpInPlace(deferred, [&]() {
      for (unsigned captureIndex : llvm::reverse(captureIndices)) {
        deferred->eraseOperand(captureIndex);
        body.eraseArgument(captureIndex);
      }
    });
    if (nested->use_empty())
      rewriter.eraseOp(nested);
    return success();
  }
};

// Form expressions without evaluating them, preserving result-name updates.
LogicalResult formDeferredFoldableExpressions(func::FuncOp function) {
  MLIRContext *context = function.getContext();
  RewritePatternSet patterns(context);
  populateFormDeferredFoldableExpressionPatterns(
      patterns, PatternBenefit(1), /*requireDeferredSource=*/true);
  onnx_mlir::ResultNamesUpdater rnUpdater;
  if (failed(applyPatternsOutsideDeferredFoldable(function, std::move(patterns),
          &rnUpdater,
          /*enableFolding=*/false)))
    return failure();
  return success();
}

struct FormDeferredFoldableExpressionsPass
    : public onnx_mlir::impl::FormDeferredFoldableExpressionsPassBase<
          FormDeferredFoldableExpressionsPass> {
  using Base::Base;

  void runOnOperation() final {
    if (failed(formDeferredFoldableExpressions(getOperation())))
      signalPassFailure();
  }
};

} // namespace

// Keep existing and newly formed bodies outside the greedy worklist. Body
// construction uses an unobserved OpBuilder, so only outer operations are
// enqueued. Repeat the walk because formation can make a transitive user match
// even when that user was not directly modified.
LogicalResult onnx_mlir::applyPatternsOutsideDeferredFoldable(
    func::FuncOp function, RewritePatternSet &&patterns,
    RewriterBase::Listener *listener, bool enableFolding) {
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps)
      .setScope(&function.getBody())
      .setListener(listener);
  config.enableFolding(enableFolding);

  for (int64_t iteration = 0; iteration < config.getMaxIterations();
      ++iteration) {
    SmallVector<Operation *> operations;
    function.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
      if (op != function)
        operations.push_back(op);
      return isa<ONNXDeferredFoldableOp>(op) ? WalkResult::skip()
                                             : WalkResult::advance();
    });

    bool changed = false;
    if (failed(applyOpPatternsGreedily(
            operations, frozenPatterns, config, &changed)))
      return failure();
    if (!changed)
      return success();
  }
  return failure();
}

FailureOr<Value> onnx_mlir::formDeferredFoldableExpression(
    Operation *op, RewriterBase &rewriter) {
  if (!canFormDeferredFoldableExpression(op) || op->getNumResults() != 1)
    return failure();
  FailureOr<SmallVector<Value>> results =
      DeferredFoldableExpressionBuilder::build(op, rewriter);
  if (failed(results))
    return failure();
  rewriter.setInsertionPointAfter(results->front().getDefiningOp());
  return results->front();
}

void onnx_mlir::populateFormDeferredFoldableExpressionPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit,
    bool requireDeferredSource) {
  patterns.insert<FormDeferredFoldableExpression>(
      patterns.getContext(), benefit, requireDeferredSource);
  patterns.insert<FlattenDeferredFoldableCapture>(
      patterns.getContext(), PatternBenefit(benefit.getBenefit() + 1));
  ONNXDeferredFoldableOp::getCanonicalizationPatterns(
      patterns, patterns.getContext());
}
