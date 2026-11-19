/*
 * SPDX-License-Identifier: Apache-2.0
 * Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
 * affiliates
 */

// Form flat deferred expressions bottom-up after ordinary constant rewrites.
// Each round rewrites only operations outside existing expression bodies.
// New operations become eligible on the next round; body operations never do.
// Captured expressions are flattened, and unshared bodies can be extended
// in place to avoid repeatedly cloning a growing chain.

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

// Import a deferred body into the current block with remapped captures.
// The source may have other users, so clone rather than move its operations.
SmallVector<Value> cloneDeferredBodyInto(
    ONNXDeferredFoldableOp source, ValueRange captures, OpBuilder &builder) {
  IRMapping mapping;
  for (auto [argument, capture] :
      llvm::zip_equal(source.getBody().getArguments(), captures))
    mapping.map(argument, capture);
  for (Operation &operation : source.getBody().front().without_terminator())
    builder.clone(operation, mapping);
  auto yield = cast<ONNXYieldOp>(source.getBody().front().getTerminator());
  SmallVector<Value> results;
  for (Value value : yield.getOperands())
    results.push_back(mapping.lookup(value));
  return results;
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
  static ONNXDeferredFoldableOp findExtensibleDeferredFoldable(Operation *op) {
    ONNXDeferredFoldableOp extensible;
    bool hasMultipleDeferredWrappers = false;
    for (Value operand : op->getOperands()) {
      auto nested =
          dyn_cast_or_null<ONNXDeferredFoldableOp>(operand.getDefiningOp());
      if (nested && nested.getNumResults() != 1)
        return {};
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
            return operand == extensible.getResult(0) ||
                   shouldInlineConstant(operand) ||
                   dominance.dominates(operand, extensible.getOperation());
          });
    }
    if (extensible && !hasMultipleDeferredWrappers &&
        capturesDominateExtensible &&
        llvm::all_of(extensible.getResult(0).getUses(),
            [&](OpOperand &use) { return use.getOwner() == op; }))
      return extensible;
    return {};
  }

  // Append the consumer to a reusable body, adding captures and updating the
  // yielded value, result type and provenance before replacing the consumer.
  static Value extendDeferredFoldable(Operation *op,
      ONNXDeferredFoldableOp extensible, RewriterBase &rewriter) {
    Block &body = extensible.getBody().front();
    auto oldYield = cast<ONNXYieldOp>(body.getTerminator());
    Value prefix = oldYield.getOperand(0);
    IRMapping mapping;
    mapping.map(extensible.getResult(0), prefix);

    OpBuilder::InsertionGuard guard(rewriter);
    OpBuilder &bodyBuilder = rewriter;
    bodyBuilder.setInsertionPoint(oldYield);

    DenseMap<Value, BlockArgument> captureArguments;
    for (auto [capture, argument] :
        llvm::zip_equal(extensible.getCaptures(), body.getArguments()))
      captureArguments.try_emplace(capture, argument);
    for (Value operand : op->getOperands()) {
      if (operand == extensible.getResult(0))
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
      extensible.getResult(0).setType(op->getResult(0).getType());
      extensible->setLoc(op->getLoc());
      for (StringRef name : {"onnx_node_name", "ResultNames"}) {
        if (Attribute attr = op->getAttr(name))
          extensible->setAttr(name, attr);
        else
          extensible->removeAttr(name);
      }
    });
    rewriter.replaceOp(op, extensible.getResult(0));
    return extensible.getResult(0);
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

  // Build an isolated body for one result, inlining nested deferred operands
  // and cloning non-tensor metadata constants needed by body verifiers.
  static ONNXDeferredFoldableOp buildDeferredFoldableForResults(
      Operation *op, ArrayRef<Value> captures, RewriterBase &rewriter) {
    DenseMap<Value, unsigned> captureIndices;
    for (auto [index, capture] : llvm::enumerate(captures))
      captureIndices.try_emplace(capture, index);
    auto deferred = rewriter.create<ONNXDeferredFoldableOp>(
        op->getLoc(), op->getResultTypes(), captures);
    if (Attribute nodeName = op->getAttr("onnx_node_name"))
      deferred->setAttr("onnx_node_name", nodeName);

    if (Attribute resultNames = op->getAttr("ResultNames"))
      deferred->setAttr("ResultNames", resultNames);

    Region &body = deferred.getBody();
    Block *block = rewriter.createBlock(&body);
    SmallVector<BlockArgument> arguments;
    arguments.reserve(captures.size());
    for (Value capture : captures)
      arguments.push_back(block->addArgument(capture.getType(), op->getLoc()));

    OpBuilder::InsertionGuard guard(rewriter);
    OpBuilder &bodyBuilder = rewriter;
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
        SmallVector<Value> nestedCaptures;
        for (Value capture : nested.getCaptures())
          nestedCaptures.push_back(arguments[captureIndices.lookup(capture)]);
        SmallVector<Value> values =
            cloneDeferredBodyInto(nested, nestedCaptures, bodyBuilder);
        for (auto [result, value] :
            llvm::zip_equal(nested.getResults(), values))
          expressionMapping.map(result, value);
        continue;
      }
      expressionMapping.map(operand, arguments[captureIndices.lookup(operand)]);
    }

    Operation *terminal = bodyBuilder.clone(*op, expressionMapping);
    bodyBuilder.create<ONNXYieldOp>(op->getLoc(), terminal->getResults());
    for (Operation &operation :
        llvm::make_early_inc_range(llvm::reverse(block->without_terminator())))
      if (operation.use_empty())
        rewriter.eraseOp(&operation);
    return deferred;
  }

  // Prefer extending an unshared operand body over cloning its computation.
  static FailureOr<Value> buildSingleResult(
      Operation *op, RewriterBase &rewriter) {
    assert(op->getNumResults() == 1);
    if (auto extensible = findExtensibleDeferredFoldable(op))
      return extendDeferredFoldable(op, extensible, rewriter);
    SmallVector<Value> captures = collectCaptures(op);
    Value replacement =
        buildDeferredFoldableForResults(op, captures, rewriter).getResult(0);
    rewriter.replaceOp(op, replacement);
    return replacement;
  }

  // Retain a multi-result producer once, preserving sharing between results.
  static FailureOr<SmallVector<Value>> buildMultiResult(
      Operation *op, RewriterBase &rewriter) {
    SmallVector<Value> captures = collectCaptures(op);
    auto deferred = buildDeferredFoldableForResults(op, captures, rewriter);
    SmallVector<Value> replacements(
        deferred.getResults().begin(), deferred.getResults().end());
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
    SmallVector<Value> nestedCaptures;
    for (Value capture : nested.getCaptures())
      nestedCaptures.push_back(getOrAddCaptureArgument(capture));

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&body);
    SmallVector<Value> nestedResults =
        cloneDeferredBodyInto(nested, nestedCaptures, rewriter);
    for (auto [captureIndex, oldArgument] :
        llvm::zip_equal(captureIndices, oldArguments)) {
      unsigned resultIndex =
          cast<OpResult>(deferred.getCaptures()[captureIndex])
              .getResultNumber();
      rewriter.replaceAllUsesWith(oldArgument, nestedResults[resultIndex]);
    }
    rewriter.modifyOpInPlace(deferred, [&]() {
      for (unsigned captureIndex : llvm::reverse(captureIndices)) {
        deferred->eraseOperand(captureIndex);
        body.eraseArgument(captureIndex);
      }
    });
    for (Operation &operation :
        llvm::make_early_inc_range(llvm::reverse(body.without_terminator())))
      if (operation.use_empty())
        rewriter.eraseOp(&operation);
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

LogicalResult onnx_mlir::applyPatternsOutsideDeferredFoldable(
    func::FuncOp function, RewritePatternSet &&patterns,
    RewriterBase::Listener *listener, bool enableFolding) {
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  GreedyRewriteConfig config;
  // Restrict the selected-op greedy driver to the collected operations so
  // insertion notifications cannot add new DeferredFoldable body operations.
  config.setStrictness(GreedyRewriteStrictness::ExistingOps)
      .setScope(&function.getBody())
      .setListener(listener);
  config.enableFolding(enableFolding);

  for (int64_t iteration = 0; iteration < config.getMaxIterations();
      ++iteration) {
    // Replace the region-based greedy driver's recursive worklist collection:
    // include DeferredFoldable operations, but skip their bodies.
    SmallVector<Operation *> operations;
    function.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
      if (op != function)
        operations.push_back(op);
      return isa<ONNXDeferredFoldableOp>(op) ? WalkResult::skip()
                                             : WalkResult::advance();
    });

    // Delegate pattern application, folding and worklist processing to greedy.
    bool changed = false;
    if (failed(applyOpPatternsGreedily(
            operations, frozenPatterns, config, &changed)))
      return failure();
    if (!changed)
      return success();

    // Reproduce the region-based greedy driver's fixed-point rescanning,
    // collecting new outer operations and revisiting transitive users that
    // the selected-op driver's notifications alone may not requeue.
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
