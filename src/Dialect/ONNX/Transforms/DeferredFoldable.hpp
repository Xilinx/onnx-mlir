/*
 * SPDX-License-Identifier: Apache-2.0
 * Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
 * affiliates
 */

#ifndef ONNX_MLIR_DEFERRED_FOLDABLE_H
#define ONNX_MLIR_DEFERRED_FOLDABLE_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::func {
class FuncOp;
}

namespace onnx_mlir {

// Add a low-benefit fallback that forms deferred expressions bottom-up and
// flattens captured expression bodies. Ordinary optimization patterns run
// first. Set requireDeferredSource=false to also retain compiler-owned constant
// expressions. Run these patterns with applyPatternsOutsideDeferredFoldable
// to keep expression bodies opaque, including newly constructed bodies.
void populateFormDeferredFoldableExpressionPatterns(
    mlir::RewritePatternSet &patterns, mlir::PatternBenefit benefit,
    bool requireDeferredSource = true);

// Form or extend a deferred expression for one newly created operation. On
// success, the insertion point is immediately after the replacement producer.
// Failure means that the operation is not eligible. Body construction notifies
// the rewriter; greedy callers must exclude bodies using outer-only seeds and
// ExistingOps strictness, as applyPatternsOutsideDeferredFoldable does.
mlir::FailureOr<mlir::Value> formDeferredFoldableExpression(
    mlir::Operation *op, mlir::RewriterBase &rewriter);

// Rewrite operations nested in `function` to a fixed point, treating each
// DeferredFoldable operation as a leaf: patterns may rewrite the operation
// itself, but neither patterns nor folding visit its body.
// Use instead of mlir::applyPatternsGreedily when a pass must retain deferred
// computations. `enableFolding` controls folding of the visited operations.
// Returns failure if rewriting does not converge.
mlir::LogicalResult applyPatternsOutsideDeferredFoldable(
    mlir::func::FuncOp function, mlir::RewritePatternSet &&patterns,
    mlir::RewriterBase::Listener *listener, bool enableFolding);

} // namespace onnx_mlir
#endif
