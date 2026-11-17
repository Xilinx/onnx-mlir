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
// expressions. Existing bodies remain opaque until explicitly evaluated.
void populateFormDeferredFoldableExpressionPatterns(
    mlir::RewritePatternSet &patterns, mlir::PatternBenefit benefit,
    bool requireDeferredSource = true);

// Form or extend a deferred expression for one newly created operation. On
// success, the insertion point is immediately after the replacement producer.
// Failure means that the operation is not eligible.
mlir::FailureOr<mlir::Value> formDeferredFoldableExpression(
    mlir::Operation *op, mlir::RewriterBase &rewriter);

// Apply patterns to a fixed point without entering DeferredFoldable bodies.
// Folding can be enabled for ordinary constant propagation or disabled when
// only forming deferred expressions. Newly created expression bodies are also
// excluded from the rewrite worklist.
mlir::LogicalResult applyPatternsOutsideDeferredFoldable(
    mlir::func::FuncOp function, mlir::RewritePatternSet &&patterns,
    mlir::RewriterBase::Listener *listener, bool enableFolding);

} // namespace onnx_mlir
#endif
