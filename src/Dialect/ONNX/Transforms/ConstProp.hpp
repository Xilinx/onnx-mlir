/*
 * SPDX-License-Identifier: Apache-2.0
 * Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
 * affiliates
 */

#ifndef ONNX_MLIR_CONST_PROP_H
#define ONNX_MLIR_CONST_PROP_H

#include <cstdint>

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace onnx_mlir {

// Exports the ConstPropONNXToONNXPass patterns.
void getConstPropONNXToONNXPatterns(mlir::RewritePatternSet &patterns,
    bool enableQDQ = false, bool enableQuantConstFold = false,
    int64_t maxLoopUnrollCount = 64, bool enableDequantConstFold = false,
    bool enableDequantConstFoldWeights = false,
    bool formDeferredFoldableExpressions = false);

// Add the fallback that forms one operation at a time from directly constant
// or deferred operands and flattens direct recipe operands. Greedy requeueing
// grows longer expressions bottom-up. A low benefit lets ordinary optimization
// patterns consume the original operations first. Ordinary callers require a
// deferred source; a pass may temporarily normalize dense constant expressions
// too when it guarantees they are evaluated or inlined again at its boundary.
void populateFormDeferredFoldableExpressionPatterns(
    mlir::RewritePatternSet &patterns, mlir::PatternBenefit benefit,
    bool requireDeferredSource = true);

// Inline recipes whose captures are all ordinary constants. Such recipes may
// be formed temporarily inside a greedy rewrite so newly created dense
// constant expressions have the same explicit producer classification as
// initializer-dependent expressions. Recipes with deferred captures remain.
// If provided, notifyInlinedOperation is called for every operation cloned
// from an inlined recipe, in program order.
mlir::LogicalResult inlineDenseDeferredFoldableExpressions(
    mlir::Operation *root,
    llvm::function_ref<void(mlir::Operation *)> notifyInlinedOperation = {});

// Form or extend a deferred recipe for one newly created operation. On
// success, the insertion point is immediately after the replacement producer.
// Failure means that the operation is not eligible for a deferred recipe.
mlir::FailureOr<mlir::Value> formDeferredFoldableExpression(
    mlir::Operation *op, mlir::RewriterBase &rewriter);

} // namespace onnx_mlir
#endif
