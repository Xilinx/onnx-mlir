/*
 * SPDX-License-Identifier: Apache-2.0
 * Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
 * affiliates
 */

#ifndef ONNX_MLIR_CONST_PROP_H
#define ONNX_MLIR_CONST_PROP_H

#include <cstdint>

#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

// Exports the ConstPropONNXToONNXPass patterns.
void getConstPropONNXToONNXPatterns(mlir::RewritePatternSet &patterns,
    bool enableQDQ = false, bool enableQuantConstFold = false,
    int64_t maxLoopUnrollCount = 64, bool enableDequantConstFold = false,
    bool enableDequantConstFoldWeights = false,
    bool formDeferredFoldableExpressions = false);

// Add the fallback that flattens constant expressions into explicit recipes.
// A low benefit lets ordinary optimization patterns consume the original
// operations before the fallback retains what remains. Ordinary callers
// require a deferred source; a pass may temporarily normalize dense constant
// expressions too when it guarantees they are evaluated or inlined again at
// that pass boundary.
void populateFormDeferredFoldableExpressionPatterns(
    mlir::RewritePatternSet &patterns, mlir::PatternBenefit benefit,
    bool requireDeferredSource = true);

// Form or extend a deferred recipe for one newly created operation. On
// success, the insertion point is immediately after the replacement producer.
// Failure means that the operation is not eligible for a deferred recipe.
mlir::FailureOr<mlir::Value> formDeferredFoldableExpression(
    mlir::Operation *op, mlir::RewriterBase &rewriter);

// Form one flattened deferred recipe for the constant-foldable DAG ending at
// op. Unlike formDeferredFoldableExpression, this explicit construction API
// discovers pure constant-producing ancestors and preserves shared operations
// within the recipe body. Failure leaves the source IR unchanged.
mlir::FailureOr<mlir::Value> formDeferredFoldableExpressionDAG(
    mlir::Operation *op, mlir::RewriterBase &rewriter);

} // namespace onnx_mlir
#endif
