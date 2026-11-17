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
// deferred source; consumers that support compiler-owned recipes can also
// normalize dense constant expressions by passing false. Existing recipes stay
// explicit until a caller evaluates or inlines them deliberately.
void populateFormDeferredFoldableExpressionPatterns(
    mlir::RewritePatternSet &patterns, mlir::PatternBenefit benefit,
    bool requireDeferredSource = true);

// Form or extend a deferred recipe for one newly created operation. On
// success, the insertion point is immediately after the replacement producer.
// Failure means that the operation is not eligible for a deferred recipe.
mlir::FailureOr<mlir::Value> formDeferredFoldableExpression(
    mlir::Operation *op, mlir::RewriterBase &rewriter);

} // namespace onnx_mlir
#endif
