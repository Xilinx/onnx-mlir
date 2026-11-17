/*
 * SPDX-License-Identifier: Apache-2.0
 * Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
 * affiliates
 */

#ifndef ONNX_MLIR_CONST_PROP_H
#define ONNX_MLIR_CONST_PROP_H

#include <cstdint>

#include "src/Dialect/ONNX/Transforms/DeferredFoldable.hpp"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace onnx_mlir {

// Exports the ConstPropONNXToONNXPass patterns.
void getConstPropONNXToONNXPatterns(mlir::RewritePatternSet &patterns,
    bool enableQDQ = false, bool enableQuantConstFold = false,
    int64_t maxLoopUnrollCount = 64, bool enableDequantConstFold = false,
    bool enableDequantConstFoldWeights = false,
    bool formDeferredFoldableExpressions = false);

} // namespace onnx_mlir
#endif
