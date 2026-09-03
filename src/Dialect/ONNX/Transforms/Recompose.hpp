/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXRecompose.hpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to recompose an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the recomposed operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point.
//
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_RECOMPOSE_H
#define ONNX_MLIR_RECOMPOSE_H

#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

// Exports the RecomposeONNXToONNXPass patterns. They are all plain rewrite
// patterns that can be used with any PatternRewriter, not conversion patterns.
// Always include patterns that use transpose to make unsuitable axes suitable
// for matching layernorm.
//
// `enableRotaryEmbeddingRecompose` enables a recomposition of a decomposed
// RotaryEmbedding into an onnx.RotaryEmbedding op. The targeted decomposition
// matches the RoPE in HuggingFaces LlamaRotaryEmbedding
// `enableReduceL2Recompositions` enables direct recomposition of complete
// Sqrt/Pow-based L2 reduction chains into an onnx.ReduceL2 op. Callers that
// also run the ReduceL2 decomposition in the same rewrite driver must clear
// this flag, otherwise the two patterns invert each other and never converge.
// `enableDepthToSpaceDecompose` mirrors the decompose flag: when the
// DepthToSpace decomposition is enabled, the DepthToSpace recompose patterns
// must be disabled so they do not immediately fold the decomposed
// reshape/transpose/reshape chain back into an onnx.DepthToSpace.
// `enableClipFromWhereMinMax` enables recomposing the explicit clamp idiom
// (a nested pair of onnx.Where implementing min(hi, max(lo, x))) back into a
// single onnx.Clip(x, lo, hi). Only fires when both bounds are finite
// single-element constants with lo <= hi.
void getRecomposeONNXToONNXPatterns(mlir::RewritePatternSet &patterns,
    bool enableRotaryEmbeddingRecompose = false,
    bool enableReduceL2Recompositions = false,
    bool enableDepthToSpaceDecompose = false,
    bool enableClipFromWhereMinMax = false);

// Adds only the RecomposeClipFromWhereMinMaxPattern (the clamp-from-Where
// recomposition) to `patterns`, so callers that just want this one pattern do
// not pull in the rest of the RecomposeONNXToONNX pattern set.
void getRecomposeClipFromWhereMinMaxPatterns(mlir::RewritePatternSet &patterns);

} // namespace onnx_mlir
#endif
