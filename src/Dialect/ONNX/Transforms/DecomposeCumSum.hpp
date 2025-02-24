/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DecomposeEinsum.hpp - Decompose Einsum op ----------------===//
//
// This file contains declarations for for a pattern that
// decomposes onnx CumSum Op into Slice, Add and Concat Ops
//
//===----------------------------------------------------------------------===//

#ifndef DECOMPOSECUMSUM_H
#define DECOMPOSECUMSUM_H

#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {

class DecomposeCumSumPattern
    : public mlir::OpRewritePattern<mlir::ONNXCumSumOp> {
public:
  using mlir::OpRewritePattern<mlir::ONNXCumSumOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::ONNXCumSumOp cumsumOp,
      mlir::PatternRewriter &rewriter) const override;
};
} // namespace onnx_mlir
#endif // DECOMPOSECUMSUM_H
