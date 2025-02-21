
#ifndef DECOMPOSECUMSUM_H
#define DECOMPOSECUMSUM_H

#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {

class DecomposeCumSumPattern : public mlir::OpRewritePattern<mlir::ONNXCumSumOp> {
public:
  using mlir::OpRewritePattern<mlir::ONNXCumSumOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::ONNXCumSumOp cumsumOp, mlir::PatternRewriter &rewriter) const override;
};
}
#endif //DECOMPOSECUMSUM_H
