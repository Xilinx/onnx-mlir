
#ifndef DECOMPOSECUMSUM_H
#define DECOMPOSECUMSUM_H

#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {

class DecomposeCumSum : public mlir::OpRewritePattern<mlir::ONNXCumSumOp> {
public:
  mlir::LogicalResult matchAndRewrite(mlir::ONNXCumSumOp cumsumOp, mlir::PatternRewriter &rewriter) const override;
};
}
#endif //DECOMPOSECUMSUM_H
