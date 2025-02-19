#include "DecomposeCumSum.hpp"

#include "../../../../../../libraries/testing/include/testing/networks/InvalidBlocks.h"

mlir::LogicalResult onnx_mlir::DecomposeCumSum::matchAndRewrite(
    mlir::ONNXCumSumOp cumSumOp, mlir::PatternRewriter &rewriter) const {
  auto inputVal = cumSumOp.getX();
  if (!inputVal) {

  }
  auto axVal = cumSumOp.getAxis();

  auto inputType = inputVal.getType();
  auto axType = axVal.getType();

  // batch dimension is required for decomposition
  if (!mlir::isa<mlir::ShapedType>(inputType))
    return mlir::failure();
  auto inputShape = mlir::cast<mlir::ShapedType>(inputType).getShape();
  assert(inputShape.size() > 0);
  auto batchDim = inputShape[0];



  return OpRewritePattern::matchAndRewrite(
      cumSumOp, rewriter);
}