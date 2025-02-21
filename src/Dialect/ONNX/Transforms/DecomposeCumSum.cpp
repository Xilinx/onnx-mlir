#include "DecomposeCumSum.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"

mlir::LogicalResult onnx_mlir::DecomposeCumSumPattern::matchAndRewrite(
    mlir::ONNXCumSumOp cumSumOp, mlir::PatternRewriter &rewriter) const {
  // NOTE: This decomposition is valid only for axis input of 0 for CumSum.
  // The behaviour is undefined for any other axis!

  // get inputs
  auto inputVal = cumSumOp.getX();
  auto axVal = cumSumOp.getAxis();

  // get input type
  auto inputType = inputVal.getType();

  // batch dimension is required for decomposition
  if (!mlir::isa<mlir::ShapedType>(inputType))
    return mlir::failure();
  // get batchDim
  auto inputShape = mlir::cast<mlir::ShapedType>(inputType).getShape();
  assert(inputShape.size() > 0);
  auto batchDim = inputShape[0];

  // Handle edge Case : If batch dimension is 1 then CumSum Op is redundant
  if (batchDim == 1) {
    rewriter.eraseOp(cumSumOp.getOperation());
    return llvm::success();
  }

  // Get dialect builder to create new onnx ops
  MultiDialectBuilder<OnnxBuilder> onnxOpBuilder(rewriter, rewriter.getFusedLoc(cumSumOp.getLoc()));

  // Create constants for 'starts' and 'ends' input for slice operator
  llvm::SmallVector<mlir::Value> constVals;
  for (int i=0; i < batchDim + 1; ++i) {
    // create const ops for slice start and end input
    auto constRawVal = llvm::SmallVector<int64_t>(1, i);
    auto constVal = onnxOpBuilder.onnx.constantInt64(mlir::ArrayRef(constRawVal));
    constVals.push_back(constVal);
  }

  // ------------ Create slice ops ------------
  // create slice output type
  llvm::SmallVector<int64_t> sliceOutputShape(1);
  sliceOutputShape[0] = 1;
  sliceOutputShape.insert(sliceOutputShape.end(), inputShape.begin() + 1, inputShape.end());
  auto sliceOutputType = mlir::RankedTensorType::get(sliceOutputShape, getElementTypeOrSelf(inputType));

  // create slice step val
  auto stepRawVal = llvm::SmallVector<int64_t>(1, 1);
  auto stepVal = onnxOpBuilder.onnx.constantInt64(mlir::ArrayRef(stepRawVal));

  // create slice vals
  llvm::SmallVector<mlir::Value> sliceVals;
  for (int i=0; i < batchDim; ++i) {
    auto sliceOp = onnxOpBuilder.onnx.slice(sliceOutputType, inputVal, constVals[i], constVals[i+1], axVal, stepVal);
    sliceVals.push_back(sliceOp);
  }
  // -------------------------------------------

  // -------------- Create Add Ops -------------
  llvm::SmallVector<mlir::Value> addVals;
  addVals.push_back(onnxOpBuilder.onnx.add(sliceVals[0], sliceVals[1]));
  for (int i=1; i<batchDim-1; ++i) {
    auto addVal = onnxOpBuilder.onnx.add(addVals[i-1], sliceVals[i+1]);
    addVals.push_back(addVal);
  }
  // -------------------------------------------

  // ------------- Create concat Op ------------
  llvm::SmallVector<mlir::Value> concatInputs;
  concatInputs.push_back(sliceVals[0]);
  concatInputs.insert(concatInputs.end(), addVals.begin(), addVals.end());
  auto concatVal = onnxOpBuilder.onnx.concat(inputType, concatInputs, 0);
  // -------------------------------------------

  rewriter.replaceOp(cumSumOp, concatVal);
  return llvm::success();
}