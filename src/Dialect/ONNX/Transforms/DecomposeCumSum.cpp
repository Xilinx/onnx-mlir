/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DecomposeCumSum.cpp - Decompose CumSum op ----------------===//
//
// This file implements the decomposition of ONNX CumSum op to simpler ops (i.e.
// Slice, Add and Concat)
//
//===----------------------------------------------------------------------===//

#include "DecomposeCumSum.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"

namespace onnx_mlir {

// =============================== NOTE ========================================
//
// 1. This decomposition works only when the axis provided to CumSum Op is 0.
// In all other cases the behaviour is undefined.
//
// 2. The reason why this decomposition cannot support dynamic axis is due to
// the use of concat op. Concat Op takes axis as an attribute and not as input.
// Therefore, the axis value must be fixed at compile time of the model, hence
// preventing support for dynamic axis.
//
// 3. This is a short-term solution because if the axis along which the
// operation has to take place is large (let's say 1000), then this pass will
// spawn 1000 Slices and 1000 Adds which isn't desirable.
//
// 4. Proper solution would be to support CumSum kernel.
//
// =============================================================================

mlir::LogicalResult DecomposeCumSumPattern::matchAndRewrite(
    mlir::ONNXCumSumOp cumSumOp, mlir::PatternRewriter &rewriter) const {
  // NOTE: This decomposition is valid only for axis input of 0 for CumSum.
  // The behaviour is undefined for any other axis value!

  // get input
  auto inputVal = cumSumOp.getX();

  // get input type
  if (!mlir::isa<mlir::ShapedType>(inputVal.getType()))
    return mlir::failure();
  auto inputType = mlir::cast<mlir::ShapedType>(inputVal.getType());

  // get batchDim - batch dimension is required for decomposition
  if (!inputType.hasRank() || inputType.getDimSize(0) < 0)
    return mlir::failure();
  auto inputShape = inputType.getShape();
  auto batchDim = inputShape[0];

  // Get dialect builder to create new onnx ops
  MultiDialectBuilder<OnnxBuilder> onnxOpBuilder(
      rewriter, rewriter.getFusedLoc(cumSumOp.getLoc()));

  // Handle edge Case : If batch dimension is 1 then CumSum Op is redundant
  if (batchDim == 1) {
    rewriter.replaceOp(cumSumOp, inputVal);
    return llvm::success();
  }

  // Create constants for 'starts' and 'ends' input of slice operator
  llvm::SmallVector<mlir::Value> constVals;
  for (int i = 0; i < batchDim + 1; ++i) {
    // create const ops for slice start and end input
    auto constRawVal = llvm::SmallVector<int64_t>(1, i);
    auto constVal =
        onnxOpBuilder.onnx.constantInt64(mlir::ArrayRef(constRawVal));
    constVals.push_back(constVal);
  }

  // ------------ Create slice ops ------------

  // We slice the input tensor into single value tensors along axis 0.
  // For example - An input X with shape <8x1x384> will led to creation of 8
  // slice Ops. The ith Slice Op will take the following as inputs:
  //    - input tensor -> X : <8x1x384>
  //    - starts -> [i] : <1>
  //    - ends -> [i+1] : <1>
  //    - axis -> [0] : <1>
  //    - step -> [1] : <1>

  // create slice output type
  llvm::SmallVector<int64_t> sliceOutputShape(1);
  sliceOutputShape[0] = 1;
  sliceOutputShape.insert(
      sliceOutputShape.end(), inputShape.begin() + 1, inputShape.end());
  auto sliceOutputType = mlir::RankedTensorType::get(
      sliceOutputShape, getElementTypeOrSelf(inputType));

  // create slice step val
  auto stepRawVal = llvm::SmallVector<int64_t>(1, 1);
  auto stepVal = onnxOpBuilder.onnx.constantInt64(mlir::ArrayRef(stepRawVal));

  // create slice vals
  llvm::SmallVector<mlir::Value> sliceVals;
  for (int i = 0; i < batchDim; ++i) {
    auto sliceOp = onnxOpBuilder.onnx.slice(sliceOutputType, inputVal,
        constVals[i], constVals[i + 1], constVals[0], stepVal);
    sliceVals.push_back(sliceOp);
  }
  // -------------------------------------------

  // -------------- Create Add Ops -------------

  // After getting the slices, we Add the first two slices i.e. slice_0 and
  // slice_1 to give add_0 output.
  // Next, we create a chain of Add ops to generate the cumulative sum. For ith
  // Add op starting from i=1 to batchDim-1, the output is calculated as
  // follows. add_{i} = Add(add_{i-1}, slice_{i+1})

  llvm::SmallVector<mlir::Value> addVals;
  addVals.push_back(onnxOpBuilder.onnx.add(sliceVals[0], sliceVals[1]));
  for (int i = 1; i < batchDim - 1; ++i) {
    auto addVal = onnxOpBuilder.onnx.add(addVals[i - 1], sliceVals[i + 1]);
    addVals.push_back(addVal);
  }
  // -------------------------------------------

  // ------------- Create concat Op ------------

  // Finally, we concatenate - slice_0, add_0, add_1, .... , add_{batchDim-2}
  // to give the final output

  llvm::SmallVector<mlir::Value> concatInputs;
  concatInputs.push_back(sliceVals[0]);
  concatInputs.insert(concatInputs.end(), addVals.begin(), addVals.end());
  auto concatVal = onnxOpBuilder.onnx.concat(inputType, concatInputs, 0);
  // -------------------------------------------

  rewriter.replaceOp(cumSumOp, concatVal);
  return llvm::success();
}
} // namespace onnx_mlir
