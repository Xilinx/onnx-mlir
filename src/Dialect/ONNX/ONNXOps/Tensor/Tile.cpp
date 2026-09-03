/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Tile.cpp - ONNX Operations ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Tile operation.
//
//===----------------------------------------------------------------------===//

#include "ONNXOps.hpp"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXTileOpShapeHelper::computeShape() {
  ONNXTileOpAdaptor operandAdaptor(operands);
  // Get info about input data operand.
  Value input = operandAdaptor.getInput();
  if (!hasShapeAndRank(input)) {
    return failure();
  }
  int64_t inputRank = createIE->getShapedTypeRank(input);
  Value repeats = operandAdaptor.getRepeats();
  // Compute outputDims
  DimsExpr outputDims;
  outputDims.resize(inputRank);
  for (int64_t i = 0; i < inputRank; i++) {
    IndexExpr dimInput = createIE->getShapeAsDim(input, i);
    IndexExpr repeatsValue =
        createIE->getIntFromArrayAsSymbol(repeats, i, inputRank);
    outputDims[i] = dimInput * repeatsValue;
  }
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXTileOp::verify() {
  if (!hasShapeAndRank(getInput()) || !hasShapeAndRank(getRepeats()))
    return success();

  auto inputType = mlir::cast<ShapedType>(getInput().getType());
  auto repeatsType = mlir::cast<ShapedType>(getRepeats().getType());

  // Repeats must be 1-D.
  if (repeatsType.getRank() != 1)
    return emitOpError("repeats must be a 1D tensor");

  int64_t inputRank = inputType.getRank();

  // Repeats length must match input rank.
  if (repeatsType.hasStaticShape()) {
    if (repeatsType.getDimSize(0) != inputRank)
      return emitOpError("repeats length must equal input rank");
  }

  // Verify output[i] == input[i] * repeats[i] when everything is static.
  if (!hasShapeAndRank(getOutput()))
    return success();

  auto outputType = mlir::cast<ShapedType>(getOutput().getType());
  if (outputType.getRank() != inputRank)
    return emitOpError("output rank must equal input rank");

  if (!inputType.hasStaticShape() || !outputType.hasStaticShape())
    return success();

  SmallVector<int64_t, 4> repeatsVals;
  if (!getI64ValuesFromONNXConstantOp(getRepeats(), repeatsVals))
    return success();

  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  for (int64_t i = 0; i < inputRank; ++i) {
    if (repeatsVals[i] < 0)
      return emitOpError("repeats values must be non-negative");

    int64_t expected = inputShape[i] * repeatsVals[i];
    if (outputShape[i] != expected)
      return emitOpError("output dimension ")
             << i << " must be " << inputShape[i] << " * " << repeatsVals[i]
             << ", got " << outputShape[i];
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXTileOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getInput()) || !hasShapeAndRank(getRepeats()))
    return success();

  // 'repeats' tensor is an 1D tensor.
  auto repeatsTensorTy = mlir::cast<RankedTensorType>(getRepeats().getType());
  if (repeatsTensorTy.getShape().size() != 1)
    return emitError("Repeats tensor must have rank one");

  Type elementType =
      mlir::cast<ShapedType>(getInput().getType()).getElementType();
  ONNXTileOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXTileOp>;
} // namespace onnx_mlir
