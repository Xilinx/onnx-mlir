/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- Col2Im.cpp - ONNX Operations ---------------------===//
//
// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Col2Im operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXCol2ImOpShapeHelper::computeShape() {
  ONNXCol2ImOp col2ImOp = llvm::cast<ONNXCol2ImOp>(op);
  ONNXCol2ImOpAdaptor adaptor(operands);
  Value input = adaptor.getInput();
  Value imageShape = adaptor.getImageShape();
  Value blockShape = adaptor.getBlockShape();
  if (!hasShapeAndRank(input) || !hasShapeAndRank(imageShape) ||
      !hasShapeAndRank(blockShape))
    return failure();
  if (createIE->getShapedTypeRank(input) != 3)
    return op->emitOpError("input must have rank 3");
  if (createIE->getShapedTypeRank(imageShape) != 1 ||
      createIE->getShapedTypeRank(blockShape) != 1)
    return op->emitOpError("image_shape and block_shape must have rank 1");

  int64_t ndim = createIE->getArraySize(imageShape);
  int64_t blockRank = createIE->getArraySize(blockShape);
  if (ShapedType::isDynamic(ndim) || ShapedType::isDynamic(blockRank))
    return failure();
  if (ndim < 1 || ndim != blockRank)
    return op->emitOpError(
        "image_shape and block_shape must have the same nonzero length");

  ArrayAttr dilations = col2ImOp.getDilationsAttr();
  ArrayAttr pads = col2ImOp.getPadsAttr();
  ArrayAttr strides = col2ImOp.getStridesAttr();
  if ((dilations && createIE->getArraySize(dilations) != ndim) ||
      (pads && createIE->getArraySize(pads) != 2 * ndim) ||
      (strides && createIE->getArraySize(strides) != ndim))
    return op->emitOpError("attribute lengths must match the spatial rank");

  IndexExpr blockSize = LitIE(1);
  IndexExpr numBlocks = LitIE(1);
  DimsExpr imageDims;
  for (int64_t i = 0; i < ndim; ++i) {
    IndexExpr image = createIE->getIntFromArrayAsSymbol(imageShape, i, ndim);
    IndexExpr block = createIE->getIntFromArrayAsSymbol(blockShape, i, ndim);
    IndexExpr dilation = createIE->getIntFromArrayAsLiteral(dilations, i, 1);
    IndexExpr stride = createIE->getIntFromArrayAsLiteral(strides, i, 1);
    IndexExpr padBegin = createIE->getIntFromArrayAsLiteral(pads, i, 0);
    IndexExpr padEnd = createIE->getIntFromArrayAsLiteral(pads, i + ndim, 0);
    if (dilation.isLiteralAndSmallerThan(1) ||
        stride.isLiteralAndSmallerThan(1) ||
        padBegin.isLiteralAndSmallerThan(0) ||
        padEnd.isLiteralAndSmallerThan(0))
      return op->emitOpError(
          "dilations and strides must be positive and pads nonnegative");
    if (image.isLiteralAndSmallerThan(1) || block.isLiteralAndSmallerThan(1))
      return op->emitOpError("image_shape and block_shape must be positive");
    IndexExpr blocks = (image + padBegin + padEnd - dilation * (block - 1) - 1)
                           .floorDiv(stride) +
                       1;
    if (blocks.isLiteralAndSmallerThan(1))
      return op->emitOpError("sliding block count must be positive");
    blockSize = blockSize * block;
    numBlocks = numBlocks * blocks;
    imageDims.emplace_back(image);
  }

  IndexExpr channels = createIE->getShapeAsDim(input, 1);
  if (channels.isLiteral() && blockSize.isLiteral() &&
      channels.getLiteral() % blockSize.getLiteral() != 0)
    return op->emitOpError(
        "input dimension 1 must be divisible by the product of block_shape");
  if (createIE->getShapeAsDim(input, 2).isLiteralAndDifferentThan(numBlocks))
    return op->emitOpError(
        "input dimension 2 must equal the product of sliding block counts");

  DimsExpr outputDims = {
      createIE->getShapeAsDim(input, 0), channels.floorDiv(blockSize)};
  outputDims.append(imageDims.begin(), imageDims.end());
  setOutputDims(outputDims);
  return success();
}

template struct ONNXNonSpecificOpShapeHelper<ONNXCol2ImOp>;

} // namespace onnx_mlir

LogicalResult ONNXCol2ImOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()) || !hasShapeAndRank(getImageShape()) ||
      !hasShapeAndRank(getBlockShape()))
    return success();
  // A runtime vector length leaves the output rank unknown.
  auto imageType = llvm::cast<ShapedType>(getImageShape().getType());
  auto blockType = llvm::cast<ShapedType>(getBlockShape().getType());
  if ((imageType.getRank() == 1 && imageType.isDynamicDim(0)) ||
      (blockType.getRank() == 1 && blockType.isDynamicDim(0)))
    return success();
  ONNXCol2ImOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(
      getElementTypeOrSelf(getInput().getType()));
}
