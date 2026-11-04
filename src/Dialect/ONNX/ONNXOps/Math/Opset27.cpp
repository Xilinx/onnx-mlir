// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates.

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

LogicalResult ONNXBitCastOp::verify() {
  auto input = dyn_cast<ShapedType>(getInput().getType());
  auto result = dyn_cast<ShapedType>(getOutput().getType());
  Type to = getTo();
  if (!input || !input.getElementType().isIntOrFloat() || !to.isIntOrFloat())
    return emitOpError("requires integer or floating-point element types");
  if (input.getElementType().getIntOrFloatBitWidth() !=
      to.getIntOrFloatBitWidth())
    return emitOpError(
        "input and output element types must have equal bit width");
  if (result && result.getElementType() != to)
    return emitOpError("result element type must match 'to'");
  return success();
}

std::vector<Type> ONNXBitCastOp::resultTypeInference() {
  return {UnrankedTensorType::get(getTo())};
}

LogicalResult ONNXBitCastOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(getOperation(), getTo());
}

LogicalResult ONNXCumProdOp::verify() {
  if (getExclusive() != 0 && getExclusive() != 1)
    return emitOpError("'exclusive' must be 0 or 1");
  if (getReverse() != 0 && getReverse() != 1)
    return emitOpError("'reverse' must be 0 or 1");
  if (hasShapeAndRank(getAxis()) &&
      cast<ShapedType>(getAxis().getType()).getRank() != 0)
    return emitOpError("'axis' must be a scalar tensor");
  if (hasShapeAndRank(getX())) {
    int64_t rank = cast<ShapedType>(getX().getType()).getRank();
    if (auto axis = readScalarConstant(getAxis()); succeeded(axis))
      if (*axis < -rank || *axis >= rank)
        return emitOpError("'axis' is outside the input rank");
  }
  return success();
}

LogicalResult ONNXCumProdOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(getOperation());
}
