/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ .cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect  operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXEyeLikeOpShapeHelper::computeShape() {
  ONNXEyeLikeOpAdaptor operandAdaptor(operands);
  DimsExpr outputDims;
  createIE->getShapeAsDims(operandAdaptor.getInput(), outputDims);
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

Type ONNXEyeLikeOp::getResultElementType() {
  const auto inputType = getInput().getType().cast<TensorType>();
  if (getDtypeAttr()) {
    auto builder = OpBuilder(getContext());
    return convertONNXTypeToMLIRType(builder,
        (onnx::TensorProto_DataType)getDtypeAttr().getValue().getSExtValue());
  }
  Type elementType = inputType.getElementType();
  if (elementType.isa<NoneType>()) {
    auto builder = OpBuilder(getContext());
    elementType = convertONNXTypeToMLIRType(
        builder, onnx::TensorProto_DataType::TensorProto_DataType_FLOAT);
  }
  return elementType;
}

std::vector<Type> ONNXEyeLikeOp::resultTypeInference() {
  Type elementType = getResultElementType();
  std::vector<Type> resultTypes;

  if (auto rankedInputType =
          getInput().getType().dyn_cast<RankedTensorType>()) {
    resultTypes.push_back(rankedInputType.clone(elementType));
  } else {
    resultTypes.push_back(UnrankedTensorType::get(elementType));
  }
  return resultTypes;
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXEyeLikeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();

  Type elementType = getResultElementType();
  ONNXEyeLikeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXEyeLikeOp>;
} // namespace onnx_mlir
