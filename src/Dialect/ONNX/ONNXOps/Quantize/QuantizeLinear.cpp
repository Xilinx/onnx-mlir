/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ QuantizeLinear.cpp.cpp - ONNX Operations ----------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect QuantizeLinear operation.
//
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
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
LogicalResult ONNXQuantizeLinearOpShapeHelper::computeShape() {
  ONNXQuantizeLinearOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  DimsExpr outputDims;
  createIE->getShapeAsDims(operandAdaptor.getX(), outputDims);
  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//
LogicalResult ONNXQuantizeLinearOp::verify() {
  switch (getOutputDtype()) {
  case onnx::TensorProto::UNDEFINED:
  case onnx::TensorProto::UINT8:
  case onnx::TensorProto::INT8:
  case onnx::TensorProto::UINT16:
  case onnx::TensorProto::INT16:
  case onnx::TensorProto::UINT4:
  case onnx::TensorProto::INT4:
  case onnx::TensorProto::FLOAT8E4M3FN:
  case onnx::TensorProto::FLOAT8E4M3FNUZ:
  case onnx::TensorProto::FLOAT8E5M2:
  case onnx::TensorProto::FLOAT8E5M2FNUZ:
    break;
  default:
    return emitOpError("output_dtype is not a supported quantized data type");
  }
  if (getPrecision() != 0 && getPrecision() != onnx::TensorProto::FLOAT &&
      getPrecision() != onnx::TensorProto::FLOAT16 &&
      getPrecision() != onnx::TensorProto::BFLOAT16)
    return emitOpError("precision must be 0, FLOAT, FLOAT16 or BFLOAT16");
  return verifyLinearQuantizationParameterShapes(getOperation(), getX(),
      getYScale(), getYZeroPoint(), getAxis(), getBlockSize());
}
//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXQuantizeLinearOp::resultTypeInference() {
  Type type;
  if (getOutputDtype()) {
    OpBuilder builder(getContext());
    type = convertONNXTypeToMLIRType(
        builder, static_cast<onnx::TensorProto_DataType>(getOutputDtype()));
  } else if (!isNoneValue(getYZeroPoint())) {
    type = cast<ShapedType>(getYZeroPoint().getType()).getElementType();
  } else {
    type = IntegerType::get(getContext(), 8, IntegerType::Unsigned);
  }
  return {UnrankedTensorType::get(type)};
}

LogicalResult ONNXQuantizeLinearOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto inTy = mlir::dyn_cast<RankedTensorType>(getX().getType());
  if (!inTy) {
    return success();
  }

  Type elementType;

  const auto outputDtype =
      static_cast<onnx::TensorProto_DataType>(getOutputDtype());
  if (outputDtype != onnx::TensorProto_DataType_UNDEFINED) {
    // If output type is specified, use it.
    auto builder = OpBuilder(getContext());
    elementType = convertONNXTypeToMLIRType(builder, outputDtype);
  } else {
    Value zero = getYZeroPoint();
    if (isNoneValue(zero)) {
      // If zero point type isn't provided, output type defaults to ui8.
      elementType = IntegerType::get(getContext(), 8, IntegerType::Unsigned);
    } else {
      // If zero point is provided, output type is same as zero point type.
      elementType = mlir::cast<ShapedType>(zero.getType()).getElementType();
    }
  }

  ONNXQuantizeLinearOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateTypes(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation; keep at the end of the file.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template struct ONNXNonSpecificOpShapeHelper<ONNXQuantizeLinearOp>;

} // namespace onnx_mlir
