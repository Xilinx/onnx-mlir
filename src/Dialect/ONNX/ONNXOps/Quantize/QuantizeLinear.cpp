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
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"
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
  // Is tensor possibly a scalar (rank 0 or rank 1 with 1 or unknown elements)?
  auto mayBeScalar = [](ShapedType t) -> bool {
    return t.getRank() == 0 ||
           (t.getRank() == 1 && (t.isDynamicDim(0) || t.getDimSize(0) == 1));
  };
  // Known length if t is a non-scalar 1-D vector, otherwise kDynamic.
  auto nonScalar1DLen = [](ShapedType t) -> int64_t {
    if (!t.hasRank() || t.getRank() != 1 || t.isDynamicDim(0) ||
        t.getDimSize(0) == 1)
      return ShapedType::kDynamic;
    return t.getDimSize(0);
  };
  const auto inputTy = mlir::cast<ShapedType>(getX().getType());
  const auto scaleTy = mlir::cast<ShapedType>(getYScale().getType());
  if (scaleTy.hasRank() && inputTy.hasRank() &&
      scaleTy.getRank() != inputTy.getRank() && scaleTy.getRank() > 1)
    return emitOpError("y_scale rank needs to be 0, 1 or match x rank");

  int64_t len = nonScalar1DLen(scaleTy);
  if (!isNoneValue(getYZeroPoint())) {
    const auto zeroTy = mlir::cast<ShapedType>(getYZeroPoint().getType());
    // Accept rank-0 and single-element rank-1 tensors as equivalent scalars,
    // as in DequantizeLinear. A scalar cannot pair with a non-scalar.
    if (zeroTy.hasRank() && scaleTy.hasRank() &&
        !(mayBeScalar(zeroTy) && mayBeScalar(scaleTy)) &&
        failed(verifyCompatibleShape(zeroTy.getShape(), scaleTy.getShape())))
      return emitOpError("y_zero_point must have the same shape as y_scale");
    if (len == ShapedType::kDynamic)
      len = nonScalar1DLen(zeroTy);
  }

  // With block quantization a 1-D y_scale holds one entry per block, not one
  // per element along the axis.
  if (getBlockSize() == 0 && len != ShapedType::kDynamic && inputTy.hasRank()) {
    int64_t rank = inputTy.getRank();
    int64_t axis = getAxis();
    if (axis < -rank || axis >= rank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "axis", axis,
          onnx_mlir::Diagnostic::Range<int64_t>(-rank, rank - 1));
    if (axis < 0)
      axis += rank;
    if (!inputTy.isDynamicDim(axis) && inputTy.getDimSize(axis) != len)
      return emitOpError("y_scale and y_zero_point 1-D tensor length must "
                         "match the input axis dim size");
  }
  return success();
}
//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

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
