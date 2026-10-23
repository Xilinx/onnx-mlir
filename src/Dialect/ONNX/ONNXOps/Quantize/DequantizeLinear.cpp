/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ DequantizeLinear.cpp - ONNX Operations ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect DequantizeLinear operation.
//
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ONNXOps"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace {

// Returns known length if ty is a non-scalar 1-D vector, otherwise
// ShapedType::kDynamic.
int64_t nonScalar1DLen(ShapedType ty) {
  if (!ty.hasRank() || ty.getRank() != 1 || ty.isDynamicDim(0))
    return ShapedType::kDynamic;
  int64_t d = ty.getDimSize(0);
  return d == 1 ? ShapedType::kDynamic
                : d; // If dim size is 1 then it's considered a scalar.
}

} // namespace

namespace onnx_mlir {

template <>
LogicalResult ONNXDequantizeLinearOpShapeHelper::computeShape() {
  ONNXDequantizeLinearOpAdaptor operandAdaptor(
      operands, op->getAttrDictionary());
  RankedTensorType xTy =
      mlir::dyn_cast<RankedTensorType>(operandAdaptor.getX().getType());
  DimsExpr outputDims;
  createIE->getShapeAsDims(operandAdaptor.getX(), outputDims);

  const auto scaleTy =
      mlir::cast<ShapedType>(operandAdaptor.getXScale().getType());
  if (isBlockedLinearQuantization(scaleTy, operandAdaptor.getBlockSize())) {
    setOutputDims(outputDims);
    return success();
  }

  // Get d.
  int64_t d = nonScalar1DLen(scaleTy);
  if (d == ShapedType::kDynamic &&
      !isNoneValue(operandAdaptor.getXZeroPoint())) {
    d = nonScalar1DLen(
        mlir::cast<ShapedType>(operandAdaptor.getXZeroPoint().getType()));
  }

  if (d != ShapedType::kDynamic) {
    int64_t r = xTy.getRank();
    int64_t a = operandAdaptor.getAxis();
    // Checked in verify:
    assert(-r <= a && a < r && "axis out of range");
    if (a < 0)
      a += r;
    if (!outputDims[a].isLiteral()) {
      outputDims[a] = LitIE(d);
    }
    LLVM_DEBUG(llvm::dbgs() << "literal: " << outputDims[a].getLiteral()
                            << " d = " << d << "\n");
    // Checked in verify.
    assert(outputDims[a].getLiteral() == d &&
           "x_scale and x_zero_point 1-D tensor length must match the input "
           "axis dim size");
  }

  // Get values.
  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXDequantizeLinearOp::verify() {
  if (failed(verifyLinearQuantizationParameterShapes(getOperation(), getX(),
          getXScale(), getXZeroPoint(), getAxis(), getBlockSize())))
    return failure();

  Value zero = getXZeroPoint();
  if (!isNoneValue(zero)) {
    const auto zeroTy = mlir::cast<ShapedType>(zero.getType());
    // TODO: Add verification of "x_zero_point and x must have same type"
    // However, some models come with x : i8 and x_zero_point : ui8.
    //
    // if (getElementType(getX().getType()) != getElementType(zero.getType()))
    //   return emitOpError("x and x_zero_point must have the same data type");

    if (getElementType(zero.getType()).isInteger(32) && zeroTy.hasRank() &&
        zeroTy.getRank() != 0) {
      if (auto values = getElementAttributeFromONNXValue(zero)) {
        WideNum zero = WideNum::widen<BType::INT32>(0);
        if (!ElementsAttrBuilder::allEqual(values, zero))
          return emitOpError("x_zero_point must be 0 for data type int32");
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXDequantizeLinearOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!mlir::dyn_cast<RankedTensorType>(getX().getType()))
    return success();
  Type elementType = mlir::cast<ShapedType>(getY().getType()).getElementType();
  ONNXDequantizeLinearOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXDequantizeLinearOp>;
} // namespace onnx_mlir
