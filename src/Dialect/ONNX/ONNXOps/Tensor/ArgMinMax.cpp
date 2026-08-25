/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ArgMinMax.cpp - ONNX Operations -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect ArgMin/Max operations.
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

template <typename OP_TYPE>
LogicalResult ONNXArgMinMaxOpShapeHelper<OP_TYPE>::computeShape() {
  // Get info about input data operand.
  OP_TYPE argOp = llvm::cast<OP_TYPE>(op);
  typename OP_TYPE::Adaptor operandAdaptor(operands);
  Value data = operandAdaptor.getData();
  int64_t dataRank = mlir::cast<ShapedType>(data.getType()).getRank();
  int64_t axisValue = argOp.getAxis();

  // axis attribute must be in the range [-r,r-1], where r = rank(data).
  assert(-dataRank <= axisValue && axisValue < dataRank && "axis out of range");

  // Negative axis means values are counted from the opposite side.
  if (axisValue < 0) {
    axisValue = dataRank + axisValue;
    auto builder = Builder(op->getContext());
    argOp.setAxisAttr(
        IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
            APInt(64, /*value=*/axisValue, /*isSigned=*/true)));
  }

  // The keepdims is a required attribute and should have default value of 1.
  int64_t keepdims = argOp.getKeepdims();
  bool isKeepdims = (keepdims == 1);

  // Compute outputDims
  DimsExpr outputDims;
  int64_t reducedRank = isKeepdims ? dataRank : dataRank - 1;
  outputDims.resize(reducedRank);
  for (int64_t i = 0; i < reducedRank; i++) {
    if (isKeepdims)
      outputDims[i] =
          (i != axisValue) ? createIE->getShapeAsDim(data, i) : LitIE(1);
    else
      outputDims[i] = (i < axisValue) ? createIE->getShapeAsDim(data, i)
                                      : createIE->getShapeAsDim(data, i + 1);
  }
  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

namespace {

// ArgMin/ArgMax output indices are in [0, s-1] where s is the input size along
// axis. For unsigned (and other narrow) result types, verify s-1 is
// representable — same [0, max] logic as GatherElements unsigned index checks.
LogicalResult verifyArgMinMaxReducedIndexFitsAxisDimension(
    Operation *op, Value data, Value reduced, int64_t axisIndex) {
  auto dataType = mlir::cast<ShapedType>(data.getType());
  auto reducedType = mlir::dyn_cast<ShapedType>(reduced.getType());
  if (!reducedType)
    return success();

  auto indexElementType =
      mlir::dyn_cast<IntegerType>(reducedType.getElementType());
  if (!indexElementType)
    return success();

  int64_t dataRank = dataType.getRank();
  int64_t axis = axisIndex < 0 ? axisIndex + dataRank : axisIndex;
  if (axis < 0 || axis >= dataRank)
    return success();

  int64_t dataDimAtAxis = dataType.getShape()[axis];
  if (dataDimAtAxis == ShapedType::kDynamic || dataDimAtAxis <= 0)
    return success();

  int64_t maxIndex = dataDimAtAxis - 1;
  int64_t maxRepresentableIndex;
  if (indexElementType.isUnsignedInteger()) {
    unsigned width = indexElementType.getWidth();
    if (width >= 64)
      return success();
    maxRepresentableIndex = static_cast<int64_t>((1ULL << width) - 1);
  } else {
    unsigned width = indexElementType.getWidth();
    if (width >= 64)
      return success();
    maxRepresentableIndex = static_cast<int64_t>((1ULL << (width - 1)) - 1);
  }

  if (maxIndex <= maxRepresentableIndex)
    return success();

  if (indexElementType.isUnsignedInteger())
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *op, "reduced", maxIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(0, maxRepresentableIndex));

  return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
      *op, "reduced", maxIndex,
      onnx_mlir::Diagnostic::Range<int64_t>(
          -maxRepresentableIndex, maxRepresentableIndex));
}

} // namespace

//===----------------------------------------------------------------------===//
// ONNXArgMaxOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXArgMaxOp::verify() {
  ONNXArgMaxOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(getOperation()))
    return success();

  int64_t rank = mlir::cast<ShapedType>(getData().getType()).getRank();
  int64_t axisIndex = getAxis();

  // axis value must be in the range [-rank, rank-1].
  if (axisIndex < -rank || axisIndex >= rank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-rank, rank - 1));

  return verifyArgMinMaxReducedIndexFitsAxisDimension(
      getOperation(), getData(), getReduced(), axisIndex);
}

LogicalResult ONNXArgMaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getData()))
    return success();

  // Default to i64 per ONNX spec, but preserve a pre-existing integer element
  // type on the result so frontend specializations to ui16/ui32 are kept.
  Builder b(getContext());
  Type elementType = b.getI64Type();
  if (auto resultType = mlir::dyn_cast<ShapedType>(getReduced().getType())) {
    Type existing = resultType.getElementType();
    if (existing && mlir::isa<IntegerType>(existing))
      elementType = existing;
  }
  ONNXArgMaxOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// ONNXArgMinOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXArgMinOp::verify() {
  ONNXArgMinOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(getOperation()))
    return success();

  int64_t rank = mlir::cast<ShapedType>(getData().getType()).getRank();
  int64_t axisIndex = getAxis();

  // axis value must be in the range [-rank, rank-1].
  if (axisIndex < -rank || axisIndex >= rank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-rank, rank - 1));

  return verifyArgMinMaxReducedIndexFitsAxisDimension(
      getOperation(), getData(), getReduced(), axisIndex);
}

LogicalResult ONNXArgMinOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getData()))
    return success();

  // Default to i64 per ONNX spec, but preserve a pre-existing integer element
  // type on the result so frontend specializations to ui16/ui32 are kept.
  Builder b(getContext());
  Type elementType = b.getI64Type();
  if (auto resultType = mlir::dyn_cast<ShapedType>(getReduced().getType())) {
    Type existing = resultType.getElementType();
    if (existing && mlir::isa<IntegerType>(existing))
      elementType = existing;
  }
  ONNXArgMinOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation; keep at the end of the file.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template struct ONNXArgMinMaxOpShapeHelper<ONNXArgMaxOp>;
template struct ONNXArgMinMaxOpShapeHelper<ONNXArgMinOp>;

} // namespace onnx_mlir
