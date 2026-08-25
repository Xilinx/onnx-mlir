/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ NonZero.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect NonZero operation.
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
LogicalResult ONNXNonZeroOpShapeHelper::computeShape() {
  ONNXNonZeroOpAdaptor operandAdaptor(operands);
  auto x = operandAdaptor.getX();
  if (!hasShapeAndRank(x)) {
    return failure();
  }
  int64_t xRank = createIE->getShapedTypeRank(x);
  // Cannot refine shape as we may otherwise loose the dynamic dim.
  return setOutputDimsFromLiterals(
      {xRank, ShapedType::kDynamic}, 0, /*refineShape*/ false);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXNonZeroOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getX()))
    return success();

  // Default to i64 per ONNX spec, but preserve a pre-existing integer element
  // type on the result so frontend specializations to ui16/ui32 are kept.
  Builder b(getContext());
  Type elementType = b.getI64Type();
  if (auto resultType = mlir::dyn_cast<ShapedType>(getY().getType())) {
    Type existing = resultType.getElementType();
    if (existing && mlir::isa<IntegerType>(existing))
      elementType = existing;
  }
  ONNXNonZeroOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXNonZeroOp>;
} // namespace onnx_mlir
