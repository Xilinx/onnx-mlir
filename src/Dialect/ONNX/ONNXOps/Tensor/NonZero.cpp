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

  Builder b(getContext());
  // Preserve any pre-existing integer element type on the Y result so any
  // widening to ui16/ui32 is not lost. Default to i64 element type.
  Type yElementType = b.getI64Type();
  if (auto yType = mlir::dyn_cast<ShapedType>(getY().getType())) {
    if (yType.getElementType()) {
      Type existing = yType.getElementType();
      if (existing && mlir::isa<IntegerType>(existing))
        yElementType = existing;
    }
  }
  ONNXNonZeroOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(yElementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXNonZeroOp>;
} // namespace onnx_mlir
