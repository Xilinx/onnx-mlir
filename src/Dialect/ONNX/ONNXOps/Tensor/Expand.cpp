/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Expand.cpp - ONNX Operations ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Expand operation.
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

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXExpandOpShapeHelper::computeShape() {
  // Get info about input operands.
  ONNXExpandOpAdaptor operandAdaptor(operands);
  Value input = operandAdaptor.getInput();
  Value shape = operandAdaptor.getShape();

  Operation *shapeDefOp = shape.getDefiningOp();
  ShapedType shapeType = mlir::dyn_cast_or_null<ShapedType>(shape.getType());
  if (!shapeType)
    return op->emitError("expected shape parameter to be defined");
  if (ShapedType::isDynamic(shapeType.getShape()[0]))
    return op->emitError("expected size of shape parameter to be defined");

  if (ONNXShapeOp shapeOp = mlir::dyn_cast_or_null<ONNXShapeOp>(shapeDefOp)) {
    assert(mlir::isa<ShapedType>(shapeOp.getData().getType()) && "expected");
    // Consider a first case where the expand.shape is produced by a shape op.
    // Infer its shape and use it as the requested shape.
    // Compute the output of the shape operation. We have to use its shape
    // helper as we need to connect to the actual expressions used to compute
    // it, not just a shape, in presence of runtime dimensions.

    // We also pass here the scope of the ExpandOp shape helper so that the
    // computations performed in the ShapeOp shape helper can be used in the
    // context of the ExpandOp.
    ONNXShapeOpShapeHelper shapeHelper(
        shapeOp.getOperation(), {}, createIE, /* important */ getScope());
    if (failed(shapeHelper.computeShape()))
      return op->emitError("failed to get shape op shape");

    // Compute the data selected by the Shape operator.
    DimsExpr selectedData;
    shapeHelper.computeSelectedDataShape(selectedData);

    // Now that we have the shape's actual computation
    if (failed(customComputeShape({input}, &selectedData)))
      return op->emitError("failed to broadcast 3");
    return success();
  }

  if (!mlir::isa<ShapedType>(shape.getType()))
    return op->emitError("Expecting a shaped type");
  SmallVector<IndexExpr, 4> constVals;
  createIE->getIntFromArrayAsSymbols(shape, constVals);
  if (failed(customComputeShape({input}, &constVals)))
    return op->emitError("failed to broadcast 4");

  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXExpandOp::verify() {
  auto operandAdaptor = ONNXExpandOpAdaptor(*this);
  // Get operands.
  auto input = operandAdaptor.getInput();
  auto shape = operandAdaptor.getShape();
  // Check input.
  auto shapeType = mlir::dyn_cast_or_null<ShapedType>(shape.getType());
  if (shapeType && shapeType.hasRank()) {
    if (shapeType.getRank() != 1)
      return emitOpError("Shape has a rank of 1");
  }

  auto inputType = mlir::dyn_cast_or_null<ShapedType>(input.getType());
  if (!inputType || !inputType.hasRank())
    return success();

  SmallVector<int64_t> shapeVals = valToVector(shape);
  if (shapeVals.empty())
    return success();

  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputIdx = inputShape.size() - 1;
  int64_t shapeIdx = shapeVals.size() - 1;
  for (; inputIdx >= 0 && shapeIdx >= 0; --inputIdx, --shapeIdx) {
    int64_t inputDim = inputShape[inputIdx];
    int64_t shapeDim = shapeVals[shapeIdx];

    // Stay conservative: only flag a pair that is provably incompatible, i.e.
    // both sides are static, positive, unequal, and neither one is 1.
    if (ShapedType::isDynamic(inputDim) || shapeDim <= 0)
      continue;
    if (inputDim != shapeDim && inputDim != 1 && shapeDim != 1)
      return mlir::emitError(getLoc(), getOperationName())
             << ": input dimension " << inputDim << " at index " << inputIdx
             << " is incompatible with target shape dimension " << shapeDim
             << " at index " << shapeIdx
             << ": two corresponding dimensions must have the same value, or "
                "one of them is equal to 1";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXExpandOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()) || !hasShapeAndRank(getShape()))
    return success();

  ShapedType shapeType =
      mlir::dyn_cast_or_null<ShapedType>(getShape().getType());
  if (!shapeType || ShapedType::isDynamic(shapeType.getShape()[0]))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getInput().getType()).getElementType();
  ONNXExpandOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
