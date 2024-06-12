/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Pad.cpp - ONNX Operations -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Pad operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Error.h"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct DimData {
  IndexExpr begin;
  IndexExpr end;
  IndexExpr dim;
};

llvm::Expected<DimData> computOutputDim(IndexExprBuilder *createIE,
    Value dataOperand, Value padsOperand, uint64_t beginEndSplit,
    uint64_t padsIndex, uint64_t padsAxis) {
  // Get begin/end pads.
  SymbolIndexExpr padBegin(
      createIE->getIntFromArrayAsSymbol(padsOperand, padsIndex));
  SymbolIndexExpr padEnd(createIE->getIntFromArrayAsSymbol(
      padsOperand, padsIndex + beginEndSplit));

  if (padBegin.isUndefined() || padEnd.isUndefined()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(), "pad parameter could not be processed");
  }

  // Get input dim.
  DimIndexExpr dimInput(createIE->getShapeAsDim(dataOperand, padsAxis));

  // Calculation for output size.
  IndexExpr dimOutputFinal = (padBegin + dimInput) + padEnd;

  return DimData{padBegin, padEnd, dimOutputFinal};
}

static llvm::Expected<DimsExpr> computeOutputShapeWithAxesOp(
    IndexExprBuilder *createIE, Value dataOperand, Value padsOperand,
    Value axesOperand) {
  DimsExpr outputDims;

  uint64_t dataRank = createIE->getShapedTypeRank(dataOperand);

  // Pad operation keeps rank and element type of dataOperand
  bool isFloat = isa<FloatType>(getElementType(dataOperand.getType()));
  outputDims.resize(dataRank, QuestionmarkIndexExpr(/*IsFloat=*/isFloat));

  // If axes is present, the size of pads is set to 2 * axes_size.
  // Axes is guaranteed to be 1-D per op definition
  auto axesSize = createIE->getArraySize(axesOperand);

  // Bail out: If axes is dynamic, output is also dynamic.
  if (axesSize == ShapedType::kDynamic) {
    return outputDims;
  }

  if (axesSize <= 0) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(), "axes size must be greater than 0");
  }

  auto beginEndSplit = (uint64_t)createIE->getArraySize(axesOperand);

  // Iterate over axesOperand to figure out to which axes the pads apply.
  for (auto axesOperandIndex : llvm::seq(axesSize)) {
    IndexExpr padsAxis =
        createIE->getIntFromArrayAsSymbol(axesOperand, axesOperandIndex);

    if (!padsAxis.isLiteral()) {
      return outputDims;
    }

    int64_t positiveAxis = padsAxis.getLiteral();
    if (positiveAxis < 0) {
      if (positiveAxis + (int)dataRank < 0) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(), "axes value is out of bounds");
      }
      positiveAxis += dataRank;
    }

    llvm::Expected<DimData> newDim = computOutputDim(createIE, dataOperand,
        padsOperand, beginEndSplit, axesOperandIndex, positiveAxis);

    if (!newDim) {
      return newDim.takeError();
    }

    if (newDim->dim.isLiteral()) {
      outputDims[positiveAxis] = newDim->dim;
    }
  }

  return outputDims;
}

LogicalResult ONNXPadOpShapeHelper::computeShape() {
  ONNXPadOpAdaptor operandAdaptor(operands);
  Value dataOperand = operandAdaptor.getData();
  Value padsOperand = operandAdaptor.getPads();
  Value axesOperand = operandAdaptor.getAxes();
  DimsExpr outputDims;

  // Get info about input data operand.
  uint64_t dataRank = createIE->getShapedTypeRank(dataOperand);

  // If axes is not present, the shape computation is simpler and the size
  // of pads in known to be 2 * dataRank.
  if (isNoneValue(axesOperand)) {

    // Initialize context and results (pads & output)
    pads.resize(2 * dataRank); // pads two sides of each axis.
    outputDims.resize(dataRank);

    // `pads` format is : [x1_begin, x2_begin,...,x1_end, x2_end,...],
    // where
    // - xi_begin: the number of pad values added at the beginning of axis `i`
    // - xi_end: the number of pad values added at the end of axis `i`.

    // Calculate output dimension sizes.
    for (auto axis : llvm::seq(dataRank)) {
      llvm::Expected<DimData> newDim = computOutputDim(
          createIE, dataOperand, padsOperand, dataRank, axis, axis);
      if (!newDim) {
        return op->emitError(llvm::toString(newDim.takeError()));
      }

      // Save results.
      pads[axis] = newDim->begin;
      pads[axis + dataRank] = newDim->end;
      outputDims[axis] = newDim->dim;
    }

    // Save the final result.
    setOutputDims(outputDims);
    return success();
  }

  assert(!isNoneValue(axesOperand) && "axes must be present");

  auto res = computeOutputShapeWithAxesOp(
      createIE, dataOperand, padsOperand, axesOperand);
  if (res) {
    setOutputDims(*res);
    return success();
  }
  return op->emitError(llvm::toString(res.takeError()));
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXPadOp::verify() {
  ShapedType dataTy = getData().getType().cast<ShapedType>();
  Type constTy = getConstantValue().getType();

  if (!isNoneValue(getConstantValue())) {
    // Check that the constant has the same element type as the input
    ShapedType shapedConstTy = constTy.cast<ShapedType>();
    if (dataTy.getElementType() != shapedConstTy.getElementType()) {
      return emitOpError("Pad with constant_value that doesn't match the "
                         "element type of the input.");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXPadOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getData()) || !hasShapeAndRank(getPads()))
    return success();

  Type elementType = getData().getType().cast<ShapedType>().getElementType();

  ONNXPadOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
