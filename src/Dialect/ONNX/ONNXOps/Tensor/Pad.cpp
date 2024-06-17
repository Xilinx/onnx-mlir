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
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Error.h"
#include <numeric>

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

LogicalResult ONNXPadOpShapeHelper::computeShape() {
  ONNXPadOpAdaptor operandAdaptor(operands);
  Value dataOperand = operandAdaptor.getData();
  Value padsOperand = operandAdaptor.getPads();
  Value axesOperand = operandAdaptor.getAxes();

  uint64_t dataRank = createIE->getShapedTypeRank(dataOperand);

  DimsExpr outputDims;
  bool isFloat = isa<FloatType>(getElementType(dataOperand.getType()));
  outputDims.resize(dataRank, QuestionmarkIndexExpr(/*IsFloat=*/isFloat));

  SmallVector<uint64_t> axes;
  // If axes is not present, the shape computation is simpler and the size
  // of pads in known to be 2 * dataRank.
  if (isNoneValue(axesOperand)) {
    axes.resize(dataRank);
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    auto axesSize = createIE->getArraySize(axesOperand);

    // Bail out: If axes is dynamic, output is also dynamic.
    if (axesSize == ShapedType::kDynamic) {
      setOutputDims(outputDims);
      return success();
    }

    if (axesSize <= 0) {
      return op->emitError("axes size must be greater than 0");
    }

    // Iterate over axesOperand to figure out the axes that will be padded
    for (auto axesOperandIndex : llvm::seq(axesSize)) {
      IndexExpr padsAxis =
          createIE->getIntFromArrayAsSymbol(axesOperand, axesOperandIndex);

      if (!padsAxis.isLiteral()) {
        continue;
      }

      int64_t positiveAxis = padsAxis.getLiteral();
      if (positiveAxis < 0) {
        positiveAxis += dataRank;
      }

      if (positiveAxis + (int)dataRank < 0 || positiveAxis >= (int)dataRank) {
        return op->emitError("axes value is out of bounds");
      }

      axes.push_back(positiveAxis);
    }
  }

  // Initialize context and results (pads & output)
  pads.resize(2 * dataRank); // pads two sides of each axis.

  // `pads` format is : [x1_begin, x2_begin,...,x1_end, x2_end,...],
  // where
  // - xi_begin: the number of pad values added at the beginning of axis `i`
  // - xi_end: the number of pad values added at the end of axis `i`.
  llvm::SmallSet<uint64_t, 4> visited;
  for (auto [idx, axis] : llvm::enumerate(axes)) {
    llvm::Expected<DimData> newDim = computOutputDim(
        createIE, dataOperand, padsOperand, axes.size(), idx, axis);

    if (!newDim) {
      return op->emitError(llvm::toString(newDim.takeError()));
    }

    visited.insert(axis);

    // Currently "pads" is only used when axes is NoneType and for constant
    // propagation
    if (isNoneValue(axesOperand)) {
      pads[axis] = newDim->begin;
      pads[axis + dataRank] = newDim->end;
    }

    outputDims[axis] = newDim->dim;
  }

  if (!axes.empty()) {
    for (auto i : llvm::seq(dataRank)) {
      if (!visited.count(i)) {
        outputDims[i] = createIE->getShapeAsLiteral(dataOperand, i);
      }
    }
  }

  // Save the final result.
  setOutputDims(outputDims);
  return success();
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
