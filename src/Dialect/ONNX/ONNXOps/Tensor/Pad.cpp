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

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

IndexExpr ONNXPadOpShapeHelper::computOutputDim(Value dataOperand,
    Value padsOperand, Value axesOperand, uint64_t padsIndex,
    uint64_t padsAxis) {
  // Get the size of the axes parameter.
  auto axesSize = createIE->getArraySize(axesOperand);

  // Get begin/end pads.
  SymbolIndexExpr padBegin(
      createIE->getIntFromArrayAsSymbol(padsOperand, padsIndex));
  SymbolIndexExpr padEnd(
      createIE->getIntFromArrayAsSymbol(padsOperand, padsIndex + axesSize));
  if (padBegin.isUndefined() || padEnd.isUndefined()) {
    // FIXME
    assert(false && "pad parameter could not be processed");
  }
  // Get input dim.
  DimIndexExpr dimInput(createIE->getShapeAsDim(dataOperand, padsAxis));

  // Calculation for output size.
  IndexExpr dimOutputFinal = (padBegin + dimInput) + padEnd;
  return dimOutputFinal;
}

LogicalResult ONNXPadOpShapeHelper::computeShape() {
  ONNXPadOpAdaptor operandAdaptor(operands);
  Value dataOperand = operandAdaptor.getData();
  Value padsOperand = operandAdaptor.getPads();
  Value axesOperand = operandAdaptor.getAxes();
  DimsExpr outputDims;

  // Get info about input data operand.
  uint64_t dataRank = createIE->getShapedTypeRank(dataOperand);

  if (!isNoneValue(axesOperand)) {
    bool isFloat = isa<FloatType>(getElementType(dataOperand.getType()));
    // Pad operation keeps the rank of dataOperand
    outputDims.resize(dataRank, QuestionmarkIndexExpr(/*IsFloat=*/isFloat));
    // Axes is guaranteed to be 1-D
    auto axesSize = createIE->getArraySize(axesOperand);

    for (auto axesIndex : llvm::seq(axesSize)) {
      IndexExpr padsAxis =
          createIE->getIntFromArrayAsSymbol(axesOperand, axesIndex);
      if (padsAxis.isLiteral()) {
        IndexExpr outputDimSize = computOutputDim(dataOperand, padsOperand,
            axesOperand, axesIndex, padsAxis.getLiteral());
        if (outputDimSize.isLiteral()) {
          llvm::errs() << "Literal: " << outputDimSize.getLiteral() << "\n";
          outputDims[padsAxis.getLiteral()] = outputDimSize;
        }
      } else {
        outputDims[axesSize] = QuestionmarkIndexExpr(/*IsFloat=*/isFloat);
      }
    }
    setOutputDims(outputDims);
    return success();
  }

  // Initialize context and results (pads & output)
  pads.resize(2 * dataRank); // pads two sides of each axis.
  outputDims.resize(dataRank);

  // `pads` format is : [x1_begin, x2_begin,...,x1_end, x2_end,...],
  // where
  // - xi_begin: the number of pad values added at the beginning of axis `i`
  // - xi_end: the number of pad values added at the end of axis `i`.

  // Calculate output dimension sizes.
  for (uint64_t i = 0; i < dataRank; i++) {
    // Get begin/end pads.
    SymbolIndexExpr padBegin(createIE->getIntFromArrayAsSymbol(padsOperand, i));
    SymbolIndexExpr padEnd(
        createIE->getIntFromArrayAsSymbol(padsOperand, i + dataRank));
    if (padBegin.isUndefined() || padEnd.isUndefined())
      return op->emitError("pad parameter could not be processed");
    // Get input dim.
    DimIndexExpr dimInput(createIE->getShapeAsDim(dataOperand, i));

    // Calculation for output size.
    IndexExpr dimOutputFinal = (padBegin + dimInput) + padEnd;
    std::string debug;
    dimOutputFinal.debugPrint(debug);
    llvm::errs() << debug << "\n";

    // Save results.
    pads[i] = padBegin;
    pads[i + dataRank] = padEnd;
    outputDims[i] = dimOutputFinal;
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
