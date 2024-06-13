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

#include "mlir/IR/Region.h"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

llvm::Expected<IndexExpr> ONNXPadOpShapeHelper::computOutputDim(
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

  // Pad operation keeps rank and element type of dataOperand
  bool isFloat = isa<FloatType>(getElementType(dataOperand.getType()));
  outputDims.resize(dataRank, QuestionmarkIndexExpr(/*IsFloat=*/isFloat));

  // If axes is present, the size of pads is set to 2 * axes_size.
  if (!isNoneValue(axesOperand)) {
    // Axes is guaranteed to be 1-D per op definition
    auto axesSize = createIE->getArraySize(axesOperand);

    // Bail out: If axes is dynamic, output is also dynamic.
    if (axesSize == ShapedType::kDynamic) {
      setOutputDims(outputDims);
      return success();
    }

    if (axesSize <= 0) {
      op->emitError("axes size must be greater than 0");
    }

    auto beginEndSplit = (uint64_t)createIE->getArraySize(axesOperand);

    // Iterate over axesOperand to figure out to which axes the pads apply.
    for (auto axesOperandIndex : llvm::seq(axesSize)) {
      // TODO: Axes values can be negative. Add special logic for that
      IndexExpr padsAxis =
          createIE->getIntFromArrayAsSymbol(axesOperand, axesOperandIndex);

      if (padsAxis.isLiteral()) {
        llvm::Expected<IndexExpr> outputDimSize =
            computOutputDim(dataOperand, padsOperand, beginEndSplit,
                axesOperandIndex, padsAxis.getLiteral());
        if (!outputDimSize) {
          return op->emitError(llvm::toString(outputDimSize.takeError()));
        }

        if (outputDimSize->isLiteral()) {
          llvm::errs() << "Literal: " << outputDimSize->getLiteral() << "\n";
          outputDims[padsAxis.getLiteral()] = *outputDimSize;
        }
      }
    }

    setOutputDims(outputDims);
    return success();
  }

  // Initialize context and results (pads & output)
  pads.resize(2 * dataRank); // pads two sides of each axis.

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
