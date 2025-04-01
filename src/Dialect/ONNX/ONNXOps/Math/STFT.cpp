/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- STFT.cpp - Lowering STFT Op
//--------------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect STFT operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXGenericSTFTOpShapeHelper<ONNXSTFTOp>::computeShape() {
  typename ONNXSTFTOp::Adaptor operandAdaptor(
      operands, op->getAttrDictionary());
  auto signal = operandAdaptor.getSignal();
  if (!hasShapeAndRank(signal))
    return failure();
  auto oneSided = operandAdaptor.getOnesided() == 1;

  auto frameLengthIE =
      createIE->getIntAsSymbol(operandAdaptor.getFrameLength());
  auto frameStepIE = createIE->getIntAsSymbol(operandAdaptor.getFrameStep());

  // ----------------------- Compute Output Dims -------------------------------
  auto batchSizeIE = createIE->getShapeAsDim(signal, 0);
  auto signalLengthIE = createIE->getShapeAsDim(signal, 1);

  // compute number of frames
  LiteralIndexExpr one(1);
  auto numFramesIE =
      (signalLengthIE - frameLengthIE).floorDiv(frameStepIE) + one;

  DimsExpr outputDims;
  outputDims.emplace_back(batchSizeIE);
  outputDims.emplace_back(numFramesIE);
  if (oneSided) {
    outputDims.emplace_back(frameLengthIE.floorDiv(2) + one);
  } else {
    outputDims.emplace_back(frameLengthIE);
  }
  outputDims.emplace_back(LitIE(2));
  // ---------------------------------------------------------------------------

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// STFT Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXSTFTOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer the output shape if the operands shape isn't known yet.
  if (!hasShapeAndRank(getOperation()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getSignal().getType()).getElementType();
  ONNXSTFTOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXGenericSTFTOpShapeHelper<ONNXSTFTOp>;
} // namespace onnx_mlir
