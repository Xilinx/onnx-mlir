/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ConcatFromSequence.cpp - ONNX Operations ----------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect ConcatFromSequence operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXConcatFromSequenceOp::verify() {
  ONNXConcatFromSequenceOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(operandAdaptor.getInputSequence()))
    return success(); // Won't be able to do any checking at this stage.

  Value inputSequence = operandAdaptor.getInputSequence();
  assert(mlir::isa<SeqType>(inputSequence.getType()) &&
         "Incorrect type for a sequence");
  auto seqType = mlir::cast<SeqType>(inputSequence.getType());
  auto elemType = mlir::cast<ShapedType>(seqType.getElementType());
  int64_t rank = elemType.getShape().size();
  int64_t axisIndex = getAxis();
  int64_t newAxisIndex = getNewAxis();

  // axis attribute must be in the range [-r,r-1], where r = rank(inputs).
  // When `new_axis` is 1, accepted range is [-r-1,r].
  if (newAxisIndex == 1) {
    if (axisIndex < (-rank - 1) || axisIndex > rank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "axis", axisIndex,
          onnx_mlir::Diagnostic::Range<int64_t>(-rank - 1, rank));
  } else {
    if (axisIndex < -rank || axisIndex >= rank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "axis", axisIndex,
          onnx_mlir::Diagnostic::Range<int64_t>(-rank, rank - 1));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXConcatFromSequenceOp::inferShapes(
    std::function<void(Region &)> /*shapeInferenceFunc*/) {
  Type inputType = getInputSequence().getType();
  if (!mlir::isa<SeqType>(inputType))
    return success();

  auto seqType = mlir::cast<SeqType>(inputType);
  Type seqElemType = seqType.getElementType();

  // If the sequence element type is not a ShapedType (e.g. during early import
  // when types are not yet fully refined), produce an unranked tensor with
  // unknown element type — we can't infer anything yet.
  if (!mlir::isa<ShapedType>(seqElemType))
    return success();

  auto elemType = mlir::cast<ShapedType>(seqElemType);

  // If the element tensors are unranked, the best we can do is an unranked
  // tensor with the same scalar element type.
  if (!elemType.hasRank()) {
    getResult().setType(UnrankedTensorType::get(elemType.getElementType()));
    return success();
  }

  int64_t elemRank = elemType.getRank();
  int64_t axisIndex = getAxis();
  int64_t newAxisIndex = getNewAxis();

  if (newAxisIndex == 1) {
    // new_axis=1: like numpy.stack — inserts a new axis.
    // Result rank is elemRank + 1.
    int64_t resultRank = elemRank + 1;
    // Normalize axis to positive.
    if (axisIndex < 0)
      axisIndex += resultRank;
    SmallVector<int64_t> resultShape;
    for (int64_t i = 0; i < resultRank; ++i) {
      if (i == axisIndex) {
        // The new axis dimension equals the number of tensors in the sequence.
        int64_t seqLen = seqType.getLength();
        resultShape.push_back(
            seqLen > 0 ? seqLen : ShapedType::kDynamic);
      } else {
        // Map back to the element dimension, accounting for the inserted axis.
        int64_t elemDim = (i < axisIndex) ? i : i - 1;
        resultShape.push_back(elemType.getDimSize(elemDim));
      }
    }
    getResult().setType(
        RankedTensorType::get(resultShape, elemType.getElementType()));
  } else {
    // new_axis=0: standard concatenation along existing axis.
    // Result rank is the same as elemRank.
    if (axisIndex < 0)
      axisIndex += elemRank;
    SmallVector<int64_t> resultShape(elemType.getShape());
    // The concat axis dimension is dynamic (sum of all tensors' dimensions
    // along this axis, which is unknown in general).
    resultShape[axisIndex] = ShapedType::kDynamic;
    getResult().setType(
        RankedTensorType::get(resultShape, elemType.getElementType()));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Helper
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXConcatFromSequenceOpShapeHelper::computeShape() {
  ONNXConcatFromSequenceOp concatOp = llvm::cast<ONNXConcatFromSequenceOp>(op);
  Type inputType = concatOp.getInputSequence().getType();
  if (!mlir::isa<SeqType>(inputType))
    return failure();

  auto seqType = mlir::cast<SeqType>(inputType);
  Type seqElemType = seqType.getElementType();
  if (!mlir::isa<ShapedType>(seqElemType))
    return failure();
  auto elemType = mlir::cast<ShapedType>(seqElemType);
  if (!elemType.hasRank())
    return failure();

  int64_t elemRank = elemType.getRank();
  int64_t axisIndex = concatOp.getAxis();
  int64_t newAxisIndex = concatOp.getNewAxis();

  DimsExpr outputDims;
  if (newAxisIndex == 1) {
    int64_t resultRank = elemRank + 1;
    if (axisIndex < 0)
      axisIndex += resultRank;
    for (int64_t i = 0; i < resultRank; ++i) {
      if (i == axisIndex) {
        int64_t seqLen = seqType.getLength();
        if (seqLen > 0)
          outputDims.emplace_back(LiteralIndexExpr(seqLen));
        else
          outputDims.emplace_back(QuestionmarkIndexExpr());
      } else {
        int64_t elemDim = (i < axisIndex) ? i : i - 1;
        int64_t d = elemType.getDimSize(elemDim);
        if (d == ShapedType::kDynamic)
          outputDims.emplace_back(QuestionmarkIndexExpr());
        else
          outputDims.emplace_back(LiteralIndexExpr(d));
      }
    }
  } else {
    if (axisIndex < 0)
      axisIndex += elemRank;
    for (int64_t i = 0; i < elemRank; ++i) {
      if (i == axisIndex) {
        outputDims.emplace_back(QuestionmarkIndexExpr());
      } else {
        int64_t d = elemType.getDimSize(i);
        if (d == ShapedType::kDynamic)
          outputDims.emplace_back(QuestionmarkIndexExpr());
        else
          outputDims.emplace_back(LiteralIndexExpr(d));
      }
    }
  }
  setOutputDims(outputDims);
  return success();
}

template struct ONNXNonSpecificOpShapeHelper<ONNXConcatFromSequenceOp>;
} // namespace onnx_mlir
