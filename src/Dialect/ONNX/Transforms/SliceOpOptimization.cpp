//===------------ SliceOpOptimization.cpp - ONNX Slice rewrites -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modifications (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its
// affiliates
//
// =============================================================================
//
// This file implements the (opt-in) Slice-through-Slice/Tile/Pad/Concat
// rewrite patterns. Passes that run canonicalization can add them via
// populateSliceOpOptimizationPatterns() when --enable-slice-canonicalization
// is set (see configureSliceCanonicalization / isSliceCanonicalizationEnabled).
//
// The Slice-through-Tile/Pad/Concat rewrites are adapted from the MLIR TOSA
// patterns in mlir/lib/Dialect/Tosa/IR/TosaCanonicalizations.cpp.
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

//===----------------------------------------------------------------------===//
// ONNXSliceOp canonicalization
//===----------------------------------------------------------------------===//

struct NormalizedSliceParams {
  SmallVector<int64_t> starts;
  SmallVector<int64_t> ends;
  SmallVector<int64_t> axes;
  SmallVector<int64_t> steps;
};

LogicalResult getNormalizedSliceParams(
    ONNXSliceOp sliceOp, NormalizedSliceParams &params) {
  auto dataType = dyn_cast<RankedTensorType>(sliceOp.getData().getType());
  if (!dataType || !dataType.hasStaticShape())
    return failure();

  ONNXSliceOpShapeHelper shapeHelper(sliceOp.getOperation(), {});
  if (failed(shapeHelper.computeShape()))
    return failure();

  auto collectLiteralValues = [](ArrayRef<IndexExpr> exprs,
                                  SmallVectorImpl<int64_t> &values) {
    values.clear();
    values.reserve(exprs.size());
    for (IndexExpr expr : exprs) {
      if (!expr.isLiteral())
        return failure();
      values.push_back(expr.getLiteral());
    }
    return success();
  };

  if (failed(collectLiteralValues(shapeHelper.starts, params.starts)) ||
      failed(collectLiteralValues(shapeHelper.ends, params.ends)) ||
      failed(collectLiteralValues(shapeHelper.steps, params.steps)))
    return failure();

  // For negative-step slices the shape helper uses -1 as the normalized
  // exclusive end sentinel. Writing that value back as an ONNX operand would be
  // reinterpreted as `dim - 1` on the next canonicalization round. Use the
  // stable raw sentinel `-dim - 1` instead.
  for (auto [idx, end] : llvm::enumerate(params.ends)) {
    if (params.steps[idx] < 0 && end == -1)
      end = -dataType.getDimSize(idx) - 1;
  }

  params.axes.clear();
  params.axes.reserve(params.starts.size());
  for (int64_t i = 0, e = params.starts.size(); i < e; ++i)
    params.axes.push_back(i);

  return success(
      params.starts.size() == static_cast<size_t>(dataType.getRank()));
}

bool hasOnlyUnitSteps(ArrayRef<int64_t> steps) {
  return llvm::all_of(steps, [](int64_t step) { return step == 1; });
}

Value createI64TensorConstant(
    PatternRewriter &rewriter, Location loc, ArrayRef<int64_t> values) {
  RankedTensorType tensorType = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, rewriter.getI64Type());
  return rewriter.create<ONNXConstantOp>(
      loc, Attribute(), DenseElementsAttr::get(tensorType, values));
}

Value createFullRankSlice(PatternRewriter &rewriter, Location loc,
    Type resultType, Value data, ArrayRef<int64_t> starts,
    ArrayRef<int64_t> ends, ArrayRef<int64_t> steps) {
  Value startsValue = createI64TensorConstant(rewriter, loc, starts);
  Value endsValue = createI64TensorConstant(rewriter, loc, ends);
  SmallVector<int64_t> axes(starts.size());
  for (auto [idx, axis] : llvm::enumerate(axes))
    axis = idx;
  Value axesValue = createI64TensorConstant(rewriter, loc, axes);
  Value stepsValue = createI64TensorConstant(rewriter, loc, steps);
  return rewriter.create<ONNXSliceOp>(
      loc, resultType, data, startsValue, endsValue, axesValue, stepsValue);
}

Value createFullRankSlice(PatternRewriter &rewriter, Location loc,
    Type resultType, Value data, ArrayRef<int64_t> starts,
    ArrayRef<int64_t> ends) {
  SmallVector<int64_t> steps(starts.size(), 1);
  return createFullRankSlice(
      rewriter, loc, resultType, data, starts, ends, steps);
}

bool hasI64Values(Value value, ArrayRef<int64_t> expected) {
  SmallVector<int64_t> values;
  return getI64ValuesFromONNXConstantOp(value, values) && values == expected;
}

bool hasCanonicalSliceOperands(
    ONNXSliceOp sliceOp, const NormalizedSliceParams &params) {
  return hasI64Values(sliceOp.getStarts(), params.starts) &&
         hasI64Values(sliceOp.getEnds(), params.ends) &&
         hasI64Values(sliceOp.getAxes(), params.axes) &&
         hasI64Values(sliceOp.getSteps(), params.steps);
}

struct NormalizeSliceOperandsPattern : public OpRewritePattern<ONNXSliceOp> {
  using OpRewritePattern<ONNXSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSliceOp sliceOp, PatternRewriter &rewriter) const override {
    NormalizedSliceParams params;
    if (failed(getNormalizedSliceParams(sliceOp, params)))
      return rewriter.notifyMatchFailure(
          sliceOp, "slice parameters must normalize to static literals");

    if (hasCanonicalSliceOperands(sliceOp, params))
      return rewriter.notifyMatchFailure(
          sliceOp, "slice operands are already canonical");

    Value startsValue =
        createI64TensorConstant(rewriter, sliceOp.getLoc(), params.starts);
    Value endsValue =
        createI64TensorConstant(rewriter, sliceOp.getLoc(), params.ends);
    Value axesValue =
        createI64TensorConstant(rewriter, sliceOp.getLoc(), params.axes);
    Value stepsValue =
        createI64TensorConstant(rewriter, sliceOp.getLoc(), params.steps);
    auto normalizedSlice = rewriter.create<ONNXSliceOp>(sliceOp.getLoc(),
        sliceOp.getOutput().getType(), sliceOp.getData(), startsValue,
        endsValue, axesValue, stepsValue);
    normalizedSlice->setAttrs(sliceOp->getAttrDictionary());
    rewriter.replaceOp(sliceOp, normalizedSlice.getOutput());
    return success();
  }
};

// Fills `lowPads[axis]`/`highPads[axis]` with the begin/end padding of a Pad op
// (zero for axes the Pad leaves untouched). Fails unless all pads are static
// and non-negative. `pads` is laid out [begin_0..begin_{k-1}, end_0..end_{k-1}]
// over `axes`.
LogicalResult getPadLowHighPads(ONNXPadOp padOp, int64_t rank,
    SmallVectorImpl<int64_t> &lowPads, SmallVectorImpl<int64_t> &highPads) {
  SmallVector<int64_t> pads;
  if (!getI64ValuesFromONNXConstantOp(padOp.getPads(), pads) ||
      llvm::any_of(pads, [](int64_t p) { return p < 0; }))
    return failure();

  SmallVector<int64_t> axes;
  if (isNoneValue(padOp.getAxes())) {
    for (int64_t axis = 0; axis < rank; ++axis)
      axes.push_back(axis);
  } else if (!getI64ValuesFromONNXConstantOp(padOp.getAxes(), axes)) {
    return failure();
  }
  if (pads.size() != 2 * axes.size())
    return failure();

  lowPads.assign(rank, 0);
  highPads.assign(rank, 0);
  for (auto [i, rawAxis] : llvm::enumerate(axes)) {
    const int64_t axis = rawAxis < 0 ? rawAxis + rank : rawAxis;
    if (axis < 0 || axis >= rank)
      return failure();
    lowPads[axis] = pads[i];
    highPads[axis] = pads[axes.size() + i];
  }
  return success();
}

struct FuseSliceSlicePattern : public OpRewritePattern<ONNXSliceOp> {
  using OpRewritePattern<ONNXSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSliceOp sliceOp, PatternRewriter &rewriter) const override {
    auto innerSliceOp = sliceOp.getData().getDefiningOp<ONNXSliceOp>();
    if (!innerSliceOp)
      return rewriter.notifyMatchFailure(
          sliceOp, "slice input must be another Slice");
    if (!innerSliceOp->hasOneUse())
      return rewriter.notifyMatchFailure(
          sliceOp, "preceding Slice must have one use");

    NormalizedSliceParams outerParams;
    NormalizedSliceParams innerParams;
    if (failed(getNormalizedSliceParams(sliceOp, outerParams)) ||
        failed(getNormalizedSliceParams(innerSliceOp, innerParams)))
      return rewriter.notifyMatchFailure(
          sliceOp, "slice parameters must normalize to static literals");
    if (!hasOnlyUnitSteps(outerParams.steps) ||
        !hasOnlyUnitSteps(innerParams.steps))
      return rewriter.notifyMatchFailure(sliceOp, "slice steps must all be 1");

    SmallVector<int64_t> starts;
    SmallVector<int64_t> ends;
    starts.reserve(outerParams.starts.size());
    ends.reserve(outerParams.ends.size());
    for (auto [innerStart, outerStart, outerEnd] : llvm::zip_equal(
             innerParams.starts, outerParams.starts, outerParams.ends)) {
      starts.push_back(innerStart + outerStart);
      ends.push_back(innerStart + outerEnd);
    }

    Value fusedSlice = createFullRankSlice(rewriter, sliceOp.getLoc(),
        sliceOp.getOutput().getType(), innerSliceOp.getData(), starts, ends);
    rewriter.replaceOp(sliceOp, fusedSlice);
    return success();
  }
};

struct TileSliceReduction {
  SmallVector<int64_t> repeats;
  SmallVector<int64_t> starts;
  SmallVector<int64_t> ends;
  SmallVector<int64_t> tileShape;
};

FailureOr<TileSliceReduction> computeTileSliceReduction(
    RankedTensorType tileInputType, const NormalizedSliceParams &params) {
  TileSliceReduction reduction;
  reduction.repeats.reserve(tileInputType.getRank());
  reduction.starts.reserve(tileInputType.getRank());
  reduction.ends.reserve(tileInputType.getRank());

  for (auto [axis, inputDim] : llvm::enumerate(tileInputType.getShape())) {
    const int64_t start = params.starts[axis];
    const int64_t size = params.ends[axis] - start;
    if (inputDim <= 0 || size <= 0)
      return failure();

    // How much of the first (partial) copy the window starts inside, then how
    // many further whole copies are needed to reach the window's end.
    const int64_t offsetInFirstCopy = start % inputDim;
    const int64_t sizeInFirstCopy =
        std::min(inputDim - offsetInFirstCopy, size);
    const int64_t multiplier =
        llvm::divideCeil(size - sizeInFirstCopy, inputDim) + 1;
    reduction.repeats.push_back(multiplier);
    reduction.starts.push_back(offsetInFirstCopy);
    reduction.ends.push_back(offsetInFirstCopy + size);
  }

  reduction.tileShape.assign(
      tileInputType.getShape().begin(), tileInputType.getShape().end());
  for (auto [dim, multiplier] :
      llvm::zip_equal(reduction.tileShape, reduction.repeats))
    dim *= multiplier;
  return reduction;
}

// A Slice reading from a Tile only needs the Tile to repeat its input enough
// times to cover the slice window. Shrink the Tile's `repeats` to that minimum
// and retarget the slice at the smaller Tile:
//
//     tile(x, 3)     +-----+-----+-----+       tile(x, 2)   +-----+-----+
//                    |  x  |  x  |  x  |  ==>               |  x  |  x  |
//                    +-----+-----+-----+                    +-----+-----+
//                      |<-want->|                              |<-want->|
//     slice       -----+--------+              slice        ---+--------+
//
// The exact-one-copy case is not special-cased: it reduces the repeats to all
// 1s, after which RemoveIdentityTilePattern drops the Tile and
// ONNXSliceOp::fold drops the now-identity slice, collapsing slice(tile(x)) to
// x on its own.
struct SliceTilePattern : public OpRewritePattern<ONNXSliceOp> {
  using OpRewritePattern<ONNXSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSliceOp sliceOp, PatternRewriter &rewriter) const override {
    auto tileOp = sliceOp.getData().getDefiningOp<ONNXTileOp>();
    if (!tileOp)
      return failure();
    if (!tileOp->hasOneUse())
      return rewriter.notifyMatchFailure(
          sliceOp, "preceding Tile must have one use");
    auto tileInputType =
        dyn_cast<RankedTensorType>(tileOp.getInput().getType());
    if (!tileInputType || !tileInputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          sliceOp, "Tile input must be a static ranked tensor");

    NormalizedSliceParams params;
    if (failed(getNormalizedSliceParams(sliceOp, params)) ||
        !hasOnlyUnitSteps(params.steps))
      return rewriter.notifyMatchFailure(
          sliceOp, "slice must have static, unit-step parameters");

    SmallVector<int64_t> repeats;
    if (!getI64ValuesFromONNXConstantOp(tileOp.getRepeats(), repeats))
      return rewriter.notifyMatchFailure(
          sliceOp, "Tile repeats must be a static constant");

    FailureOr<TileSliceReduction> reduction =
        computeTileSliceReduction(tileInputType, params);
    if (failed(reduction))
      return rewriter.notifyMatchFailure(sliceOp, "degenerate slice or tile");

    if (reduction->repeats == repeats)
      return rewriter.notifyMatchFailure(
          sliceOp, "Tile repeats cannot be reduced");

    Value newRepeatsValue =
        createI64TensorConstant(rewriter, tileOp.getLoc(), reduction->repeats);
    Value newTile = rewriter.create<ONNXTileOp>(tileOp.getLoc(),
        tileInputType.clone(reduction->tileShape), tileOp.getInput(),
        newRepeatsValue);
    Value newSlice = createFullRankSlice(rewriter, sliceOp.getLoc(),
        sliceOp.getOutput().getType(), newTile, reduction->starts,
        reduction->ends);
    rewriter.replaceOp(sliceOp, newSlice);
    return success();
  }
};

// A Slice reading from a Pad only needs the padding that still falls inside its
// window. Slice the Pad's input down to the region the window covers and re-pad
// with just the leftover padding, i.e. slice(pad(x)) becomes pad(slice(x)):
//
//                   lowPad          highPad
//                  |<--->|          |<--->|
//     pad(x)       +-----+----------+-----+
//                  | pad |    x     | pad |
//                  +-----+----------+-----+
//                     |<----- want ----->|
//     slice        ---+------------------+       ==>   pad(slice(x))
//
// When no padding is left the Pad is dropped and only slice(x) remains; if that
// slice is also an identity ONNXSliceOp::fold collapses it to x. Re-padding is
// only valid for `constant` mode: reflect/edge/wrap derive padded values from
// the input edges, which move once the input is sliced. The pure-drop case is
// safe for any mode since the padded values are never read.

struct PadSliceReduction {
  SmallVector<int64_t> inputStarts;
  SmallVector<int64_t> inputEnds;
  SmallVector<int64_t> lowPads;
  SmallVector<int64_t> highPads;
  bool needsInputSlice = false;
  bool changed = false;

  [[nodiscard]] bool padRemains() const {
    return llvm::any_of(lowPads, [](int64_t p) { return p != 0; }) ||
           llvm::any_of(highPads, [](int64_t p) { return p != 0; });
  }

  [[nodiscard]] SmallVector<int64_t> inputShape() const {
    SmallVector<int64_t> shape(inputStarts.size());
    for (size_t i = 0, e = shape.size(); i < e; ++i)
      shape[i] = inputEnds[i] - inputStarts[i];
    return shape;
  }

  [[nodiscard]] SmallVector<int64_t> pads() const {
    SmallVector<int64_t> result;
    result.reserve(lowPads.size() + highPads.size());
    result.append(lowPads.begin(), lowPads.end());
    result.append(highPads.begin(), highPads.end());
    return result;
  }
};

FailureOr<PadSliceReduction> computePadSliceReduction(
    RankedTensorType padInputType, const NormalizedSliceParams &params,
    ArrayRef<int64_t> oldLowPads, ArrayRef<int64_t> oldHighPads) {
  const int64_t rank = padInputType.getRank();
  PadSliceReduction reduction;
  reduction.inputStarts.resize(rank);
  reduction.inputEnds.resize(rank);
  reduction.lowPads.assign(rank, 0);
  reduction.highPads.assign(rank, 0);

  for (auto [axis, inDim] : llvm::enumerate(padInputType.getShape())) {
    const int64_t start = params.starts[axis];
    const int64_t size = params.ends[axis] - start;
    if (size <= 0)
      return failure();
    const int64_t lowPad = oldLowPads[axis];
    const int64_t highPad = oldHighPads[axis];
    const int64_t windowEnd = start + size;

    // Portion of the leading/trailing padding the window still reads.
    const int64_t newLow = std::clamp(lowPad - start, int64_t(0), size);
    const int64_t newHigh =
        std::max(int64_t(0), windowEnd - std::max(start, lowPad + inDim));
    // Sub-region of the (unpadded) input the window reads.
    const int64_t inputStart = std::max(start - lowPad, int64_t(0));
    const int64_t inputEnd = std::min(windowEnd - lowPad, inDim);
    if (inputEnd - inputStart <= 0)
      return failure();

    reduction.inputStarts[axis] = inputStart;
    reduction.inputEnds[axis] = inputEnd;
    reduction.lowPads[axis] = newLow;
    reduction.highPads[axis] = newHigh;
    reduction.needsInputSlice |= inputStart != 0 || inputEnd != inDim;
    reduction.changed |=
        reduction.needsInputSlice || newLow != lowPad || newHigh != highPad;
  }

  return reduction;
}

struct SlicePadPattern : public OpRewritePattern<ONNXSliceOp> {
  using OpRewritePattern<ONNXSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSliceOp sliceOp, PatternRewriter &rewriter) const override {
    auto padOp = sliceOp.getData().getDefiningOp<ONNXPadOp>();
    if (!padOp)
      return failure();
    if (!padOp->hasOneUse())
      return rewriter.notifyMatchFailure(
          sliceOp, "preceding Pad must have one use");
    auto padInputType = dyn_cast<RankedTensorType>(padOp.getData().getType());
    if (!padInputType || !padInputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          sliceOp, "Pad input must be a static ranked tensor");

    NormalizedSliceParams params;
    if (failed(getNormalizedSliceParams(sliceOp, params)) ||
        !hasOnlyUnitSteps(params.steps))
      return rewriter.notifyMatchFailure(
          sliceOp, "slice must have static, unit-step parameters");

    const int64_t rank = padInputType.getRank();
    SmallVector<int64_t> lowPads;
    SmallVector<int64_t> highPads;
    if (failed(getPadLowHighPads(padOp, rank, lowPads, highPads)))
      return rewriter.notifyMatchFailure(
          sliceOp, "Pad pads/axes must be static non-negative literals");

    FailureOr<PadSliceReduction> reduction =
        computePadSliceReduction(padInputType, params, lowPads, highPads);
    if (failed(reduction))
      return rewriter.notifyMatchFailure(sliceOp, "degenerate slice");
    if (!reduction->changed)
      return rewriter.notifyMatchFailure(sliceOp, "Pad cannot be reduced");

    const bool padRemains = reduction->padRemains();
    if (padRemains && padOp.getMode() != "constant")
      return rewriter.notifyMatchFailure(
          sliceOp, "non-constant Pad cannot be re-padded");

    Value newInput = padOp.getData();
    if (reduction->needsInputSlice) {
      newInput = createFullRankSlice(rewriter, sliceOp.getLoc(),
          padInputType.clone(reduction->inputShape()), padOp.getData(),
          reduction->inputStarts, reduction->inputEnds);
    }

    if (!padRemains) {
      rewriter.replaceOp(sliceOp, newInput);
      return success();
    }

    // pads laid out [begin_0..begin_{rank-1}, end_0..end_{rank-1}].
    Value newPadsValue =
        createI64TensorConstant(rewriter, padOp.getLoc(), reduction->pads());
    Value noneAxes = rewriter.create<ONNXNoneOp>(padOp.getLoc());
    Value newPad = rewriter.create<ONNXPadOp>(padOp.getLoc(),
        sliceOp.getOutput().getType(), newInput, newPadsValue,
        padOp.getConstantValue(), noneAxes, rewriter.getStringAttr("constant"));
    rewriter.replaceOp(sliceOp, newPad);
    return success();
  }
};

// A Slice reading from a Concat only needs the operands its window overlaps.
// Drop the others, shrink the Concat, and shift the slice onto the reduced
// Concat, i.e. slice(concat(A, B, C)) becomes slice(concat(B, C)):
//
//     concat(A, B, C)         offset:  0   4         10  13
//                                      +---+---------+---+
//                                      | A |    B    | C |
//                                      +---+---------+---+
//                                          |<--- want--->|
//     slice[4:13] on concat axis  ---------+-------------+
//
// The exact-one-operand case is not special-cased: keeping a single operand
// yields a single-input Concat that ConcatSingleOperandPattern removes, after
// which ONNXSliceOp::fold drops the now-identity slice, giving back that
// operand.

struct ConcatSliceReduction {
  SmallVector<Value> inputs;
  int64_t droppedSize = 0;
  int64_t concatDim = 0;
};

FailureOr<ConcatSliceReduction> computeConcatSliceReduction(
    ONNXConcatOp concatOp, int64_t concatAxis, int64_t start, int64_t end) {
  ConcatSliceReduction reduction;
  int64_t offset = 0;
  for (Value operand : concatOp.getInputs()) {
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandType || !operandType.hasStaticShape())
      return failure();
    const int64_t operandDim = operandType.getDimSize(concatAxis);
    if (offset < end && offset + operandDim > start) {
      if (reduction.inputs.empty())
        reduction.droppedSize = offset;
      reduction.inputs.push_back(operand);
      reduction.concatDim += operandDim;
    }
    offset += operandDim;
  }
  return reduction;
}

struct SliceConcatPattern : public OpRewritePattern<ONNXSliceOp> {
  using OpRewritePattern<ONNXSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSliceOp sliceOp, PatternRewriter &rewriter) const override {
    auto concatOp = sliceOp.getData().getDefiningOp<ONNXConcatOp>();
    if (!concatOp)
      return failure();
    auto concatType = dyn_cast<RankedTensorType>(sliceOp.getData().getType());
    if (!concatType || !concatType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          sliceOp, "Concat result must be a static ranked tensor");

    NormalizedSliceParams params;
    if (failed(getNormalizedSliceParams(sliceOp, params)) ||
        !hasOnlyUnitSteps(params.steps))
      return rewriter.notifyMatchFailure(
          sliceOp, "slice must have static, unit-step parameters");

    int64_t concatAxis = concatOp.getAxis();
    if (concatAxis < 0)
      concatAxis += concatType.getRank();

    for (auto [axis, dim] : llvm::enumerate(concatType.getShape())) {
      if (static_cast<int64_t>(axis) != concatAxis &&
          (params.starts[axis] != 0 || params.ends[axis] != dim))
        return rewriter.notifyMatchFailure(
            sliceOp, "slice is not full-range on a non-concat axis");
    }

    const int64_t start = params.starts[concatAxis];
    const int64_t end = params.ends[concatAxis];
    FailureOr<ConcatSliceReduction> reduction =
        computeConcatSliceReduction(concatOp, concatAxis, start, end);
    if (failed(reduction))
      return rewriter.notifyMatchFailure(
          sliceOp, "Concat inputs must be static ranked tensors");

    if (reduction->inputs.empty())
      return rewriter.notifyMatchFailure(sliceOp, "degenerate slice");
    if (reduction->inputs.size() == concatOp.getNumOperands())
      return rewriter.notifyMatchFailure(
          sliceOp, "Concat inputs cannot be reduced");
    // Keeping more than one operand rebuilds a Concat; only do that when the
    // original Concat has no other users, so we never introduce a new Concat.
    if (reduction->inputs.size() != 1 && !concatOp->hasOneUse())
      return rewriter.notifyMatchFailure(
          sliceOp, "would introduce an additional Concat");

    SmallVector<int64_t> newStarts(params.starts.begin(), params.starts.end());
    SmallVector<int64_t> newEnds(params.ends.begin(), params.ends.end());
    newStarts[concatAxis] = start - reduction->droppedSize;
    newEnds[concatAxis] = end - reduction->droppedSize;

    SmallVector<int64_t> newConcatShape(concatType.getShape());
    newConcatShape[concatAxis] = reduction->concatDim;
    Value newConcat = rewriter.create<ONNXConcatOp>(concatOp.getLoc(),
        concatType.clone(newConcatShape), reduction->inputs, concatAxis);
    Value newSlice = createFullRankSlice(rewriter, sliceOp.getLoc(),
        sliceOp.getOutput().getType(), newConcat, newStarts, newEnds);
    rewriter.replaceOp(sliceOp, newSlice);
    return success();
  }
};

} // namespace

void populateSliceOpOptimizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.insert<NormalizeSliceOperandsPattern, FuseSliceSlicePattern,
      SliceTilePattern, SlicePadPattern, SliceConcatPattern>(context);
}

} // namespace onnx_mlir
