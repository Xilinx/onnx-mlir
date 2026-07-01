/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ XFECanonicalize.cpp - XFE Op Canonicalizers ------------===//
//
// Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
//
// =============================================================================
//
// This file implements canonicalizers for the com.amd.xfe (channel-last) ONNX
// ops. It is kept separate from ONNXOps/Canonicalize.cpp so that the core ONNX
// canonicalizer file does not keep growing.
//
// When adding a canonicalizer for a new XFE operation, please add that
// operation to the OpsWithCanonicalizer list in utils/gen_onnx_mlir.py.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallVector.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {

// =============================================================================
// Folds XFEAveragePool with ceil_mode=1 into an equivalent floor-mode pool by
// enlarging the end-padding so the floor output-shape formula reproduces the
// original (ceil) output shape.
//
// The window positions are unchanged, and count_include_pad is preserved, so
// the numerical result is identical: the extra end-padding is treated exactly
// like the ceil overhang. The trailing ceil window always starts inside the
// real input, so no all-padding (divide-by-zero) window is introduced.
//
// XFEAveragePool uses channel-last layout: X is [N, spatial..., C], and pads
// follow [begin_0..begin_{n-1}, end_0..end_{n-1}].
// =============================================================================
struct XFEAveragePoolCeilModeToPadPattern
    : public OpRewritePattern<XFEAveragePoolOp> {
  using OpRewritePattern<XFEAveragePoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      XFEAveragePoolOp poolOp, PatternRewriter &rewriter) const override {
    auto ceilModeAttr = poolOp.getCeilModeAttr();
    if (!ceilModeAttr || ceilModeAttr.getSInt() == 0)
      return failure();

    // auto_pad SAME_* derives the output shape independently of ceil_mode, so
    // there is nothing to fold; only handle explicit padding.
    if (poolOp.getAutoPad() != "NOTSET")
      return rewriter.notifyMatchFailure(poolOp, "auto_pad is not NOTSET");

    auto inputType = mlir::dyn_cast<RankedTensorType>(poolOp.getX().getType());
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(poolOp.getResult().getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(poolOp, "unranked input/output");

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    int64_t rank = inputType.getRank();
    if (rank < 3 || outputType.getRank() != rank)
      return rewriter.notifyMatchFailure(poolOp, "unexpected rank");
    int64_t numSpatialDims = rank - 2;

    // Reads element `i` of an optional i64 array attribute (kernel_shape,
    // strides, dilations, pads), falling back to `dflt` when the attribute is
    // absent or shorter than expected.
    auto attrOr = [](std::optional<ArrayAttr> arr, int64_t i, int64_t dflt) {
      if (arr && i < static_cast<int64_t>(arr->size()))
        return mlir::cast<IntegerAttr>((*arr)[i]).getInt();
      return dflt;
    };

    std::optional<ArrayAttr> kernelShape = poolOp.getKernelShape();
    std::optional<ArrayAttr> strides = poolOp.getStrides();
    std::optional<ArrayAttr> dilations = poolOp.getDilations();
    std::optional<ArrayAttr> pads = poolOp.getPads();

    // Pads are laid out [begin_0..begin_{n-1}, end_0..end_{n-1}]; only the end
    // pads grow, begins pass through unchanged.
    SmallVector<int64_t, 4> beginPads;
    SmallVector<int64_t, 4> endPads;
    for (int64_t i = 0; i < numSpatialDims; ++i) {
      int64_t inputDim = inputShape[i + 1];
      int64_t outputDim = outputShape[i + 1];
      if (inputDim == ShapedType::kDynamic || outputDim == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(poolOp, "dynamic spatial dim");

      int64_t kernel = attrOr(kernelShape, i, 1);
      int64_t stride = attrOr(strides, i, 1);
      int64_t dilation = attrOr(dilations, i, 1);
      int64_t beginPad = attrOr(pads, i, 0);
      int64_t endPad = attrOr(pads, numSpatialDims + i, 0);

      // Extra end padding that turns the trailing ceil-mode overhang into
      // explicit padding, so the floor output-shape formula reproduces the
      // op's ceil output dim -- keeping the result type unchanged.
      int64_t effectiveKernel = (kernel - 1) * dilation + 1;
      int64_t delta = (outputDim - 1) * stride + effectiveKernel -
                      (inputDim + beginPad + endPad);

      beginPads.push_back(beginPad);
      endPads.push_back(delta > 0 ? endPad + delta : endPad);
    }

    SmallVector<int64_t, 8> newPads(beginPads.begin(), beginPads.end());
    newPads.append(endPads.begin(), endPads.end());

    auto zeroCeilMode = rewriter.getIntegerAttr(
        rewriter.getIntegerType(64, /*isSigned=*/true), 0);

    rewriter.modifyOpInPlace(poolOp, [&]() {
      poolOp->setAttr(
          poolOp.getPadsAttrName(), rewriter.getI64ArrayAttr(newPads));
      poolOp->setAttr(poolOp.getCeilModeAttrName(), zeroCeilMode);
    });

    return success();
  }
};

} // namespace

/// on the XFEAveragePoolOp.
void XFEAveragePoolOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<XFEAveragePoolCeilModeToPadPattern>(context);
}
