/*
 * SPDX-License-Identifier: Apache-2.0
 * Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
 * affiliates
 */

#include "XFECast.hpp"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

/// Remove an unobservable intermediate cast while retaining the outer XFE
/// policy, which governs the only conversion that can lose information.
struct FoldConsecutiveXFECastPattern : OpRewritePattern<XFECastOp> {
  using OpRewritePattern<XFECastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      XFECastOp outerCast, PatternRewriter &rewriter) const override {
    Operation *innerCast = outerCast.getInput().getDefiningOp();
    Value source;
    if (auto xfeCast = dyn_cast_or_null<XFECastOp>(innerCast))
      source = xfeCast.getInput();
    else if (auto onnxCast = dyn_cast_or_null<ONNXCastOp>(innerCast))
      source = onnxCast.getInput();
    else
      return failure();

    if (!innerCast->getResult(0).hasOneUse())
      return failure();

    Type src = getElementTypeOrSelf(source.getType());
    Type mid = getElementTypeOrSelf(innerCast->getResult(0).getType());
    Type dst = getElementTypeOrSelf(outerCast.getOutput().getType());
    if (!canElideIntermediateCast(src, mid, dst))
      return failure();

    rewriter.modifyOpInPlace(
        outerCast, [&]() { outerCast.getInputMutable().assign(source); });
    return success();
  }
};

} // namespace

XFECastOp createXFECastFromONNXCast(OpBuilder &builder, ONNXCastOp castOp,
    Value input, ONNXCastToXFEOptions options) {
  Type inputElementType = getElementTypeOrSelf(input.getType());
  Type outputElementType = castOp.getTo();

  StringRef roundMode = "nearest_even";
  if (isa<FloatType>(inputElementType) && isa<IntegerType>(outputElementType) &&
      !options.roundFPToIntToNearestEven)
    roundMode = "toward_zero";

  // ONNX saturation only affects float8 destinations. XFECast deliberately
  // extends it to every floating-point destination, so translating any other
  // ONNX Cast must disable it explicitly.
  StringRef saturationMode = "none";
  if (auto floatType = dyn_cast<FloatType>(outputElementType);
      floatType && floatType.getWidth() == 8)
    saturationMode = castOp.getSaturate() != 0 ? "finite" : "none";

  return builder.create<XFECastOp>(castOp.getLoc(), castOp.getType(), input,
      "propagate", roundMode, saturationMode, outputElementType);
}

} // namespace onnx_mlir

void mlir::XFECastOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<onnx_mlir::FoldConsecutiveXFECastPattern>(context);
}
