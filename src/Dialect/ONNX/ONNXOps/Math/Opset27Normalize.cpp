// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates.

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

Type arithmeticType(MLIRContext *ctx, int64_t dtype) {
  switch (dtype) {
  case onnx::TensorProto::FLOAT:
    return Float32Type::get(ctx);
  case onnx::TensorProto::DOUBLE:
    return Float64Type::get(ctx);
  case onnx::TensorProto::FLOAT16:
    return Float16Type::get(ctx);
  case onnx::TensorProto::BFLOAT16:
    return BFloat16Type::get(ctx);
  default:
    return {};
  }
}

Type elementType(Value v) {
  auto type = dyn_cast<ShapedType>(v.getType());
  return type ? type.getElementType() : Type();
}

// Make arithmetic precision explicit using existing casts so downstream Q/DQ
// patterns need not infer it from independent input, scale and result types.
struct NormalizeQuantize : OpRewritePattern<ONNXQuantizeLinearOp> {
  explicit NormalizeQuantize(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/20) {}

  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp op, PatternRewriter &rewriter) const override {
    Type scaleType = elementType(op.getYScale());
    Type computeType = op.getPrecision()
                           ? arithmeticType(op.getContext(), op.getPrecision())
                           : scaleType;
    if (!computeType)
      return failure(); // The verifier diagnoses unsupported precision.
    if (!op.getPrecision() && elementType(op.getX()) == scaleType)
      return failure();
    OnnxBuilder create(rewriter, op.getLoc());
    Value x = op.getX();
    Value scale = op.getYScale();
    if (elementType(x) != computeType)
      x = create.cast(x, computeType);
    if (elementType(scale) != computeType)
      scale = create.cast(scale, computeType);
    rewriter.modifyOpInPlace(op, [&] {
      op->setOperand(0, x);
      op->setOperand(1, scale);
      op.setPrecision(0);
    });
    return success();
  }
};

struct NormalizeDequantize : OpRewritePattern<ONNXDequantizeLinearOp> {
  explicit NormalizeDequantize(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/20) {}

  LogicalResult matchAndRewrite(
      ONNXDequantizeLinearOp op, PatternRewriter &rewriter) const override {
    Type scaleType = elementType(op.getXScale());
    Type computeType = op.getOutputDtype() ? arithmeticType(op.getContext(),
                                                 op.getOutputDtype())
                                           : scaleType;
    if (!computeType || !op.getOutputDtype())
      return failure();
    OnnxBuilder create(rewriter, op.getLoc());
    Value scale = op.getXScale();
    if (scaleType != computeType)
      scale = create.cast(scale, computeType);
    rewriter.modifyOpInPlace(op, [&] {
      op->setOperand(1, scale);
      op.setOutputDtype(0);
    });
    return success();
  }
};

struct NormalizeRange : OpRewritePattern<ONNXRangeOp> {
  explicit NormalizeRange(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/20) {}

  LogicalResult matchAndRewrite(
      ONNXRangeOp op, PatternRewriter &rewriter) const override {
    Type inputType = elementType(op.getStart());
    if (!inputType || (!inputType.isF16() && !inputType.isBF16()))
      return failure();
    Type computeType = arithmeticType(op.getContext(), op.getStashType());
    if (!computeType || computeType == inputType)
      return failure();
    OnnxBuilder create(rewriter, op.getLoc());
    auto resultType = cast<ShapedType>(op.getOutput().getType());
    Type widenedType = resultType.clone(computeType);
    auto range = rewriter.create<ONNXRangeOp>(op.getLoc(), widenedType,
        create.cast(op.getStart(), computeType),
        create.cast(op.getLimit(), computeType),
        create.cast(op.getDelta(), computeType), op.getStashTypeAttr());
    rewriter.replaceOp(op, create.cast(range.getOutput(), inputType));
    return success();
  }
};

} // namespace

void populateOpset27NormalizationPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeQuantize, NormalizeDequantize, NormalizeRange>(
      patterns.getContext());
}
} // namespace onnx_mlir
