/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXAveragePoolOp.cpp - ONNXAveragePoolOp --------------===//
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNXAveragePoolOp operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/ADT/ArrayRef.h"
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

LogicalResult handleIncludePadAttr(
    ConversionPatternRewriter &rewriter, Operation *op, Value input) {
  Location loc = op->getLoc();

  // Get shape.
  IndexExprBuilderForTosa createTosaIE(rewriter, loc);
  ONNXGenericPoolOpShapeHelper<ONNXAveragePoolOp> shapeHelper(
      op, {}, &createTosaIE);
  if (shapeHelper.computeShape().failed()) {
    return rewriter.notifyMatchFailure(op, "Could not infer shapes");
  }

  auto inputType = cast<mlir::TensorType>(input.getType());
  if (inputType.getShape().size() != 4) {
    return rewriter.notifyMatchFailure(op, "TOSA only supports 2d pooling");
  }

  llvm::SmallVector<int64_t, 4> pads =
      tosa::createOrderedPadAttrForWindowBasedOps(rewriter,
          cast<mlir::TensorType>(input.getType()).getShape(), shapeHelper,
          /*ceilMode*/ 0, {0, 1, 2, 3});

  // ONNX Pad uses [N_begin, C_begin, H_begin, W_begin, N_end, C_end,
  // H_end, W_end], with no padding on the batch or channel dimensions.
  OnnxBuilder onnxBuilder(rewriter, loc);
  Value padding = onnxBuilder.constantInt64(
      {0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]});
  auto zeroType = RankedTensorType::get({}, inputType.getElementType());
  Value zero = onnxBuilder.constant(rewriter.getZeroAttr(zeroType));
  Value padOp = onnxBuilder.pad(input, padding, zero, "constant");

  // In-place update of AveragePool by setting operand to PadOp
  // and pads attribute to {0, 0, 0, 0}.
  rewriter.modifyOpInPlace(op, [&]() { op->setOperand(0, padOp); });
  rewriter.modifyOpInPlace(op,
      [&]() { op->setAttr("pads", rewriter.getI32ArrayAttr({0, 0, 0, 0})); });
  return success();
}

class ONNXAveragePoolOpLoweringToTOSA
    : public OpConversionPattern<ONNXAveragePoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXAveragePoolOp averagePoolOp,
      ONNXAveragePoolOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    // With 'countIncludePad' and 'ceilMode', there are 4 cases of attributes
    // that have an impact on the padding of the TOSA AveragePoolOp. The only
    // configuration that is natively supported by TOSA is if both are 0. In
    // that case we will have a 1-1 conversion. If only 'countIncludePad' is
    // set, we pull the padding out of the averagepool op into its own ONNX
    // Pad op with the padding value 0. That way the averagepool treats the
    // padded values as native input values. The padding attribute is then set
    // to 0. If 'ceilMode' is set, we add padding in the pad attribute of the
    // TOSA AveragePoolOp if necessary. Looking into
    // https://github.com/pytorch/pytorch/blob/efc7c366f4fbccf649454726a96df291c8a9df43/aten/src/ATen/native/cuda/AveragePool2d.cu#L243
    // 'ceilMode' does not seem to add implicit padding but just changes the
    // output size. If both are set, we first pull out the padding into its own
    // op and then check if we need to add padding for the ceilMode.

    const int64_t includePad = adaptor.getCountIncludePad();

    if (includePad != 0) {
      // When attribute include_pad is set, create an ONNXPadOp before lowering
      // AveragePool to TOSA. We use ONNX format for it so that the AveragePool
      // lowering still generates transposes between ONNX and TOSA formats, and
      // implementation doesn't diverge much. This will modify the original onnx
      // op.
      if (failed(handleIncludePadAttr(rewriter, averagePoolOp, adaptor.getX())))
        return failure();
    }

    FailureOr<Value> newAveragePoolOp =
        tosa::convertPoolOp<ONNXAveragePoolOp, mlir::tosa::AvgPool2dOp>(
            rewriter, averagePoolOp);

    if (failed(newAveragePoolOp)) {
      return rewriter.notifyMatchFailure(
          averagePoolOp, "Could not create averagepool op.");
    }
    rewriter.replaceOp(averagePoolOp, *newAveragePoolOp);
    return success();
  }
};

} // namespace

void populateLoweringONNXAveragePoolOpToTOSAPattern(
    ConversionTarget & /*target*/, RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXAveragePoolOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
