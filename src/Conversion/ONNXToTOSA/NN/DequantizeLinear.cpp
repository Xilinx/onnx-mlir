/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXDequantizeLinearOp.cpp - ONNXDequantizeLinearOp-----===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNXDequantizeLinearOp operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXDequantizeLinearOpLoweringToTOSA
    : public OpConversionPattern<ONNXDequantizeLinearOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXDequantizeLinearOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    Value x = op.getX();
    auto resultType = dyn_cast_if_present<ShapedType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          loc, "expected valid tensor result type");
    }

    if (auto zpTy = dyn_cast<ShapedType>(adaptor.getXZeroPoint().getType());
        !zpTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          loc, "expected zero point to have static shape");
    }

    if (auto zpTy = dyn_cast<ShapedType>(adaptor.getXScale().getType());
        !zpTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          loc, "expected scale to have static shape");
    }

    // Since tosa.add and tosa.mul don't allow different ranks, get the value
    // from the constants, and create a new constant of the same rank as the
    // input out of it in order to have a correct add and mul.
    auto zpConst = tosa::expandShape(rewriter, loc, adaptor.getXZeroPoint(),
        op.getAxis(), resultType.getRank());
    auto scaleFactorConst = tosa::expandShape(
        rewriter, loc, adaptor.getXScale(), op.getAxis(), resultType.getRank());

    // Dequantization formula is (x - zero_point) * scale
    // Cast into the destination type first
    Value subOp = tosa::CreateOpAndInfer<mlir::tosa::SubOp>(
        rewriter, loc, x.getType(), x, zpConst)
                      .getResult();
    Value castOp = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
        rewriter, loc, resultType, subOp)
                       .getResult();
    Value mulOp = tosa::CreateOpAndInfer<mlir::tosa::MulOp>(
        rewriter, loc, resultType, castOp, scaleFactorConst, 0)
                      .getResult();

    rewriter.replaceOp(op, mulOp);
    return success();
  }
};

} // namespace

void populateLoweringONNXDequantizeLinearOpToTOSAPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXDequantizeLinearOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
