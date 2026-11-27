/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Shrink.cpp - Shrink Op-------------------------------===//
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers the ONNX Shrink operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXShrinkOpLoweringToTOSA : public OpConversionPattern<ONNXShrinkOp> {
public:
  using OpConversionPattern<ONNXShrinkOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXShrinkOp shrinkOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = shrinkOp->getLoc();

    auto lambd = adaptor.getLambdAttr();
    auto bias = adaptor.getBiasAttr();
    auto input = adaptor.getInput();

    TosaBuilder tosaBuilder(rewriter, loc);
    OnnxBuilder onnxBuilder(rewriter, loc);

    auto inputRankedTensorTy = dyn_cast<RankedTensorType>(input.getType());
    if (!inputRankedTensorTy) {
      return rewriter.notifyMatchFailure(
          loc, "Expected RankedTensorType for input data of ShrinkOp");
    }

    // lambd and bias have float type so it is safe to conver it to float
    const float lambdAsFloat = lambd.getValue().convertToFloat();
    const float biasAsFloat = bias.getValue().convertToFloat();
    auto lambdConstOp = tosaBuilder.getSplattedConst(lambdAsFloat,
        inputRankedTensorTy.getElementType(), inputRankedTensorTy.getRank());
    auto negatedLambdConstOp = tosaBuilder.getSplattedConst(-lambdAsFloat,
        inputRankedTensorTy.getElementType(), inputRankedTensorTy.getRank());
    auto biasConstOp = tosaBuilder.getSplattedConst(biasAsFloat,
        inputRankedTensorTy.getElementType(), inputRankedTensorTy.getRank());
    auto zeroConstOp = tosaBuilder.getSplattedConst(
        0, inputRankedTensorTy.getElementType(), inputRankedTensorTy.getRank());

    // Formula to be implemented:
    // { x < -lambd, then y = x + bias
    // { x > lambd,  then y = x - bias
    // { otherwise,  then y = 0

    Value firstCmp = onnxBuilder.createOpAndInferShapes<ONNXGreaterOp>(
        negatedLambdConstOp, input);
    auto firstFormula = onnxBuilder.add(input, biasConstOp);
    auto firstSelect =
        onnxBuilder.where(input.getType(), firstCmp, firstFormula, zeroConstOp);

    Value secondCmp =
        onnxBuilder.createOpAndInferShapes<ONNXGreaterOp>(input, lambdConstOp);
    auto secondFormula = onnxBuilder.sub(input, biasConstOp);
    auto secondSelect = onnxBuilder.where(
        input.getType(), secondCmp, secondFormula, firstSelect);

    rewriter.replaceOp(shrinkOp, secondSelect);

    return success();
  }
};

} // namespace

void populateLoweringONNXShrinkOpToTOSAPattern(ConversionTarget & /*target*/,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXShrinkOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
