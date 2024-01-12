/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include <src/Conversion/ONNXToTOSA/DialectBuilder.hpp>

using namespace mlir;

namespace onnx_mlir {

template <>
struct TOSADialectOp<ONNXNegOp> {
  using Op = mlir::tosa::NegateOp;
};

struct AbsIOSupportedTypes {
  static LogicalResult checkType(
      ConversionPatternRewriter &rewriter, Type scalarType, Operation *op) {
    if (!mlir::isa<mlir::BFloat16Type, mlir::Float16Type, mlir::Float32Type>(
            scalarType) &&
        !scalarType.isSignlessInteger(/*width=*/32)) {
      return rewriter.notifyMatchFailure(op,
          "this operation only supports signless 32 integer or fp16, fp32"
          " or bf16");
    }
    return success();
  }
};

struct ErfIOSupportedTypes {
  static LogicalResult checkType(
      ConversionPatternRewriter &rewriter, Type scalarType, Operation *op) {
    if (!mlir::isa<mlir::BFloat16Type, mlir::Float16Type, mlir::Float32Type>(
            scalarType)) {
      return rewriter.notifyMatchFailure(
          op, "this operation only supports fp16, fp32 or bf16");
    }
    return success();
  }
};

struct IsIntOrFloat {
  static LogicalResult checkType(
      ConversionPatternRewriter &rewriter, Type scalarType, Operation *op) {
    if (!isTOSAFloat(scalarType) && !isTOSASignedInt(scalarType)) {
      return rewriter.notifyMatchFailure(
          op, "this operation only supports signed integer or float types");
    }
    return success();
  }
};

struct IsInt {
  static LogicalResult checkType(
      ConversionPatternRewriter &rewriter, Type scalarType, Operation *op) {
    if (!isTOSASignedInt(scalarType)) {
      return rewriter.notifyMatchFailure(
          op, "this operation only supports int types");
    }
    return success();
  }
};

struct IsFloat {
  static LogicalResult checkType(
      ConversionPatternRewriter &rewriter, Type scalarType, Operation *op) {
    if (!isTOSAFloat(scalarType)) {
      return rewriter.notifyMatchFailure(
          op, "this operation only supports float types");
    }
    return success();
  }
};

struct IsBool {
  static LogicalResult checkType(
      ConversionPatternRewriter &rewriter, Type scalarType, Operation *op) {
    if (!isTOSABool(scalarType)) {
      return rewriter.notifyMatchFailure(
          op, "this operation only supports bool type");
    }
    return success();
  }
};

template <typename OpAdaptorT, typename TypeChecker>
LogicalResult checkBasicTosaRequirementsForBinaryOps(
    ConversionPatternRewriter &rewriter, Operation *op, OpAdaptorT adaptor,
    Type resultType) {
  Value lhs = adaptor.getOperands()[0];
  auto lhsType = lhs.getType().dyn_cast<TensorType>();

  Value rhs = adaptor.getOperands()[1];
  auto rhsType = rhs.getType().dyn_cast<TensorType>();

  auto resultTensorType = resultType.dyn_cast<TensorType>();
  if (!lhsType || !rhsType || !resultTensorType) {
    return rewriter.notifyMatchFailure(op, "Tosa only supports TensorTypes");
  }

  Type resultElementType = resultTensorType.getElementType();

  if (failed(TypeChecker::checkType(rewriter, resultElementType, op))) {
    return failure();
  }

  return success();
}

// Element-wise unary ops lowering to custom op of TOSA dialect.
//===----------------------------------------------------------------------===//
template <typename ONNXOp>
class ConvertONNXUnaryOpToTosaCustomOp : public OpConversionPattern<ONNXOp> {
public:
  using OpConversionPattern<ONNXOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXOp::Adaptor;

  ConvertONNXUnaryOpToTosaCustomOp(TypeConverter &typeConverter,
      MLIRContext *context, std::string opName,
      std::string implementedWithOpAttr = "UNDEF")
      : OpConversionPattern<ONNXOp>(typeConverter, context),
        opName(std::move(opName)),
        implementedWithOpAttr(std::move(implementedWithOpAttr)) {}

  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Set tosa.custom_op attributes.
    // Only identifier needs to be known. Other attributes are not used.
    auto *ctx = op->getContext();
    auto identifier = StringAttr::get(ctx, opName);
    auto implementAttr = StringAttr::get(ctx, implementedWithOpAttr);
    auto config = StringAttr::get(ctx, "UNDEF");

    rewriter.replaceOpWithNewOp<mlir::tosa::CustomOp>(op,
        TypeRange{OpConversionPattern<ONNXOp>::getTypeConverter()->convertType(
            op.getType())},
        identifier, config, implementAttr, adaptor.getOperands());
    return success();
  }

private:
  std::string opName;
  std::string implementedWithOpAttr;
};

// Element-wise unary ops lowering to TOSA dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOpONNX, typename ElementwiseUnaryOpTOSA,
    typename InputType, typename OutputType>
class ONNXElementwiseUnaryOpLoweringToTOSA
    : public OpConversionPattern<ElementwiseUnaryOpONNX> {
public:
  using OpConversionPattern<ElementwiseUnaryOpONNX>::OpConversionPattern;
  using OpAdaptor = typename ElementwiseUnaryOpONNX::Adaptor;
  LogicalResult matchAndRewrite(ElementwiseUnaryOpONNX op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value input = *adaptor.getODSOperands(0).begin();
    auto inputType = input.getType().dyn_cast<TensorType>();
    Value output = op.getResult();
    auto outputType = output.getType().dyn_cast<TensorType>();

    if (!inputType || !outputType) {
      return rewriter.notifyMatchFailure(op, "Tosa only supports TensorTypes");
    }

    Type inputElementType = inputType.getElementType();
    Type outputElementType = outputType.getElementType();

    if (failed(InputType::checkType(rewriter, inputElementType, op)))
      return failure();

    if (failed(InputType::checkType(rewriter, outputElementType, op)))
      return failure();

    rewriter.replaceOpWithNewOp<ElementwiseUnaryOpTOSA>(
        op, op.getType(), *adaptor.getODSOperands(0).begin());
    return success();
  }
};

template <typename ONNXOpT, typename TosaOpT, typename TypeChecker>
class ONNXBinaryElementwiseOpLoweringToTOSA
    : public OpConversionPattern<ONNXOpT> {
public:
  using OpConversionPattern<ONNXOpT>::OpConversionPattern;
  using OpAdaptor = typename ONNXOpT::Adaptor;
  LogicalResult matchAndRewrite(ONNXOpT op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    if (failed(checkBasicTosaRequirementsForBinaryOps<OpAdaptor, TypeChecker>(
            rewriter, op, adaptor, op.getResult().getType())))
      return failure();

    auto loc = op.getLoc();
    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];

    if (TosaOpT::template hasTrait<
            mlir::OpTrait::ResultsBroadcastableShape>()) {

      IndexExprBuilderForTosa createTosaIE(rewriter, op->getLoc());
      ONNXBroadcastOpShapeHelper shapeHelper(op, {}, &createTosaIE);
      shapeHelper.computeShapeAndAssertOnFailure();

      if (shapeHelper.hasRankBroadcast()) {
        TosaBuilder tosaBuilder(rewriter, loc);
        llvm::SmallVector<Value, 4> newValues =
            tosaBuilder.equalizeRanks({lhs, rhs});
        lhs = newValues[0];
        rhs = newValues[1];
      }
    }

    rewriter.replaceOpWithNewOp<TosaOpT>(op, op.getType(), lhs, rhs);

    return success();
  }
};

class ONNXMulOpLoweringToTosa : public OpConversionPattern<ONNXMulOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (failed(checkBasicTosaRequirementsForBinaryOps<OpAdaptor, IsIntOrFloat>(
            rewriter, op, adaptor, op.getResult().getType())))
      return failure();

    Value lhs = adaptor.getA();
    Value rhs = adaptor.getB();

    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    Value mulOp = tosaBuilder.mul(lhs, rhs);
    rewriter.replaceOp(op, {mulOp});

    return success();
  }
};

class ONNXReluOpLoweringToTOSA : public OpConversionPattern<ONNXReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getX();

    // Quantized types are not supported right now (in type conversion).
    // Once they are, the input should be rescaled for quantized types. (TBD)
    // Maps to `tosa.clamp` which has both int and fp limits.
    rewriter.replaceOpWithNewOp<mlir::tosa::ClampOp>(op, op.getType(), input,
        rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(std::numeric_limits<int32_t>::max()),
        rewriter.getF32FloatAttr(0.0f),
        rewriter.getF32FloatAttr(std::numeric_limits<float>::max()));
    return success();
  }
};

// Support for prelu/leakyrelu adapted from tensorflow to tosa implementation
static LogicalResult legalizeFloatingPointPrelu(Operation *op,
    PatternRewriter &rewriter, Value input, Value alphaOrSlope,
    TensorType outputType) {
  auto loc = op->getLoc();
  TosaBuilder tosaBuilder(rewriter, loc);
  Value constZero = tosaBuilder.getSplattedConst(
      0.0, outputType.getShape(), outputType.getElementType());

  auto mul = tosaBuilder.mul(input, alphaOrSlope);
  auto greaterEqual = tosaBuilder.greaterEqual(input, constZero);
  auto select = tosaBuilder.select(greaterEqual, input, mul);

  rewriter.replaceOp(op, {select});
  return success();
}

class ONNXLeakyReluOpLoweringToTOSA
    : public OpConversionPattern<ONNXLeakyReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = ONNXLeakyReluOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXLeakyReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto outputType = op.getResult().getType().cast<TensorType>();
    if (failed(IsIntOrFloat::checkType(
            rewriter, outputType.getElementType(), op))) {
      return failure();
    }

    // ONNX docs: alpha : float (default 0.01)
    float alpha = 0.01;
    FloatAttr alphaAttr = adaptor.getAlphaAttr();
    if (alphaAttr) {
      // No easy interface in MLIR to get value as float
      alpha = alphaAttr.getValueAsDouble();
    }
    auto loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);
    return legalizeFloatingPointPrelu(op, rewriter, adaptor.getX(),
        tosaBuilder.getSplattedConst(
            alpha, outputType.getShape(), outputType.getElementType()),
        outputType);
  }
};

class ONNXClipOpLoweringToTOSA : public OpConversionPattern<ONNXClipOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = ONNXClipOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXClipOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto res = adaptor.getInput();
    auto min = adaptor.getMin();
    auto max = adaptor.getMax();

    auto matchIntOrFloat = [&](Value val) -> std::tuple<bool, int64_t, float> {
      APInt valueInt(64, 0);
      APFloat valueFloat(0.0f);
      if (matchPattern(val, m_ConstantInt(&valueInt))) {
        auto intVal = valueInt.getSExtValue();
        return {true, intVal, static_cast<float>(intVal)};
      }
      if (matchPattern(val, m_ConstantFloat(&valueFloat))) {
        float floatVal = valueFloat.convertToFloat();
        return {true, static_cast<int64_t>(floatVal), floatVal};
      }
      return {false, 0, 0.0};
    };

    // Use ClampOp if min and max are splat constants.
    // Otherwise, MaximumOp and MinimumOp to clamp min and max, respectively.
    auto [isSplatConstMin, minInt, minFloat] = matchIntOrFloat(min);
    auto [isSplatConstMax, maxInt, maxFloat] = matchIntOrFloat(max);
    if (isSplatConstMin && isSplatConstMax) {
      rewriter.replaceOpWithNewOp<mlir::tosa::ClampOp>(op, op.getType(), res,
          rewriter.getI64IntegerAttr(minInt),
          rewriter.getI64IntegerAttr(maxInt),
          rewriter.getF32FloatAttr(minFloat),
          rewriter.getF32FloatAttr(maxFloat));
    } else {
      if (!isNoneValue(min)) {
        res = tosa::CreateOpAndInfer<mlir::tosa::MaximumOp>(
            rewriter, op->getLoc(), op.getType(), res, min);
      }
      if (!isNoneValue(max)) {
        res = tosa::CreateOpAndInfer<mlir::tosa::MinimumOp>(
            rewriter, op->getLoc(), op.getType(), res, max);
      }
      rewriter.replaceOp(op, res);
    }
    return success();
  }
};

class ONNXDivOpLoweringToTOSA : public OpConversionPattern<ONNXDivOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXDivOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getA();
    Value rhs = adaptor.getB();
    auto resultType = op.getResult().getType().template cast<TensorType>();
    Type resultElementType = resultType.getElementType();

    TosaBuilder tosaBuilder(rewriter, op->getLoc());

    if (resultElementType.isSignlessInteger(32)) {
      // tosa::DivOp takes 32-but signless integers as inputs
      Value divOp = tosaBuilder.intdiv(lhs, rhs);
      rewriter.replaceOp(op, {divOp});
      return success();
    }
    // If it is not a 32-bit signless integer, decompose ONNXDivOp into
    // tosa::ReciprocalOp and tosa::MulOp
    Value reciprocalOp = tosaBuilder.unaryOp<mlir::tosa::ReciprocalOp>(rhs);
    Value mulOp = tosaBuilder.mul(lhs, reciprocalOp);
    rewriter.replaceOp(op, {mulOp});
    return success();
  }
};

class ONNXSqrtOpLoweringToTOSA : public OpConversionPattern<ONNXSqrtOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXSqrtOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto resultTensorType = op.getResult().getType().cast<TensorType>();
    if (failed(IsFloat::checkType(
            rewriter, resultTensorType.getElementType(), op))) {
      return failure();
    }

    Value input = op.getX();
    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    Value sqrtOp = tosaBuilder.sqrt(input);
    rewriter.replaceOp(op, {sqrtOp});
    return success();
  }
};

class ONNXEluOpLoweringToTOSA : public OpConversionPattern<ONNXEluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXEluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // ELU(x) = x                     if x >= 0
    //         alpha * (exp(x) - 1.)  if x <  0

    auto resultTensorType = op.getResult().getType().cast<TensorType>();
    if (failed(IsFloat::checkType(
            rewriter, resultTensorType.getElementType(), op))) {
      return failure();
    }

    Value input = op.getX();

    TosaBuilder tosaBuilder(rewriter, op->getLoc());

    Value one = tosaBuilder.getSplattedConst(
        1.0, resultTensorType.getShape(), resultTensorType.getElementType());
    Value alpha =
        tosaBuilder.getSplattedConst(adaptor.getAlpha().convertToDouble(),
            resultTensorType.getShape(), resultTensorType.getElementType());
    Value constZero = tosaBuilder.getSplattedConst(
        0.0, resultTensorType.getShape(), resultTensorType.getElementType());

    Value exp = tosaBuilder.unaryOp<mlir::tosa::ExpOp>(input);
    Value expMinusOne = tosaBuilder.binaryOp<mlir::tosa::SubOp>(exp, one);
    Value alphaTimesExpMinusOne = tosaBuilder.mul(expMinusOne, alpha);
    Value greaterEqual = tosaBuilder.greaterEqual(input, constZero);
    auto select =
        tosaBuilder.select(greaterEqual, input, alphaTimesExpMinusOne);

    rewriter.replaceOp(op, {select});
    return success();
  }
};

class ONNXHardSigmoidOpLoweringToTOSA
    : public OpConversionPattern<ONNXHardSigmoidOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXHardSigmoidOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // ONNXHardSigmoid -> TOSA:
    // - tosa.add(input, beta/alpha)
    // - tosa.clamp(add) with min = 0, and max = 1/alpha
    // - tosa.mul(clamp, alpha)
    Value input = adaptor.getX();

    auto resultType = op.getResult().getType().template cast<TensorType>();
    auto resultElementType = resultType.getElementType();

    TosaBuilder tosaBuilder(rewriter, op->getLoc());

    auto alpha = adaptor.getAlpha();

    auto betaOverAlpha = adaptor.getBeta();
    betaOverAlpha.divide(alpha, APFloat::rmNearestTiesToEven);

    APFloat oneOverAlpha(alpha.getSemantics(), 1);
    oneOverAlpha.divide(alpha, APFloat::rmNearestTiesToEven);

    Value constBetaOverAlpha =
        tosaBuilder.getSplattedConst(betaOverAlpha.convertToDouble(),
            resultType.getShape(), resultElementType);
    Value constAlpha = tosaBuilder.getSplattedConst(
        alpha.convertToDouble(), resultType.getShape(), resultElementType);

    auto addOp =
        tosaBuilder.binaryOp<mlir::tosa::AddOp>(input, constBetaOverAlpha);
    Value clampOp = tosa::CreateOpAndInfer<mlir::tosa::ClampOp>(rewriter,
        op->getLoc(), resultType, addOp, rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(oneOverAlpha.convertToDouble()),
        rewriter.getF32FloatAttr(0),
        rewriter.getF32FloatAttr(oneOverAlpha.convertToDouble()));
    auto mulOp = tosaBuilder.mul(clampOp, constAlpha);

    rewriter.replaceOp(op, {mulOp});
    return success();
  }
};

class ONNXPReluOpLoweringToTOSA : public OpConversionPattern<ONNXPReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = ONNXPReluOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXPReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto outputType = op.getResult().getType().cast<TensorType>();
    if (failed(IsIntOrFloat::checkType(
            rewriter, outputType.getElementType(), op))) {
      return failure();
    }

    return legalizeFloatingPointPrelu(
        op, rewriter, adaptor.getX(), adaptor.getSlope(), outputType);
  }
};

class ONNXSoftplusOpLoweringToTOSA
    : public OpConversionPattern<ONNXSoftplusOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXSoftplusOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto outputType = op.getResult().getType().cast<TensorType>();
    if (failed(IsFloat::checkType(rewriter, outputType.getElementType(), op))) {
      return failure();
    }

    Value input = adaptor.getX();

    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    auto one = tosaBuilder.getSplattedConst(
        1.0, outputType.getShape(), outputType.getElementType());

    auto expOp = tosaBuilder.unaryOp<mlir::tosa::ExpOp>(input);
    auto expPlusOne = tosaBuilder.binaryOp<mlir::tosa::AddOp>(expOp, one);
    auto logOp = tosaBuilder.unaryOp<mlir::tosa::LogOp>(expPlusOne);
    rewriter.replaceOp(op, {logOp});
    return success();
  }
};

class ONNXSeluOpLoweringToTOSA : public OpConversionPattern<ONNXSeluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXSeluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto outputType = op.getResult().getType().cast<TensorType>();
    if (failed(IsFloat::checkType(rewriter, outputType.getElementType(), op))) {
      return failure();
    }

    Value input = adaptor.getX();

    TosaBuilder tosaBuilder(rewriter, op->getLoc());

    Value alpha =
        tosaBuilder.getSplattedConst(adaptor.getAlpha().convertToDouble(),
            outputType.getShape(), outputType.getElementType());
    Value gamma =
        tosaBuilder.getSplattedConst(adaptor.getGamma().convertToDouble(),
            outputType.getShape(), outputType.getElementType());
    Value constZero = tosaBuilder.getSplattedConst(
        0.0, outputType.getShape(), outputType.getElementType());

    Value exp = tosaBuilder.unaryOp<mlir::tosa::ExpOp>(input);
    Value expTimesAlpha = tosaBuilder.mul(exp, alpha);
    Value expTimesAlphaMinusAlpha =
        tosaBuilder.binaryOp<mlir::tosa::SubOp>(expTimesAlpha, alpha);

    Value greater = tosaBuilder.greater(input, constZero);
    auto select = tosaBuilder.select(greater, input, expTimesAlphaMinusAlpha);
    Value valTimesGamma = tosaBuilder.mul(select, gamma);

    rewriter.replaceOp(op, {valTimesGamma});
    return success();
  }
};

class ONNXThresholdedReluOpLoweringToTOSA
    : public OpConversionPattern<ONNXThresholdedReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXThresholdedReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto outputType = op.getResult().getType().cast<TensorType>();
    if (failed(IsIntOrFloat::checkType(
            rewriter, outputType.getElementType(), op))) {
      return failure();
    }

    Value input = adaptor.getX();

    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    auto alpha =
        tosaBuilder.getSplattedConst(adaptor.getAlpha().convertToDouble(),
            outputType.getShape(), outputType.getElementType());
    auto zero = tosaBuilder.getSplattedConst(
        0.0, outputType.getShape(), outputType.getElementType());

    auto greater = tosaBuilder.greater(input, alpha);
    auto select = tosaBuilder.select(greater, input, zero);

    rewriter.replaceOp(op, {select});
    return success();
  }
};

static void populateLoweringONNXElementwiseBinaryTemplateOpToTOSAPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXBinaryElementwiseOpLoweringToTOSA<ONNXAndOp,
                      mlir::tosa::LogicalAndOp, IsBool>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXBitwiseAndOp,
          mlir::tosa::BitwiseAndOp, IsInt>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXOrOp, mlir::tosa::LogicalOrOp,
          IsBool>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXBitwiseOrOp,
          mlir::tosa::BitwiseOrOp, IsInt>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXXorOp, mlir::tosa::LogicalXorOp,
          IsBool>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXBitwiseXorOp,
          mlir::tosa::BitwiseXorOp, IsInt>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXAddOp, mlir::tosa::AddOp,
          IsIntOrFloat>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXSubOp, mlir::tosa::SubOp,
          IsIntOrFloat>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXPowOp, mlir::tosa::PowOp,
          IsFloat>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXMinOp, mlir::tosa::MinimumOp,
          IsIntOrFloat>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXMaxOp, mlir::tosa::MaximumOp,
          IsIntOrFloat>>(typeConverter, ctx);
}

static void populateLoweringONNXElementwiseUnaryTemplateOpToTOSAPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXElementwiseUnaryOpLoweringToTOSA<ONNXNegOp,
                      mlir::tosa::NegateOp, IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXCeilOp, mlir::tosa::CeilOp,
          IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXFloorOp, mlir::tosa::FloorOp,
          IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXExpOp, mlir::tosa::ExpOp,
          IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXLogOp, mlir::tosa::LogOp,
          IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXReciprocalOp,
          mlir::tosa::ReciprocalOp, IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXTanhOp, mlir::tosa::TanhOp,
          IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXSigmoidOp, mlir::tosa::SigmoidOp,
          IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXBitwiseNotOp,
          mlir::tosa::BitwiseNotOp, IsInt, IsInt>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXNotOp, mlir::tosa::LogicalNotOp,
          IsBool, IsBool>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXAbsOp, mlir::tosa::AbsOp,
          AbsIOSupportedTypes, AbsIOSupportedTypes>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXErfOp, mlir::tosa::ErfOp,
          ErfIOSupportedTypes, ErfIOSupportedTypes>>(typeConverter, ctx);

// Tosa custom ops
#define INSERT_ONNX_UNARY_TO_TOSA_CUSTOMOP_PATTERN(                            \
    ONNXOp, opName, implementedWith)                                           \
  patterns.add<ConvertONNXUnaryOpToTosaCustomOp<ONNXOp>>(                      \
      typeConverter, ctx, opName, implementedWith);

  INSERT_ONNX_UNARY_TO_TOSA_CUSTOMOP_PATTERN(
      ONNXSinOp, "math.sin", "linalg.generic");
  INSERT_ONNX_UNARY_TO_TOSA_CUSTOMOP_PATTERN(
      ONNXCosOp, "math.cos", "linalg.generic");
#undef INSERT_ONNX_UNARY_TO_TOSA_CUSTOMOP_PATTERN
}

void populateLoweringONNXElementwiseOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXReluOpLoweringToTOSA, ONNXLeakyReluOpLoweringToTOSA,
      ONNXMulOpLoweringToTosa, ONNXClipOpLoweringToTOSA,
      ONNXDivOpLoweringToTOSA, ONNXHardSigmoidOpLoweringToTOSA,
      ONNXSqrtOpLoweringToTOSA, ONNXEluOpLoweringToTOSA,
      ONNXPReluOpLoweringToTOSA, ONNXThresholdedReluOpLoweringToTOSA,
      ONNXSoftplusOpLoweringToTOSA, ONNXSeluOpLoweringToTOSA>(
      typeConverter, ctx);

  populateLoweringONNXElementwiseBinaryTemplateOpToTOSAPattern(
      patterns, typeConverter, ctx);
  populateLoweringONNXElementwiseUnaryTemplateOpToTOSAPattern(
      patterns, typeConverter, ctx);
}

} // namespace onnx_mlir
