// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass quantizes float constant operands in onnx.Where ops after
// QuantTypesPass. When QuantTypesPass folds DQ/Q into quant.uniform types,
// onnx.Where can have mixed types: one value operand carries quant.uniform
// while the other remains a plain float constant. This pass quantizes those
// float constants using the output's scale/zero-point and wraps them with
// quant.scast, ensuring all inputs have consistent quant.uniform types for
// downstream conversion (WhereQuantConversion in ONNXToXIRDialectPass).
//
// Also promotes all-float Where ops to quantized if downstream consumers
// produce quant.uniform types.
//
// Handles Cast(int→i1) conditions by replacing the Cast with
// XCOMPILERFusedEltwise(REQUANTIZE), preserving the original type conversion
// (e.g. i32 → i1) and attaching identity scale/zp attributes (1.0/0).
// Downstream XIR conversion produces:
//   xir.qlinear_eltwise {a_scale=[1.0], a_zero_point=[0],
//     b_scale=[1.0], b_zero_point=[0], y_scale=[1.0], y_zero_point=[0],
//     op_type="REQUANTIZE"} : (tensor<...xi32>) -> tensor<...xi1>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/Support/Debug.h"

#include <cmath>

#define DEBUG_TYPE "replace-qdq-where"

using namespace mlir;

namespace {

static mlir::quant::UniformQuantizedType getUniformQuantType(Type type) {
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type))
    return mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
        tensorType.getElementType());
  return nullptr;
}

static bool isQuantizedType(Type type) {
  return getUniformQuantType(type) != nullptr;
}

// When an onnx.Where has a float result but its downstream consumers produce
// a quant.uniform type, retrieve that type so we can quantize the Where.
// All quant-typed users must agree on the same quant type.
static mlir::quant::UniformQuantizedType getDownstreamQuantType(
    ONNXWhereOp op) {
  mlir::quant::UniformQuantizedType found = nullptr;
  for (Operation *user : op.getResult().getUsers()) {
    for (auto resultType : user->getResultTypes()) {
      if (auto qtype = getUniformQuantType(resultType)) {
        if (!found)
          found = qtype;
        else if (found != qtype)
          return nullptr;
      }
    }
  }
  return found;
}

// Quantize a float value: q = clamp(round(f / scale) + zp, storageMin,
// storageMax)
static int64_t quantizeFloat(float val, double scale, int64_t zp,
    int64_t storageMin, int64_t storageMax) {
  int64_t q = static_cast<int64_t>(std::llround(val / scale)) + zp;
  return std::clamp(q, storageMin, storageMax);
}

// Given a float DenseElementsAttr and quant params, produce a quantized
// DenseElementsAttr with the storage integer type.
static DenseElementsAttr quantizeDenseAttr(DenseElementsAttr floatAttr,
    mlir::quant::UniformQuantizedType qtype, ArrayRef<int64_t> shape) {
  double scale = qtype.getScale();
  int64_t zp = qtype.getZeroPoint();
  int64_t storageMin = qtype.getStorageTypeMin();
  int64_t storageMax = qtype.getStorageTypeMax();
  Type storageType = qtype.getStorageType();
  unsigned bitWidth = storageType.getIntOrFloatBitWidth();

  auto resultTensorType = RankedTensorType::get(shape, storageType);

  SmallVector<APInt> quantizedValues;
  for (APFloat fVal : floatAttr.getValues<APFloat>()) {
    float f = fVal.convertToFloat();
    int64_t q = quantizeFloat(f, scale, zp, storageMin, storageMax);
    quantizedValues.push_back(
        APInt(bitWidth, q, /*isSigned=*/qtype.isSigned()));
  }

  return DenseElementsAttr::get(resultTensorType, quantizedValues);
}

// Quantize a float constant operand: create a new constant with quantized
// integer values and wrap it with quant.scast to produce the quant.uniform
// type.
static Value quantizeConstantOperand(PatternRewriter &rewriter, Location loc,
    ONNXConstantOp constOp, mlir::quant::UniformQuantizedType qtype,
    RankedTensorType targetQuantTensorType) {
  auto floatAttr = mlir::dyn_cast<DenseElementsAttr>(constOp.getValueAttr());
  if (!floatAttr)
    return nullptr;

  auto shape = targetQuantTensorType.getShape();
  auto quantizedAttr = quantizeDenseAttr(floatAttr, qtype, shape);

  auto newConst = rewriter.create<ONNXConstantOp>(loc,
      /*sparse_value=*/Attribute(),
      /*value=*/quantizedAttr);

  auto scast = rewriter.create<quant::StorageCastOp>(
      loc, targetQuantTensorType, newConst);
  return scast.getResult();
}

struct ReplaceQDQWherePattern : public OpRewritePattern<ONNXWhereOp> {
  using OpRewritePattern<ONNXWhereOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXWhereOp op, PatternRewriter &rewriter) const override {
    auto resultType = mlir::dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    auto resultQType = getUniformQuantType(resultType);
    bool promoteToQuant = false;

    if (!resultQType) {
      resultQType = getDownstreamQuantType(op);
      if (!resultQType)
        return failure();
      promoteToQuant = true;
    }

    Value condition = op.getCondition();
    Value xOperand = op.getX();
    Value yOperand = op.getY();

    bool changed = false;
    Value newX = xOperand;
    Value newY = yOperand;

    auto tryQuantize = [&](Value operand) -> Value {
      if (isQuantizedType(operand.getType()))
        return operand;

      auto constOp = operand.getDefiningOp<ONNXConstantOp>();
      if (!constOp)
        return operand;

      auto operandTensorType =
          mlir::dyn_cast<RankedTensorType>(operand.getType());
      if (!operandTensorType)
        return operand;

      auto targetQuantTensorType =
          RankedTensorType::get(operandTensorType.getShape(), resultQType);

      Value quantized = quantizeConstantOperand(
          rewriter, op.getLoc(), constOp, resultQType, targetQuantTensorType);
      if (!quantized)
        return operand;

      changed = true;
      return quantized;
    };

    newX = tryQuantize(xOperand);
    newY = tryQuantize(yOperand);

    if (promoteToQuant) {
      if (!isQuantizedType(newX.getType()) || !isQuantizedType(newY.getType()))
        return failure();
      changed = true;
    }

    auto newResultType =
        promoteToQuant
            ? RankedTensorType::get(resultType.getShape(), resultQType)
            : resultType;

    if (!changed)
      return failure();

    auto newWhere = rewriter.create<ONNXWhereOp>(
        op.getLoc(), newResultType, condition, newX, newY);
    onnx_mlir::ResultNamesUpdater().notifyOperationReplaced(
        op, newWhere->getResults());
    rewriter.replaceOp(op, newWhere);
    return success();
  }
};

// Replace Cast(int→i1) feeding Where condition with
// XCOMPILERFusedEltwise(REQUANTIZE). Preserves the Cast's original type
// conversion (e.g. i32 → i1) and adds identity scale/zp attributes so
// the downstream XIR conversion produces:
//   xir.qlinear_eltwise {a_scale=[1.0], a_zero_point=[0],
//     b_scale=[1.0], b_zero_point=[0], y_scale=[1.0], y_zero_point=[0],
//     op_type="REQUANTIZE"} : (tensor<...xi32>) -> tensor<...xi1>
struct ReplaceCastCondWithRequantize : public OpRewritePattern<ONNXWhereOp> {
  using OpRewritePattern<ONNXWhereOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXWhereOp op, PatternRewriter &rewriter) const override {
    auto resultType = mlir::dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    auto resultQType = getUniformQuantType(resultType);
    if (!resultQType)
      return failure();

    Value condition = op.getCondition();
    auto condRTT = mlir::dyn_cast<RankedTensorType>(condition.getType());
    if (!condRTT || !condRTT.getElementType().isSignlessInteger(1))
      return failure();

    auto castOp = condition.getDefiningOp<ONNXCastOp>();
    if (!castOp)
      return failure();

    Value castInput = castOp.getInput();
    auto castInputRTT = mlir::dyn_cast<RankedTensorType>(castInput.getType());
    if (!castInputRTT)
      return failure();

    auto loc = op.getLoc();

    // Keep the Cast's output type (i1) as the REQUANTIZE output type.
    auto requantOutputType = condRTT;

    Value noneB = rewriter.create<ONNXNoneOp>(loc).getResult();
    auto requantize = rewriter.create<XCOMPILERFusedEltwiseOp>(loc,
        requantOutputType, castInput, noneB,
        /*clip_max=*/IntegerAttr(),
        /*clip_min=*/IntegerAttr(),
        /*enable_lut_sigmoid=*/rewriter.getBoolAttr(false),
        /*leakyrelu_alpha=*/FloatAttr(),
        /*mul_y=*/FloatAttr(),
        /*nonlinear=*/rewriter.getStringAttr("NONE"),
        /*nonlinear_in_scales=*/FloatAttr(),
        /*nonlinear_in_zeropoints=*/IntegerAttr(),
        /*prelu_in=*/IntegerAttr(),
        /*prelu_shift=*/IntegerAttr(),
        /*type=*/rewriter.getStringAttr("REQUANTIZE"));

    // Identity scale/zp for the REQUANTIZE — carried through to XIR conversion.
    requantize->setAttr(
        "a_scale", rewriter.getF32ArrayAttr({1.0f}));
    requantize->setAttr(
        "a_zero_point", rewriter.getI64ArrayAttr({0}));
    requantize->setAttr(
        "b_scale", rewriter.getF32ArrayAttr({1.0f}));
    requantize->setAttr(
        "b_zero_point", rewriter.getI64ArrayAttr({0}));
    requantize->setAttr(
        "y_scale", rewriter.getF32ArrayAttr({1.0f}));
    requantize->setAttr(
        "y_zero_point", rewriter.getI64ArrayAttr({0}));

    auto newWhere = rewriter.create<ONNXWhereOp>(
        loc, resultType, requantize.getResult(), op.getX(), op.getY());
    onnx_mlir::ResultNamesUpdater().notifyOperationReplaced(
        op, newWhere->getResults());
    rewriter.replaceOp(op, newWhere);
    return success();
  }
};

} // end anonymous namespace

namespace onnx_mlir {

struct ReplaceQDQWherePass
    : public PassWrapper<ReplaceQDQWherePass, OperationPass<func::FuncOp>> {
  ReplaceQDQWherePass() = default;
  ReplaceQDQWherePass(const ReplaceQDQWherePass &pass) : PassWrapper(pass) {}

  StringRef getArgument() const override { return "replace-qdq-where"; }
  StringRef getDescription() const override {
    return "Quantize float constant operands in onnx.Where ops "
           "(post-QuantTypesPass) to ensure consistent quant.uniform types";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ReplaceQDQWherePattern>(ctx);
    patterns.add<ReplaceCastCondWithRequantize>(ctx);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = 10;

    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createReplaceQDQWherePass() {
  return std::make_unique<ReplaceQDQWherePass>();
}

} // namespace onnx_mlir
