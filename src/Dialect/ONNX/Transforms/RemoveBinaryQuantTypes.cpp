//===- RemoveBinaryQuantTypes.cpp - Remove binary ops on quant types ------===//
//
// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//
//
// Quant-types variant of DQBinaryQOpt. After create-quant-types converts Q/DQ
// to quant::StorageCastOp with !quant.uniform element types, scalar arithmetic
// (Add/Sub/Mul/Div) between a quantized activation and a constant can be
// absorbed into the scale or zero-point of the UniformQuantizedType:
//
//   Add/Sub with scalar k  ->  zp_new = zp +/- k/scale
//   Mul/Div with scalar k  ->  scale_new = scale * k  or  scale / k
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <set>

using namespace mlir;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct QuantParams {
  double scale;
  int64_t zeroPoint;
  quant::UniformQuantizedType quantType;
};

static std::optional<QuantParams> getQuantParams(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return std::nullopt;
  auto qt = dyn_cast<quant::UniformQuantizedType>(tensorType.getElementType());
  if (!qt)
    return std::nullopt;
  return QuantParams{qt.getScale(), qt.getZeroPoint(), qt};
}

static RankedTensorType updateQuantType(RankedTensorType tensorType,
                                        double newScale, int64_t newZeroPoint) {
  auto oldQt =
      dyn_cast<quant::UniformQuantizedType>(tensorType.getElementType());
  if (!oldQt)
    return tensorType;
  auto newQt = quant::UniformQuantizedType::get(
      oldQt.getFlags(), oldQt.getStorageType(), oldQt.getExpressedType(),
      newScale, newZeroPoint, oldQt.getStorageTypeMin(),
      oldQt.getStorageTypeMax());
  return RankedTensorType::get(tensorType.getShape(), newQt);
}

template <typename T>
static std::optional<T> getScalarTensorValue(ONNXConstantOp constOp) {
  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(constOp.getValueAttr());
  if (!elementsAttr)
    return std::nullopt;

  Type elementType = elementsAttr.getElementType();

  if (elementsAttr.isSplat()) {
    if (isa<FloatType>(elementType)) {
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
        APFloat splatValue = elementsAttr.getSplatValue<APFloat>();
        return static_cast<T>(splatValue.convertToDouble());
      }
    }
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
      if constexpr (std::is_integral_v<T>) {
        APInt splatValue = elementsAttr.getSplatValue<APInt>();
        if (intType.isUnsigned())
          return static_cast<T>(splatValue.getZExtValue());
        else
          return static_cast<T>(splatValue.getSExtValue());
      }
    }
    return std::nullopt;
  }

  auto shapedTy = dyn_cast<ShapedType>(elementsAttr.getType());
  if (!shapedTy || !shapedTy.hasStaticShape())
    return std::nullopt;

  if (shapedTy.getRank() == 0) {
    auto firstAttr = *elementsAttr.getValues<Attribute>().begin();
    if (auto fAttr = dyn_cast<FloatAttr>(firstAttr)) {
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>)
        return static_cast<T>(fAttr.getValueAsDouble());
    }
    if (auto iAttr = dyn_cast<IntegerAttr>(firstAttr)) {
      if constexpr (std::is_integral_v<T>)
        return static_cast<T>(iAttr.getInt());
    }
    return std::nullopt;
  }

  if (isa<FloatType>(elementType)) {
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
      std::set<double> vals;
      for (auto a : elementsAttr.getValues<FloatAttr>())
        vals.insert(a.getValueAsDouble());
      if (vals.size() == 1)
        return static_cast<T>(*vals.begin());
    }
  } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
    if constexpr (std::is_integral_v<T>) {
      std::set<int64_t> vals;
      for (auto a : elementsAttr.getValues<IntegerAttr>())
        vals.insert(intType.isUnsigned() ? a.getUInt() : a.getInt());
      if (vals.size() == 1)
        return static_cast<T>(*vals.begin());
    }
  }
  return std::nullopt;
}

static bool isValuePreservingOp(Operation *op) {
  if (!op)
    return false;
  return isa<ONNXIdentityOp, ONNXReshapeOp, ONNXSqueezeOp, ONNXUnsqueezeOp,
             ONNXTransposeOp>(op);
}

static bool hasBranchOnValue(Value v) {
  llvm::SmallPtrSet<Operation *, 8> uniq;
  for (auto *u : v.getUsers())
    uniq.insert(u);
  return uniq.size() > 1;
}

static LogicalResult checkNewParamsFit(PatternRewriter &rewriter,
                                       Operation *op,
                                       quant::UniformQuantizedType qt,
                                       double newScale, int64_t newZp) {
  auto storageType = qt.getStorageType();
  if (auto intType = dyn_cast<IntegerType>(storageType)) {
    int64_t zpMin, zpMax;
    unsigned bw = intType.getWidth();
    if (bw == 4) {
      zpMin = intType.isUnsigned() ? 0 : -8;
      zpMax = intType.isUnsigned() ? 15 : 7;
    } else if (intType.isUnsigned()) {
      zpMin = 0;
      zpMax = (bw == 64) ? INT64_MAX : ((int64_t(1) << bw) - 1);
    } else {
      zpMin = (bw == 64) ? INT64_MIN : (-(int64_t(1) << (bw - 1)));
      zpMax = (bw == 64) ? INT64_MAX : ((int64_t(1) << (bw - 1)) - 1);
    }
    if (newZp < zpMin || newZp > zpMax)
      return rewriter.notifyMatchFailure(op, "new zero point overflows");
  }

  auto expressedType = qt.getExpressedType();
  if (auto floatType = dyn_cast<FloatType>(expressedType)) {
    double scaleMax;
    if (floatType.isF16())
      scaleMax = 65504.0;
    else if (floatType.isBF16() || floatType.isF32())
      scaleMax = std::numeric_limits<float>::max();
    else if (floatType.isF64())
      scaleMax = std::numeric_limits<double>::max();
    else
      return rewriter.notifyMatchFailure(op,
                                         "unsupported float type for scale");
    if (newScale < -scaleMax || newScale > scaleMax)
      return rewriter.notifyMatchFailure(op, "new scale overflows");
  }

  return success();
}

// ---------------------------------------------------------------------------
// Match state
// ---------------------------------------------------------------------------

struct MatchState {
  Value activationValue;
  double kValue = 0.0;
  QuantParams dstParams;
  bool foldIntoInput = false;
};

// ---------------------------------------------------------------------------
// Try to extract a scalar float constant from a value.
// ---------------------------------------------------------------------------

static std::optional<double> tryGetScalarConstant(Value val) {
  if (!val)
    return std::nullopt;

  if (auto constOp = val.getDefiningOp<ONNXConstantOp>())
    return getScalarTensorValue<double>(constOp);

  if (auto scast = val.getDefiningOp<quant::StorageCastOp>()) {
    auto qp = getQuantParams(scast.getResult().getType());
    if (!qp)
      return std::nullopt;
    Value storageVal = scast.getInput();
    Operation *defOp = storageVal.getDefiningOp();
    if (defOp && isValuePreservingOp(defOp))
      storageVal = defOp->getOperand(0);
    auto constOp = storageVal.getDefiningOp<ONNXConstantOp>();
    if (!constOp)
      return std::nullopt;
    auto intVal = getScalarTensorValue<int64_t>(constOp);
    if (!intVal)
      return std::nullopt;
    return (*intVal - qp->zeroPoint) * qp->scale;
  }

  Operation *defOp = val.getDefiningOp();
  if (defOp && isValuePreservingOp(defOp)) {
    if (auto constOp =
            defOp->getOperand(0).getDefiningOp<ONNXConstantOp>())
      return getScalarTensorValue<double>(constOp);
  }

  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Pattern: RemoveBinaryQuantTypesPattern
// ---------------------------------------------------------------------------

template <typename BinOp>
struct RemoveBinaryQuantTypesPattern : public OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse())
      return rewriter.notifyMatchFailure(op, "binary op has multiple users");

    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    Value result = op->getResult(0);

    auto lhsQP = getQuantParams(lhs.getType());
    auto rhsQP = getQuantParams(rhs.getType());
    auto outQP = getQuantParams(result.getType());

    if (!outQP)
      return rewriter.notifyMatchFailure(op, "output is not quant-typed");

    MatchState state;

    // Case A: lhs is quant activation, rhs is scalar constant
    if (lhsQP) {
      auto kOpt = tryGetScalarConstant(rhs);
      if (kOpt) {
        state.activationValue = lhs;
        state.kValue = *kOpt;
      }
    }
    // Case A-reversed: rhs is quant activation, lhs is scalar constant
    if (!state.activationValue && rhsQP) {
      auto kOpt = tryGetScalarConstant(lhs);
      if (kOpt) {
        if constexpr (std::is_same_v<BinOp, ONNXSubOp> ||
                      std::is_same_v<BinOp, ONNXDivOp>) {
          return rewriter.notifyMatchFailure(
              op, "Sub/Div not supported when constant is first operand");
        }
        state.activationValue = rhs;
        state.kValue = *kOpt;
      }
    }
    // Case B: both quant-typed, one traces to a constant
    if (!state.activationValue && lhsQP && rhsQP) {
      auto kLhs = tryGetScalarConstant(lhs);
      auto kRhs = tryGetScalarConstant(rhs);
      if (kRhs) {
        state.activationValue = lhs;
        state.kValue = *kRhs;
      } else if (kLhs) {
        if constexpr (std::is_same_v<BinOp, ONNXSubOp> ||
                      std::is_same_v<BinOp, ONNXDivOp>) {
          return rewriter.notifyMatchFailure(
              op, "Sub/Div not supported when constant is first operand");
        }
        state.activationValue = rhs;
        state.kValue = *kLhs;
      }
    }

    if (!state.activationValue)
      return rewriter.notifyMatchFailure(
          op, "could not identify activation + scalar constant pair");

    if (!state.activationValue.getDefiningOp())
      return rewriter.notifyMatchFailure(
          op, "activation is a block argument, cannot modify its type");

    auto actQP = getQuantParams(state.activationValue.getType());

    if (actQP && hasBranchOnValue(state.activationValue)) {
      state.foldIntoInput = true;
      state.dstParams = *actQP;
    } else {
      state.foldIntoInput = false;
      state.dstParams = *outQP;
    }

    // Safety checks
    if (state.kValue == 0.0) {
      bool divByK = false;
      if constexpr (std::is_same_v<BinOp, ONNXDivOp>)
        divByK = state.foldIntoInput;
      if constexpr (std::is_same_v<BinOp, ONNXMulOp>)
        divByK = !state.foldIntoInput;
      if (divByK)
        return rewriter.notifyMatchFailure(op, "k=0 would cause div-by-zero");
    }
    if (state.dstParams.scale == 0.0) {
      if constexpr (std::is_same_v<BinOp, ONNXAddOp> ||
                    std::is_same_v<BinOp, ONNXSubOp>) {
        return rewriter.notifyMatchFailure(op,
                                           "scale=0 would cause div-by-zero");
      }
    }

    // Compute new scale and zero-point
    double newScale = state.dstParams.scale;
    double newZpFloat = static_cast<double>(state.dstParams.zeroPoint);
    const double kVal = state.kValue;
    const bool dstIsInput = state.foldIntoInput;

    if constexpr (std::is_same_v<BinOp, ONNXAddOp>) {
      if (dstIsInput)
        newZpFloat -= (kVal / newScale);
      else
        newZpFloat += (kVal / newScale);
    } else if constexpr (std::is_same_v<BinOp, ONNXSubOp>) {
      if (dstIsInput)
        newZpFloat += (kVal / newScale);
      else
        newZpFloat -= (kVal / newScale);
    } else if constexpr (std::is_same_v<BinOp, ONNXMulOp>) {
      if (dstIsInput)
        newScale *= kVal;
      else
        newScale /= kVal;
    } else if constexpr (std::is_same_v<BinOp, ONNXDivOp>) {
      if (dstIsInput)
        newScale /= kVal;
      else
        newScale *= kVal;
    }

    int64_t newZp = (newZpFloat >= 0.0)
                        ? static_cast<int64_t>(std::floor(newZpFloat))
                        : static_cast<int64_t>(std::ceil(newZpFloat));

    if (newScale <= 0.0)
      return rewriter.notifyMatchFailure(op,
                                         "new scale would be non-positive");

    if (failed(checkNewParamsFit(rewriter, op, state.dstParams.quantType,
                                 newScale, newZp)))
      return failure();

    auto actTensorType =
        cast<RankedTensorType>(state.activationValue.getType());
    auto newType = updateQuantType(actTensorType, newScale, newZp);
    state.activationValue.setType(newType);

    rewriter.replaceOp(op, state.activationValue);
    return success();
  }
};

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

class RemoveBinaryQuantTypesPass
    : public PassWrapper<RemoveBinaryQuantTypesPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveBinaryQuantTypesPass)

  StringRef getArgument() const override {
    return "remove-binary-quant-types";
  }

  StringRef getDescription() const override {
    return "Remove scalar binary ops (Add/Sub/Mul/Div) by absorbing them "
           "into quantization parameters on quant-typed tensors";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ONNXDialect>();
    registry.insert<quant::QuantDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    patterns.add<RemoveBinaryQuantTypesPattern<ONNXAddOp>>(
        patterns.getContext());
    patterns.add<RemoveBinaryQuantTypesPattern<ONNXSubOp>>(
        patterns.getContext());
    patterns.add<RemoveBinaryQuantTypesPattern<ONNXMulOp>>(
        patterns.getContext());
    patterns.add<RemoveBinaryQuantTypesPattern<ONNXDivOp>>(
        patterns.getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace onnx_mlir {

std::unique_ptr<mlir::Pass> createRemoveBinaryQuantTypesPass() {
  return std::make_unique<RemoveBinaryQuantTypesPass>();
}

} // namespace onnx_mlir
