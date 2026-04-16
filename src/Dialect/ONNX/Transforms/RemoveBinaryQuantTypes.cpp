//===- RemoveBinaryQuantTypes.cpp - Remove binary ops on quant types ------===//
//
// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//
//
// Quant-types variant of DQBinaryQOpt (remove-binary). Absorbs scalar binary
// ops (Add/Sub/Mul/Div) into quantization parameters after create-quant-types
// has replaced Q/DQ with quant::StorageCastOp + !quant.uniform element types.
//
// Handles two IR shapes:
//   Boundary: QuantizeLinear -> scast -> Binary(k) -> scast
//   Interior: scast -> Binary(k) -> scast  (no surviving Q/DQ)
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

static RankedTensorType updateQuantType(
    RankedTensorType tensorType, double newScale, int64_t newZeroPoint) {
  auto oldQt =
      dyn_cast<quant::UniformQuantizedType>(tensorType.getElementType());
  if (!oldQt)
    return tensorType;
  auto newQt = quant::UniformQuantizedType::get(oldQt.getFlags(),
      oldQt.getStorageType(), oldQt.getExpressedType(), newScale, newZeroPoint,
      oldQt.getStorageTypeMin(), oldQt.getStorageTypeMax());
  return RankedTensorType::get(tensorType.getShape(), newQt);
}

// Identical to getScalarTensorValue in DQBinaryQOpt.cpp.
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

// Identical to isValuePreservingOp in DQBinaryQOpt.cpp.
static bool isValuePreservingOp(Operation *op) {
  if (!op)
    return false;
  return isa<ONNXIdentityOp, ONNXReshapeOp, ONNXSqueezeOp, ONNXUnsqueezeOp,
      ONNXTransposeOp>(op);
}

// Identical to checkNewQDQParameterFits in DQBinaryQOpt.cpp but reads
// storage/expressed types from UniformQuantizedType instead of Value operands.
static LogicalResult checkNewParamsFit(PatternRewriter &rewriter, Operation *op,
    quant::UniformQuantizedType qt, double newScale, int64_t newZp) {
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
      return rewriter.notifyMatchFailure(
          op, "unsupported float type for scale");
    if (newScale < -scaleMax || newScale > scaleMax)
      return rewriter.notifyMatchFailure(op, "new scale overflows");
  }

  return success();
}

// Identical to makeScalarDEA in DQBinaryQOpt.cpp.
static DenseElementsAttr makeScalarDEA(ShapedType likeTy, double d) {
  auto ranked = dyn_cast<RankedTensorType>(likeTy);
  if (!ranked || !ranked.hasStaticShape() || ranked.getNumElements() != 1)
    return {};

  Type outET = ranked.getElementType();

  if (auto outFT = dyn_cast<FloatType>(outET)) {
    llvm::APFloat ap(d);
    bool loses = false;
    ap.convert(
        outFT.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &loses);
    return DenseElementsAttr::get(ranked, FloatAttr::get(outFT, ap));
  }

  if (auto outIT = dyn_cast<IntegerType>(outET)) {
    int64_t iv = static_cast<int64_t>(std::llround(d));
    unsigned bw = outIT.getWidth();
    bool isSigned = outIT.isSigned();
    int64_t minV = isSigned ? (-(int64_t(1) << (bw - 1))) : 0;
    int64_t maxV =
        isSigned ? ((int64_t(1) << (bw - 1)) - 1) : ((int64_t(1) << bw) - 1);
    iv = std::min<int64_t>(std::max<int64_t>(iv, minV), maxV);
    return DenseElementsAttr::get(ranked, IntegerAttr::get(outIT, iv));
  }
  return {};
}

// Identical to updateInitializer in DQBinaryQOpt.cpp.
static void updateInitializer(PatternRewriter &rewriter, Operation *targetOp,
    Value oldInit, double newScalar) {
  if (!targetOp || !oldInit)
    return;
  auto oldCst = oldInit.getDefiningOp<ONNXConstantOp>();
  if (!oldCst)
    return;
  auto likeTy = dyn_cast<ShapedType>(oldInit.getType());
  if (!likeTy || !likeTy.hasStaticShape() || likeTy.getNumElements() != 1)
    return;
  DenseElementsAttr payload = makeScalarDEA(likeTy, newScalar);
  if (!payload)
    return;

  auto singleUseByTarget = [&]() -> bool {
    auto it = oldInit.use_begin(), e = oldInit.use_end();
    if (it == e)
      return false;
    auto *owner = it->getOwner();
    ++it;
    return (it == e) && (owner == targetOp);
  };

  if (singleUseByTarget()) {
    rewriter.modifyOpInPlace(oldCst, [&] {
      oldCst->setAttr("value", payload);
      oldCst->removeAttr("sparse_value");
      oldCst->removeAttr("value_float");
      oldCst->removeAttr("value_floats");
      oldCst->removeAttr("value_int");
      oldCst->removeAttr("value_ints");
      oldCst->removeAttr("value_string");
      oldCst->removeAttr("value_strings");
    });
    return;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(targetOp);
  OperationState st(targetOp->getLoc(), ONNXConstantOp::getOperationName());
  st.addTypes(likeTy);
  st.addAttribute("value", payload);
  Operation *raw = Operation::create(st);
  rewriter.insert(raw);
  auto newCst = dyn_cast<ONNXConstantOp>(raw);
  if (!newCst)
    return;
  for (unsigned i = 0, e = targetOp->getNumOperands(); i < e; ++i) {
    if (targetOp->getOperand(i) == oldInit) {
      targetOp->setOperand(i, newCst.getOutput());
      break;
    }
  }
}

// Trace through quant.scast to find a surviving Q or DQ boundary op.
static Operation *findBoundaryQDQOp(Value quantVal) {
  auto scast = quantVal.getDefiningOp<quant::StorageCastOp>();
  if (!scast)
    return nullptr;
  Value storageVal = scast.getInput();
  if (auto qOp = storageVal.getDefiningOp<ONNXQuantizeLinearOp>())
    return qOp;
  if (auto dqOp = storageVal.getDefiningOp<ONNXDequantizeLinearOp>())
    return dqOp;
  return nullptr;
}

struct MatchState {
  Value activationValue;
  double kValue = 0.0;
  double dstScale = 0.0;
  int64_t dstZeroPoint = 0;
};

// Extract a scalar float constant from a value. Handles plain float constants,
// quant-typed constants (after --quant-types folds DQ into the constant),
// constants behind quant.scast, and constants behind value-preserving ops.
static std::optional<double> tryGetScalarConstant(Value val) {
  if (!val)
    return std::nullopt;

  if (auto constOp = val.getDefiningOp<ONNXConstantOp>()) {
    auto floatVal = getScalarTensorValue<double>(constOp);
    if (floatVal)
      return floatVal;
    auto qp = getQuantParams(val.getType());
    if (qp) {
      auto intVal = getScalarTensorValue<int64_t>(constOp);
      if (intVal)
        return (*intVal - qp->zeroPoint) * qp->scale;
    }
    return std::nullopt;
  }

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
    if (auto constOp = defOp->getOperand(0).getDefiningOp<ONNXConstantOp>())
      return getScalarTensorValue<double>(constOp);
  }

  return std::nullopt;
}

template <typename BinOp>
struct RemoveBinaryQuantTypesPattern : public OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      BinOp op, PatternRewriter &rewriter) const override {
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

    if (lhsQP) {
      auto kOpt = tryGetScalarConstant(rhs);
      if (kOpt) {
        state.activationValue = lhs;
        state.kValue = *kOpt;
      }
    }
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
          op, "activation is a block argument, cannot modify");

    state.dstScale = outQP->scale;
    state.dstZeroPoint = outQP->zeroPoint;

    // Avoid division by zero: Mul folds as scale/k, so k=0 is invalid.
    // Div folds as scale*k, so k=0 gives scale=0, caught by newScale<=0 below.
    if (state.kValue == 0.0) {
      if constexpr (std::is_same_v<BinOp, ONNXMulOp>)
        return rewriter.notifyMatchFailure(op, "k=0 would cause div-by-zero");
    }
    if (state.dstScale == 0.0) {
      if constexpr (std::is_same_v<BinOp, ONNXAddOp> ||
                    std::is_same_v<BinOp, ONNXSubOp>) {
        return rewriter.notifyMatchFailure(
            op, "scale=0 would cause div-by-zero");
      }
    }

    // Same formulas as compute_new_scale_and_zp_values in DQBinaryQOpt.cpp
    // (fold-into-Q / fold-into-output direction).
    double newScale = state.dstScale;
    double newZpFloat = static_cast<double>(state.dstZeroPoint);
    const double kVal = state.kValue;

    if constexpr (std::is_same_v<BinOp, ONNXAddOp>) {
      newZpFloat += (kVal / newScale);
    } else if constexpr (std::is_same_v<BinOp, ONNXSubOp>) {
      newZpFloat -= (kVal / newScale);
    } else if constexpr (std::is_same_v<BinOp, ONNXMulOp>) {
      newScale /= kVal;
    } else if constexpr (std::is_same_v<BinOp, ONNXDivOp>) {
      newScale *= kVal;
    }

    // Same rounding as DQBinaryQOpt: floor for non-negative, ceil for negative.
    int64_t newZp = (newZpFloat >= 0.0)
                        ? static_cast<int64_t>(std::floor(newZpFloat))
                        : static_cast<int64_t>(std::ceil(newZpFloat));

    if (newScale <= 0.0)
      return rewriter.notifyMatchFailure(op, "new scale would be non-positive");

    if (failed(
            checkNewParamsFit(rewriter, op, outQP->quantType, newScale, newZp)))
      return failure();

    // Boundary case: surviving Q behind scast — update Q's constants directly.
    auto *boundaryQ = findBoundaryQDQOp(state.activationValue);
    if (boundaryQ) {
      auto upstreamQ = cast<ONNXQuantizeLinearOp>(boundaryQ);
      updateInitializer(
          rewriter, boundaryQ, upstreamQ->getOperand(1), newScale);
      updateInitializer(rewriter, boundaryQ, upstreamQ->getOperand(2),
          static_cast<double>(newZp));

      Value qResult = upstreamQ.getResult();
      auto upstreamScast =
          state.activationValue.getDefiningOp<quant::StorageCastOp>();
      auto downstreamScast =
          dyn_cast<quant::StorageCastOp>(*result.getUsers().begin());

      if (downstreamScast) {
        rewriter.replaceOp(downstreamScast, qResult);
        rewriter.eraseOp(op);
      } else {
        rewriter.replaceOp(op, state.activationValue);
      }
      if (upstreamScast && upstreamScast->use_empty())
        rewriter.eraseOp(upstreamScast);
    } else {
      // Interior case: no surviving Q/DQ — update the quant type annotation.
      auto actType = cast<RankedTensorType>(state.activationValue.getType());
      state.activationValue.setType(updateQuantType(actType, newScale, newZp));
      rewriter.replaceOp(op, state.activationValue);
    }

    return success();
  }
};

class RemoveBinaryQuantTypesPass
    : public PassWrapper<RemoveBinaryQuantTypesPass,
          OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveBinaryQuantTypesPass)

  StringRef getArgument() const override { return "remove-binary-quant-types"; }

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
