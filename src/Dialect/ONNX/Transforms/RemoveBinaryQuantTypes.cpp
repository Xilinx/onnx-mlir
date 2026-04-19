// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//
//
// Absorbs scalar binary ops (Add/Sub/Mul/Div) into quantization parameters
// on quant-typed tensors.  Runs after --quant-types has replaced Q/DQ with
// quant::StorageCastOp + !quant.uniform element types.
//
// In quant-types IR the pattern is simply:
//   --qType--> BinaryOp(k) --qType-->
// There is no DQ or Q surrounding the binary op.
//
// Three apply strategies:
//   A. Single-use activation, no boundary Q:
//        setType() on the activation, erase binary.
//   B. Multi-use activation (branch):
//        Insert scast -> intType -> scast -> newQType, erase binary.
//   C. Boundary Q still present (QuantizeLinear -> scast -> Binary):
//        Update Q's scale/zp constants, erase binary.
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
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
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

  if (oldInit.hasOneUse()) {
    rewriter.modifyOpInPlace(
        oldCst, [&] { oldCst->setAttr("value", payload); });
    return;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(targetOp);
  auto newCst =
      rewriter.create<ONNXConstantOp>(targetOp->getLoc(), Attribute(), payload);
  for (unsigned i = 0, e = targetOp->getNumOperands(); i < e; ++i) {
    if (targetOp->getOperand(i) == oldInit) {
      targetOp->setOperand(i, newCst.getOutput());
      break;
    }
  }
}

// Trace through quant.scast to find a surviving QuantizeLinear boundary op.
static ONNXQuantizeLinearOp findBoundaryQ(Value quantVal) {
  auto scast = quantVal.getDefiningOp<quant::StorageCastOp>();
  if (!scast)
    return nullptr;
  return scast.getInput().getDefiningOp<ONNXQuantizeLinearOp>();
}

// Extract a scalar float constant from a value. Handles plain float constants
// and quant-typed constants (after --quant-types folds DQ into the constant).
// By the time this pass runs (after --quant-types --onnx-hybrid-transform),
// there are no scast wrappers or value-preserving ops around scalar constants.
static std::optional<double> tryGetScalarConstant(Value val) {
  if (!val)
    return std::nullopt;

  auto constOp = val.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return std::nullopt;

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

// Build a storage-type tensor type from a quant-typed tensor (strips the
// quant element type down to its storage integer type).
static RankedTensorType getStorageTensorType(RankedTensorType qTensorType) {
  auto qt = dyn_cast<quant::UniformQuantizedType>(qTensorType.getElementType());
  if (!qt)
    return {};
  return RankedTensorType::get(qTensorType.getShape(), qt.getStorageType());
}

template <typename BinOp>
struct RemoveBinaryQuantTypesPattern : public OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      BinOp op, PatternRewriter &rewriter) const override {
    // Multiple users of the binary op are fine: when we fold, replaceOp
    // updates all uses to the new value at once.

    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    Value result = op->getResult(0);

    auto lhsQP = getQuantParams(lhs.getType());
    auto outQP = getQuantParams(result.getType());

    if (!outQP)
      return rewriter.notifyMatchFailure(op, "output is not quant-typed");

    Value activationValue;
    double kValue = 0.0;

    // After canonicalization, the constant is always on the rhs for
    // commutative ops (Add, Mul). For Sub/Div the constant must be rhs
    // (constant-first Sub/Div cannot be folded into a scale/zp shift).
    if (lhsQP) {
      auto kOpt = tryGetScalarConstant(rhs);
      if (kOpt) {
        activationValue = lhs;
        kValue = *kOpt;
      }
    }

    if (!activationValue)
      return rewriter.notifyMatchFailure(
          op, "could not identify activation + scalar constant pair");

    if (!activationValue.getDefiningOp())
      return rewriter.notifyMatchFailure(
          op, "activation is a block argument, cannot modify");

    // Always fold into the output quant params. In quant-types IR there is
    // no DQ/Q dichotomy — we simply absorb k into the output's scale/zp.
    double dstScale = outQP->scale;
    int64_t dstZeroPoint = outQP->zeroPoint;
    quant::UniformQuantizedType dstQuantType = outQP->quantType;

    // Safety: Mul with k=0 means new_scale = scale/0 (div-by-zero).
    if (kValue == 0.0) {
      if constexpr (std::is_same_v<BinOp, ONNXMulOp>)
        return rewriter.notifyMatchFailure(op, "k=0 would cause div-by-zero");
    }
    // Safety: Add/Sub with scale=0 means zp += k/0 (div-by-zero).
    if (dstScale == 0.0) {
      if constexpr (std::is_same_v<BinOp, ONNXAddOp> ||
                    std::is_same_v<BinOp, ONNXSubOp>) {
        return rewriter.notifyMatchFailure(
            op, "scale=0 would cause div-by-zero");
      }
    }

    // Compute new scale and zero point.
    // Folding always into output: Add -> zp += k/s, Sub -> zp -= k/s,
    // Mul -> s /= k, Div -> s *= k.
    double newScale = dstScale;
    double newZpFloat = static_cast<double>(dstZeroPoint);

    if constexpr (std::is_same_v<BinOp, ONNXAddOp>) {
      newZpFloat += (kValue / newScale);
    } else if constexpr (std::is_same_v<BinOp, ONNXSubOp>) {
      newZpFloat -= (kValue / newScale);
    } else if constexpr (std::is_same_v<BinOp, ONNXMulOp>) {
      newScale /= kValue;
    } else if constexpr (std::is_same_v<BinOp, ONNXDivOp>) {
      newScale *= kValue;
    }

    int64_t newZp = (newZpFloat >= 0.0)
                        ? static_cast<int64_t>(std::floor(newZpFloat))
                        : static_cast<int64_t>(std::ceil(newZpFloat));

    // Reject zero or negative scale. Zero produces NaN/Inf in dequantization;
    // negative scales arise when folding a Mul/Div with a negative constant
    // (e.g. attention mask bias) and are not handled by downstream passes.
    if (newScale <= 0.0)
      return rewriter.notifyMatchFailure(
          op, "new scale would be zero or negative");

    if (failed(checkNewParamsFit(rewriter, op, dstQuantType, newScale, newZp)))
      return failure();

    // --- Apply the fold ---

    // Check if a surviving QuantizeLinear boundary op feeds the activation
    // through a scast.
    auto boundaryQ = findBoundaryQ(activationValue);
    bool activationHasSingleUse = activationValue.hasOneUse();
    // Pattern C mutates boundaryQ's scale/zp constants, which changes the
    // integer values boundaryQ emits. That is only safe when boundaryQ's
    // storage-int result feeds exactly one consumer (the upstream scast on
    // our branch). If boundaryQ is shared with other branches, those
    // branches would silently observe the shifted integers reinterpreted
    // with their unchanged qType -> wrong values. Fall through to Pattern B
    // in that case; Pattern B isolates the change behind two fresh scasts
    // and never touches upstream ops.
    bool boundaryQHasSingleUse = boundaryQ && boundaryQ.getResult().hasOneUse();

    if (boundaryQ && activationHasSingleUse && boundaryQHasSingleUse) {
      // Pattern C: QuantizeLinear -> scast -> Binary. Update Q's constants.
      updateInitializer(
          rewriter, boundaryQ, boundaryQ->getOperand(1), newScale);
      updateInitializer(rewriter, boundaryQ, boundaryQ->getOperand(2),
          static_cast<double>(newZp));

      Value qResult = boundaryQ.getResult();
      auto upstreamScast =
          activationValue.getDefiningOp<quant::StorageCastOp>();
      auto downstreamScast =
          dyn_cast<quant::StorageCastOp>(*result.getUsers().begin());
      if (downstreamScast) {
        rewriter.replaceOp(downstreamScast, qResult);
        rewriter.eraseOp(op);
      } else {
        rewriter.replaceOp(op, activationValue);
      }
      if (upstreamScast && upstreamScast->use_empty())
        rewriter.eraseOp(upstreamScast);
    } else if (activationHasSingleUse) {
      // Pattern A: Single-use, no boundary Q. Update type in-place.
      auto actType = cast<RankedTensorType>(activationValue.getType());
      activationValue.setType(updateQuantType(actType, newScale, newZp));
      rewriter.replaceOp(op, activationValue);
    } else {
      // Pattern B: Multi-use (branch). Cannot mutate activation type.
      // Insert scast -> intType -> scast -> newQType.
      auto actType = cast<RankedTensorType>(activationValue.getType());
      auto storageTensorType = getStorageTensorType(actType);
      if (!storageTensorType)
        return rewriter.notifyMatchFailure(
            op, "cannot derive storage type for scast pair");
      auto newQTensorType = updateQuantType(actType, newScale, newZp);

      auto toStorage = rewriter.create<quant::StorageCastOp>(
          op.getLoc(), storageTensorType, activationValue);
      auto toNewQuant = rewriter.create<quant::StorageCastOp>(
          op.getLoc(), newQTensorType, toStorage.getResult());
      rewriter.replaceOp(op, toNewQuant.getResult());
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

    onnx_mlir::ResultNamesUpdater rnUpdater;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
            GreedyRewriteConfig{.listener = &rnUpdater})))
      signalPassFailure();
  }
};

} // namespace

namespace onnx_mlir {

std::unique_ptr<mlir::Pass> createRemoveBinaryQuantTypesPass() {
  return std::make_unique<RemoveBinaryQuantTypesPass>();
}

} // namespace onnx_mlir
