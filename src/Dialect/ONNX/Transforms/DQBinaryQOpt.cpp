//===- foldDqBinaryQPattern.cpp - Remove DQ-Binary-Q chains -----*- C++ -*-===//
//
// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/STLExtras.h"
#include <cmath> // For std::llround
#include <optional>
#include <variant>

using namespace mlir;
using namespace onnx_mlir;

namespace {

static ElementsAttr getElementAttributeFromConstant(Value val) {
  if (!val)
    return nullptr;
  if (auto constOp = val.getDefiningOp<ONNXConstantOp>())
    return mlir::dyn_cast<ElementsAttr>(constOp.getValueAttr());
  return nullptr;
}

template <typename T>
std::optional<T> getScalarOrSplatValue(ONNXConstantOp constOp) {
  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(constOp.getValueAttr());
  if (!elementsAttr || !elementsAttr.isSplat()) {
    return std::nullopt;
  }
  mlir::Type elementType = elementsAttr.getElementType();
  // Case 1: Floating-point types (f32, f64).
  if (elementType.isF32() || elementType.isF64()) {
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
      // APFloat can handle both f32 and f64.
      APFloat splatValue = elementsAttr.getSplatValue<APFloat>();
      return static_cast<T>(splatValue.convertToDouble());
    }
  }
  // Case 2: Integer types (i8, ui16, etc.).
  if (auto intType = elementType.dyn_cast<IntegerType>()) {
    if constexpr (std::is_integral_v<T>) {
      APInt splatValue = elementsAttr.getSplatValue<APInt>();
      if (intType.isUnsigned()) {
        return static_cast<T>(splatValue.getZExtValue());
      } else {
        return static_cast<T>(splatValue.getSExtValue());
      }
    }
  }
  return std::nullopt;
}

template <typename T>
std::optional<T> getScalarOrSplatConstant(Value value) {
  if (!value) {
    return std::nullopt;
  }
  auto constOp = value.getDefiningOp<ONNXConstantOp>();
  if (!constOp) {
    return std::nullopt;
  }
  return getScalarOrSplatValue<T>(constOp);
}

static mlir::DenseElementsAttr buildScalarDEA(
    mlir::ShapedType likeTy, double d) {
  using namespace mlir;
  auto ranked = likeTy.dyn_cast<RankedTensorType>();
  if (!ranked || !ranked.hasStaticShape() || ranked.getNumElements() != 1)
    return DenseElementsAttr();
  Type et = ranked.getElementType();
  // Floats: let MLIR handle semantics (f8/f16/bf16/f32/f64, etc).
  if (auto ft = et.dyn_cast<FloatType>()) {
    auto fa = FloatAttr::get(ft, d);
    return DenseElementsAttr::get(ranked, fa);
  }
  // Integers: round & clamp, then use IntegerAttr.
  if (auto it = et.dyn_cast<IntegerType>()) {
    int64_t iv = static_cast<int64_t>(std::llround(d));
    const unsigned bw = it.getWidth();
    const bool isSigned = it.isSigned();
    int64_t minV = isSigned ? (-(int64_t(1) << (bw - 1))) : 0;
    int64_t maxV =
        isSigned ? ((int64_t(1) << (bw - 1)) - 1) : ((int64_t(1) << bw) - 1);
    iv = std::min<int64_t>(std::max<int64_t>(iv, minV), maxV);
    auto ia = IntegerAttr::get(it, iv);
    return DenseElementsAttr::get(ranked, ia);
  }
  return DenseElementsAttr();
}

static bool hasSingleUseBy(mlir::Value v, mlir::Operation *who) {
  bool seen = false;
  for (mlir::OpOperand &use : v.getUses()) {
    if (use.getOwner() == who && !seen) {
      seen = true;
      continue;
    }
    return false;
  }
  return seen;
}

static void updateInitializerScalar(mlir::PatternRewriter &rewriter,
    mlir::Operation *targetOp, mlir::Value oldInit, double newScalar) {
  using namespace mlir;
  auto oldCst = oldInit.getDefiningOp<ONNXConstantOp>();
  if (!oldCst)
    return;

  auto likeTy = oldInit.getType().dyn_cast<ShapedType>();
  if (!likeTy || !likeTy.hasStaticShape() || likeTy.getNumElements() != 1)
    return;

  DenseElementsAttr payload = buildScalarDEA(likeTy, newScalar);
  if (!payload)
    return;

  // Case A: mutate in place iff the *only* use is by targetOp.
  if (hasSingleUseBy(oldInit, targetOp)) {
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

  // Case B: multi-use → clone a fresh Constant with the same result type.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(targetOp);

  OperationState st(targetOp->getLoc(), ONNXConstantOp::getOperationName());
  st.addTypes(likeTy);
  st.addAttribute("value", payload);
  Operation *raw = Operation::create(st);
  rewriter.insert(raw);
  auto newCst = llvm::dyn_cast<ONNXConstantOp>(raw);
  if (!newCst)
    return;
  // Replace exactly this operand
  for (unsigned i = 0, e = targetOp->getNumOperands(); i < e; ++i)
    if (targetOp->getOperand(i) == oldInit) {
      targetOp->setOperand(i, newCst.getOutput());
      break;
    }
}

static LogicalResult tryRemoveQThenDQChain(
    mlir::PatternRewriter &rewriter, mlir::ONNXDequantizeLinearOp dqOp) {
  using namespace mlir;

  // Match Q -> DQ
  auto qOp = dqOp.getX().template getDefiningOp<ONNXQuantizeLinearOp>();
  if (!qOp) {
    return failure();
  }

  // 1) Axis / block_size must match
  if (qOp.getAxis() != dqOp.getAxis()) {
    return failure();
  }
  if (qOp.getBlockSize() != dqOp.getBlockSize()) {
    return failure();
  }

  // 2) Zero-points must match scalars/splats
  auto zpQ = getElementAttributeFromConstant(qOp.getYZeroPoint());
  auto zpDQ = getElementAttributeFromConstant(dqOp.getXZeroPoint());
  if (!zpQ || !zpDQ || zpQ != zpDQ) {
    return failure();
  }

  // 3) Scales must match scalars/splats
  auto sQ = getElementAttributeFromConstant(qOp.getYScale());
  auto sDQ = getElementAttributeFromConstant(dqOp.getXScale());
  if (!sQ || !sDQ || sQ != sDQ) {
    return failure();
  }

  // 4) Data type consistency: input of Q and output of DQ must have same elem
  // type.
  auto qInTypeOp = qOp.getX().getType();
  auto dqOutTypeOp = dqOp.getResult().getType();

  if (auto qInTensorType = qInTypeOp.dyn_cast<TensorType>()) {
    if (auto dqOutTensorType = dqOutTypeOp.dyn_cast<TensorType>()) {
      if (dqOutTensorType.getElementType() != qInTensorType.getElementType()) {
        return failure();
      }
    } else {
      return failure();
    }
  } else {
    return failure();
  }

  // Replace DQ with Q's float input; erase Q if it becomes dead.
  rewriter.replaceOp(dqOp, qOp.getX());
  if (qOp->use_empty()) {
    rewriter.eraseOp(qOp);
  }

  return success();
}

template <typename BinOp>
struct FoldBinaryThroughQDQ : public OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;

private:
  struct MatchState {
    ONNXDequantizeLinearOp dequantActivationOp = nullptr;
    ONNXConstantOp constantOp = nullptr;
    mlir::Type dstScaleDtype;
    mlir::Type dstZeroPointDtype;
    double kValue = 0.0;
    double dstScale = 0.0;
    int64_t dstZeroPoint = 0;
    double newScale = 0.0;
    int64_t newZp = 0;
  };

  LogicalResult match_qdq(MatchState &state, ONNXDequantizeLinearOp dq1,
      ONNXDequantizeLinearOp dq2) const {

    ONNXDequantizeLinearOp constantDqOp = nullptr;
    ONNXConstantOp constantSourceOp = nullptr;

    // Case 1: Direct ConstantOp as input to the DQ.
    if (auto constOp = dq1.getX().getDefiningOp<ONNXConstantOp>()) {
      constantDqOp = dq1;
      state.dequantActivationOp = dq2;
      constantSourceOp = constOp;
    } else if (auto constOp = dq2.getX().getDefiningOp<ONNXConstantOp>()) {
      constantDqOp = dq2;
      state.dequantActivationOp = dq1;
      constantSourceOp = constOp;
    }
    // Case 2: The input to the DQ op comes from a chain whose input is a
    // constant.
    else if (auto intermediateOp = dq1.getX().getDefiningOp()) {
      if (auto constOp =
              intermediateOp->getOperand(0).getDefiningOp<ONNXConstantOp>()) {
        constantDqOp = dq1;
        state.dequantActivationOp = dq2;
        constantSourceOp = constOp;
      }
    } else if (auto intermediateOp = dq2.getX().getDefiningOp()) {
      if (auto constOp =
              intermediateOp->getOperand(0).getDefiningOp<ONNXConstantOp>()) {
        constantDqOp = dq2;
        state.dequantActivationOp = dq1;
        constantSourceOp = constOp;
      }
    }

    if (!constantDqOp) {
      return failure();
    }

    // Use the templated helper to get the scalar value of the constant source.
    {
      auto scalar_value_opt = getScalarOrSplatValue<int64_t>(constantSourceOp);
      if (!scalar_value_opt) {
        return failure();
      }
      // Use the templated helper to get the scale and zero-point values.
      Value scaleVal = constantDqOp.getXScale();
      Value zpVal = constantDqOp.getXZeroPoint();
      auto scale_value_opt = getScalarOrSplatConstant<double>(scaleVal);
      auto zp_value_opt = getScalarOrSplatConstant<int64_t>(zpVal);
      if (!scale_value_opt || !zp_value_opt) {
        return failure();
      }
      // Calculate and store kValue.
      state.kValue = (*scalar_value_opt - *zp_value_opt) * *scale_value_opt;
    }
    {
      // Use the templated helper to get the scale and zero-point values.
      Value scaleVal = state.dequantActivationOp.getXScale();
      Value zpVal = state.dequantActivationOp.getXZeroPoint();
      auto scale_value_opt = getScalarOrSplatConstant<double>(scaleVal);
      auto zp_value_opt = getScalarOrSplatConstant<int64_t>(zpVal);
      if (!scale_value_opt || !zp_value_opt) {
        return failure();
      }
      state.dstScale = *scale_value_opt;
      state.dstZeroPoint = *zp_value_opt;
      // Store the data types.
      state.dstScaleDtype =
          scaleVal.getType().cast<TensorType>().getElementType();
      state.dstZeroPointDtype =
          zpVal.getType().cast<TensorType>().getElementType();
    }
    return success();
  }

  LogicalResult match_binary_op(MatchState &state, BinOp binaryOp) const {
    Value lhs = binaryOp.getOperand(0);
    Value rhs = binaryOp.getOperand(1);

    // -------- Case A: lhs is DQ, rhs is Constant --------
    if (auto dqOp = lhs.getDefiningOp<ONNXDequantizeLinearOp>()) {
      if (auto constOp = rhs.getDefiningOp<ONNXConstantOp>()) {
        state.dequantActivationOp = dqOp;
        state.constantOp = constOp;
      }
    }
    // -------- Case A reversed --------
    else if (auto dqOp = rhs.getDefiningOp<ONNXDequantizeLinearOp>()) {
      if (auto constOp = lhs.getDefiningOp<ONNXConstantOp>()) {
        state.dequantActivationOp = dqOp;
        state.constantOp = constOp;
      }
    }

    // -------- Fill kValue for Case A --------
    if (state.dequantActivationOp && state.constantOp) {
      auto kValueOpt = getScalarOrSplatValue<double>(state.constantOp);
      if (!kValueOpt.has_value()) {
        return failure();
      }
      state.kValue = kValueOpt.value();
      return success();
    }

    // -------- Case B: both inputs are DQ --------
    auto dqOp1 = lhs.getDefiningOp<ONNXDequantizeLinearOp>();
    auto dqOp2 = rhs.getDefiningOp<ONNXDequantizeLinearOp>();

    if (dqOp1 && dqOp2) {
      if (failed(match_qdq(state, dqOp1, dqOp2)))
        return failure();
      return success();
    }
    return failure();
  }

  LogicalResult check_needed_values(
      const MatchState &state, Operation *binaryOp) const {
    if (state.kValue == 0.0) {
      if (isa<ONNXDivOp>(binaryOp)) {
        return failure();
      }
    }
    if (state.dstScale == 0.0) {
      if (isa<ONNXAddOp, ONNXSubOp>(binaryOp)) {
        return failure();
      }
    }
    return success();
  }

  static bool compute_new_scale_and_zp_values(
      MatchState &state, Operation *binaryOp) {
    double newScale = state.dstScale;
    double newZpFloat = static_cast<double>(state.dstZeroPoint);
    const double kVal = state.kValue;

    if (isa<ONNXAddOp>(binaryOp)) {
      newZpFloat -= (kVal / newScale);

    } else if (isa<ONNXSubOp>(binaryOp)) {
      newZpFloat += (kVal / newScale);

    } else if (isa<ONNXMulOp>(binaryOp)) {
      newScale *= kVal;

    } else if (isa<ONNXDivOp>(binaryOp)) {
      newScale /= kVal;

    } else {
      return false;
    }

    int64_t newZp = static_cast<int64_t>(std::llround(newZpFloat));
    state.newScale = newScale;
    state.newZp = newZp;

    return true;
  }

public:
  LogicalResult matchAndRewrite(
      BinOp op, PatternRewriter &rewriter) const override {

    // STEP 1: Find the Quantize op after the binary op. Assuming only one user
    auto quantOutputOp = dyn_cast<ONNXQuantizeLinearOp>(*op->user_begin());
    if (!quantOutputOp) {
      return failure();
    }

    // Instantiate the state struct
    MatchState state;

    // STEP 2
    if (failed(match_binary_op(state, op))) {
      return failure();
    }

    // STEP 3
    if (failed(check_needed_values(state, op))) {
      return failure();
    }

    // STEP 4
    if (!compute_new_scale_and_zp_values(state, op)) {
      return failure();
    }

    // STEP 5: call initializer based on the binary op
    ONNXDequantizeLinearOp dqAct = state.dequantActivationOp;
    if constexpr (std::is_same_v<BinOp, ONNXAddOp> ||
                  std::is_same_v<BinOp, ONNXSubOp>) {
      Value zpVal = dqAct.getXZeroPoint();
      if (!zpVal) {
        return failure();
      }
      updateInitializerScalar(rewriter, dqAct.getOperation(), zpVal,
          static_cast<double>(state.newZp));

    } else if constexpr (std::is_same_v<BinOp, ONNXMulOp> ||
                         std::is_same_v<BinOp, ONNXDivOp>) {
      Value scaleVal = dqAct.getXScale();
      if (!scaleVal) {
        return failure();
      }
      updateInitializerScalar(
          rewriter, dqAct.getOperation(), scaleVal, state.newScale);
    }

    // STEP 6: Remove binary op
    rewriter.replaceOp(op, dqAct.getResult());

    // STEP 7: Remove Q->DQ chain
    for (Operation *user : quantOutputOp.getY().getUsers()) {
      if (auto tailDQ = llvm::dyn_cast<ONNXDequantizeLinearOp>(user)) {
        (void)tryRemoveQThenDQChain(rewriter, tailDQ);
      }
    }

    return success();
  }
};

struct FoldDQBinaryQPass
    : public PassWrapper<FoldDQBinaryQPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldDQBinaryQPass)

  StringRef getArgument() const final { return "dq-binary-q-opt-onnx-to-onnx"; }
  StringRef getDescription() const final {
    return "Fold Add/Sub/Mul/Div through Q/DQ by updating scale/zero_point, "
           "then remove trivial Q->DQ chains when safe.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns
        .add<FoldBinaryThroughQDQ<ONNXDivOp>, FoldBinaryThroughQDQ<ONNXSubOp>,
            FoldBinaryThroughQDQ<ONNXMulOp>, FoldBinaryThroughQDQ<ONNXAddOp>>(
            &getContext());
    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createFoldDQBinaryQPass() {
  return std::make_unique<FoldDQBinaryQPass>();
}
} // namespace onnx_mlir