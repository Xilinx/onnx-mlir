// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass fuses the quantized L2-norm subgraph into a single onnx.ReduceL2
// op. It runs after QuantTypesPass, so the graph uses native
// !quant.uniform types and there are no explicit Q/DQ ops to handle.
//
// Two shapes are recognized (the eps=0 Add is optional):
//
//   Pattern A:  x -> Square -> ReduceSum -> Add(eps=0) -> Sqrt
//   Pattern B:  x -> Square -> ReduceSum -> Sqrt
//
// where "Square" is Pow(x, 2) or Mul(x, x) (onnx-mlir canonicalizes Pow(x,2) to
// Mul(x,x) before the xmc passes). Both compute sqrt(reduce_sum(x^2)) ==
// ReduceL2(x). The pass anchors on the terminal Sqrt, walks upstream through an
// optional eps=0 Add and the ReduceSum to the square, and replaces the Sqrt
// with onnx.ReduceL2(x, axes, keepdims) -- keeping the Sqrt's result type so
// any downstream consumer sees the same (quantized) output.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "replace-qdq-reduce-l2"

using namespace mlir;

namespace {

// Element type of a (shaped) value. For tensors of !quant.uniform this carries
// the scale/zero_point, so type equality implies matching quant params.
Type valueElementType(Value v) {
  if (auto st = dyn_cast<ShapedType>(v.getType()))
    return st.getElementType();
  return v.getType();
}

std::optional<double> getConstFloatSplat(Value v) {
  auto cst = v.getDefiningOp<ONNXConstantOp>();
  if (!cst)
    return std::nullopt;
  auto dense = dyn_cast_or_null<DenseElementsAttr>(cst.getValueAttr());
  if (!dense || !isa<FloatType>(dense.getElementType()))
    return std::nullopt;
  if (!dense.isSplat() && dense.getNumElements() != 1)
    return std::nullopt;
  return dense.getSplatValue<APFloat>().convertToDouble();
}

std::optional<Value> getSquareInput(Value v) {
  if (auto powOp = v.getDefiningOp<ONNXPowOp>()) {
    if (!powOp.getZ().hasOneUse())
      return std::nullopt;
    auto exponent = getConstFloatSplat(powOp.getY());
    if (!exponent || *exponent != 2.0)
      return std::nullopt;
    return powOp.getX();
  }
  if (auto mulOp = v.getDefiningOp<ONNXMulOp>()) {
    if (!mulOp.getC().hasOneUse())
      return std::nullopt;
    if (mulOp.getA() != mulOp.getB())
      return std::nullopt;
    return mulOp.getA();
  }
  return std::nullopt;
}

bool isEpsZero(Value v) {
  if (auto f = getConstFloatSplat(v))
    return *f == 0.0;
  auto cst = v.getDefiningOp<ONNXConstantOp>();
  if (!cst)
    return false;
  auto shaped = dyn_cast<ShapedType>(cst.getResult().getType());
  if (!shaped)
    return false;
  auto qType = dyn_cast<quant::UniformQuantizedType>(shaped.getElementType());
  if (!qType)
    return false;
  auto dense = dyn_cast_or_null<DenseElementsAttr>(cst.getValueAttr());
  if (!dense || !dense.getElementType().isIntOrIndex() ||
      !(dense.isSplat() || dense.getNumElements() == 1))
    return false;
  return dense.getSplatValue<APInt>().getSExtValue() == qType.getZeroPoint();
}

// Match the quantized L2-norm subgraph anchored on Sqrt and replace with
// onnx.ReduceL2 (native quant types; runs after QuantTypesPass).
struct ReplaceQDQReduceL2Pattern : public OpRewritePattern<ONNXSqrtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSqrtOp sqrtOp, PatternRewriter &rewriter) const override {
    Value cur = sqrtOp.getX();

    // Optional eps=0 add: one operand is a zero constant, the other continues
    // the chain up to ReduceSum.
    if (auto addOp = cur.getDefiningOp<ONNXAddOp>()) {
      if (!addOp.getResult().hasOneUse())
        return rewriter.notifyMatchFailure(sqrtOp, "eps-add has multiple uses");
      if (isEpsZero(addOp.getA()))
        cur = addOp.getB();
      else if (isEpsZero(addOp.getB()))
        cur = addOp.getA();
      else
        return rewriter.notifyMatchFailure(sqrtOp, "add is not an eps=0 add");

      if (valueElementType(cur) != valueElementType(addOp.getResult()))
        return rewriter.notifyMatchFailure(
            sqrtOp, "eps-add is not a transparent requant (scale/zp mismatch)");
    }

    auto reduceSumOp = cur.getDefiningOp<ONNXReduceSumOp>();
    if (!reduceSumOp)
      return rewriter.notifyMatchFailure(sqrtOp, "no ReduceSum feeding sqrt");
    if (!reduceSumOp.getReduced().hasOneUse())
      return rewriter.notifyMatchFailure(sqrtOp, "ReduceSum has multiple uses");

    // ReduceSum input must be x^2 (Pow(x,2) or Mul(x,x)).
    std::optional<Value> squareIn = getSquareInput(reduceSumOp.getData());
    if (!squareIn)
      return rewriter.notifyMatchFailure(
          sqrtOp, "ReduceSum input is not x^2 (Pow(x,2) or Mul(x,x))");

    LLVM_DEBUG(llvm::dbgs() << "replace-qdq-reduce-l2: matched L2-norm at "
                            << sqrtOp.getLoc() << "\n");

    Value reduceL2 = rewriter.create<ONNXReduceL2Op>(sqrtOp.getLoc(),
        sqrtOp.getResult().getType(), *squareIn, reduceSumOp.getAxes(),
        reduceSumOp.getKeepdimsAttr(), reduceSumOp.getNoopWithEmptyAxesAttr());

    rewriter.replaceOp(sqrtOp, reduceL2);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct ReplaceQDQReduceL2Pass
    : public PassWrapper<ReplaceQDQReduceL2Pass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "replace-qdq-reduce-l2"; }
  StringRef getDescription() const override {
    return "Fuse quantized Square->ReduceSum->[eps-Add]->Sqrt (L2-norm) into "
           "onnx.ReduceL2.";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReplaceQDQReduceL2Pattern>(context);
    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.setListener(&rnUpdater);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createReplaceQDQReduceL2Pass() {
  return std::make_unique<ReplaceQDQReduceL2Pass>();
}

} // namespace onnx_mlir
