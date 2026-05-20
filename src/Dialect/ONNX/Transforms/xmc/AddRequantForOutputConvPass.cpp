// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
// AddRequantForOutputConvPass
//
// When a quantized "qlinear" producer (Conv / DepthwiseConv / FusedEltwise)
// feeds a DequantizeLinear that produces a graph output (or more generally,
// the floating-point boundary), AND the producer has more than one use, insert
// an identity XCOMPILERRequantize between the producer and the
// DequantizeLinear.
//
// Before:
//   %q  = onnx.Conv(...)            : tensor<...x!quant.uniform<...>>
//   %dq = onnx.DequantizeLinear(%q) : tensor<...xf32>
//   %u  = onnx.SomeOp(%q)            ; %q has >1 use
//
// After:
//   %q  = onnx.Conv(...)               : tensor<...x!quant.uniform<...>>
//   %rq = "onnx.XCOMPILERRequantize"(%q)
//          {a_scale = [s], a_zero_point = [zp],
//           y_scale = [s], y_zero_point = [zp]} :
//           tensor<...x!quant.uniform<...>>
//   %dq = onnx.DequantizeLinear(%rq) : tensor<...xf32>
//   %u  = onnx.SomeOp(%q)
//
// The inserted Requantize is mathematically an identity (a_scale == y_scale and
// a_zero_point == y_zero_point), serving as a buffer node between the producer
// and the f32 boundary on hardware targets that require it.
//
// Mirrors xcompiler's AddRequantForOutputConvPass which matches the template
//   qconv (qlinear-conv2d / qlinear-l2_normalize / qlinear-eltwise) -> DQ
// with a "non-single-fanout" filter on the qconv.
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

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "add-requant-for-output-conv"

using namespace mlir;

namespace {

/// Build F32ArrayAttr from a UniformQuantizedType's scale.
ArrayAttr buildScaleAttr(
    PatternRewriter &rewriter, quant::UniformQuantizedType qType) {
  return rewriter.getArrayAttr(
      {rewriter.getF32FloatAttr(static_cast<float>(qType.getScale()))});
}

/// Build I64ArrayAttr from a UniformQuantizedType's zero point.
ArrayAttr buildZeroPointAttr(
    PatternRewriter &rewriter, quant::UniformQuantizedType qType) {
  return rewriter.getI64ArrayAttr({qType.getZeroPoint()});
}

/// Build F32ArrayAttr from a UniformQuantizedPerAxisType's scales.
ArrayAttr buildScaleAttr(
    PatternRewriter &rewriter, quant::UniformQuantizedPerAxisType qType) {
  SmallVector<Attribute> attrs;
  for (double s : qType.getScales())
    attrs.push_back(rewriter.getF32FloatAttr(static_cast<float>(s)));
  return rewriter.getArrayAttr(attrs);
}

/// Build I64ArrayAttr from a UniformQuantizedPerAxisType's zero points.
ArrayAttr buildZeroPointAttr(
    PatternRewriter &rewriter, quant::UniformQuantizedPerAxisType qType) {
  SmallVector<int64_t> zps(
      qType.getZeroPoints().begin(), qType.getZeroPoints().end());
  return rewriter.getI64ArrayAttr(zps);
}

/// Return true if the op is an accepted "qlinear" producer that this pass
/// targets: Conv, XCOMPILERDepthwiseConv, or XCOMPILERFusedEltwise.
/// These mirror xcompiler's qlinear-conv2d / qlinear-eltwise categories.
bool isAcceptedProducer(Operation *op) {
  return isa<ONNXConvOp, XCOMPILERDepthwiseConvOp, XCOMPILERFusedEltwiseOp>(op);
}

/// Extract identity (a == y) quantization attrs from a quantized tensor type.
/// Returns false if the element type is not a recognized quantized type.
bool extractIdentityQuantAttrs(PatternRewriter &rewriter, Type tensorTy,
    ArrayAttr &scaleAttr, ArrayAttr &zpAttr) {
  auto rtt = dyn_cast<RankedTensorType>(tensorTy);
  if (!rtt)
    return false;
  Type elem = rtt.getElementType();
  if (auto qPerTensor = dyn_cast<quant::UniformQuantizedType>(elem)) {
    scaleAttr = buildScaleAttr(rewriter, qPerTensor);
    zpAttr = buildZeroPointAttr(rewriter, qPerTensor);
    return true;
  }
  if (auto qPerAxis = dyn_cast<quant::UniformQuantizedPerAxisType>(elem)) {
    scaleAttr = buildScaleAttr(rewriter, qPerAxis);
    zpAttr = buildZeroPointAttr(rewriter, qPerAxis);
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Pattern: Insert identity XCOMPILERRequantize between a multi-use quantized
// producer and a DequantizeLinear consumer.
//===----------------------------------------------------------------------===//

struct AddRequantForOutputConvPattern
    : public OpRewritePattern<ONNXDequantizeLinearOp> {
  using OpRewritePattern<ONNXDequantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXDequantizeLinearOp dqOp, PatternRewriter &rewriter) const override {
    Value dqInput = dqOp.getX();
    Operation *producer = dqInput.getDefiningOp();
    if (!producer)
      return rewriter.notifyMatchFailure(dqOp, "DQ input has no defining op");

    if (!isAcceptedProducer(producer))
      return rewriter.notifyMatchFailure(
          dqOp, "DQ producer is not a Conv/DepthwiseConv/FusedEltwise");

    // The producer must have more than one use (mirrors xcompiler's
    // !if_single_fanout filter). Single use means inserting a buffer would be
    // a no-op transformation, so we skip.
    if (dqInput.hasOneUse())
      return rewriter.notifyMatchFailure(
          dqOp, "Producer has a single use; no buffer needed");

    // Producer output must carry a quantized element type so we can derive
    // the identity quant parameters from it.
    ArrayAttr scaleAttr;
    ArrayAttr zpAttr;
    if (!extractIdentityQuantAttrs(
            rewriter, dqInput.getType(), scaleAttr, zpAttr))
      return rewriter.notifyMatchFailure(
          dqOp, "Producer output does not have a quantized element type");

    // Avoid running on a Requantize chain: if the existing producer already is
    // an identity-ish requantize feeding this DQ, do not keep inserting more.
    if (isa<XCOMPILERRequantizeOp>(producer))
      return rewriter.notifyMatchFailure(
          dqOp, "Producer is already a Requantize");

    // Create the identity XCOMPILERRequantize. Input/output type matches the
    // producer's output type so the inserted op is a true integer identity.
    auto requantizeOp = rewriter.create<XCOMPILERRequantizeOp>(dqOp.getLoc(),
        dqInput.getType(), dqInput, /*a_scale=*/scaleAttr,
        /*a_zero_point=*/zpAttr, /*y_scale=*/scaleAttr,
        /*y_zero_point=*/zpAttr);

    LLVM_DEBUG(llvm::dbgs() << "add-requant-for-output-conv: inserted "
                            << requantizeOp << " before " << dqOp << "\n");

    rewriter.modifyOpInPlace(
        dqOp, [&]() { dqOp->setOperand(0, requantizeOp.getResult()); });
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct AddRequantForOutputConvPass
    : public PassWrapper<AddRequantForOutputConvPass,
          OperationPass<func::FuncOp>> {
  [[nodiscard]] StringRef getArgument() const override {
    return "add-requant-for-output-conv";
  }
  [[nodiscard]] StringRef getDescription() const override {
    return "Insert an identity XCOMPILERRequantize between a multi-use "
           "quantized Conv/DepthwiseConv/FusedEltwise producer and a "
           "DequantizeLinear consumer at the f32 boundary";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<quant::QuantDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<AddRequantForOutputConvPattern>(context);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    // Restrict to existing ops to prevent the pattern from re-matching on its
    // own newly-inserted Requantize, ensuring the rewrite converges.
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createAddRequantForOutputConvPass() {
  return std::make_unique<AddRequantForOutputConvPass>();
}

} // namespace onnx_mlir
