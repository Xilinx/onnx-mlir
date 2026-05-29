// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass handles quantized pool operation patterns, matching the behavior
// of the XCompiler ReplaceQDQPoolPass.
//
// After the quant-types pass, pool ops already carry quantized tensor types
// (no explicit DequantizeLinear/QuantizeLinear wrappers). The pool ops
// (onnx.AveragePool, onnx.MaxPoolSingleOut) remain as standard ONNX ops
// with quantized element types.
//
// Pattern:
//   AvgPool(quantized) -> Mul(DPU coefficient const) -> remove Mul
//   The DPU computes avgpool with integer arithmetic using a fixed-point
//   approximation. When the ONNX graph contains an explicit Mul by the DPU
//   coefficient after quantized AvgPool, it is redundant and can be removed.
//

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

#define DEBUG_TYPE "replace-qdq-pool"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

bool isQuantizedType(Type type) {
  if (auto tensorType = mlir::dyn_cast<TensorType>(type))
    return mlir::isa<mlir::quant::UniformQuantizedType>(
        tensorType.getElementType());
  return false;
}

bool padValuesAreNonNegative(ArrayAttr padsAttr) {
  if (!padsAttr)
    return true;
  for (auto attr : padsAttr) {
    if (mlir::cast<IntegerAttr>(attr).getInt() < 0)
      return false;
  }
  return true;
}

std::optional<float> getConstScalarF32(Value v) {
  if (!v || isa<NoneType>(v.getType()))
    return std::nullopt;
  auto cst = v.getDefiningOp<ONNXConstantOp>();
  if (!cst)
    return std::nullopt;
  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(cst.getValueAttr());
  if (!elementsAttr)
    return std::nullopt;
  if (elementsAttr.isSplat()) {
    Type et = elementsAttr.getElementType();
    if (isa<FloatType>(et)) {
      auto apf = elementsAttr.getSplatValue<APFloat>();
      return static_cast<float>(apf.convertToDouble());
    }
    return std::nullopt;
  }
  auto shapedTy = dyn_cast<ShapedType>(elementsAttr.getType());
  if (!shapedTy || !shapedTy.hasStaticShape() || shapedTy.getNumElements() != 1)
    return std::nullopt;
  Attribute firstAttr = *elementsAttr.getValues<Attribute>().begin();
  if (auto f = dyn_cast<FloatAttr>(firstAttr))
    return static_cast<float>(f.getValueAsDouble());
  return std::nullopt;
}

/// Compute the DPU fixed-point factors for average pooling.
/// Returns (multi_factor, shift_factor) pair matching xcompiler's
/// get_avgpool_dpu_factors.
std::pair<int32_t, int32_t> getAvgPoolDpuFactors(int64_t kH, int64_t kW) {
  auto rec = static_cast<int32_t>(kH * kW);
  int32_t multiFactor = 0;
  int32_t shiftFactor = 0;

  if (kH == 3 && kW == 3) {
    multiFactor = 7;
    shiftFactor = 6;
  } else if (kH == 5 && kW == 5) {
    multiFactor = 10;
    shiftFactor = 8;
  } else if (kH == 6 && kW == 6) {
    multiFactor = 7;
    shiftFactor = 8;
  } else if (kH == 7 && kW == 7) {
    multiFactor = 21;
    shiftFactor = 10;
  } else if (kH == 14 && kW == 14) {
    multiFactor = 21;
    shiftFactor = 12;
  } else {
    auto maxFactor = static_cast<int32_t>(std::ceil(std::log2(rec * 128)));
    float diff = 1.0f;
    for (int32_t sf = 0; sf < maxFactor; sf++) {
      auto factor = static_cast<int32_t>(std::round(std::exp2(sf) / rec));
      float diff_ =
          std::abs(static_cast<float>(factor) / std::exp2(sf) - 1.0f / rec);
      if (diff_ < diff) {
        multiFactor = factor;
        diff = diff_;
        shiftFactor = sf;
      }
    }
  }
  return {multiFactor, shiftFactor};
}

/// Compute the DPU coefficient for average pooling given kernel dimensions.
/// Matches xcompiler's get_avgpool_dpu_coefficient.
float getAvgPoolDpuCoefficient(int64_t kH, int64_t kW) {
  auto [multiFactor, shiftFactor] = getAvgPoolDpuFactors(kH, kW);
  return static_cast<float>(kH * kW * multiFactor) / std::exp2(shiftFactor);
}

SmallVector<int64_t, 2> extractKernelShape(ArrayAttr attr) {
  SmallVector<int64_t, 2> kernel;
  for (auto a : attr)
    kernel.push_back(mlir::cast<IntegerAttr>(a).getInt());
  return kernel;
}

//===----------------------------------------------------------------------===//
// Pattern: Remove redundant DPU coefficient Mul after quantized AvgPool
//
// AvgPool(quantized) -> Mul(DPU_coeff_const) -> ...
// When the Mul constant matches the DPU coefficient for the kernel,
// the Mul is removed since the DPU implicitly applies this factor.
//
// This matches xcompiler's ReplaceQDQPoolPass::replace_with_mul.
//===----------------------------------------------------------------------===//

struct RemoveAvgPoolDpuCoefficientMul : public OpRewritePattern<ONNXMulOp> {
  using OpRewritePattern<ONNXMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMulOp mulOp, PatternRewriter &rewriter) const override {
    Value a = mulOp.getA();
    Value b = mulOp.getB();

    // Find the AvgPool and const operands (either order).
    auto avgPoolOp = a.getDefiningOp<ONNXAveragePoolOp>();
    Value constVal = b;
    if (!avgPoolOp) {
      avgPoolOp = b.getDefiningOp<ONNXAveragePoolOp>();
      constVal = a;
    }
    if (!avgPoolOp)
      return rewriter.notifyMatchFailure(mulOp, "no AvgPool input");

    // Require quantized types.
    if (!isQuantizedType(avgPoolOp.getX().getType()) ||
        !isQuantizedType(mulOp.getResult().getType()))
      return rewriter.notifyMatchFailure(mulOp, "not quantized");

    // Require single fanout from avgpool to mul.
    if (!avgPoolOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(mulOp, "avgpool has multiple uses");

    // Require single fanout from mul.
    if (!mulOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(mulOp, "mul has multiple uses");

    // Require non-negative pads.
    if (!padValuesAreNonNegative(avgPoolOp.getPadsAttr()))
      return rewriter.notifyMatchFailure(mulOp, "negative pad values");

    // Extract the constant scalar value.
    std::optional<float> mulConstVal = getConstScalarF32(constVal);
    if (!mulConstVal)
      return rewriter.notifyMatchFailure(
          mulOp, "mul operand is not constant scalar");

    // Extract kernel shape.
    auto kernelShapeAttr = avgPoolOp.getKernelShapeAttr();
    if (!kernelShapeAttr || kernelShapeAttr.size() != 2)
      return rewriter.notifyMatchFailure(mulOp, "kernel_shape not 2D");

    auto kernel = extractKernelShape(kernelShapeAttr);
    float dpuCoeff = getAvgPoolDpuCoefficient(kernel[0], kernel[1]);

    if (std::abs(*mulConstVal - dpuCoeff) > 1e-6f)
      return rewriter.notifyMatchFailure(
          mulOp, "mul constant does not match DPU coefficient");

    LLVM_DEBUG(llvm::dbgs()
               << "replace-qdq-pool: Removing redundant DPU coefficient "
                  "Mul after AvgPool\n");

    // Replace mul output with avgpool output (remove the mul).
    rewriter.replaceOp(mulOp, avgPoolOp.getResult());

    // Erase the constant if it has no other uses.
    if (auto *constOp = constVal.getDefiningOp())
      if (constOp->use_empty())
        rewriter.eraseOp(constOp);

    return success();
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct ReplaceQDQPoolPass
    : public PassWrapper<ReplaceQDQPoolPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceQDQPoolPass)

  ReplaceQDQPoolPass() = default;
  ReplaceQDQPoolPass(const ReplaceQDQPoolPass &pass) = default;

  [[nodiscard]] StringRef getArgument() const override {
    return "replace-qdq-pool";
  }
  [[nodiscard]] StringRef getDescription() const override {
    return "Remove redundant DPU coefficient Mul after quantized AvgPool";
  }

  void runOnOperation() override {
    auto function = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    // Remove redundant DPU coefficient Mul after quantized AvgPool.
    patterns.add<RemoveAvgPoolDpuCoefficientMul>(context);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = 10;

    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

    if (failed(applyPatternsGreedily(function, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createReplaceQDQPoolPass() {
  return std::make_unique<ReplaceQDQPoolPass>();
}

} // namespace onnx_mlir
