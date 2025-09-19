//===- ConvToQLinearConvPass.cpp ---------------------------------*- C++
//-*-===//
//
// Convert pattern: DQ(X_q) -> DQ(W_q) -> [DQ(B_q)] -> Conv -> Q
// into: QLinearConv
//
// This pass looks for the exact shape of a quantized conv subgraph and replaces
// it with a single QLinearConv that consumes the real quantization parameters
// already present in the graph.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include <cmath>

using namespace mlir;
using namespace onnx_mlir;

namespace {

static bool extractScalarFloatFromConst(mlir::Value v, float &out) {
  auto def = v.getDefiningOp<ONNXConstantOp>();
  if (!def)
    return false;

  mlir::Attribute raw;
  if (def.getValue().has_value())
    raw = *def.getValue();
  else
    raw = def.getValueAttr();

  if (auto elts = raw.dyn_cast<mlir::ElementsAttr>()) {
    for (auto apf : elts.getValues<llvm::APFloat>()) {
      out = apf.convertToFloat();
      return true;
    }
    return false;
  }

  return false;
}

static mlir::DenseElementsAttr extractDenseFloatFromConst(
    ONNXConstantOp constBiasOp, mlir::Value biasFloatValue) {
  mlir::DenseElementsAttr denseBiasF;

  if (!constBiasOp.getValue().has_value())
    return denseBiasF; // empty, caller must check

  mlir::Attribute opt = *constBiasOp.getValue();

  if (auto elts = opt.dyn_cast<mlir::ElementsAttr>()) {
    // Collect floats
    llvm::SmallVector<float, 32> floats;
    floats.reserve(elts.getNumElements());
    for (auto v : elts.getValues<llvm::APFloat>())
      floats.push_back(v.convertToFloat());

    // Ensure correct tensor type shape
    auto desiredTy =
        biasFloatValue.getType().dyn_cast<mlir::RankedTensorType>();
    if (!desiredTy)
      return mlir::DenseElementsAttr(); // invalid → return empty

    // Build DenseElementsAttr
    denseBiasF =
        mlir::DenseElementsAttr::get(desiredTy, llvm::ArrayRef<float>(floats));
  }

  return denseBiasF;
}

static llvm::SmallVector<mlir::Attribute, 32> createBiasI32Attrs(
    mlir::DenseElementsAttr denseBiasF, double xScaleS, double wScaleS,
    mlir::PatternRewriter &rewriter) {
  llvm::SmallVector<mlir::Attribute, 32> biasI32Attrs;

  double denom = static_cast<double>(xScaleS) * static_cast<double>(wScaleS);
  if (denom == 0.0)
    return biasI32Attrs; // empty, caller should check

  biasI32Attrs.reserve(denseBiasF.getNumElements());
  for (auto apf : denseBiasF.getValues<llvm::APFloat>()) {
    double f = apf.convertToFloat();
    double qd = std::nearbyint(f / denom);
    int64_t qi = static_cast<int64_t>(qd);

    // clamp to int32 range
    if (qi > std::numeric_limits<int32_t>::max())
      qi = std::numeric_limits<int32_t>::max();
    if (qi < std::numeric_limits<int32_t>::min())
      qi = std::numeric_limits<int32_t>::min();

    biasI32Attrs.push_back(
        rewriter.getI32IntegerAttr(static_cast<int32_t>(qi)));
  }

  return biasI32Attrs;
}

/// Pattern that matches Conv fed by DequantizeLinear(s) and consumed by
/// QuantizeLinear.
struct ConvToQLinearConvPattern : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();

    // Conv has bias B and its defining op is DequantizeLinear
    Value biasVal = convOp.getB();
    if (!biasVal || isa<ONNXNoneOp>(biasVal.getDefiningOp()))
      return failure();

    auto dqBiasOp = dyn_cast<ONNXDequantizeLinearOp>(biasVal.getDefiningOp());
    if (!dqBiasOp) {
      return failure();
    }

    // QuantizeLinear that produces the quantized bias (input to DQ)
    auto qBiasOp =
        dyn_cast<ONNXQuantizeLinearOp>(dqBiasOp.getX().getDefiningOp());
    if (!qBiasOp) {
      return failure();
    }

    // Conv input X must come from DequantizeLinear ----
    auto dqInputOp =
        dyn_cast_or_null<ONNXDequantizeLinearOp>(convOp.getX().getDefiningOp());
    if (!dqInputOp) {
      return failure();
    }

    Value qInput = dqInputOp.getX();
    Value xScale = dqInputOp.getXScale();
    Value xZp = dqInputOp.getXZeroPoint();

    // Conv weight W must come from DequantizeLinear ----
    auto dqWeightOp =
        dyn_cast_or_null<ONNXDequantizeLinearOp>(convOp.getW().getDefiningOp());
    if (!dqWeightOp) {
      return failure();
    }
    Value qWeight = dqWeightOp.getX();
    Value wScale = dqWeightOp.getXScale();
    Value wZp = dqWeightOp.getXZeroPoint();

    // Conv output consumed by QuantizeLinear (qOutOp) ----
    if (convOp->getUsers().empty())
      return failure();
    Operation *firstUser = *convOp->getUsers().begin();
    auto qOutOp = dyn_cast<ONNXQuantizeLinearOp>(firstUser);
    if (!qOutOp)
      return failure();

    Value yScale = qOutOp.getYScale();
    Value yZp = qOutOp.getYZeroPoint();
    // ---- Extract float bias values from qBiasOp.getX() (which points to the
    // float constant) ----
    Value biasFloatValue = qBiasOp.getX();
    auto biasFloatDefOp = biasFloatValue.getDefiningOp();
    if (!biasFloatDefOp)
      return failure();

    auto constBiasOp = dyn_cast<ONNXConstantOp>(biasFloatDefOp);
    if (!constBiasOp)
      return failure();

    Value biasQ = dqBiasOp.getX();
    auto biasQType = biasQ.getType().dyn_cast<mlir::RankedTensorType>();
    if (!biasQType)
      return failure();

    Value biasInt32Val;

    // Case 1: Bias is already int32 -----------------------
    if (biasQType.getElementType().isInteger(32)) {
      biasInt32Val = biasQ;
    } else {
      // Case 2: Bias is int8. float32 → quantize -----------------

      // Try to get the ElementsAttr
      auto denseBiasF = extractDenseFloatFromConst(constBiasOp, biasFloatValue);

      if (!denseBiasF)
        return failure();

      float xScaleS = 0.0f;
      if (!extractScalarFloatFromConst(xScale, xScaleS))
        return failure();

      float wScaleS = 0.0f;
      if (!extractScalarFloatFromConst(wScale, wScaleS))
        return failure();

      auto biasI32Attrs =
          createBiasI32Attrs(denseBiasF, xScaleS, wScaleS, rewriter);
      if (biasI32Attrs.empty()) {
        return failure();
      }

      // ---- Build tensor type for i32 bias with same shape as denseBiasF ----
      auto biasTensorTy = denseBiasF.getType().cast<mlir::RankedTensorType>();
      auto biasTypeI32 = mlir::RankedTensorType::get(
          biasTensorTy.getShape(), rewriter.getIntegerType(32));

      // Build DenseElementsAttr<i32> from integer attrs
      auto denseAttrI32 =
          mlir::DenseElementsAttr::get(biasTypeI32, biasI32Attrs);

      // ---- Create ONNXConstantOp for i32 bias (pass the full argument list as
      // required) ----
      auto biasConstI32 = rewriter.create<ONNXConstantOp>(loc, biasTypeI32,
          mlir::Attribute(), denseAttrI32, mlir::FloatAttr(), mlir::ArrayAttr(),
          mlir::IntegerAttr(), mlir::ArrayAttr(), mlir::StringAttr(),
          mlir::ArrayAttr());

      biasInt32Val = biasConstI32.getResult();
    }

    // ---- Create QLinearConv: operand order:
    SmallVector<Value, 9> qconvOperands{
        qInput, xScale, xZp, qWeight, wScale, wZp, yScale, yZp, biasInt32Val};

    Type qOutType = qOutOp.getResult().getType();

    // Create QLinearConv
    auto qconv =
        rewriter.create<ONNXQLinearConvOp>(loc, qOutType, qconvOperands);

    // Replace the QuantizeLinear (qOutOp) result with qconv result
    rewriter.replaceOp(qOutOp, qconv.getResult());

    // Erase dead ops
    if (dqBiasOp->use_empty())
      rewriter.eraseOp(dqBiasOp);
    if (qBiasOp && qBiasOp->use_empty())
      rewriter.eraseOp(qBiasOp);
    if (dqInputOp && dqInputOp->use_empty())
      rewriter.eraseOp(dqInputOp);
    if (dqWeightOp && dqWeightOp->use_empty())
      rewriter.eraseOp(dqWeightOp);
    if (convOp->use_empty())
      rewriter.eraseOp(convOp);

    return success();
  }
};

/// The pass wrapper
struct ConvToQLinearConvPass
    : public PassWrapper<ConvToQLinearConvPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvToQLinearConvPass)
  StringRef getArgument() const override {
    return "conv-to-qlinearconv-onnx-to-onnx";
  }
  StringRef getDescription() const override {
    return "Replace Conv operation with a QlinearConv op";
  }
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    patterns.add<ConvToQLinearConvPattern>(&ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace
namespace onnx_mlir {
// Factory to create the pass (useful for registration)
std::unique_ptr<Pass> createConvToQLinearConvPass() {
  return std::make_unique<ConvToQLinearConvPass>();
}
} // namespace onnx_mlir