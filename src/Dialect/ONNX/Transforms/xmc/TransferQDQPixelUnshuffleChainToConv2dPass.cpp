// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// TransferQDQPixelUnshuffleChainToConv2dPass
//
// Folds the model-specific 5-op Reshape/Transpose chain that algebraically
// computes pixel-unshuffle (NCHW SpaceToDepth) into an identity-weight strided
// onnx.Conv:
//
//   reshape -> transpose(perm=[0,2,1,3]) ->
//   reshape -> transpose(perm=[0,3,1,2]) ->
//   reshape (NCHW SpaceToDepth output)
//
// becomes a single
//
//   onnx.Conv (kernel = strides = [B,B], identity-like weights, zero bias)
//
// where B is derived from the chain-input/chain-output spatial-dim ratio.  Any
// trailing layout-switch transpose (e.g. NCHW->NHWC) is left untouched and
// simply re-routed onto the new Conv result.  This combines the work of two
// xcompiler frontend passes:
//   TransferQDQPatternToPixelUnShufflePass  (refolds chain ->
//   pixel-shuffle-fix) TransferPixelUnshuffleToTileConvPass    (lowers to
//   gstiling + channel-shuffle)
// into a single ONNX-MLIR pass.  The DPU lowers a stride-B identity conv into
// exactly the same gstiling + channel-shuffle primitives the xir-level pair
// would have emitted, so the end state is equivalent.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

using namespace mlir;

namespace {

// Build an onnx.Constant carrying a (possibly-quantized) value attribute.
static Value createOnnxConstant(PatternRewriter &rewriter, Location loc,
    Type resultType, Attribute valueAttr) {
  OperationState st(loc, "onnx.Constant");
  st.addTypes(resultType);
  st.addAttribute("value", valueAttr);
  Operation *op = rewriter.create(st);
  return op->getResult(0);
}

// Read the perm attribute of an onnx.Transpose op into a vector.
static SmallVector<int64_t> getPerm(ONNXTransposeOp t) {
  SmallVector<int64_t> v;
  if (auto a = t.getPermAttr())
    for (Attribute aa : a)
      v.push_back(llvm::cast<IntegerAttr>(aa).getInt());
  return v;
}

static bool isPerm(ONNXTransposeOp t, ArrayRef<int64_t> expected) {
  auto p = getPerm(t);
  return ArrayRef<int64_t>(p) == expected;
}

// Build identity-like NCHW weights of shape [C*B*B, C, B, B] with a 1 placed at
// (oc, ic, kh, kw) when oc == (kh*B + kw)*C + ic.
static std::vector<int8_t> buildIdentityWeights(
    int64_t inputChannels, int64_t blockSize) {
  int64_t numel = inputChannels * inputChannels * blockSize * blockSize *
                  blockSize * blockSize;
  std::vector<int8_t> weights(numel, 0);
  for (int64_t kh = 0; kh < blockSize; ++kh) {
    for (int64_t kw = 0; kw < blockSize; ++kw) {
      for (int64_t ic = 0; ic < inputChannels; ++ic) {
        int64_t oc = (kh * blockSize + kw) * inputChannels + ic;
        int64_t idx = oc * inputChannels * blockSize * blockSize +
                      ic * blockSize * blockSize + kh * blockSize + kw;
        weights[idx] = 1;
      }
    }
  }
  return weights;
}

struct TransferQDQPixelUnshuffleChainToConv2dPattern
    : public OpRewritePattern<ONNXReshapeOp> {
  using OpRewritePattern<ONNXReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReshapeOp rsh2, PatternRewriter &rewriter) const override {
    // Anchor on the 5th op (rsh2 = NCHW SpaceToDepth output reshape).
    // Walk back: rsh2 <- t1 <- rsh1 <- t0 <- rsh0 <- chainIn.
    auto t1 = rsh2.getData().getDefiningOp<ONNXTransposeOp>();
    if (!t1 || !t1->hasOneUse() || !isPerm(t1, {0, 3, 1, 2}))
      return failure();
    auto rsh1 = t1.getData().getDefiningOp<ONNXReshapeOp>();
    if (!rsh1 || !rsh1->hasOneUse())
      return failure();
    auto t0 = rsh1.getData().getDefiningOp<ONNXTransposeOp>();
    if (!t0 || !t0->hasOneUse() || !isPerm(t0, {0, 2, 1, 3}))
      return failure();
    auto rsh0 = t0.getData().getDefiningOp<ONNXReshapeOp>();
    if (!rsh0 || !rsh0->hasOneUse())
      return failure();

    Value chainIn = rsh0.getData();
    auto inTy = dyn_cast<RankedTensorType>(chainIn.getType());
    auto outTy = dyn_cast<RankedTensorType>(rsh2.getResult().getType());
    if (!inTy || !inTy.hasStaticShape() || inTy.getRank() != 4)
      return failure();
    if (!outTy || !outTy.hasStaticShape() || outTy.getRank() != 4)
      return failure();

    ArrayRef<int64_t> inShape = inTy.getShape();
    ArrayRef<int64_t> outShape = outTy.getShape();

    // NCHW SpaceToDepth(B): chainIn = [N, C,      H,     W]
    //                      rsh2-out = [N, C*B*B,  H/B,   W/B]
    int64_t N = inShape[0], C = inShape[1], H = inShape[2], W = inShape[3];
    int64_t oN = outShape[0], oC = outShape[1], oH = outShape[2],
            oW = outShape[3];

    if (N != oN)
      return rewriter.notifyMatchFailure(rsh2, "N mismatch");
    if (oH <= 0 || oW <= 0 || H % oH != 0 || W % oW != 0)
      return rewriter.notifyMatchFailure(rsh2, "non-integral spatial ratio");
    int64_t bH = H / oH;
    int64_t bW = W / oW;
    if (bH != bW || bH < 2)
      return rewriter.notifyMatchFailure(rsh2, "non-square or <2 blocksize");
    int64_t B = bH;
    if (oC != C * B * B)
      return rewriter.notifyMatchFailure(
          rsh2, "out_channels != in_channels * blocksize^2");

    Location loc = rsh2.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Identity weights/bias (int8 storage) following the same construction
    // that TransferSpaceToDepthToConv2dPass uses for the canonical
    // SpaceToDepth -> Conv lowering.
    SmallVector<int64_t, 4> wShape = {C * B * B, C, B, B};
    std::vector<int8_t> wData = buildIdentityWeights(C, B);

    bool quantAvailable = (ctx->getLoadedDialect("quant") != nullptr) ||
                          (ctx->getOrLoadDialect("quant") != nullptr);
    Type inElt = inTy.getElementType();

    Value wConst, bConst;

    if (quantAvailable) {
      auto wQTy =
          quant::UniformQuantizedType::get(quant::QuantizationFlags::Signed,
              /*storage=*/rewriter.getI8Type(),
              /*expressed=*/rewriter.getF32Type(),
              /*scale=*/1.0, /*zp=*/0,
              /*storageMin=*/-128, /*storageMax=*/127);
      auto wResTy = RankedTensorType::get(wShape, wQTy);
      auto wAttrTy = RankedTensorType::get(wShape, rewriter.getI8Type());
      auto wAttr =
          DenseIntElementsAttr::get(wAttrTy, llvm::ArrayRef<int8_t>(wData));
      wConst = createOnnxConstant(rewriter, loc, wResTy, wAttr);

      // Bias follows the ONNX QLinearConv convention: int32 storage with
      // scale = x_scale * w_scale (here w_scale == 1.0) and zero point 0.
      // Using i32 (not i8) matches every other quantized Conv in the XIR
      // lowering, which is what the downstream QDQ_ConvA8W8Wrapper kernel
      // expects.
      double inScale = 1.0;
      if (auto qIn = dyn_cast<quant::UniformQuantizedType>(inElt))
        inScale = qIn.getScale();
      Type biasStorageTy = rewriter.getIntegerType(32);
      auto bQTy = quant::UniformQuantizedType::get(
          quant::QuantizationFlags::Signed, biasStorageTy,
          rewriter.getF32Type(), static_cast<float>(inScale), 0,
          /*storageMin=*/std::numeric_limits<int32_t>::min(),
          /*storageMax=*/std::numeric_limits<int32_t>::max());
      auto bResTy = RankedTensorType::get({C * B * B}, bQTy);
      auto bAttrTy = RankedTensorType::get({C * B * B}, biasStorageTy);
      std::vector<int32_t> bData(C * B * B, 0);
      auto bAttr =
          DenseIntElementsAttr::get(bAttrTy, llvm::ArrayRef<int32_t>(bData));
      bConst = createOnnxConstant(rewriter, loc, bResTy, bAttr);
    } else {
      // Non-quant fallback: emit weights/bias in the input element type.
      auto wResTy = RankedTensorType::get(wShape, inElt);
      Attribute wAttr;
      if (auto fTy = dyn_cast<FloatType>(inElt)) {
        SmallVector<APFloat> fv;
        fv.reserve(wData.size());
        for (size_t i = 0; i < wData.size(); ++i)
          fv.emplace_back(
              fTy.getFloatSemantics(), APInt::getZero(fTy.getWidth()));
        for (size_t i = 0; i < wData.size(); ++i)
          if (wData[i] != 0)
            fv[i] = APFloat(1.0f);
        wAttr = DenseFPElementsAttr::get(wResTy, fv);
      } else if (auto iTy = dyn_cast<IntegerType>(inElt)) {
        SmallVector<APInt> iv;
        iv.reserve(wData.size());
        for (size_t i = 0; i < wData.size(); ++i)
          iv.emplace_back(iTy.getWidth(), static_cast<uint64_t>(wData[i]),
              /*isSigned=*/iTy.isSigned());
        wAttr = DenseIntElementsAttr::get(wResTy, iv);
      } else {
        return rewriter.notifyMatchFailure(
            rsh2, "unsupported input element type for non-quant path");
      }
      wConst = createOnnxConstant(rewriter, loc, wResTy, wAttr);

      auto bResTy = RankedTensorType::get({C * B * B}, inElt);
      Attribute bAttr;
      if (auto fTy = dyn_cast<FloatType>(inElt)) {
        SmallVector<APFloat> fv;
        fv.reserve(C * B * B);
        for (int64_t i = 0; i < C * B * B; ++i)
          fv.emplace_back(
              fTy.getFloatSemantics(), APInt::getZero(fTy.getWidth()));
        bAttr = DenseFPElementsAttr::get(bResTy, fv);
      } else if (auto iTy = dyn_cast<IntegerType>(inElt)) {
        SmallVector<APInt> iv(
            C * B * B, APInt(iTy.getWidth(), 0, /*isSigned=*/iTy.isSigned()));
        bAttr = DenseIntElementsAttr::get(bResTy, iv);
      } else {
        return rewriter.notifyMatchFailure(
            rsh2, "unsupported input element type for non-quant path");
      }
      bConst = createOnnxConstant(rewriter, loc, bResTy, bAttr);
    }

    SmallVector<int64_t, 2> kernel = {B, B};
    SmallVector<int64_t, 2> strides = {B, B};
    SmallVector<int64_t, 4> pads = {0, 0, 0, 0};
    SmallVector<int64_t, 2> dilations = {1, 1};

    onnx_mlir::OnnxBuilder ob(rewriter, loc);
    Value y = ob.conv(outTy, chainIn, wConst, bConst, /*autoPad=*/"NOTSET",
        dilations, /*group=*/1, kernel, pads, strides);

    // Replace the 5th op (rsh2) with the Conv result.  Any downstream layout
    // transpose stays in place and is now applied to the Conv's NCHW output -
    // semantically identical to the original chain.  The earlier 4 ops
    // (rsh0/t0/rsh1/t1) become dead and are DCE'd.
    rewriter.replaceOp(rsh2, y);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct TransferQDQPixelUnshuffleChainToConv2dPass
    : public PassWrapper<TransferQDQPixelUnshuffleChainToConv2dPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "transfer-qdq-pixel-unshuffle-chain-to-conv2d";
  }
  StringRef getDescription() const override {
    return "Refold the 6-op Reshape/Transpose pixel-unshuffle decomposition "
           "into a stride-B identity onnx.Conv (combined replacement for the "
           "xcompiler-frontend TransferQDQPatternToPixelUnShuffle + "
           "TransferPixelUnshuffleToTileConv passes).";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<TransferQDQPixelUnshuffleChainToConv2dPattern>(ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createTransferQDQPixelUnshuffleChainToConv2dPass() {
  return std::make_unique<TransferQDQPixelUnshuffleChainToConv2dPass>();
}

} // namespace onnx_mlir
