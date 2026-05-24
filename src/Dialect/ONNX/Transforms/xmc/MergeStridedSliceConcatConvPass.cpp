// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include <cassert>
#include <optional>

#define DEBUG_TYPE "onnx-merge-strided-slice-concat-conv"

using namespace mlir;

namespace {

// Helper function to get constant integers
std::optional<SmallVector<int64_t>> getConstantIntegers(Value value) {
  auto constOp = value.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return std::nullopt;

  auto valueAttr = constOp.getValueAttr();
  if (!valueAttr)
    return std::nullopt;

  auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(valueAttr);
  if (!denseAttr)
    return std::nullopt;

  SmallVector<int64_t> result;
  for (auto val : denseAttr.getValues<int64_t>()) {
    result.push_back(val);
  }
  return result;
}

// Weight reshaping function implementing pixel shuffle / depth-to-space
// transformation NHWC: maps weights from [O, H, W, C] (OHWI) to [O, H*stride,
// W*stride, C/(stride*stride)] NCHW: maps weights from [O, C, H, W] (OIHW) to
// [O, C/(stride*stride), H*stride, W*stride]
template <typename T>
SmallVector<T> reshapeWeightsData(ArrayRef<T> data,
    ArrayRef<int64_t> inputShape, int64_t stride, bool isNCHW = false) {

  assert(inputShape.size() == 4 && "Expected 4D weight tensor");

  SmallVector<T> outputData(data.size());

  if (isNCHW) {
    // OIHW format: [O, I, H, W]
    int64_t n = inputShape[0]; // output channels
    int64_t c = inputShape[1]; // input channels
    int64_t h = inputShape[2]; // kernel height
    int64_t w = inputShape[3]; // kernel width

    assert(c % (stride * stride) == 0 &&
           "Input channel dimension must be divisible by stride^2");

    SmallVector<int64_t> newShape = {
        n, c / (stride * stride), h * stride, w * stride};

    // Iterate through original weight tensor
    for (int64_t on = 0; on < n; on++) {
      for (int64_t ic = 0; ic < c; ic++) {
        for (int64_t ih = 0; ih < h; ih++) {
          for (int64_t iw = 0; iw < w; iw++) {
            // Calculate offset and new position.
            // The mapping off→(oh_offset, ow_offset) must match the concat
            // order {rr, lr, rl, ll}: off=0→(0,0), off=1→(1,0),
            // off=2→(0,1), off=3→(1,1). This is oh=off%stride, ow=off/stride
            // — same formula as the golden xcompiler and our NHWC path.
            int64_t off = ic / newShape[1]; // newShape[1] is new InChannels
            int64_t oh = ih * stride + off % stride;
            int64_t ow = iw * stride + off / stride;
            int64_t oc = ic % newShape[1];

            // Calculate linear indices
            int64_t idxIn = on * c * h * w + ic * h * w + ih * w + iw;
            int64_t idxOut = on * newShape[1] * newShape[2] * newShape[3] +
                             oc * newShape[2] * newShape[3] + oh * newShape[3] +
                             ow;
            outputData[idxOut] = data[idxIn];
          }
        }
      }
    }
  } else {
    // OHWI format: [O, H, W, I]
    int64_t n = inputShape[0]; // output channels
    int64_t h = inputShape[1]; // kernel height
    int64_t w = inputShape[2]; // kernel width
    int64_t c = inputShape[3]; // input channels

    assert(c % (stride * stride) == 0 &&
           "Input channel dimension must be divisible by stride^2");

    SmallVector<int64_t> newShape = {
        n, h * stride, w * stride, c / (stride * stride)};

    // Iterate through original weight tensor
    for (int64_t on = 0; on < n; on++) {
      for (int64_t ih = 0; ih < h; ih++) {
        for (int64_t iw = 0; iw < w; iw++) {
          for (int64_t ic = 0; ic < c; ic++) {
            // Calculate offset and new position
            int64_t off = ic / newShape[3];
            int64_t oh = ih * stride + off % stride;
            int64_t ow = iw * stride + off / stride;
            int64_t oc = ic % newShape[3];

            // Calculate linear indices
            int64_t idxIn = on * h * w * c + ih * w * c + iw * c + ic;
            int64_t idxOut = on * newShape[1] * newShape[2] * newShape[3] +
                             oh * newShape[2] * newShape[3] + ow * newShape[3] +
                             oc;

            outputData[idxOut] = data[idxIn];
          }
        }
      }
    }
  }

  return outputData;
}

// Check that an op's input and output have matching quantized element types.
// Returns false when either side is not quantized (golden only matches
// quantized ops like strided_slice-fix / concat-fix / qlinear-conv2d),
// or when both are quantized but with different scale/zero_point.
static bool hasMatchingQuantParams(Value input, Value output) {
  auto inType = mlir::dyn_cast<RankedTensorType>(input.getType());
  auto outType = mlir::dyn_cast<RankedTensorType>(output.getType());
  if (!inType || !outType)
    return false;
  auto inQ =
      mlir::dyn_cast<mlir::quant::QuantizedType>(inType.getElementType());
  auto outQ =
      mlir::dyn_cast<mlir::quant::QuantizedType>(outType.getElementType());
  if (!inQ || !outQ)
    return false;
  return inQ == outQ;
}

// Verify that all slice and concat ops in the chain have matching quantization
// parameters between their inputs and outputs. If any op has an implicit
// requantization, the merge would silently drop it and degrade accuracy.
// Mirrors the filter in the golden xcompiler MergeStridedSliceConcatConvPass.
static bool allQuantParamsMatch(
    ArrayRef<ONNXSliceOp> slices, ONNXConcatOp concatOp) {
  for (auto sliceOp : slices) {
    if (!hasMatchingQuantParams(sliceOp.getData(), sliceOp.getResult()))
      return false;
  }
  for (auto input : concatOp.getInputs()) {
    if (!hasMatchingQuantParams(input, concatOp.getResult()))
      return false;
  }
  return true;
}

// Helper to check if slice matches specific begin/strides pattern
bool checkSlicePattern(ONNXSliceOp sliceOp, ArrayRef<int64_t> expectedBegin,
    ArrayRef<int64_t> expectedStrides) {
  auto starts = getConstantIntegers(sliceOp.getStarts());
  auto steps = getConstantIntegers(sliceOp.getSteps());

  if (!starts || !steps)
    return false;

  if (starts->size() != expectedBegin.size() ||
      steps->size() != expectedStrides.size())
    return false;

  for (size_t i = 0; i < starts->size(); i++) {
    if ((*starts)[i] != expectedBegin[i])
      return false;
  }

  for (size_t i = 0; i < steps->size(); i++) {
    if ((*steps)[i] != expectedStrides[i])
      return false;
  }

  return true;
}

// Merge StridedSlice->Concat->Conv in NCHW layout.
// This pass runs before ConvertToChannelLastPass, so everything is NCHW.
// Mirrors the golden xcompiler's MergeStridedSliceConcatConvPass with all
// guards: conv dilation/stride=1, concat axis=1 with single consumer,
// matching slice hierarchy, shared input, and quantization parameter matching.
struct MergeStridedSliceConcatConvPureNCHW
    : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, PatternRewriter &rewriter) const override {

    // Check Conv is suitable: dilation=1, stride=1
    auto dilations = convOp.getDilations();
    auto strides = convOp.getStrides();
    if (!dilations || !strides)
      return rewriter.notifyMatchFailure(
          convOp, "conv dilations or strides attribute missing");

    // Extract values from ArrayAttr
    auto dilationArray = dilations->getValue();
    auto strideArray = strides->getValue();

    for (auto attr : dilationArray) {
      if (mlir::cast<IntegerAttr>(attr).getInt() != 1)
        return rewriter.notifyMatchFailure(
            convOp, "conv dilation must be 1 for all dimensions");
    }
    for (auto attr : strideArray) {
      if (mlir::cast<IntegerAttr>(attr).getInt() != 1)
        return rewriter.notifyMatchFailure(convOp,
            "conv stride must be 1 for all dimensions (pattern expects "
            "stride-2 in slices)");
    }

    // Check input comes directly from concat (NO transpose)
    auto concatOp = convOp.getX().getDefiningOp<ONNXConcatOp>();
    if (!concatOp)
      return rewriter.notifyMatchFailure(convOp,
          "conv input does not come from Concat operation (pure NCHW "
          "expects no Transpose)");

    // Check concat axis = 1 (NCHW channel axis)
    auto axisAttr = concatOp.getAxisAttr();
    if (!axisAttr || axisAttr.getValue().getSExtValue() != 1)
      return rewriter.notifyMatchFailure(
          convOp, "concat axis must be 1 (NCHW channel axis)");

    // Concat must have a single consumer (the conv).
    if (!concatOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(
          convOp, "concat has multiple consumers");

    // Check concat has exactly 4 inputs in order: {rr, lr, rl, ll}
    auto concatInputs = concatOp.getInputs();
    if (concatInputs.size() != 4)
      return rewriter.notifyMatchFailure(convOp,
          "concat must have exactly 4 inputs "
          "(expected rr, lr, rl, ll pattern)");

    // Expected slice patterns for NCHW layout [N, C, H, W]
    // First level: stride on H dimension (axis 2)
    SmallVector<int64_t> begin_l_nchw = {0, 0, 1, 0};  // Start at H=1
    SmallVector<int64_t> begin_r_nchw = {0, 0, 0, 0};  // Start at H=0
    SmallVector<int64_t> stride_h_nchw = {1, 1, 2, 1}; // Stride=2 on H (axis 2)

    // Second level: stride on W dimension (axis 3)
    SmallVector<int64_t> begin_w1_nchw = {0, 0, 0, 1}; // Start at W=1
    SmallVector<int64_t> begin_w0_nchw = {0, 0, 0, 0}; // Start at W=0
    SmallVector<int64_t> stride_w_nchw = {1, 1, 1, 2}; // Stride=2 on W (axis 3)

    // Extract second-level slices
    auto slice_rr = concatInputs[0].getDefiningOp<ONNXSliceOp>();
    auto slice_lr = concatInputs[1].getDefiningOp<ONNXSliceOp>();
    auto slice_rl = concatInputs[2].getDefiningOp<ONNXSliceOp>();
    auto slice_ll = concatInputs[3].getDefiningOp<ONNXSliceOp>();

    if (!slice_rr || !slice_lr || !slice_rl || !slice_ll)
      return rewriter.notifyMatchFailure(
          convOp, "concat inputs are not Slice operations");

    // Verify second-level slice patterns
    if (!checkSlicePattern(slice_rr, begin_w0_nchw, stride_w_nchw) ||
        !checkSlicePattern(slice_lr, begin_w0_nchw, stride_w_nchw) ||
        !checkSlicePattern(slice_rl, begin_w1_nchw, stride_w_nchw) ||
        !checkSlicePattern(slice_ll, begin_w1_nchw, stride_w_nchw))
      return rewriter.notifyMatchFailure(convOp,
          "second-level slices do not match "
          "expected W-dimension stride pattern");

    // Get first-level slices
    auto slice_l_from_lr = slice_lr.getData().getDefiningOp<ONNXSliceOp>();
    auto slice_r_from_rr = slice_rr.getData().getDefiningOp<ONNXSliceOp>();
    auto slice_l_from_ll = slice_ll.getData().getDefiningOp<ONNXSliceOp>();
    auto slice_r_from_rl = slice_rl.getData().getDefiningOp<ONNXSliceOp>();

    if (!slice_l_from_lr || !slice_r_from_rr || !slice_l_from_ll ||
        !slice_r_from_rl)
      return rewriter.notifyMatchFailure(convOp,
          "second-level slice inputs are not first-level Slice operations");

    // Verify first-level slice patterns
    if (!checkSlicePattern(slice_l_from_lr, begin_l_nchw, stride_h_nchw) ||
        !checkSlicePattern(slice_l_from_ll, begin_l_nchw, stride_h_nchw))
      return rewriter.notifyMatchFailure(convOp,
          "left first-level slices do not match "
          "expected H-dimension stride pattern");

    if (!checkSlicePattern(slice_r_from_rr, begin_r_nchw, stride_h_nchw) ||
        !checkSlicePattern(slice_r_from_rl, begin_r_nchw, stride_h_nchw))
      return rewriter.notifyMatchFailure(convOp,
          "right first-level slices do not match expected H-dimension "
          "stride pattern");

    // All first-level slices must share the same input
    Value commonInput = slice_l_from_lr.getData();
    if (slice_r_from_rr.getData() != commonInput ||
        slice_l_from_ll.getData() != commonInput ||
        slice_r_from_rl.getData() != commonInput)
      return rewriter.notifyMatchFailure(
          convOp, "first-level slices do not share the same input");

    // Verify slice_l operations are the same
    if (slice_l_from_lr.getOperation() != slice_l_from_ll.getOperation())
      return rewriter.notifyMatchFailure(
          convOp, "left first-level slice operations are not identical");

    // Verify slice_r operations are the same
    if (slice_r_from_rr.getOperation() != slice_r_from_rl.getOperation())
      return rewriter.notifyMatchFailure(
          convOp, "right first-level slice operations are not identical");

    // Verify all slices and concat have matching quantization parameters.
    SmallVector<ONNXSliceOp> allSlicesNCHW = {slice_rr, slice_lr, slice_rl,
        slice_ll, slice_l_from_lr, slice_r_from_rr};
    if (!allQuantParamsMatch(allSlicesNCHW, concatOp))
      return rewriter.notifyMatchFailure(
          convOp, "slice/concat chain has mismatched quantization parameters");

    LLVM_DEBUG(
        llvm::dbgs()
        << "MergeStridedSliceConcatConvPureNCHW: Exact pattern matched\n");

    // Get conv weights
    auto weightsOp = convOp.getW().getDefiningOp<ONNXConstantOp>();
    if (!weightsOp)
      return rewriter.notifyMatchFailure(
          convOp, "conv weights are not from a Constant operation");

    auto weightsAttr = weightsOp.getValueAttr();
    if (!weightsAttr)
      return rewriter.notifyMatchFailure(
          convOp, "conv weights constant has no value attribute");

    auto weightsData = mlir::dyn_cast<DenseElementsAttr>(weightsAttr);
    if (!weightsData)
      return rewriter.notifyMatchFailure(
          convOp, "conv weights are not dense elements");

    auto weightsType = mlir::cast<RankedTensorType>(weightsOp.getType());
    if (!weightsType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          convOp, "conv weights do not have static shape");

    // Weight shape for NCHW: [O, I, H, W] (OIHW format)
    auto weightsShape = weightsType.getShape();
    if (weightsShape.size() != 4)
      return rewriter.notifyMatchFailure(convOp,
          "conv weights are not 4D (expected OIHW format for PureNCHW)");

    // Hardcoded stride=2
    int64_t sliceStride = 2;

    // New weight shape after pixel shuffle (OIHW format)
    SmallVector<int64_t> newWeightsShape = {
        weightsShape[0],                               // output channels
        weightsShape[1] / (sliceStride * sliceStride), // I / 4
        weightsShape[2] * sliceStride,                 // H * 2
        weightsShape[3] * sliceStride                  // W * 2
    };

    // Reshape weights using pixel shuffle transformation for NCHW (OIHW)
    auto weightsElementType = weightsType.getElementType();
    DenseElementsAttr newWeightsAttr;

    // Get the actual element type from the DenseElementsAttr (what's actually
    // stored) This may differ from weightsElementType if quantized types are
    // involved
    auto actualDataType = weightsData.getElementType();

    // Process the weight data based on actual stored type
    if (actualDataType.isF32()) {
      auto values = weightsData.getValues<float>();
      SmallVector<float> originalData(values.begin(), values.end());
      auto reshapedData = reshapeWeightsData<float>(
          originalData, weightsShape, sliceStride, true);
      auto newWeightsType =
          RankedTensorType::get(newWeightsShape, actualDataType);
      newWeightsAttr =
          DenseElementsAttr::get(newWeightsType, llvm::ArrayRef(reshapedData));
    } else if (actualDataType.isInteger(8)) {
      if (actualDataType.isSignedInteger()) {
        auto values = weightsData.getValues<int8_t>();
        SmallVector<int8_t> originalData(values.begin(), values.end());
        auto reshapedData = reshapeWeightsData<int8_t>(
            originalData, weightsShape, sliceStride, true);
        auto newWeightsType =
            RankedTensorType::get(newWeightsShape, actualDataType);
        newWeightsAttr = DenseElementsAttr::get(
            newWeightsType, llvm::ArrayRef(reshapedData));
      } else {
        auto values = weightsData.getValues<uint8_t>();
        SmallVector<uint8_t> originalData(values.begin(), values.end());
        auto reshapedData = reshapeWeightsData<uint8_t>(
            originalData, weightsShape, sliceStride, true);
        auto newWeightsType =
            RankedTensorType::get(newWeightsShape, actualDataType);
        newWeightsAttr = DenseElementsAttr::get(
            newWeightsType, llvm::ArrayRef(reshapedData));
      }
    } else if (actualDataType.isInteger(16)) {
      if (actualDataType.isSignedInteger()) {
        auto values = weightsData.getValues<int16_t>();
        SmallVector<int16_t> originalData(values.begin(), values.end());
        auto reshapedData = reshapeWeightsData<int16_t>(
            originalData, weightsShape, sliceStride, true);
        auto newWeightsType =
            RankedTensorType::get(newWeightsShape, actualDataType);
        newWeightsAttr = DenseElementsAttr::get(
            newWeightsType, llvm::ArrayRef(reshapedData));
      } else {
        auto values = weightsData.getValues<uint16_t>();
        SmallVector<uint16_t> originalData(values.begin(), values.end());
        auto reshapedData = reshapeWeightsData<uint16_t>(
            originalData, weightsShape, sliceStride, true);
        auto newWeightsType =
            RankedTensorType::get(newWeightsShape, actualDataType);
        newWeightsAttr = DenseElementsAttr::get(
            newWeightsType, llvm::ArrayRef(reshapedData));
      }
    } else {
      return rewriter.notifyMatchFailure(convOp,
          "unsupported weight data type (expected f32, i8, u8, i16, or u16)");
    }

    // Create new weights constant with quantized result type
    // Build the operation manually to specify quantized result type
    auto newWeightsResultType =
        RankedTensorType::get(newWeightsShape, weightsElementType);
    auto newWeightsOp = rewriter.create<ONNXConstantOp>(weightsOp.getLoc(),
        newWeightsResultType, /*sparse_value=*/Attribute(),
        /*value=*/newWeightsAttr,
        /*value_float=*/FloatAttr(), /*value_floats=*/ArrayAttr(),
        /*value_int=*/IntegerAttr(), /*value_ints=*/ArrayAttr(),
        /*value_string=*/StringAttr(), /*value_strings=*/ArrayAttr());

    // Adjust conv parameters
    auto kernelShape = convOp.getKernelShape();
    if (!kernelShape || kernelShape->size() != 2)
      return rewriter.notifyMatchFailure(
          convOp, "conv kernel shape must be 2D");

    auto kernelArray = kernelShape->getValue();
    SmallVector<int64_t> newKernel = {
        mlir::cast<IntegerAttr>(kernelArray[0]).getInt() * sliceStride,
        mlir::cast<IntegerAttr>(kernelArray[1]).getInt() * sliceStride};

    SmallVector<int64_t> newStride = {sliceStride, sliceStride};

    // Adjust padding
    auto pads = convOp.getPads();
    SmallVector<int64_t> newPads;
    if (pads) {
      auto padArray = pads->getValue();
      for (auto attr : padArray) {
        newPads.push_back(mlir::cast<IntegerAttr>(attr).getInt() * sliceStride);
      }
    } else {
      newPads = {0, 0, 0, 0};
    }

    // Build dilations array (keep as 1,1)
    SmallVector<int64_t> newDilations = {1, 1};

    // Get the expected output type - preserve quantized type information
    auto originalOutputType = convOp.getType();

    // Create new Conv with modified parameters
    auto newConv = rewriter.create<ONNXConvOp>(convOp.getLoc(),
        originalOutputType, commonInput, // Use original input (before slices)
        newWeightsOp.getResult(), convOp.getB(), convOp.getAutoPadAttr(),
        rewriter.getI64ArrayAttr(newDilations), convOp.getGroupAttr(),
        rewriter.getI64ArrayAttr(newKernel), rewriter.getI64ArrayAttr(newPads),
        rewriter.getI64ArrayAttr(newStride));

    rewriter.replaceOp(convOp, newConv.getResult());

    // Cleanup if possible
    if (weightsOp->use_empty())
      rewriter.eraseOp(weightsOp);
    if (concatOp->use_empty())
      rewriter.eraseOp(concatOp);

    // Cleanup second-level slices
    if (slice_rr->use_empty())
      rewriter.eraseOp(slice_rr);
    if (slice_lr->use_empty())
      rewriter.eraseOp(slice_lr);
    if (slice_rl->use_empty())
      rewriter.eraseOp(slice_rl);
    if (slice_ll->use_empty())
      rewriter.eraseOp(slice_ll);

    // Cleanup first-level slices
    if (slice_l_from_lr->use_empty())
      rewriter.eraseOp(slice_l_from_lr);
    if (slice_r_from_rr->use_empty())
      rewriter.eraseOp(slice_r_from_rr);

    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct MergeStridedSliceConcatConvPass
    : public PassWrapper<MergeStridedSliceConcatConvPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "merge-strided-slice-concat-conv";
  }
  StringRef getDescription() const override {
    return "Merge StridedSlice->Concat->Conv patterns by reshaping weights";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // In the onnx-mlir pipeline, this pass runs BEFORE ConvertToChannelLast,
    // so everything is still NCHW. Only PureNCHW can match at this point.
    // The golden xcompiler has NHWC + TransposeConv patterns because its
    // layout conversion runs earlier. All patterns carry the golden's
    // quantization-parameter-matching guard.
    patterns.add<MergeStridedSliceConcatConvPureNCHW>(context);

    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createMergeStridedSliceConcatConvPass() {
  return std::make_unique<MergeStridedSliceConcatConvPass>();
}

} // namespace onnx_mlir
