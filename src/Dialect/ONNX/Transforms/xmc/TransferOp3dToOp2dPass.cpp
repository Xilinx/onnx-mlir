// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Expand per-axis quantization scales/zeroPoints by repeating each entry
/// `factor` times. Used when reshaping axis 0 from OC to OC*D — each original
/// output channel maps to `factor` consecutive channels in the reshaped weight.
/// Returns the original type unchanged if not per-axis quantized.
static Type expandPerAxisQuantType(Type elementType, int64_t factor) {
  auto perAxisType = dyn_cast<quant::UniformQuantizedPerAxisType>(elementType);
  if (!perAxisType)
    return elementType;

  auto scales = perAxisType.getScales();
  auto zeroPoints = perAxisType.getZeroPoints();

  SmallVector<double> expandedScales;
  SmallVector<int64_t> expandedZeroPoints;
  expandedScales.reserve(scales.size() * factor);
  expandedZeroPoints.reserve(zeroPoints.size() * factor);
  for (size_t i = 0; i < scales.size(); ++i) {
    for (int64_t d = 0; d < factor; ++d) {
      expandedScales.push_back(scales[i]);
      expandedZeroPoints.push_back(zeroPoints[i]);
    }
  }

  return quant::UniformQuantizedPerAxisType::get(perAxisType.getFlags(),
      perAxisType.getStorageType(), perAxisType.getExpressedType(),
      expandedScales, expandedZeroPoints, perAxisType.getQuantizedDimension(),
      perAxisType.getStorageTypeMin(), perAxisType.getStorageTypeMax());
}

/// Extract int64_t value from an IntegerAttr
inline int64_t getIntAttrValue(Attribute attr) {
  return cast<IntegerAttr>(attr).getValue().getSExtValue();
}

/// Extract int64_t value from ArrayAttr at given index
inline int64_t getArrayAttrValue(ArrayAttr arr, size_t idx) {
  return getIntAttrValue(arr.getValue()[idx]);
}

/// Check if all elements in ArrayAttr equal the expected value
bool allArrayAttrEqual(std::optional<ArrayAttr> arr, int64_t expected) {
  if (!arr)
    return false;
  for (auto attr : arr->getValue()) {
    if (getIntAttrValue(attr) != expected)
      return false;
  }
  return true;
}

/// 5D shape dimensions helper struct
struct Shape5D {
  int64_t N, C, D, H, W;

  Shape5D(llvm::ArrayRef<int64_t> shape)
      : N(shape[0]), C(shape[1]), D(shape[2]), H(shape[3]), W(shape[4]) {}

  /// Get flattened 4D shape: [N, C*D, H, W]
  [[nodiscard]] llvm::SmallVector<int64_t, 4> to4D() const {
    return {N, C * D, H, W};
  }

  /// Get as 5D vector
  [[nodiscard]] llvm::SmallVector<int64_t, 5> toVector() const {
    return {N, C, D, H, W};
  }
};

/// Create a shape constant for ONNX Reshape
Value createShapeConstant(
    PatternRewriter &rewriter, Location loc, llvm::ArrayRef<int64_t> shape) {
  onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
  return onnxBuilder.constantInt64(shape);
}

/// Extract 2D attributes from 3D ArrayAttr (drop first dimension)
/// 3D: [D, H, W] -> 2D: [H, W]
llvm::SmallVector<int64_t, 2> extract2DFrom3D(ArrayAttr arr) {
  return {getArrayAttrValue(arr, 1), getArrayAttrValue(arr, 2)};
}

/// Extract 2D pads from 3D pads
/// 3D: [D_front, D_back, H_top, H_bottom, W_left, W_right]
/// 2D: [H_top, H_bottom, W_left, W_right]
llvm::SmallVector<int64_t, 4> extract2DPadsFrom3D(ArrayAttr arr) {
  return {getArrayAttrValue(arr, 2), getArrayAttrValue(arr, 3),
      getArrayAttrValue(arr, 4), getArrayAttrValue(arr, 5)};
}

/// Check if Conv is matmul-like (kernel=1, stride=1, pad=0, dilation=1)
bool isMatmulLikeConv(ONNXConvOp convOp) {
  return allArrayAttrEqual(convOp.getKernelShape(), 1) &&
         allArrayAttrEqual(convOp.getStrides(), 1) &&
         allArrayAttrEqual(convOp.getPads(), 0) &&
         allArrayAttrEqual(convOp.getDilations(), 1);
}

/// Get the group attribute from Conv op
int64_t getConvGroup(ONNXConvOp convOp) {
  auto groupAttr = convOp.getGroupAttr();
  return groupAttr ? groupAttr.getValue().getSExtValue() : 1;
}

/// Pattern to convert Conv3d to Conv2d (without bias, without relu)
struct Conv3dToConv2dPattern : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, PatternRewriter &rewriter) const override {
    // Only match 3D convolutions (5D tensors: NDHWC format)
    auto inputType = dyn_cast<RankedTensorType>(convOp.getX().getType());
    if (!inputType || inputType.getRank() != 5)
      return failure();

    // Check group == 1 (no grouped convolution)
    if (getConvGroup(convOp) != 1)
      return failure();

    // Skip if this conv feeds into Add or Relu (other patterns will handle)
    if (convOp.getResult().hasOneUse()) {
      auto users = convOp.getResult().getUsers();
      if (!users.empty()) {
        auto *firstUser = *users.begin();
        if (isa<ONNXAddOp, ONNXReluOp>(firstUser))
          return failure();
      }
    }

    return transformConv3dToConv2d(convOp, rewriter, nullptr, nullptr);
  }

  static LogicalResult transformConv3dToConv2d(ONNXConvOp convOp,
      PatternRewriter &rewriter, Value bias, ONNXReluOp reluOp) {
    auto loc = convOp.getLoc();
    auto input = convOp.getX();
    auto weight = convOp.getW();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto weightType = cast<RankedTensorType>(weight.getType());

    // Input shape: [N, C, D, H, W] (ONNX channel-first format)
    Shape5D inShape(inputType.getShape());

    // Weight shape: [OC, IC/group, D_k, H_k, W_k]
    auto weightShape = weightType.getShape();
    int64_t OC = weightShape[0];
    int64_t IC = weightShape[1];
    int64_t H_k = weightShape[3];
    int64_t W_k = weightShape[4];

    bool matmulLike = isMatmulLikeConv(convOp);

    // Input: [N, C*D, H, W] for both cases
    auto newInputShape = inShape.to4D();

    // Weight shape depends on matmul-like vs standard
    llvm::SmallVector<int64_t, 4> newWeightShape;
    Type newWeightElemType = weightType.getElementType();
    if (matmulLike) {
      // Weight: [OC, IC*D, 1, 1] — axis 0 stays OC, per-axis quant unchanged.
      newWeightShape = {OC, IC * inShape.D, 1, 1};
    } else {
      // Weight: [OC*D_out, IC*D, H_k, W_k] — axis 0 grows by factor D.
      // Expand per-axis quant scales/zp: each OC entry repeats D times.
      newWeightShape = {OC * inShape.D, IC * inShape.D, H_k, W_k};
      newWeightElemType = expandPerAxisQuantType(newWeightElemType, inShape.D);
    }

    // Reshape input
    auto newInputType =
        RankedTensorType::get(newInputShape, inputType.getElementType());
    auto inputShapeConst = createShapeConstant(rewriter, loc, newInputShape);
    auto reshapedInput = rewriter.create<ONNXReshapeOp>(
        loc, newInputType, input, inputShapeConst);

    // Reshape weight
    auto newWeightType =
        RankedTensorType::get(newWeightShape, newWeightElemType);
    auto weightShapeConst = createShapeConstant(rewriter, loc, newWeightShape);
    auto reshapedWeight = rewriter.create<ONNXReshapeOp>(
        loc, newWeightType, weight, weightShapeConst);

    // Extract 2D attributes from 3D (drop depth dimension)
    auto kernel = convOp.getKernelShape();
    auto strides = convOp.getStrides();
    auto pads = convOp.getPads();
    auto dilations = convOp.getDilations();

    // 3D [D, H, W] -> 2D [H, W], Pads: 3D [6 values] -> 2D [4 values]
    auto kernel2d =
        kernel ? extract2DFrom3D(*kernel) : llvm::SmallVector<int64_t, 2>{1, 1};
    auto strides2d = strides ? extract2DFrom3D(*strides)
                             : llvm::SmallVector<int64_t, 2>{1, 1};
    auto dilations2d = dilations ? extract2DFrom3D(*dilations)
                                 : llvm::SmallVector<int64_t, 2>{1, 1};
    auto pads2d = pads ? extract2DPadsFrom3D(*pads)
                       : llvm::SmallVector<int64_t, 4>{0, 0, 0, 0};

    // Handle bias if provided
    Value reshapedBias;
    if (bias) {
      auto biasType = cast<RankedTensorType>(bias.getType());
      // Bias shape must match Conv2D output channels: OC*D
      llvm::SmallVector<int64_t, 1> newBiasShape = {OC * inShape.D};
      // Bias goes from [OC] to [OC*D] on both paths — expand per-axis quant.
      auto newBiasElemType =
          expandPerAxisQuantType(biasType.getElementType(), inShape.D);
      auto newBiasType = RankedTensorType::get(newBiasShape, newBiasElemType);
      auto biasShapeConst = createShapeConstant(rewriter, loc, newBiasShape);
      reshapedBias = rewriter.create<ONNXReshapeOp>(
          loc, newBiasType, bias, biasShapeConst);
    } else {
      onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
      reshapedBias = onnxBuilder.none();
    }

    // Output: [N, OC*D, H, W] (D_out = D for kernel=1)
    llvm::SmallVector<int64_t, 4> conv2dOutputShape = {
        inShape.N, OC * inShape.D, inShape.H, inShape.W};

    auto conv2dOutputType =
        RankedTensorType::get(conv2dOutputShape, inputType.getElementType());

    // Create Conv2d
    auto conv2dOp = rewriter.create<ONNXConvOp>(loc, conv2dOutputType,
        reshapedInput, reshapedWeight, reshapedBias,
        /*auto_pad=*/convOp.getAutoPadAttr(),
        /*dilations=*/rewriter.getI64ArrayAttr(dilations2d),
        /*group=*/
        IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true), 1),
        /*kernel_shape=*/rewriter.getI64ArrayAttr(kernel2d),
        /*pads=*/rewriter.getI64ArrayAttr(pads2d),
        /*strides=*/rewriter.getI64ArrayAttr(strides2d));

    // Apply Relu if needed
    Value finalOutput = conv2dOp.getResult();
    if (reluOp) {
      auto reluOutput = rewriter.create<ONNXReluOp>(
          loc, conv2dOp.getResult().getType(), finalOutput);
      finalOutput = reluOutput.getResult();
    }

    // Reshape output back to 5D
    auto outputType =
        reluOp ? reluOp.getResult().getType() : convOp.getResult().getType();
    auto outputShape = cast<RankedTensorType>(outputType).getShape();
    llvm::SmallVector<int64_t, 5> finalOutputShape(
        outputShape.begin(), outputShape.end());

    auto outputShapeConst =
        createShapeConstant(rewriter, loc, finalOutputShape);
    auto finalReshape = rewriter.create<ONNXReshapeOp>(
        loc, outputType, finalOutput, outputShapeConst);

    if (reluOp) {
      rewriter.replaceOp(reluOp, finalReshape.getResult());
    } else {
      rewriter.replaceOp(convOp, finalReshape.getResult());
    }

    return success();
  }
};

/// Pattern to convert Conv3d+Relu to Conv2d+Relu
struct Conv3dReluToConv2dPattern : public OpRewritePattern<ONNXReluOp> {
  using OpRewritePattern<ONNXReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReluOp reluOp, PatternRewriter &rewriter) const override {
    // Check if input is Conv3d
    auto convOp = reluOp.getX().getDefiningOp<ONNXConvOp>();
    if (!convOp || !convOp.getResult().hasOneUse())
      return failure();

    // Check Conv3d (5D input)
    auto inputType = dyn_cast<RankedTensorType>(convOp.getX().getType());
    if (!inputType || inputType.getRank() != 5)
      return failure();

    // Check group == 1
    if (getConvGroup(convOp) != 1)
      return failure();

    // Use the Conv3dToConv2dPattern transformation logic
    Conv3dToConv2dPattern conv3dPattern(rewriter.getContext());
    return conv3dPattern.transformConv3dToConv2d(
        convOp, rewriter, nullptr, reluOp);
  }
};

/// Broadcast a constant DenseElementsAttr (storage-typed) from inShape to
/// outShape following numpy/ONNX broadcast rules (size-1 dims replicated).
/// Generic over rank; integer storage values are replicated by index mapping.
static DenseElementsAttr broadcastDenseTo(DenseElementsAttr in,
    llvm::ArrayRef<int64_t> inShape, llvm::ArrayRef<int64_t> outShape) {
  auto storageElem = in.getType().getElementType();
  auto outType = RankedTensorType::get(outShape, storageElem);
  if (in.isSplat())
    return DenseElementsAttr::get(outType, in.getSplatValue<mlir::Attribute>());

  int rank = static_cast<int>(outShape.size());
  llvm::SmallVector<int64_t> inStrides(rank, 1), outStrides(rank, 1);
  for (int d = rank - 2; d >= 0; --d) {
    inStrides[d] = inStrides[d + 1] * inShape[d + 1];
    outStrides[d] = outStrides[d + 1] * outShape[d + 1];
  }
  int64_t outCount = 1;
  for (auto s : outShape)
    outCount *= s;

  auto inVals = llvm::to_vector(in.getValues<llvm::APInt>());
  llvm::SmallVector<llvm::APInt> outVals;
  outVals.reserve(outCount);
  for (int64_t o = 0; o < outCount; ++o) {
    int64_t inIdx = 0;
    for (int d = 0; d < rank; ++d) {
      int64_t coord = (o / outStrides[d]) % outShape[d];
      int64_t inCoord = (inShape[d] == 1) ? 0 : coord;
      inIdx += inCoord * inStrides[d];
    }
    outVals.push_back(inVals[inIdx]);
  }
  return DenseElementsAttr::get(outType, outVals);
}

/// Broadcast an eltwise operand up to the 5D result shape *before* flattening,
/// so the subsequent 4D reshape is count-preserving. A broadcast (size-1 dim)
/// operand cannot be turned into a larger shape by Reshape; for constants we
/// materialize a proper tiled constant (per Rachit's suggestion), otherwise we
/// insert an Expand. Operands already at the full shape are returned as-is.
static Value broadcastOperandTo5D(Value v, llvm::ArrayRef<int64_t> targetShape,
    PatternRewriter &rewriter, Location loc) {
  auto vType = dyn_cast<RankedTensorType>(v.getType());
  if (!vType || vType.getShape() == targetShape)
    return v;

  // Constant operand: build a properly-shaped (tiled) constant.
  if (auto cst = v.getDefiningOp<ONNXConstantOp>()) {
    if (auto dense = dyn_cast_or_null<DenseElementsAttr>(cst.getValueAttr())) {
      auto tiledStorage =
          broadcastDenseTo(dense, vType.getShape(), targetShape);
      auto tiledResultType =
          RankedTensorType::get(targetShape, vType.getElementType());
      auto valueNamedAttr = rewriter.getNamedAttr("value", tiledStorage);
      return rewriter
          .create<ONNXConstantOp>(loc, tiledResultType, mlir::ValueRange{},
              mlir::ArrayRef<mlir::NamedAttribute>{valueNamedAttr})
          .getResult();
    }
  }

  // Non-constant broadcast operand: insert an Expand to the target shape.
  auto targetType = RankedTensorType::get(targetShape, vType.getElementType());
  auto shapeConst = createShapeConstant(rewriter, loc, targetShape);
  return rewriter.create<ONNXExpandOp>(loc, targetType, v, shapeConst);
}

/// Helper to transform 5D eltwise Add to 4D
/// Returns the 4D Add result, caller handles optional Relu and final reshape
Value transformAdd5Dto4D(
    ONNXAddOp addOp, PatternRewriter &rewriter, Location loc) {
  auto outputType = cast<RankedTensorType>(addOp.getResult().getType());
  auto resShape = outputType.getShape();
  Shape5D shape(resShape);
  auto new4dShape = shape.to4D();

  auto shapeConst = createShapeConstant(rewriter, loc, new4dShape);
  auto elemType = outputType.getElementType();
  auto new4dType = RankedTensorType::get(new4dShape, elemType);

  // A broadcast operand (e.g. a [1,1,H,W,C] grid vs [1,3,H,W,C] activation)
  // cannot be reshaped to the flattened [N,C*D,H,W] shape (count changes).
  // First broadcast each operand up to the full 5D result shape, then the 4D
  // reshape is count-preserving.
  Value a = broadcastOperandTo5D(addOp.getA(), resShape, rewriter, loc);
  Value b = broadcastOperandTo5D(addOp.getB(), resShape, rewriter, loc);

  auto reshape1 = rewriter.create<ONNXReshapeOp>(loc, new4dType, a, shapeConst);
  auto reshape2 = rewriter.create<ONNXReshapeOp>(loc, new4dType, b, shapeConst);
  return rewriter.create<ONNXAddOp>(loc, new4dType, reshape1, reshape2);
}

/// Check if Add is fed by Conv3d
bool isConvFedAdd(ONNXAddOp addOp) {
  return addOp.getA().getDefiningOp<ONNXConvOp>() ||
         addOp.getB().getDefiningOp<ONNXConvOp>();
}

/// Pattern to convert 3D element-wise Add to 2D
struct Eltwise3dToEltwise2dPattern : public OpRewritePattern<ONNXAddOp> {
  using OpRewritePattern<ONNXAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &rewriter) const override {
    auto outputType = dyn_cast<RankedTensorType>(addOp.getResult().getType());
    if (!outputType || outputType.getRank() != 5)
      return failure();

    // Skip Conv3d+Add pattern and Add+Relu pattern
    if (isConvFedAdd(addOp))
      return failure();

    if (addOp.getResult().hasOneUse()) {
      if (isa<ONNXReluOp>(*addOp.getResult().getUsers().begin()))
        return failure();
    }

    auto loc = addOp.getLoc();
    auto add4d = transformAdd5Dto4D(addOp, rewriter, loc);

    // Reshape back to 5D
    Shape5D shape(outputType.getShape());
    auto outputShapeConst =
        createShapeConstant(rewriter, loc, shape.toVector());
    auto finalReshape = rewriter.create<ONNXReshapeOp>(
        loc, outputType, add4d, outputShapeConst);

    rewriter.replaceOp(addOp, finalReshape.getResult());
    return success();
  }
};

/// Pattern to convert Eltwise3D Add + Relu to Eltwise2D Add + Relu
struct Eltwise3dReluToEltwise2dPattern : public OpRewritePattern<ONNXReluOp> {
  using OpRewritePattern<ONNXReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReluOp reluOp, PatternRewriter &rewriter) const override {
    auto addOp = reluOp.getX().getDefiningOp<ONNXAddOp>();
    if (!addOp || !addOp.getResult().hasOneUse())
      return failure();

    auto addOutputType =
        dyn_cast<RankedTensorType>(addOp.getResult().getType());
    if (!addOutputType || addOutputType.getRank() != 5)
      return failure();

    if (isConvFedAdd(addOp))
      return failure();

    auto loc = reluOp.getLoc();
    auto add4d = transformAdd5Dto4D(addOp, rewriter, loc);

    // Create ReLU on 4D
    auto relu4d = rewriter.create<ONNXReluOp>(loc, add4d.getType(), add4d);

    // Reshape back to 5D
    Shape5D shape(addOutputType.getShape());
    auto outputShapeConst =
        createShapeConstant(rewriter, loc, shape.toVector());
    auto finalReshape = rewriter.create<ONNXReshapeOp>(
        loc, reluOp.getResult().getType(), relu4d, outputShapeConst);

    rewriter.replaceOp(reluOp, finalReshape.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct TransferOp3dToOp2dPass
    : public PassWrapper<TransferOp3dToOp2dPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "transfer-op-3d-to-op-2d"; }
  StringRef getDescription() const override {
    return "Transfer 3D operations to 2D operations by reshaping tensors";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Add all patterns
    patterns.add<Conv3dToConv2dPattern>(context);
    patterns.add<Conv3dReluToConv2dPattern>(context);
    patterns.add<Eltwise3dReluToEltwise2dPattern>(context);
    patterns.add<Eltwise3dToEltwise2dPattern>(context);

    // Apply patterns greedily
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    ResultNamesUpdater rnUpdater;
    config.setListener(&rnUpdater);
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createTransferOp3dToOp2dPass() {
  return std::make_unique<TransferOp3dToOp2dPass>();
}

} // namespace onnx_mlir
