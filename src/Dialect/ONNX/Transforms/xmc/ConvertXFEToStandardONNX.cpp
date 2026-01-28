// (c) Copyright 2025 Advanced Micro Devices, Inc. All Rights reserved.
//
// ConvertXFEToStandardONNX.cpp - Convert XFE ops back to standard ONNX ops
//
// This pass converts XFE (channel-last) operations back to standard ONNX
// (channel-first) operations for CPU validation purposes.

#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Helper to create transpose operation
Value createTranspose(PatternRewriter &rewriter, Location loc, Value input,
                      ArrayRef<int64_t> perm) {
  auto inputType = mlir::cast<ShapedType>(input.getType());
  
  // IMPORTANT: Preserve the element type (including quantized types)
  Type elementType = inputType.getElementType();
  
  // Use unranked type if input is unranked, will be inferred later
  Type outputType;
  if (inputType.hasRank()) {
    // Compute output shape
    SmallVector<int64_t> outputShape;
    for (int64_t p : perm) {
      outputShape.push_back(inputType.getDimSize(p));
    }
    // Preserve quantized element type in the output
    outputType = RankedTensorType::get(outputShape, elementType);
  } else {
    // Preserve quantized element type even for unranked
    outputType = UnrankedTensorType::get(elementType);
  }
  
  auto permAttr = rewriter.getI64ArrayAttr(perm);
  
  // Create the transpose op with the explicitly computed output type
  auto transposeOp = rewriter.create<ONNXTransposeOp>(loc, outputType, input, permAttr);
  
  // For non-quantized unranked tensors, try to infer shapes
  // For quantized types, NEVER call inferShapes as it strips the quantization
  bool isQuantized = elementType.template isa<mlir::quant::QuantizedType>();
  if (!inputType.hasRank() && !isQuantized) {
    if (failed(transposeOp.inferShapes(nullptr))) {
      // Shape inference failed, keep the unranked type
    }
  }
  
  // CRITICAL: Verify and force-correct the result type to preserve quantization
  auto actualResultType = mlir::cast<ShapedType>(transposeOp.getResult().getType());
  Type actualElemType = actualResultType.getElementType();
  
  // Check if the element type was corrupted (e.g., !quant.uniform<u8:f32, ...> became ui8)
  if (actualElemType != elementType) {
    // The op creation or shape inference corrupted the type - forcibly fix it!
    Type correctedType;
    if (actualResultType.hasRank()) {
      correctedType = RankedTensorType::get(actualResultType.getShape(), elementType);
    } else {
      correctedType = UnrankedTensorType::get(elementType);
    }
    
    // Force set the corrected type
    transposeOp.getResult().setType(correctedType);
  }
  
  return transposeOp.getResult();
}

// Helper to create a constant for quantization parameters
Value createQuantParamConstant(PatternRewriter &rewriter, Location loc, 
                               float scale, int64_t zeroPoint, bool isUnsigned) {
  auto scaleType = RankedTensorType::get({}, rewriter.getF32Type());
  auto scaleAttr = DenseElementsAttr::get(scaleType, rewriter.getF32FloatAttr(scale));
  return rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(), scaleAttr).getResult();
}

Value createZeroPointConstant(PatternRewriter &rewriter, Location loc,
                              int64_t zeroPoint, bool isUnsigned) {
  // ONNX Constant only supports ui8 (unsigned) or i8 (signless), not si8 (signed)
  // For signed zero-points, we use signless i8
  Type elementType;
  if (isUnsigned) {
    elementType = rewriter.getIntegerType(8, /*isSigned=*/false); // ui8
  } else {
    elementType = rewriter.getI8Type(); // i8 (signless)
  }
  
  auto zpType = RankedTensorType::get({}, elementType);
  
  // Create integer attribute with the correct type
  auto zpIntAttr = IntegerAttr::get(elementType, zeroPoint);
  auto zpAttr = DenseElementsAttr::get(zpType, zpIntAttr);
  
  return rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(), zpAttr).getResult();
}

// Convert XFEConv back to standard Conv  
// XFEConv uses NHWC layout, Conv uses NCHW layout
// NOTE: This pass preserves !quant.uniform types
struct ConvertXFEConvPattern : public OpRewritePattern<XFEConvOp> {
  using OpRewritePattern<XFEConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XFEConvOp xfeConvOp,
                                PatternRewriter &rewriter) const override {
    Location loc = xfeConvOp.getLoc();
    
    // Get inputs: input is NHWC
    Value input = xfeConvOp.getX();
    Value weight = xfeConvOp.getW();
    Value bias = xfeConvOp.getB();
    
    auto inputType = mlir::cast<ShapedType>(input.getType());
    auto rank = inputType.getRank();
    
    if (rank != 4) {
      return rewriter.notifyMatchFailure(xfeConvOp, 
          "Only 4D convolutions supported for conversion");
    }
    
    // Transpose input from NHWC to NCHW: [0,3,1,2]
    Value inputNCHW = createTranspose(rewriter, loc, input, {0, 3, 1, 2});
    
    // Transpose weight from OHWI to OIHW: [0,3,1,2]
    Value weightOIHW = createTranspose(rewriter, loc, weight, {0, 3, 1, 2});
    
    // If weight is storage type (i8/ui8), insert DequantizeLinear to convert to f32
    // Standard onnx.Conv requires float or !quant.uniform, not raw storage types
    auto weightType = mlir::cast<ShapedType>(weightOIHW.getType());
    if (weightType.getElementType().isInteger(8) || 
        weightType.getElementType().isUnsignedInteger(8)) {
      // Weight is raw i8/ui8 - need to dequantize
      // Use scale from bias (they share quantization params typically)
      auto biasType = mlir::cast<ShapedType>(bias.getType());
      if (auto biasQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
              biasType.getElementType())) {
        // Extract scale/zp from bias quantization
        float scale = biasQuantType.getScale();
        int64_t zeroPoint = 0;  // Weights typically have zp=0
        Type storageType = weightType.getElementType();
        
        // Create scale and zero-point constants for DQ
        auto scaleType = RankedTensorType::get({}, rewriter.getF32Type());
        auto scaleAttr = DenseElementsAttr::get(scaleType, rewriter.getF32FloatAttr(scale));
        Value scaleConst = rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(), scaleAttr).getResult();
        
        Type zpElemType = storageType.isUnsignedInteger(8) ? 
          rewriter.getIntegerType(8, /*isSigned=*/false) : rewriter.getI8Type();
        auto zpType = RankedTensorType::get({}, zpElemType);
        auto zpIntAttr = IntegerAttr::get(zpElemType, zeroPoint);
        auto zpAttr = DenseElementsAttr::get(zpType, zpIntAttr);
        Value zpConst = rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(), zpAttr).getResult();
        
        Type dqOutputType = weightType.hasRank() ?
          (Type)RankedTensorType::get(weightType.getShape(), rewriter.getF32Type()) :
          (Type)UnrankedTensorType::get(rewriter.getF32Type());
        
        auto dqOp = rewriter.create<ONNXDequantizeLinearOp>(
            loc, dqOutputType, weightOIHW, scaleConst, zpConst);
        
        dqOp.inferShapes(nullptr);
        weightOIHW = dqOp.getResult();
      }
    }
    
    // Get conv attributes
    StringRef autoPad = xfeConvOp.getAutoPad();
    int64_t group = xfeConvOp.getGroup();
    ArrayAttr dilationsAttr = xfeConvOp.getDilations().value_or(ArrayAttr());
    ArrayAttr kernelShapeAttr = xfeConvOp.getKernelShape().value_or(ArrayAttr());
    ArrayAttr padsAttr = xfeConvOp.getPads().value_or(ArrayAttr());
    ArrayAttr stridesAttr = xfeConvOp.getStrides().value_or(ArrayAttr());
    
    // Conv should output the same type as its inputs (after DQ conversion, this will be f32)
    // Get the actual element type from input (could be f32 if already dequantized, or !quant.uniform)
    auto inputElemType = inputType.getElementType();
    
    // If input is quantized, use the expressed type (f32) for output
    Type outputElemType;
    if (auto quantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(inputElemType)) {
      outputElemType = quantType.getExpressedType();  // f32
    } else {
      outputElemType = inputElemType;  // Already f32 or other float type
    }
    
    auto convOp = rewriter.create<ONNXConvOp>(
        loc, UnrankedTensorType::get(outputElemType),
        inputNCHW, weightOIHW, bias,
        autoPad, dilationsAttr, group, kernelShapeAttr, padsAttr, stridesAttr);
    
    // Infer shapes
    if (failed(convOp.inferShapes(nullptr))) {
      return failure();
    }
    
    // Transpose output from NCHW back to NHWC: [0,2,3,1]
    Value outputNHWC = createTranspose(rewriter, loc, convOp.getResult(), {0, 2, 3, 1});
    
    rewriter.replaceOp(xfeConvOp, outputNHWC);
    return success();
  }
};

// Convert XFEInstanceNormalization back to standard InstanceNormalization
struct ConvertXFEInstanceNormPattern : public OpRewritePattern<XFEInstanceNormalizationOp> {
  using OpRewritePattern<XFEInstanceNormalizationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XFEInstanceNormalizationOp xfeNormOp,
                                PatternRewriter &rewriter) const override {
    Location loc = xfeNormOp.getLoc();
    
    Value input = xfeNormOp.getInput();
    Value scale = xfeNormOp.getScale();
    Value B = xfeNormOp.getB();
    
    auto inputType = mlir::cast<ShapedType>(input.getType());
    auto rank = inputType.getRank();
    
    if (rank < 3) {
      return rewriter.notifyMatchFailure(xfeNormOp, 
          "InstanceNorm requires at least 3D input");
    }
    
    // Transpose input from NHWC to NCHW
    SmallVector<int64_t> toNCHW;
    toNCHW.push_back(0); // N
    toNCHW.push_back(rank - 1); // C (last dim)
    for (int64_t i = 1; i < rank - 1; ++i) {
      toNCHW.push_back(i); // spatial dims
    }
    Value inputNCHW = createTranspose(rewriter, loc, input, toNCHW);
    
    // Create standard InstanceNormalization
    llvm::APFloat epsilon = xfeNormOp.getEpsilon();
    
    auto elemType = inputType.getElementType();
    bool isQuantized = elemType.isa<mlir::quant::QuantizedType>();
    
    auto normOp = rewriter.create<ONNXInstanceNormalizationOp>(
        loc,
        UnrankedTensorType::get(elemType),
        inputNCHW,
        scale,
        B,
        epsilon);
    
    // Only infer shapes for non-quantized types
    if (!isQuantized && failed(normOp.inferShapes(nullptr))) {
      return failure();
    }
    
    // Transpose output from NCHW back to NHWC
    SmallVector<int64_t> toNHWC;
    toNHWC.push_back(0); // N
    for (int64_t i = 2; i < rank; ++i) {
      toNHWC.push_back(i); // spatial dims
    }
    toNHWC.push_back(1); // C
    Value outputNHWC = createTranspose(rewriter, loc, normOp.getResult(), toNHWC);
    
    rewriter.replaceOp(xfeNormOp, outputNHWC);
    return success();
  }
};

// Helper to permute a 1D tensor (scales/sizes) from NHWC order to NCHW order
// For 4D: [N, H, W, C] -> [N, C, H, W] using indices [0, 3, 1, 2]
Value permuteScalesOrSizes(PatternRewriter &rewriter, Location loc, 
                           Value tensor, int64_t rank) {
  if (!tensor || tensor.getType().isa<NoneType>())
    return tensor;
    
  auto tensorType = mlir::dyn_cast<ShapedType>(tensor.getType());
  if (!tensorType || !tensorType.hasRank())
    return tensor;
  
  // Build permutation indices: NHWC -> NCHW
  // For rank 4: [0, 3, 1, 2]
  SmallVector<int64_t> perm;
  perm.push_back(0);         // N stays at 0
  perm.push_back(rank - 1);  // C moves from last to position 1
  for (int64_t i = 1; i < rank - 1; ++i) {
    perm.push_back(i);       // H, W move to positions 2, 3
  }
  
  // Try to fold constant tensors directly to preserve static shapes
  if (auto constOp = tensor.getDefiningOp<ONNXConstantOp>()) {
    if (auto denseAttr = constOp.getValueAttr().dyn_cast_or_null<DenseElementsAttr>()) {
      auto elemType = denseAttr.getElementType();
      
      if (elemType.isF32()) {
        // Extract float values and permute
        auto values = denseAttr.getValues<float>();
        SmallVector<float> permuted;
        for (int64_t i = 0; i < rank; ++i) {
          permuted.push_back(values[perm[i]]);
        }
        auto newType = RankedTensorType::get({rank}, elemType);
        auto newAttr = DenseElementsAttr::get(newType, llvm::ArrayRef(permuted));
        return rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(), newAttr).getResult();
      } else if (elemType.isInteger(64)) {
        // Extract i64 values and permute
        auto values = denseAttr.getValues<int64_t>();
        SmallVector<int64_t> permuted;
        for (int64_t i = 0; i < rank; ++i) {
          permuted.push_back(values[perm[i]]);
        }
        auto newType = RankedTensorType::get({rank}, elemType);
        auto newAttr = DenseElementsAttr::get(newType, llvm::ArrayRef(permuted));
        return rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(), newAttr).getResult();
      }
    }
  }
  
  // Fallback to Gather for non-constant tensors
  auto indicesType = RankedTensorType::get({rank}, rewriter.getI64Type());
  auto indicesAttr = DenseElementsAttr::get(indicesType, 
      llvm::ArrayRef<int64_t>(perm));
  Value indices = rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(), 
      indicesAttr).getResult();
  
  // Use Gather to permute: output[i] = input[perm[i]]
  auto outputType = tensorType;  // Same type as input
  auto gatherOp = rewriter.create<ONNXGatherOp>(loc, outputType, tensor, 
      indices, /*axis=*/0);
  
  return gatherOp.getResult();
}

// Convert XFEResize back to standard Resize
struct ConvertXFEResizePattern : public OpRewritePattern<XFEResizeOp> {
  using OpRewritePattern<XFEResizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XFEResizeOp xfeResizeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = xfeResizeOp.getLoc();
    
    Value input = xfeResizeOp.getX();
    Value roi = xfeResizeOp.getRoi();
    Value scales = xfeResizeOp.getScales();
    Value sizes = xfeResizeOp.getSizes();
    
    auto inputType = mlir::cast<ShapedType>(input.getType());
    auto rank = inputType.getRank();
    
    // Transpose input from NHWC to NCHW
    SmallVector<int64_t> toNCHW;
    toNCHW.push_back(0);
    toNCHW.push_back(rank - 1);
    for (int64_t i = 1; i < rank - 1; ++i) {
      toNCHW.push_back(i);
    }
    Value inputNCHW = createTranspose(rewriter, loc, input, toNCHW);
    
    // Permute scales/sizes from NHWC order to NCHW order
    // NHWC scales [N, H, W, C] must become NCHW [N, C, H, W]
    Value scalesNCHW = permuteScalesOrSizes(rewriter, loc, scales, rank);
    Value sizesNCHW = permuteScalesOrSizes(rewriter, loc, sizes, rank);
    
    // Get raw values for builder
    int64_t antialias = xfeResizeOp.getAntialias();
    StringRef coordMode = xfeResizeOp.getCoordinateTransformationMode();
    llvm::APFloat cubicCoeff = xfeResizeOp.getCubicCoeffA();
    int64_t excludeOutside = xfeResizeOp.getExcludeOutside();
    llvm::APFloat extrapolation = xfeResizeOp.getExtrapolationValue();
    StringRef keepAspect = xfeResizeOp.getKeepAspectRatioPolicy();
    StringRef mode = xfeResizeOp.getMode();
    StringRef nearestMode = xfeResizeOp.getNearestMode();
    
    // Unwrap optional axes attribute
    ArrayAttr axesAttr = xfeResizeOp.getAxes().value_or(ArrayAttr());
    
    // Create standard Resize with all attributes
    auto resizeOp = rewriter.create<ONNXResizeOp>(
        loc,
        UnrankedTensorType::get(inputType.getElementType()),
        inputNCHW,
        roi,
        scalesNCHW,
        sizesNCHW,
        antialias,
        axesAttr,
        coordMode,
        cubicCoeff,
        excludeOutside,
        extrapolation,
        keepAspect,
        mode,
        nearestMode);
    
    if (failed(resizeOp.inferShapes(nullptr))) {
      return failure();
    }
    
    // Transpose output from NCHW back to NHWC
    SmallVector<int64_t> toNHWC;
    toNHWC.push_back(0);
    for (int64_t i = 2; i < rank; ++i) {
      toNHWC.push_back(i);
    }
    toNHWC.push_back(1);
    Value outputNHWC = createTranspose(rewriter, loc, resizeOp.getResult(), toNHWC);
    
    rewriter.replaceOp(xfeResizeOp, outputNHWC);
    return success();
  }
};

// Convert XFEGlobalAveragePool back to standard GlobalAveragePool
struct ConvertXFEGlobalAveragePoolPattern : public OpRewritePattern<XFEGlobalAveragePoolOp> {
  using OpRewritePattern<XFEGlobalAveragePoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XFEGlobalAveragePoolOp xfePoolOp,
                                PatternRewriter &rewriter) const override {
    Location loc = xfePoolOp.getLoc();
    Value input = xfePoolOp.getX();
    auto inputType = mlir::cast<ShapedType>(input.getType());
    auto rank = inputType.getRank();
    
    // Transpose input from NHWC to NCHW
    SmallVector<int64_t> toNCHW;
    toNCHW.push_back(0);
    toNCHW.push_back(rank - 1);
    for (int64_t i = 1; i < rank - 1; ++i) {
      toNCHW.push_back(i);
    }
    Value inputNCHW = createTranspose(rewriter, loc, input, toNCHW);
    
    auto poolOp = rewriter.create<ONNXGlobalAveragePoolOp>(
        loc,
        UnrankedTensorType::get(inputType.getElementType()),
        inputNCHW);
    
    if (failed(poolOp.inferShapes(nullptr))) {
      return failure();
    }
    
    // Transpose output from NCHW back to NHWC
    SmallVector<int64_t> toNHWC;
    toNHWC.push_back(0);
    for (int64_t i = 2; i < rank; ++i) {
      toNHWC.push_back(i);
    }
    toNHWC.push_back(1);
    Value outputNHWC = createTranspose(rewriter, loc, poolOp.getResult(), toNHWC);
    
    rewriter.replaceOp(xfePoolOp, outputNHWC);
    return success();
  }
};

// Convert XFEDepthToSpace back to standard DepthToSpace
struct ConvertXFEDepthToSpacePattern : public OpRewritePattern<XFEDepthToSpaceOp> {
  using OpRewritePattern<XFEDepthToSpaceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XFEDepthToSpaceOp xfeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = xfeOp.getLoc();
    Value input = xfeOp.getInput();
    auto inputType = mlir::cast<ShapedType>(input.getType());
    auto rank = inputType.getRank();
    
    if (rank != 4) {
      return rewriter.notifyMatchFailure(xfeOp, 
          "Only 4D DepthToSpace supported for conversion");
    }
    
    // Transpose input from NHWC to NCHW: [0,3,1,2]
    Value inputNCHW = createTranspose(rewriter, loc, input, {0, 3, 1, 2});
    
    // Get attributes
    int64_t blocksize = xfeOp.getBlocksize().value_or(2);
    StringRef mode = xfeOp.getMode();
    
    // Create standard DepthToSpace
    auto depthToSpaceOp = rewriter.create<ONNXDepthToSpaceOp>(
        loc,
        UnrankedTensorType::get(inputType.getElementType()),
        inputNCHW,
        blocksize,
        mode);
    
    if (failed(depthToSpaceOp.inferShapes(nullptr))) {
      return failure();
    }
    
    // Transpose output from NCHW back to NHWC: [0,2,3,1]
    Value outputNHWC = createTranspose(rewriter, loc, depthToSpaceOp.getResult(), {0, 2, 3, 1});
    
    rewriter.replaceOp(xfeOp, outputNHWC);
    return success();
  }
};

// Convert XFESpaceToDepth back to standard SpaceToDepth
struct ConvertXFESpaceToDepthPattern : public OpRewritePattern<XFESpaceToDepthOp> {
  using OpRewritePattern<XFESpaceToDepthOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XFESpaceToDepthOp xfeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = xfeOp.getLoc();
    Value input = xfeOp.getInput();
    auto inputType = mlir::cast<ShapedType>(input.getType());
    auto rank = inputType.getRank();
    
    if (rank != 4) {
      return rewriter.notifyMatchFailure(xfeOp, 
          "Only 4D SpaceToDepth supported for conversion");
    }
    
    // Transpose input from NHWC to NCHW: [0,3,1,2]
    Value inputNCHW = createTranspose(rewriter, loc, input, {0, 3, 1, 2});
    
    int64_t blocksize = xfeOp.getBlocksize().value_or(2);
    
    auto spaceToDepthOp = rewriter.create<ONNXSpaceToDepthOp>(
        loc,
        UnrankedTensorType::get(inputType.getElementType()),
        inputNCHW,
        blocksize);
    
    if (failed(spaceToDepthOp.inferShapes(nullptr))) {
      return failure();
    }
    
    // Transpose output from NCHW back to NHWC: [0,2,3,1]
    Value outputNHWC = createTranspose(rewriter, loc, spaceToDepthOp.getResult(), {0, 2, 3, 1});
    
    rewriter.replaceOp(xfeOp, outputNHWC);
    return success();
  }
};

// Convert XFEAveragePool back to standard AveragePool
struct ConvertXFEAveragePoolPattern : public OpRewritePattern<XFEAveragePoolOp> {
  using OpRewritePattern<XFEAveragePoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XFEAveragePoolOp xfePoolOp,
                                PatternRewriter &rewriter) const override {
    Location loc = xfePoolOp.getLoc();
    Value input = xfePoolOp.getX();
    auto inputType = mlir::cast<ShapedType>(input.getType());
    auto rank = inputType.getRank();
    
    if (rank != 4) {
      return rewriter.notifyMatchFailure(xfePoolOp, 
          "Only 4D pooling supported for conversion");
    }
    
    // Transpose input from NHWC to NCHW: [0,3,1,2]
    Value inputNCHW = createTranspose(rewriter, loc, input, {0, 3, 1, 2});
    
    // Extract attributes
    auto autoPad = xfePoolOp.getAutoPadAttr();
    auto ceilMode = xfePoolOp.getCeilModeAttr();
    auto countIncludePad = xfePoolOp.getCountIncludePadAttr();
    auto dilations = xfePoolOp.getDilationsAttr();
    auto kernelShape = xfePoolOp.getKernelShapeAttr();
    auto pads = xfePoolOp.getPadsAttr();
    auto strides = xfePoolOp.getStridesAttr();
    
    auto poolOp = rewriter.create<ONNXAveragePoolOp>(
        loc,
        UnrankedTensorType::get(inputType.getElementType()),
        inputNCHW,
        autoPad,
        ceilMode,
        countIncludePad,
        dilations,
        kernelShape,
        pads,
        strides);
    
    if (failed(poolOp.inferShapes(nullptr))) {
      return failure();
    }
    
    // Transpose output from NCHW back to NHWC: [0,2,3,1]
    Value outputNHWC = createTranspose(rewriter, loc, poolOp.getResult(), {0, 2, 3, 1});
    
    rewriter.replaceOp(xfePoolOp, outputNHWC);
    return success();
  }
};

// Convert XFEMaxPool back to standard MaxPool
struct ConvertXFEMaxPoolPattern : public OpRewritePattern<XFEMaxPoolOp> {
  using OpRewritePattern<XFEMaxPoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XFEMaxPoolOp xfePoolOp,
                                PatternRewriter &rewriter) const override {
    Location loc = xfePoolOp.getLoc();
    Value input = xfePoolOp.getX();
    auto inputType = mlir::cast<ShapedType>(input.getType());
    auto rank = inputType.getRank();
    
    if (rank != 4) {
      return rewriter.notifyMatchFailure(xfePoolOp, 
          "Only 4D pooling supported for conversion");
    }
    
    // Transpose input from NHWC to NCHW: [0,3,1,2]
    Value inputNCHW = createTranspose(rewriter, loc, input, {0, 3, 1, 2});
    
    // Extract and prepare attributes
    auto autoPad = xfePoolOp.getAutoPadAttr();
    auto ceilMode = xfePoolOp.getCeilModeAttr();
    auto dilations = xfePoolOp.getDilationsAttr();
    auto kernelShape = xfePoolOp.getKernelShapeAttr();
    auto pads = xfePoolOp.getPadsAttr();
    auto storageOrder = xfePoolOp.getStorageOrderAttr();
    auto strides = xfePoolOp.getStridesAttr();
    
    auto poolOp = rewriter.create<ONNXMaxPoolSingleOutOp>(
        loc,
        UnrankedTensorType::get(inputType.getElementType()),
        inputNCHW,
        autoPad,
        ceilMode,
        dilations,
        kernelShape,
        pads,
        storageOrder,
        strides);
    
    if (failed(poolOp.inferShapes(nullptr))) {
      return failure();
    }
    
    // Transpose output from NCHW back to NHWC: [0,2,3,1]
    Value outputNHWC = createTranspose(rewriter, loc, poolOp.getResult(), {0, 2, 3, 1});
    
    rewriter.replaceOp(xfePoolOp, outputNHWC);
    return success();
  }
};

// Convert XFEConvTranspose back to standard ConvTranspose
// XFEConvTranspose uses NHWC layout for input/output and OHWI for weight
// ConvTranspose uses NCHW layout for input/output and IOHW for weight
struct ConvertXFEConvTransposePattern : public OpRewritePattern<XFEConvTransposeOp> {
  using OpRewritePattern<XFEConvTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XFEConvTransposeOp xfeConvTransposeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = xfeConvTransposeOp.getLoc();
    
    // Get inputs: input is NHWC, weight is OHWI
    Value input = xfeConvTransposeOp.getX();
    Value weight = xfeConvTransposeOp.getW();
    Value bias = xfeConvTransposeOp.getB();
    
    auto inputType = mlir::cast<ShapedType>(input.getType());
    auto weightType = mlir::cast<ShapedType>(weight.getType());
    auto rank = inputType.getRank();
    
    if (rank != 4) {
      return rewriter.notifyMatchFailure(xfeConvTransposeOp, 
          "Only 4D ConvTranspose supported for conversion");
    }
    
    // Transpose input from NHWC to NCHW: [0,3,1,2]
    Value inputNCHW = createTranspose(rewriter, loc, input, {0, 3, 1, 2});
    
    // Transpose weight from OHWI to IOHW: [3,0,1,2]
    // XFEConvTranspose weight: O=dim0, H=dim1, W=dim2, I=dim3
    // ConvTranspose weight:    I=dim0, O=dim1, H=dim2, W=dim3
    Value weightIOHW = createTranspose(rewriter, loc, weight, {3, 0, 1, 2});
    
    // If weight is storage type (i8/ui8), insert DequantizeLinear to convert to f32
    if (weightType.getElementType().isInteger(8) || 
        weightType.getElementType().isUnsignedInteger(8)) {
      // Weight is raw i8/ui8 - need to dequantize
      auto weightTransposedType = mlir::cast<ShapedType>(weightIOHW.getType());
      float scale = 1.0f;  // Default scale
      int64_t zeroPoint = 0;
      Type storageType = weightType.getElementType();
      
      // Try to get scale from bias if it has quant type
      auto biasType = mlir::cast<ShapedType>(bias.getType());
      if (auto biasQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
              biasType.getElementType())) {
        scale = biasQuantType.getScale();
      }
      
      // Create scale and zero-point constants for DQ
      auto scaleType = RankedTensorType::get({}, rewriter.getF32Type());
      auto scaleAttr = DenseElementsAttr::get(scaleType, rewriter.getF32FloatAttr(scale));
      Value scaleConst = rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(), scaleAttr).getResult();
      
      Type zpElemType = storageType.isUnsignedInteger(8) ? 
        rewriter.getIntegerType(8, /*isSigned=*/false) : rewriter.getI8Type();
      auto zpType = RankedTensorType::get({}, zpElemType);
      auto zpIntAttr = IntegerAttr::get(zpElemType, zeroPoint);
      auto zpAttr = DenseElementsAttr::get(zpType, zpIntAttr);
      Value zpConst = rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(), zpAttr).getResult();
      
      Type dqOutputType = weightTransposedType.hasRank() ?
        (Type)RankedTensorType::get(weightTransposedType.getShape(), rewriter.getF32Type()) :
        (Type)UnrankedTensorType::get(rewriter.getF32Type());
      
      auto dqOp = rewriter.create<ONNXDequantizeLinearOp>(
          loc, dqOutputType, weightIOHW, scaleConst, zpConst);
      
      dqOp.inferShapes(nullptr);
      weightIOHW = dqOp.getResult();
    }
    
    // Get ConvTranspose attributes
    StringRef autoPad = xfeConvTransposeOp.getAutoPad();
    int64_t group = xfeConvTransposeOp.getGroup();
    ArrayAttr dilationsAttr = xfeConvTransposeOp.getDilations().value_or(ArrayAttr());
    ArrayAttr kernelShapeAttr = xfeConvTransposeOp.getKernelShape().value_or(ArrayAttr());
    ArrayAttr outputPaddingAttr = xfeConvTransposeOp.getOutputPadding().value_or(ArrayAttr());
    ArrayAttr outputShapeAttr = xfeConvTransposeOp.getOutputShape().value_or(ArrayAttr());
    ArrayAttr padsAttr = xfeConvTransposeOp.getPads().value_or(ArrayAttr());
    ArrayAttr stridesAttr = xfeConvTransposeOp.getStrides().value_or(ArrayAttr());
    
    // Determine output element type
    auto inputElemType = inputType.getElementType();
    Type outputElemType;
    if (auto quantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(inputElemType)) {
      outputElemType = quantType.getExpressedType();  // f32
    } else {
      outputElemType = inputElemType;  // Already f32 or other float type
    }
    
    // Create standard ConvTranspose
    auto convTransposeOp = rewriter.create<ONNXConvTransposeOp>(
        loc, UnrankedTensorType::get(outputElemType),
        inputNCHW, weightIOHW, bias,
        autoPad, dilationsAttr, group, kernelShapeAttr,
        outputPaddingAttr, outputShapeAttr, padsAttr, stridesAttr);
    
    // Infer shapes
    if (failed(convTransposeOp.inferShapes(nullptr))) {
      return failure();
    }
    
    // Transpose output from NCHW back to NHWC: [0,2,3,1]
    Value outputNHWC = createTranspose(rewriter, loc, convTransposeOp.getResult(), {0, 2, 3, 1});
    
    rewriter.replaceOp(xfeConvTransposeOp, outputNHWC);
    return success();
  }
};

// Convert XFEGlobalMaxPool back to standard GlobalMaxPool
struct ConvertXFEGlobalMaxPoolPattern : public OpRewritePattern<XFEGlobalMaxPoolOp> {
  using OpRewritePattern<XFEGlobalMaxPoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XFEGlobalMaxPoolOp xfePoolOp,
                                PatternRewriter &rewriter) const override {
    Location loc = xfePoolOp.getLoc();
    Value input = xfePoolOp.getX();
    auto inputType = mlir::cast<ShapedType>(input.getType());
    auto rank = inputType.getRank();
    
    // Transpose input from NHWC to NCHW
    SmallVector<int64_t> toNCHW;
    toNCHW.push_back(0);
    toNCHW.push_back(rank - 1);
    for (int64_t i = 1; i < rank - 1; ++i) {
      toNCHW.push_back(i);
    }
    Value inputNCHW = createTranspose(rewriter, loc, input, toNCHW);
    
    auto poolOp = rewriter.create<ONNXGlobalMaxPoolOp>(
        loc,
        UnrankedTensorType::get(inputType.getElementType()),
        inputNCHW);
    
    if (failed(poolOp.inferShapes(nullptr))) {
      return failure();
    }
    
    // Transpose output from NCHW back to NHWC
    SmallVector<int64_t> toNHWC;
    toNHWC.push_back(0);
    for (int64_t i = 2; i < rank; ++i) {
      toNHWC.push_back(i);
    }
    toNHWC.push_back(1);
    Value outputNHWC = createTranspose(rewriter, loc, poolOp.getResult(), toNHWC);
    
    rewriter.replaceOp(xfePoolOp, outputNHWC);
    return success();
  }
};

// Pass definition
struct ConvertXFEToStandardONNXPass
    : public PassWrapper<ConvertXFEToStandardONNXPass, OperationPass<func::FuncOp>> {
  
  StringRef getArgument() const override { return "convert-xfe-to-standard-onnx"; }
  
  StringRef getDescription() const override {
    return "Convert XFE (channel-last) operations to standard ONNX (channel-first) for CPU execution";
  }

  void runOnOperation() override {
    auto function = getOperation();
    MLIRContext *context = &getContext();
    
    RewritePatternSet patterns(context);
    patterns.add<ConvertXFEConvPattern>(context);
    patterns.add<ConvertXFEConvTransposePattern>(context);
    patterns.add<ConvertXFEInstanceNormPattern>(context);
    patterns.add<ConvertXFEResizePattern>(context);
    patterns.add<ConvertXFEGlobalAveragePoolPattern>(context);
    patterns.add<ConvertXFEGlobalMaxPoolPattern>(context);
    patterns.add<ConvertXFEDepthToSpacePattern>(context);
    patterns.add<ConvertXFESpaceToDepthPattern>(context);
    patterns.add<ConvertXFEAveragePoolPattern>(context);
    patterns.add<ConvertXFEMaxPoolPattern>(context);
    
    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createConvertXFEToStandardONNXPass() {
  return std::make_unique<ConvertXFEToStandardONNXPass>();
}

} // namespace onnx_mlir
