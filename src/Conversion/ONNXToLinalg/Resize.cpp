/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Resize.cpp - ONNX Resize to Linalg lowering --------------===//
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace onnx_mlir {
namespace {

Value createFloatConstant(
    OpBuilder &builder, Location loc, Type type, double value) {
  return builder.create<arith::ConstantOp>(
      loc, builder.getFloatAttr(type, value));
}

Value createIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

Value castIndexToFloat(
    OpBuilder &builder, Location loc, Value value, Type floatType) {
  Value intValue =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), value);
  return builder.create<arith::SIToFPOp>(loc, floatType, intValue);
}

Value castFloatToIndex(OpBuilder &builder, Location loc, Value value) {
  Value intValue =
      builder.create<arith::FPToSIOp>(loc, builder.getI64Type(), value);
  return builder.create<arith::IndexCastOp>(
      loc, builder.getIndexType(), intValue);
}

Value clampIndex(OpBuilder &builder, Location loc, Value value, int64_t upper) {
  Value zero = createIndexConstant(builder, loc, 0);
  Value upperValue = createIndexConstant(builder, loc, upper);
  Value isBelowZero = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, value, zero);
  Value lowerClamped =
      builder.create<arith::SelectOp>(loc, isBelowZero, zero, value);
  Value isAboveUpper = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sgt, lowerClamped, upperValue);
  return builder.create<arith::SelectOp>(
      loc, isAboveUpper, upperValue, lowerClamped);
}

Value convertFloatValue(
    OpBuilder &builder, Location loc, Value value, Type dstType) {
  Type srcType = value.getType();
  if (srcType == dstType)
    return value;

  auto srcFloat = dyn_cast<FloatType>(srcType);
  auto dstFloat = dyn_cast<FloatType>(dstType);
  assert(srcFloat && dstFloat && "expected floating-point types");
  if (srcFloat.getWidth() < dstFloat.getWidth())
    return builder.create<arith::ExtFOp>(loc, dstType, value);
  return builder.create<arith::TruncFOp>(loc, dstType, value);
}

class ONNXResizeOpLoweringToLinalg : public OpRewritePattern<ONNXResizeOp> {
public:
  using OpRewritePattern<ONNXResizeOp>::OpRewritePattern;

  struct ResizeParams {
    Value input;
    RankedTensorType inputType;
    RankedTensorType resultType;
    ArrayRef<float> scales;
    StringRef coordinateTransformationMode;
    StringRef nearestMode;
  };

  static Value getCoordinate(const ResizeParams &params, OpBuilder &builder,
      Location loc, int64_t dim, Value outputIndex, Type calcType) {
    Value x = castIndexToFloat(builder, loc, outputIndex, calcType);
    Value scale =
        createFloatConstant(builder, loc, calcType, params.scales[dim]);

    if (params.coordinateTransformationMode == "half_pixel") {
      Value half = createFloatConstant(builder, loc, calcType, 0.5);
      Value shifted = builder.create<arith::AddFOp>(loc, x, half);
      Value scaled = builder.create<arith::DivFOp>(loc, shifted, scale);
      return builder.create<arith::SubFOp>(loc, scaled, half);
    }

    if (params.coordinateTransformationMode == "pytorch_half_pixel") {
      if (params.resultType.getDimSize(dim) == 1)
        return createFloatConstant(builder, loc, calcType, 0.0);
      Value half = createFloatConstant(builder, loc, calcType, 0.5);
      Value shifted = builder.create<arith::AddFOp>(loc, x, half);
      Value scaled = builder.create<arith::DivFOp>(loc, shifted, scale);
      return builder.create<arith::SubFOp>(loc, scaled, half);
    }

    if (params.coordinateTransformationMode == "asymmetric")
      return builder.create<arith::DivFOp>(loc, x, scale);

    assert(params.coordinateTransformationMode == "align_corners" &&
           "unexpected coordinate transformation mode");
    if (params.resultType.getDimSize(dim) == 1)
      return createFloatConstant(builder, loc, calcType, 0.0);
    Value inputMinusOne = createFloatConstant(
        builder, loc, calcType, params.inputType.getDimSize(dim) - 1);
    Value outputMinusOne = createFloatConstant(
        builder, loc, calcType, params.resultType.getDimSize(dim) - 1);
    Value numerator = builder.create<arith::MulFOp>(loc, x, inputMinusOne);
    return builder.create<arith::DivFOp>(loc, numerator, outputMinusOne);
  }

  static Value getNearestIndex(const ResizeParams &params, OpBuilder &builder,
      Location loc, int64_t dim, Value coordinate, Type calcType) {
    Value floorCoordinate = builder.create<math::FloorOp>(loc, coordinate);
    Value floorIndex = castFloatToIndex(builder, loc, floorCoordinate);
    Value selectedIndex = floorIndex;

    if (params.nearestMode == "round_prefer_ceil" ||
        params.nearestMode == "round_prefer_floor") {
      Value fraction =
          builder.create<arith::SubFOp>(loc, coordinate, floorCoordinate);
      Value half = createFloatConstant(builder, loc, calcType, 0.5);
      arith::CmpFPredicate predicate = params.nearestMode == "round_prefer_ceil"
                                           ? arith::CmpFPredicate::OGE
                                           : arith::CmpFPredicate::OGT;
      Value takeUpper =
          builder.create<arith::CmpFOp>(loc, predicate, fraction, half);
      Value one = createIndexConstant(builder, loc, 1);
      Value upperIndex = builder.create<arith::AddIOp>(loc, floorIndex, one);
      selectedIndex = builder.create<arith::SelectOp>(
          loc, takeUpper, upperIndex, floorIndex);
    }

    return clampIndex(
        builder, loc, selectedIndex, params.inputType.getDimSize(dim) - 1);
  }

  static Value buildNearest(const ResizeParams &params, OpBuilder &builder,
      Location loc, Type calcType) {
    SmallVector<Value> indices;
    for (int64_t dim = 0; dim < params.inputType.getRank(); ++dim) {
      Value outputIndex = builder.create<linalg::IndexOp>(loc, dim);
      Value coordinate =
          getCoordinate(params, builder, loc, dim, outputIndex, calcType);
      indices.push_back(
          getNearestIndex(params, builder, loc, dim, coordinate, calcType));
    }
    return builder.create<tensor::ExtractOp>(loc, params.input, indices);
  }

  static Value buildLinear(const ResizeParams &params, OpBuilder &builder,
      Location loc, Type calcType, Type resultElementType) {
    int64_t rank = params.inputType.getRank();
    SmallVector<Value> baseIndices(rank);
    SmallVector<Value> lowIndices(rank);
    SmallVector<Value> highIndices(rank);
    SmallVector<Value> lowWeights(rank);
    SmallVector<Value> highWeights(rank);
    SmallVector<int64_t> interpolationDims;
    Value oneFloat = createFloatConstant(builder, loc, calcType, 1.0);
    Value oneIndex = createIndexConstant(builder, loc, 1);

    for (int64_t dim = 0; dim < rank; ++dim) {
      Value outputIndex = builder.create<linalg::IndexOp>(loc, dim);
      baseIndices[dim] = outputIndex;
      if (params.scales[dim] == 1.0f &&
          params.inputType.getDimSize(dim) == params.resultType.getDimSize(dim))
        continue;

      Value coordinate =
          getCoordinate(params, builder, loc, dim, outputIndex, calcType);
      Value floorCoordinate = builder.create<math::FloorOp>(loc, coordinate);
      Value lowIndex = castFloatToIndex(builder, loc, floorCoordinate);
      Value highIndex = builder.create<arith::AddIOp>(loc, lowIndex, oneIndex);
      Value highWeight =
          builder.create<arith::SubFOp>(loc, coordinate, floorCoordinate);
      Value lowWeight =
          builder.create<arith::SubFOp>(loc, oneFloat, highWeight);
      lowIndices[dim] = clampIndex(
          builder, loc, lowIndex, params.inputType.getDimSize(dim) - 1);
      highIndices[dim] = clampIndex(
          builder, loc, highIndex, params.inputType.getDimSize(dim) - 1);
      lowWeights[dim] = lowWeight;
      highWeights[dim] = highWeight;
      interpolationDims.push_back(dim);
    }

    Value accumulator = createFloatConstant(builder, loc, calcType, 0.0);
    int64_t cornerCount = 1LL << interpolationDims.size();
    for (int64_t mask = 0; mask < cornerCount; ++mask) {
      SmallVector<Value> extractIndices(baseIndices);
      Value weight = createFloatConstant(builder, loc, calcType, 1.0);
      for (auto [index, dim] : llvm::enumerate(interpolationDims)) {
        bool useHigh = mask & (1LL << index);
        extractIndices[dim] = useHigh ? highIndices[dim] : lowIndices[dim];
        weight = builder.create<arith::MulFOp>(
            loc, weight, useHigh ? highWeights[dim] : lowWeights[dim]);
      }
      Value sample =
          builder.create<tensor::ExtractOp>(loc, params.input, extractIndices);
      sample = convertFloatValue(builder, loc, sample, calcType);
      Value weighted = builder.create<arith::MulFOp>(loc, sample, weight);
      accumulator = builder.create<arith::AddFOp>(loc, accumulator, weighted);
    }

    return convertFloatValue(builder, loc, accumulator, resultElementType);
  }

  LogicalResult matchAndRewrite(
      ONNXResizeOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !inputType.hasStaticShape() || !resultType ||
        !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "requires static ranked input and result tensors");

    int64_t rank = inputType.getRank();
    if ((rank != 4 && rank != 5) || resultType.getRank() != rank)
      return rewriter.notifyMatchFailure(
          op, "requires rank-4 or rank-5 input and result tensors");
    if (llvm::any_of(
            inputType.getShape(), [](int64_t dim) { return dim <= 0; }) ||
        llvm::any_of(
            resultType.getShape(), [](int64_t dim) { return dim <= 0; }))
      return rewriter.notifyMatchFailure(
          op, "requires positive input and result dimensions");

    if (op.getAntialias() != 0 || op.getExcludeOutside() != 0 ||
        op.getKeepAspectRatioPolicy() != "stretch" || op.getAxes())
      return rewriter.notifyMatchFailure(op,
          "unsupported antialias, exclude_outside, "
          "keep_aspect_ratio_policy, or axes attribute");

    StringRef mode = op.getMode();
    if (mode != "nearest" && mode != "linear")
      return rewriter.notifyMatchFailure(op, "unsupported resize mode");

    StringRef coordinateTransformationMode =
        op.getCoordinateTransformationMode();
    if (coordinateTransformationMode != "half_pixel" &&
        coordinateTransformationMode != "pytorch_half_pixel" &&
        coordinateTransformationMode != "asymmetric" &&
        coordinateTransformationMode != "align_corners")
      return rewriter.notifyMatchFailure(
          op, "unsupported coordinate transformation mode");

    StringRef nearestMode = op.getNearestMode();
    if (nearestMode != "floor" && nearestMode != "round_prefer_ceil" &&
        nearestMode != "round_prefer_floor")
      return rewriter.notifyMatchFailure(op, "unsupported nearest mode");

    Type elementType = inputType.getElementType();
    if (mode == "linear" && !isa<FloatType>(elementType))
      return rewriter.notifyMatchFailure(
          op, "linear mode requires a floating-point element type");
    if (mode == "nearest" && !isa<IntegerType, FloatType>(elementType))
      return rewriter.notifyMatchFailure(op,
          "nearest mode requires an integer or floating-point element type");

    bool hasScales = !isa<NoneType>(op.getScales().getType());
    bool hasSizes = !isa<NoneType>(op.getSizes().getType());
    if (hasScales == hasSizes)
      return rewriter.notifyMatchFailure(
          op, "requires exactly one of scales or sizes");

    SmallVector<float> scales;
    scales.reserve(rank);
    if (hasScales) {
      auto scalesAttr = dyn_cast_or_null<DenseElementsAttr>(
          getElementAttributeFromONNXValue(op.getScales()));
      if (!scalesAttr || !isa<FloatType>(scalesAttr.getElementType()) ||
          scalesAttr.getNumElements() != rank)
        return rewriter.notifyMatchFailure(
            op, "requires constant floating-point scales matching the rank");
      for (const APFloat &scale : scalesAttr.getValues<APFloat>()) {
        float value = scale.convertToFloat();
        if (value <= 0.0f)
          return rewriter.notifyMatchFailure(
              op, "requires positive scale values");
        scales.push_back(value);
      }
    } else {
      auto sizesAttr = dyn_cast_or_null<DenseElementsAttr>(
          getElementAttributeFromONNXValue(op.getSizes()));
      if (!sizesAttr || !isa<IntegerType>(sizesAttr.getElementType()) ||
          sizesAttr.getNumElements() != rank)
        return rewriter.notifyMatchFailure(
            op, "requires constant integer sizes matching the rank");

      for (auto [inputDim, resultDim, size] :
          llvm::zip_equal(inputType.getShape(), resultType.getShape(),
              sizesAttr.getValues<APInt>())) {
        int64_t sizeValue = size.getSExtValue();
        if (sizeValue != resultDim)
          return rewriter.notifyMatchFailure(
              op, "constant sizes must match the result shape");
        scales.push_back(
            static_cast<float>(sizeValue) / static_cast<float>(inputDim));
      }
    }

    Type calcType = elementType;
    if (auto floatType = dyn_cast<FloatType>(elementType)) {
      if (floatType.getWidth() < 32)
        calcType = rewriter.getF32Type();
    } else {
      calcType = rewriter.getF32Type();
    }

    ResizeParams params{op.getX(), inputType, resultType, scales,
        coordinateTransformationMode, nearestMode};
    Value output = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
        resultType.getElementType(), ValueRange{}, resultType.getEncoding());
    AffineMap outputMap = rewriter.getMultiDimIdentityMap(resultType.getRank());
    SmallVector<utils::IteratorType> iteratorTypes(
        resultType.getRank(), utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc,
        TypeRange{resultType}, ValueRange{}, output,
        SmallVector<AffineMap>{outputMap}, iteratorTypes,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange) {
          Value result = mode == "nearest"
                             ? buildNearest(params, builder, bodyLoc, calcType)
                             : buildLinear(params, builder, bodyLoc, calcType,
                                   elementType);
          builder.create<linalg::YieldOp>(bodyLoc, result);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct ConvertONNXResizeToLinalgPass
    : public PassWrapper<ConvertONNXResizeToLinalgPass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertONNXResizeToLinalgPass)

  [[nodiscard]] StringRef getArgument() const override {
    return "convert-onnx-resize-to-linalg";
  }

  [[nodiscard]] StringRef getDescription() const override {
    return "Lower supported ONNX Resize operations to Linalg";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
        math::MathDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ONNXResizeOpLoweringToLinalg>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertONNXResizeToLinalgPass() {
  return std::make_unique<ConvertONNXResizeToLinalgPass>();
}

} // namespace onnx_mlir
