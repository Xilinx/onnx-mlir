// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct BatchReductionToReshapeReductionPattern
    : public OpRewritePattern<ONNXReduceSumOp> {
  using OpRewritePattern<ONNXReduceSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReduceSumOp reduceOp, PatternRewriter &rewriter) const override {
    Location loc = reduceOp.getLoc();

    // Match only quantized ReduceSum.
    Value inputValue = reduceOp.getData();
    auto inputType = dyn_cast<RankedTensorType>(inputValue.getType());
    if (!inputType)
      return failure();

    auto inputQType =
        dyn_cast<quant::UniformQuantizedType>(inputType.getElementType());
    if (!inputQType)
      return failure();

    if (inputType.getRank() != 4)
      return failure();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    if (inputShape[0] == 1)
      return failure();

    Value axesValue = reduceOp.getAxes();
    if (!axesValue)
      return failure();
    ElementsAttr axesAttr =
        onnx_mlir::getDenseOrDisposableConstLikeElements(axesValue);
    if (!axesAttr || !axesAttr.getElementType().isIntOrIndex())
      return failure();
    SmallVector<int64_t, 4> axesVec;
    for (APInt v : axesAttr.getValues<APInt>())
      axesVec.push_back(v.getSExtValue());
    if (axesVec.size() != 1)
      return failure();
    constexpr int64_t kInputRank = 4;
    int64_t canonicalInputAxis = axesVec[0];
    if (canonicalInputAxis < 0)
      canonicalInputAxis += kInputRank;
    if (canonicalInputAxis != 3)
      return failure();

    auto outputType =
        dyn_cast<RankedTensorType>(reduceOp.getResult().getType());
    if (!outputType)
      return failure();
    auto outputQType =
        dyn_cast<quant::UniformQuantizedType>(outputType.getElementType());
    if (!outputQType)
      return failure();

    // Shapes:
    //   preReshape  [1, dim0*dim1, dim2, dim3]
    //   reduceShape [1, dim0*dim1, dim2]      when keepdims = 0
    //               [1, dim0*dim1, dim2, 1]   when keepdims = 1
    //   postReshape original ReduceSum output shape
    int64_t dim0 = inputShape[0];
    int64_t dim1 = inputShape[1];
    int64_t dim2 = inputShape[2];
    int64_t dim3 = inputShape[3];
    int64_t flattenedDims = dim0 * dim1;

    // Inherit keepdims so the new ReduceSum's result rank stays consistent.
    int64_t origKeepdims = reduceOp.getKeepdims();

    SmallVector<int64_t, 4> preReshapeShape = {1, flattenedDims, dim2, dim3};

    // Normalize negative axis against the new (post-flatten) rank.
    int64_t reduceAxis = axesVec[0];
    if (reduceAxis < 0)
      reduceAxis += static_cast<int64_t>(preReshapeShape.size());

    // keepdims=1: keep reduced axis as size-1; keepdims=0: drop it.
    SmallVector<int64_t, 4> reduceShape(
        preReshapeShape.begin(), preReshapeShape.end());
    if (origKeepdims != 0)
      reduceShape[reduceAxis] = 1;
    else
      reduceShape.erase(reduceShape.begin() + reduceAxis);
    SmallVector<int64_t, 4> postReshapeShape(
        outputType.getShape().begin(), outputType.getShape().end());

    rewriter.setInsertionPoint(reduceOp);
    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
    Value preShapeConst = onnxBuilder.constantInt64(preReshapeShape);

    auto preReshapeType =
        RankedTensorType::get(preReshapeShape, inputType.getElementType());
    Value preReshape = rewriter
                           .create<ONNXReshapeOp>(
                               loc, preReshapeType, inputValue, preShapeConst)
                           .getResult();

    // Clone ReduceSum op generically to preserve attributes.
    Operation *oldReduce = reduceOp.getOperation();
    OperationState reduceState(loc, oldReduce->getName().getStringRef());
    reduceState.addTypes(
        RankedTensorType::get(reduceShape, outputType.getElementType()));

    SmallVector<Value, 4> newOperands;
    for (Value v : oldReduce->getOperands())
      newOperands.push_back(v);
    if (!newOperands.empty())
      newOperands[0] = preReshape;
    reduceState.addOperands(newOperands);
    for (NamedAttribute attr : oldReduce->getAttrs())
      reduceState.addAttribute(attr.getName(), attr.getValue());

    Operation *newReduce = rewriter.create(reduceState);

    rewriter.setInsertionPointAfter(newReduce);
    Value postShapeConst = onnxBuilder.constantInt64(postReshapeShape);
    Value postReshape = rewriter
                            .create<ONNXReshapeOp>(loc, outputType,
                                newReduce->getResult(0), postShapeConst)
                            .getResult();

    rewriter.replaceOp(reduceOp, postReshape);
    return success();
  }
};

static bool hasStaticPositiveShape(RankedTensorType type) {
  return llvm::all_of(type.getShape(),
      [](int64_t d) { return !ShapedType::isDynamic(d) && d > 0; });
}

// Rewrites a quantized ReduceSum over axis 1 of a [B, R, 1, C] tensor with
// B > 1 so that the backend sees a batch of 1:
//   [B, R, 1, C] -> Reshape [1, B, R, C]
//                -> ReduceSum(axes = [2], keepdims = 1) [1, B, 1, C]
//                -> Reshape original output shape
// Dimension 2 being 1 makes [B, R, 1, C] and [1, B, R, C] share the same
// row-major layout, and reducing R per (b, c) is unchanged.
struct BatchMiddleAxisReductionToReshapeReductionPattern
    : public OpRewritePattern<ONNXReduceSumOp> {
  using OpRewritePattern<ONNXReduceSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReduceSumOp reduceOp, PatternRewriter &rewriter) const override {
    constexpr int64_t kInputRank = 4;
    Location loc = reduceOp.getLoc();

    Value inputValue = reduceOp.getData();
    auto inputType = dyn_cast<RankedTensorType>(inputValue.getType());
    if (!inputType || inputType.getRank() != kInputRank ||
        !hasStaticPositiveShape(inputType))
      return failure();
    if (!isa<quant::UniformQuantizedType>(inputType.getElementType()))
      return failure();

    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t batch = inputShape[0];
    int64_t reduceDim = inputShape[1];
    int64_t channels = inputShape[3];
    if (batch <= 1 || inputShape[2] != 1)
      return failure();

    Value axesValue = reduceOp.getAxes();
    if (!axesValue)
      return failure();
    ElementsAttr axesAttr =
        onnx_mlir::getDenseOrDisposableConstLikeElements(axesValue);
    if (!axesAttr || !axesAttr.getElementType().isIntOrIndex() ||
        axesAttr.size() != 1)
      return failure();
    int64_t axis = (*axesAttr.getValues<APInt>().begin()).getSExtValue();
    if (axis < 0)
      axis += kInputRank;
    if (axis != 1)
      return failure();

    auto outputType =
        dyn_cast<RankedTensorType>(reduceOp.getResult().getType());
    if (!outputType || !hasStaticPositiveShape(outputType))
      return failure();
    if (!isa<quant::UniformQuantizedType>(outputType.getElementType()))
      return failure();

    // The original result must be [B, 1, 1, C] (keepdims = 1) or [B, 1, C]
    // (keepdims = 0); it is restored by the final reshape.
    SmallVector<int64_t, 4> expectedOutputShape =
        reduceOp.getKeepdims() != 0
            ? SmallVector<int64_t, 4>{batch, 1, 1, channels}
            : SmallVector<int64_t, 4>{batch, 1, channels};
    if (outputType.getShape() != ArrayRef<int64_t>(expectedOutputShape))
      return failure();

    SmallVector<int64_t, 4> preReshapeShape = {1, batch, reduceDim, channels};
    SmallVector<int64_t, 4> reduceShape = {1, batch, 1, channels};

    rewriter.setInsertionPoint(reduceOp);
    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
    Value preShapeConst = onnxBuilder.constantInt64(preReshapeShape);
    auto preReshapeType =
        RankedTensorType::get(preReshapeShape, inputType.getElementType());
    Value preReshape = rewriter.create<ONNXReshapeOp>(
        loc, preReshapeType, inputValue, preShapeConst);

    // The reduced axis moves from 1 to 2, so the original axes operand cannot
    // be reused. ResultNames is intentionally not copied: the updater infers
    // the inner names (with the inverse reshape) from the final reshape.
    Value newAxes = onnxBuilder.constantInt64({2});
    auto reduceType =
        RankedTensorType::get(reduceShape, outputType.getElementType());
    Value newReduce = rewriter.create<ONNXReduceSumOp>(loc, reduceType,
        preReshape, newAxes, /*keepdims=*/1, reduceOp.getNoopWithEmptyAxes());

    Value postShapeConst = onnxBuilder.constantInt64(outputType.getShape());
    Value postReshape = rewriter.create<ONNXReshapeOp>(
        loc, outputType, newReduce, postShapeConst);

    rewriter.replaceOp(reduceOp, postReshape);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct BatchReductionToReshapeReductionPass
    : public PassWrapper<BatchReductionToReshapeReductionPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "batch-reduction-to-reshape-reduction";
  }
  StringRef getDescription() const override {
    return "Convert batch ReduceSum on quantized tensors into "
           "reshape-optimized ReduceSum";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<BatchReductionToReshapeReductionPattern,
        BatchMiddleAxisReductionToReshapeReductionPattern>(context);

    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    ResultNamesUpdater rnUpdater;
    config.setListener(&rnUpdater);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createBatchReductionToReshapeReductionPass() {
  return std::make_unique<BatchReductionToReshapeReductionPass>();
}

} // namespace onnx_mlir
