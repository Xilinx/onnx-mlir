/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

#include <algorithm>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

namespace onnx_mlir {
#define GEN_PASS_DEF_CANONICALIZEELEMENTWISEOPERANDRANKSPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"
} // namespace onnx_mlir

using namespace mlir;
using namespace onnx_mlir;

namespace {

struct CanonicalizeElementwiseOperandRanksPattern : RewritePattern {
  explicit CanonicalizeElementwiseOperandRanksPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    const auto operands = getRankedElementwiseBroadcastOperands(op);
    if (operands.empty())
      return failure();

    int64_t maxRank = 0;
    for (OpOperand *operand : operands)
      maxRank = std::max(
          maxRank, cast<RankedTensorType>(operand->get().getType()).getRank());

    SmallVector<std::pair<OpOperand *, RankedTensorType>> reshapes;
    for (OpOperand *operand : operands) {
      const auto type = cast<RankedTensorType>(operand->get().getType());
      const int64_t leadingDims = maxRank - type.getRank();
      if (leadingDims == 0)
        continue;
      const Type elementType = type.getElementType();
      if (type.getEncoding() ||
          (isa<quant::QuantizedType>(elementType) &&
              !isa<quant::UniformQuantizedType>(elementType)))
        return rewriter.notifyMatchFailure(op,
            "cannot prepend dimensions to the operand's layout or "
            "quantization");
      SmallVector<int64_t> shape(leadingDims, 1);
      llvm::append_range(shape, type.getShape());
      reshapes.emplace_back(operand, RankedTensorType::get(shape, elementType));
    }
    if (reshapes.empty())
      return failure();

    OnnxBuilder builder(rewriter, op->getLoc());
    SmallVector<Value> newOperands(op->getOperands());
    for (const auto &[operand, resultType] : reshapes) {
      const Value input = operand->get();
      const ArrayRef<int64_t> shape = resultType.getShape();
      bool allowZero = llvm::is_contained(shape, 0);
      Value shapeValue;
      if (resultType.hasStaticShape()) {
        shapeValue = builder.constantInt64(shape);
      } else {
        const int64_t leadingDims = maxRank - getRank(input.getType());
        const Value prefix =
            builder.constantInt64(shape.take_front(leadingDims));
        const Value inputShape = builder.shape(input);
        shapeValue = builder.concat(ValueRange{prefix, inputShape}, 0);
        allowZero = true;
      }
      newOperands[operand->getOperandNumber()] = ONNXReshapeOp::create(rewriter,
          op->getLoc(), resultType, input, shapeValue,
          rewriter.getIntegerAttr(
              rewriter.getIntegerType(64, /*isSigned=*/true), allowZero));
    }
    rewriter.modifyOpInPlace(op, [&] { op->setOperands(newOperands); });
    return success();
  }
};

struct CanonicalizeElementwiseOperandRanksPass
    : onnx_mlir::impl::CanonicalizeElementwiseOperandRanksPassBase<
          CanonicalizeElementwiseOperandRanksPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateElementwiseOperandRankCanonicalizationPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void onnx_mlir::populateElementwiseOperandRankCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CanonicalizeElementwiseOperandRanksPattern>(
      patterns.getContext());
}
