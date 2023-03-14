/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ReduceMean.cpp - ReduceMean Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX reduce mean operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include <numeric>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXReduceMeanLoweringToTOSA
    : public OpConversionPattern<ONNXReduceMeanV13Op> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXReduceMeanV13Op::Adaptor;
  LogicalResult matchAndRewrite(ONNXReduceMeanV13Op op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);
    Value input = adaptor.getData();
    auto axes = adaptor.getAxes();
    auto keepDims = adaptor.getKeepdims();

    auto resultType = getTypeConverter()
                          ->convertType(op.getResult().getType())
                          .cast<RankedTensorType>();

    // axes is mandatoy for tosa
    if (!axes) {
      // if not present all axes are reduced
      const int64_t numberOfAxes = input.getType().cast<ShapedType>().getRank();
      std::vector<int64_t> allDims(numberOfAxes);
      std::iota(std::begin(allDims), std::end(allDims), 0);
      axes = rewriter.getI64ArrayAttr(allDims);
    }
    // Tosa needs a DenseElementsAttr
    auto vecValues = extractFromI64ArrayAttr(axes.value());
    const int64_t vecValuesSize = vecValues.size();
    DenseElementsAttr newAxesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({vecValuesSize}, rewriter.getI64Type()),
        vecValues);

    auto output = tosa::convertReduceMeanOp(rewriter, op, tosaBuilder,
        resultType, input, newAxesAttr, (bool)keepDims);

    if (!output) {
      return rewriter.notifyMatchFailure(op, "could not be converted");
    }
    // Shape inference is handled by the helper functions
    rewriter.replaceOp(op, {output.value()});
    return success();
  }
};

} // namespace

void populateLoweringONNXReduceMeanOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXReduceMeanLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
