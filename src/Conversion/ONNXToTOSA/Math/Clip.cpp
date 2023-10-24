/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- Clip.cpp - ClipOp --------------===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNXClipOp operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <limits>
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXClipOpLoweringToTOSA : public ConversionPattern {
public:
  ONNXClipOpLoweringToTOSA(MLIRContext *ctx)
      : ConversionPattern(ONNXClipOp::getOperationName(), 1, ctx) {}
  using OpAdaptor = typename ONNXClipOp::Adaptor;
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    OpAdaptor adaptor(operands, op->getAttrDictionary());
    Value input = adaptor.getInput();
    Value max = adaptor.getMax();
    Value min = adaptor.getMin();
    auto maxInt = rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
    auto minInt = rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::lowest());
    auto maxFp = rewriter.getF32FloatAttr(std::numeric_limits<float>::max());
    auto minFp = rewriter.getF32FloatAttr(std::numeric_limits<float>::lowest());

    if (!isNoneValue(max) && max.getDefiningOp<mlir::tosa::ConstOp>()) {
        auto maxType = cast<TensorType>(max.getType()).getElementType();
        if (maxType.isF32()) {
            maxFp = tosa::getValueFromTosaConst<FloatAttr>(max);
        }
        else {
            maxInt = tosa::getValueFromTosaConst<IntegerAttr>(max);
        }
    }
    if (!isNoneValue(min) && min.getDefiningOp<mlir::tosa::ConstOp>()) {
        auto minType = cast<TensorType>(min.getType()).getElementType();
        if (minType.isF32()) {
            minFp = tosa::getValueFromTosaConst<FloatAttr>(min);
        }
        else {
            minInt = tosa::getValueFromTosaConst<IntegerAttr>(min);
        }
    }
    
    auto outputType = getTypeConverter()->convertType(op->getResult(0).getType());

    rewriter.replaceOpWithNewOp<mlir::tosa::ClampOp>(op, outputType, input, minInt, maxInt, minFp, maxFp);
    return success();
  }
};

} // namespace

void populateLoweringONNXClipOpToTOSAPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXClipOpLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
