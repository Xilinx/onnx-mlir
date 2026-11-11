// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#include "XFEFold.hpp"
#include "XFECastUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <cstdint>
#include <iterator>
#include <optional>

using namespace mlir;

namespace {

std::optional<APFloat::roundingMode> getAPFloatRoundingMode(StringRef mode) {
  return StringSwitch<std::optional<APFloat::roundingMode>>(mode)
      .Case("nearest_even", APFloat::rmNearestTiesToEven)
      .Case("nearest_away", APFloat::rmNearestTiesToAway)
      .Case("toward_zero", APFloat::rmTowardZero)
      .Case("toward_positive", APFloat::rmTowardPositive)
      .Case("toward_negative", APFloat::rmTowardNegative)
      .Default(std::nullopt);
}

bool shouldSaturate(StringRef saturationMode, bool negative) {
  return saturationMode == "finite" ||
         (saturationMode == "neg_finite" && negative);
}

APFloat convertF32ToBf16(APFloat value, StringRef roundMode,
    StringRef saturationMode, StringRef nanMode) {
  uint32_t bits = value.bitcastToAPInt().getZExtValue();
  return APFloat(
      APFloat::BFloat(), APInt(16, onnx_mlir::xfe_cast::f32BitsToBf16(bits,
                                       roundMode, saturationMode, nanMode)));
}

APFloat convertFloat(APFloat value, FloatType outputType,
    APFloat::roundingMode roundMode, StringRef saturationMode) {
  bool negative = value.isNegative();
  bool isNaN = value.isNaN();
  bool isInfinity = value.isInfinity();
  bool losesInfo;
  APFloat::opStatus status =
      value.convert(outputType.getFloatSemantics(), roundMode, &losesInfo);
  if (shouldSaturate(saturationMode, negative) && !isNaN &&
      ((status & APFloat::opOverflow) != 0 || isInfinity))
    return APFloat::getLargest(outputType.getFloatSemantics(), negative);
  return value;
}

} // namespace

OpFoldResult mlir::XFECastOpFold(Operation *op, ArrayRef<Attribute> operands) {
  auto castOp = cast<XFECastOp>(op);
  auto input = dyn_cast_or_null<ElementsAttr>(operands.front());
  if (!input)
    return {};

  auto outputType = cast<ShapedType>(castOp.getOutput().getType());
  auto inputType = cast<ShapedType>(castOp.getInput().getType());
  StringRef roundMode = castOp.getRoundMode();
  StringRef saturationMode = castOp.getSaturationMode();
  StringRef nanMode = castOp.getNanMode();
  Type inputElementType = inputType.getElementType();
  Type outputElementType = outputType.getElementType();
  ShapedType outputAttrType =
      cast<ShapedType>(input.getType()).clone(outputElementType);

  if (auto outputFloatType = dyn_cast<FloatType>(outputElementType)) {
    auto inputFloatType = dyn_cast<FloatType>(inputElementType);
    std::optional<APFloat::roundingMode> apRoundMode =
        getAPFloatRoundingMode(roundMode);
    if (!apRoundMode)
      return {};

    if (!inputFloatType) {
      auto inputIntegerType = dyn_cast<IntegerType>(inputElementType);
      if (!inputIntegerType)
        return {};
      // ONNX bool is represented by signless i1, whose set bit denotes true
      // rather than the signed integer -1.
      bool isSigned = !inputIntegerType.isUnsigned() &&
                      !inputIntegerType.isSignlessInteger(1);
      auto convert = [=](APInt value) {
        APFloat result(outputFloatType.getFloatSemantics());
        APFloat::opStatus status =
            result.convertFromAPInt(value, isSigned, *apRoundMode);
        if ((status & APFloat::opOverflow) != 0 &&
            shouldSaturate(saturationMode, result.isNegative()))
          return APFloat::getLargest(
              outputFloatType.getFloatSemantics(), result.isNegative());
        return result;
      };
      if (input.isSplat())
        return DenseElementsAttr::get(
            outputAttrType, convert(input.getSplatValue<APInt>()));
      SmallVector<APFloat> output;
      output.reserve(input.getNumElements());
      llvm::transform(
          input.getValues<APInt>(), std::back_inserter(output), convert);
      return DenseElementsAttr::get(outputAttrType, output);
    }

    bool useOptimizedF32ToBF16 =
        inputFloatType.isF32() && outputFloatType.isBF16() &&
        (roundMode == "toward_zero" || roundMode == "nearest_even");
    auto convert = [=](APFloat value) {
      if (useOptimizedF32ToBF16)
        return convertF32ToBf16(value, roundMode, saturationMode, nanMode);
      return convertFloat(value, outputFloatType, *apRoundMode, saturationMode);
    };
    if (input.isSplat())
      return DenseElementsAttr::get(
          outputAttrType, convert(input.getSplatValue<APFloat>()));

    SmallVector<APFloat> output;
    output.reserve(input.getNumElements());
    llvm::transform(
        input.getValues<APFloat>(), std::back_inserter(output), convert);
    return DenseElementsAttr::get(outputAttrType, output);
  }

  if (auto outputIntegerType = dyn_cast<IntegerType>(outputElementType)) {
    if (roundMode != "toward_zero" && roundMode != "nearest_even")
      return {};
    return onnx_mlir::OnnxElementsAttrBuilder(op->getContext())
        .castToIntElementType(
            input, outputIntegerType, roundMode == "nearest_even");
  }

  return {};
}
