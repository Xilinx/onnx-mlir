// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#include "XFEFold.hpp"
#include "XFECastUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/StringSwitch.h"

#include <cstdint>
#include <optional>
#include <type_traits>

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

template <typename T>
T apFloatToCpp(APFloat value) {
  if constexpr (std::is_same_v<T, double>)
    return value.convertToDouble();
  else if constexpr (std::is_same_v<T, float>)
    return value.convertToFloat();
  else
    return T::fromAPFloat(value);
}

template <typename InputType, typename Convert>
ElementsAttr convertElementsToFloat(ElementsAttr input, ShapedType outputType,
    onnx_mlir::OnnxElementsAttrBuilder &builder, Convert convert) {
  return onnx_mlir::dispatchByBType(
      onnx_mlir::btypeOfMlirType(outputType.getElementType()),
      [&](auto outputBType) -> ElementsAttr {
        if constexpr (onnx_mlir::BTypeTrait<outputBType>::isFloat) {
          using OutputType = onnx_mlir::CppType<outputBType>;
          return builder.fromArray<OutputType>(
              outputType, [&](MutableArrayRef<OutputType> output) {
                size_t i = 0;
                for (InputType value : input.getValues<InputType>())
                  output[i++] = apFloatToCpp<OutputType>(convert(value));
              });
        } else {
          llvm_unreachable("expected a floating-point output type");
        }
      });
}

ElementsAttr convertIntegerElements(ElementsAttr input, ShapedType outputShape,
    IntegerType inputType, IntegerType outputElementType,
    onnx_mlir::OnnxElementsAttrBuilder &builder) {
  bool inputIsSigned =
      !inputType.isUnsigned() && !inputType.isSignlessInteger(1);
  unsigned outputWidth = outputElementType.getWidth();
  return onnx_mlir::dispatchByBType(
      onnx_mlir::btypeOfMlirType(outputElementType),
      [&](auto outputBType) -> ElementsAttr {
        if constexpr (onnx_mlir::BTypeTrait<outputBType>::isInt) {
          using OutputType = onnx_mlir::CppType<outputBType>;
          return builder.fromArray<
              OutputType>(outputShape, [&](MutableArrayRef<OutputType> output) {
            size_t i = 0;
            for (APInt value : input.getValues<APInt>()) {
              if constexpr (std::is_same_v<OutputType, bool>) {
                output[i++] = !value.isZero();
              } else {
                APInt converted = inputIsSigned
                                      ? value.sextOrTrunc(outputWidth)
                                      : value.zextOrTrunc(outputWidth);
                if constexpr (onnx_mlir::BTypeTrait<outputBType>::isSignedInt)
                  output[i++] = OutputType(converted.getSExtValue());
                else
                  output[i++] = OutputType(converted.getZExtValue());
              }
            }
          });
        } else {
          llvm_unreachable("expected an integer output type");
        }
      });
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
  onnx_mlir::OnnxElementsAttrBuilder elementsBuilder(op->getContext());

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
          result = APFloat::getLargest(
              outputFloatType.getFloatSemantics(), result.isNegative());
        return result;
      };
      return convertElementsToFloat<APInt>(
          input, outputAttrType, elementsBuilder, convert);
    }

    bool useOptimizedF32ToBF16 =
        inputFloatType.isF32() && outputFloatType.isBF16() &&
        (roundMode == "toward_zero" || roundMode == "nearest_even");
    auto convert = [=](APFloat inputValue) {
      if (useOptimizedF32ToBF16)
        return convertF32ToBf16(inputValue, roundMode, saturationMode, nanMode);
      return convertFloat(
          inputValue, outputFloatType, *apRoundMode, saturationMode);
    };
    return convertElementsToFloat<APFloat>(
        input, outputAttrType, elementsBuilder, convert);
  }

  if (auto outputIntegerType = dyn_cast<IntegerType>(outputElementType)) {
    if (roundMode != "toward_zero" && roundMode != "nearest_even")
      return {};
    if (auto inputIntegerType = dyn_cast<IntegerType>(inputElementType))
      return convertIntegerElements(input, outputAttrType, inputIntegerType,
          outputIntegerType, elementsBuilder);
    return elementsBuilder.castToIntElementType(
        input, outputIntegerType, roundMode == "nearest_even");
  }

  return {};
}
