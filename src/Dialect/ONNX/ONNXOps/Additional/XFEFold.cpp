// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#include "XFEFold.hpp"
#include "XFECastUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <iterator>

using namespace mlir;

namespace {

APFloat convertF32ToBf16(APFloat value, StringRef roundMode, bool saturate) {
  uint32_t bits = value.bitcastToAPInt().getZExtValue();
  return APFloat(APFloat::BFloat(),
      APInt(16, onnx_mlir::xfe_cast::f32BitsToBf16(bits, roundMode, saturate)));
}

} // namespace

OpFoldResult mlir::XFECastOpFold(Operation *op, ArrayRef<Attribute> operands) {
  auto castOp = cast<XFECastOp>(op);
  auto input = dyn_cast_or_null<ElementsAttr>(operands.front());
  if (!input)
    return {};

  auto outputType = cast<ShapedType>(castOp.getOutput().getType());
  StringRef roundMode = castOp.getRoundMode();
  bool saturate = castOp.getSaturate() != 0;
  auto convert = [roundMode, saturate](APFloat value) {
    return convertF32ToBf16(value, roundMode, saturate);
  };
  if (input.isSplat())
    return DenseElementsAttr::get(
        outputType, convert(input.getSplatValue<APFloat>()));

  SmallVector<APFloat> output;
  output.reserve(input.getNumElements());
  llvm::transform(
      input.getValues<APFloat>(), std::back_inserter(output), convert);
  return DenseElementsAttr::get(outputType, output);
}
