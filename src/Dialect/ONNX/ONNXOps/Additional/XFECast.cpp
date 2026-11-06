// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#include "src/Dialect/ONNX/ONNXOps/Additional/XFECast.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <cstdint>
#include <iterator>

using namespace mlir;

namespace {

uint16_t convertF32ToBf16(uint32_t bits, StringRef policy) {
  if (policy == onnx_mlir::xfe_cast::legacyBf16TruncatePolicy)
    return onnx_mlir::xfe_cast::f32BitsToBf16Truncate(bits);

  assert(policy == onnx_mlir::xfe_cast::runtimeBf16RnePolicy &&
         "unexpected XFECast policy");
  return onnx_mlir::xfe_cast::f32BitsToBf16RuntimeRne(bits);
}

APFloat convertF32ToBf16(APFloat value, StringRef policy) {
  uint32_t bits = value.bitcastToAPInt().getZExtValue();
  return APFloat(APFloat::BFloat(), APInt(16, convertF32ToBf16(bits, policy)));
}

} // namespace

LogicalResult XFECastOp::verify() {
  StringRef policy = getPolicy();
  if (policy != onnx_mlir::xfe_cast::legacyBf16TruncatePolicy &&
      policy != onnx_mlir::xfe_cast::runtimeBf16RnePolicy)
    return emitOpError("has unsupported policy '") << policy << "'";
  return success();
}

OpFoldResult XFECastOp::fold(FoldAdaptor adaptor) {
  auto input = dyn_cast_or_null<ElementsAttr>(adaptor.getInput());
  if (!input)
    return {};

  auto outputType = cast<ShapedType>(getOutput().getType());
  StringRef policy = getPolicy();
  auto convert = [policy](
                     APFloat value) { return convertF32ToBf16(value, policy); };
  if (input.isSplat())
    return DenseElementsAttr::get(
        outputType, convert(input.getSplatValue<APFloat>()));

  SmallVector<APFloat> output;
  output.reserve(input.getNumElements());
  llvm::transform(
      input.getValues<APFloat>(), std::back_inserter(output), convert);
  return DenseElementsAttr::get(outputType, output);
}
