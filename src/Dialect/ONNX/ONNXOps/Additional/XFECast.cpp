/*
 * SPDX-License-Identifier: Apache-2.0
 * Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
 * affiliates
 */

#include "XFECast.hpp"

#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace onnx_mlir {

XFECastOp createXFECastFromONNXCast(OpBuilder &builder, ONNXCastOp castOp,
    Value input, ONNXCastToXFEOptions options) {
  Type inputElementType = getElementTypeOrSelf(input.getType());
  Type outputElementType = castOp.getTo();

  StringRef roundMode = "nearest_even";
  if (isa<FloatType>(inputElementType) && isa<IntegerType>(outputElementType) &&
      !options.roundFPToIntToNearestEven)
    roundMode = "toward_zero";

  // ONNX saturation only affects float8 destinations. XFECast deliberately
  // extends it to every floating-point destination, so translating any other
  // ONNX Cast must disable it explicitly.
  StringRef saturationMode = "none";
  if (auto floatType = dyn_cast<FloatType>(outputElementType);
      floatType && floatType.getWidth() == 8)
    saturationMode = castOp.getSaturate() != 0 ? "finite" : "none";

  return builder.create<XFECastOp>(castOp.getLoc(), castOp.getType(), input,
      "propagate", roundMode, saturationMode, outputElementType);
}

} // namespace onnx_mlir
