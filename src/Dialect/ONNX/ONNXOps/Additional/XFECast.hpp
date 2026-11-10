/*
 * SPDX-License-Identifier: Apache-2.0
 * Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
 * affiliates
 */

#ifndef ONNX_MLIR_XFE_CAST_H
#define ONNX_MLIR_XFE_CAST_H

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/IR/Builders.h"

namespace onnx_mlir {

struct ONNXCastToXFEOptions {
  // ONNX does not fully specify the rounding of an in-range floating-point
  // value to an integer. Select the implementation policy explicitly when
  // translating the operation.
  bool roundFPToIntToNearestEven = false;
};

// Build the explicitly parameterized XFE equivalent of an ONNX Cast. This is
// intentionally independent of any particular rewrite pass so later XFE
// legalization can use the same semantic mapping as constant propagation.
mlir::XFECastOp createXFECastFromONNXCast(mlir::OpBuilder &builder,
    mlir::ONNXCastOp castOp, mlir::Value input,
    ONNXCastToXFEOptions options = {});

} // namespace onnx_mlir

#endif
