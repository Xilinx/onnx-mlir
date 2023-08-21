/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DialectBuilder.hpp.inc - Builder for ONNX dialects -------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Region.h"

template <typename OnnxOpType, typename... Args>
OnnxOpType OnnxBuilder::createOpAndInferShapes(Args &&... args) {
  OnnxOpType op = create<OnnxOpType>(args...);
  if (!op)
    llvm_unreachable("unexpected failure to create op");
  if (failed(op.inferShapes([](mlir::Region &region) {})))
    llvm_unreachable("unexpected inferShapes failure");
  return op;
}

// There is technically no need for this function call as the one above could
// also be used. However, this call ensure that a mlir::Type parameter is
// given, so it provide a bit more typechecking than the more generic call
// above.
template <typename OnnxOpType, typename... Args>
OnnxOpType OnnxBuilder::createTypedOpAndInferShapes(
    mlir::Type result_ty, Args &&... args) {
  return createOpAndInferShapes<OnnxOpType>(result_ty, args...);
}
