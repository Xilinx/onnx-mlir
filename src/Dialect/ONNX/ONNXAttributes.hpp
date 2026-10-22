/*
 * SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
 */

//===--------------------- ONNXAttributes.hpp -----------------------------===//
//
// This file defines attributes in the ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_ATTRIBUTES_H
#define ONNX_MLIR_ONNX_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "src/Dialect/ONNX/CallerOwnedDataAttr.hpp"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/ONNX/ONNXAttributes.hpp.inc"
#endif
