/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- ONNXTraits.hpp - ONNX Op Traits --------------------===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file defines traits of ONNX ops.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {

template <int version>
class OpVersionTrait {
public:
  template <typename ConcreteType>
  class Impl : public OpTrait::TraitBase<ConcreteType, Impl> {
  public:
    int getOpVersion() { return version; }
  };
};

template <typename ConcreteType>
class ONNXElementwise
    : public OpTrait::TraitBase<ConcreteType, ONNXElementwise> {};

} // namespace OpTrait
} // namespace mlir
