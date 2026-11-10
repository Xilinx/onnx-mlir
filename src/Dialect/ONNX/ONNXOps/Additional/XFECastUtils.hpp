// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#pragma once

#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace onnx_mlir::xfe_cast {

/// Convert f32 storage bits to bf16 using the XFE Cast attributes.
[[nodiscard]] inline uint16_t f32BitsToBf16(
    uint32_t bits, llvm::StringRef roundMode, bool saturate) {
  uint16_t converted;
  if (roundMode == "toward_zero") {
    converted = static_cast<uint16_t>(bits >> 16);
  } else {
    assert(roundMode == "nearest_even" && "unexpected XFE Cast round mode");
    converted =
        static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
  }
  return saturate ? std::min(converted, uint16_t{0xff7f}) : converted;
}

} // namespace onnx_mlir::xfe_cast
