// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#pragma once

#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <cstdint>

namespace onnx_mlir::xfe_cast {

/// Convert f32 storage bits to bf16 using the XFE Cast attributes.
[[nodiscard]] inline uint16_t f32BitsToBf16(uint32_t bits,
    llvm::StringRef roundMode, llvm::StringRef saturationMode,
    llvm::StringRef nanMode) {
  bool isNaN = (bits & 0x7f800000U) == 0x7f800000U && (bits & 0x007fffffU) != 0;
  if (isNaN && nanMode == "propagate")
    return static_cast<uint16_t>((bits >> 16) | 0x0040U);

  uint16_t converted;
  if (roundMode == "toward_zero") {
    converted = static_cast<uint16_t>(bits >> 16);
  } else {
    assert(roundMode == "nearest_even" && "unexpected XFE Cast round mode");
    converted =
        static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
  }
  if (saturationMode == "finite" && !isNaN && (converted & 0x7fffU) >= 0x7f80U)
    return static_cast<uint16_t>((converted & 0x8000U) | 0x7f7fU);
  if (saturationMode == "neg_finite" && !isNaN && converted > 0xff7fU)
    return 0xff7fU;
  return converted;
}

} // namespace onnx_mlir::xfe_cast
