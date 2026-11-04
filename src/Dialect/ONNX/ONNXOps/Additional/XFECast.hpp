// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#pragma once

#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdint>

namespace onnx_mlir::xfe_cast {

inline constexpr llvm::StringLiteral legacyBf16TruncatePolicy =
    "legacy_bf16_truncate";
inline constexpr llvm::StringLiteral runtimeBf16RnePolicy = "runtime_bf16_rne";

/// Convert f32 bits to bf16 using the compiler's legacy truncation policy.
[[nodiscard]] inline uint16_t f32BitsToBf16Truncate(uint32_t bits) {
  return static_cast<uint16_t>(bits >> 16);
}

/// Convert f32 bits to bf16 using FlexML runtime-compatible RNE semantics.
///
/// This intentionally preserves the runtime's saturation and NaN behavior.
[[nodiscard]] inline uint16_t f32BitsToBf16RuntimeRne(uint32_t bits) {
  return std::min(
      static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16),
      uint16_t{0xff7f});
}

} // namespace onnx_mlir::xfe_cast
