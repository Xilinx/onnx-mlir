/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- Int4.hpp -----------------------------===//
//
// 4 bits integer type
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_INT_4_H
#define ONNX_MLIR_INT_4_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>
#define ONNX_MLIR_INT4_SSE2 1
#endif

namespace onnx_mlir {

namespace detail {

template <typename IT>
class Int4Base {
public:
  using StorageType = uint8_t;

  constexpr Int4Base() : ui() {}
  constexpr explicit Int4Base(const IT &it) : ui(it.ui) {}

  /// Extracts the 4-bit integer from a packed representation.
  /// The packed representation is expected to be packed in the LSB byte in the
  /// integer T. If `isFirst` is true, it extracts the LSB 4 bits; otherwise,
  /// it extracts the MSB 4 bits from this byte.
  template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>>>
  static constexpr IT extractFromPacked(T packed, bool isFirst) {
    IT result;
    if (isFirst) {
      result.ui = static_cast<StorageType>(packed & 0x0F);
    } else {
      result.ui = static_cast<StorageType>((packed >> 4) & 0x0F);
    }
    return result;
  }

  bool operator==(IT other) const { return ui == other.ui; }

  bool operator!=(IT other) const { return !(*this == other); }

protected:
  StorageType ui;
};

} // namespace detail

template <class T>
constexpr bool isAnyInt4Type = std::is_base_of_v<detail::Int4Base<T>, T>;

class int_4 : public detail::Int4Base<int_4> {
  using Base = detail::Int4Base<int_4>;

public:
  using Base::Int4Base;

  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, int_4>>>
  explicit int_4(T value) {
    auto expanded = static_cast<int64_t>(value);
    ui = expanded & 0x0F;
  }

  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, int_4>>>
  explicit operator T() const {
    if (ui & 0x08) {
      // If the sign bit is set, extend the sign.
      return static_cast<T>(static_cast<int8_t>(ui | 0xF0));
    }
    return static_cast<T>(ui);
  }
};
static_assert(sizeof(int_4) == 1, "int_4 must be 1 byte in size");

class uint_4 : public detail::Int4Base<uint_4> {
  using Base = detail::Int4Base<uint_4>;

public:
  using Base::Int4Base;

  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, uint_4>>>
  explicit uint_4(T value) {
    const auto expanded = static_cast<uint64_t>(value);
    ui = expanded & 0x0F;
  }

  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, uint_4>>>
  explicit operator T() const {
    return static_cast<T>(ui);
  }
};
static_assert(sizeof(uint_4) == 1, "int_4 must be 1 byte in size");

/// Unpacks `count` 4-bit values of a packed buffer, starting at element
/// `first`, to one byte each (int_4 and uint_4 hold just the nibble). The
/// buffer holds two elements per byte, the first one in the low nibble. This is
/// the hot loop when a large packed weight is read or sliced, so whole bytes
/// are handled 16 at a time where SSE2 is available.
inline void unpackInt4Nibbles(
    const char *packed, size_t first, size_t count, char *dst) {
  if (count == 0)
    return;
  if (first % 2 == 1) {
    // The first element is the high nibble of its byte.
    *dst++ = static_cast<char>((static_cast<uint8_t>(packed[first / 2]) >> 4));
    ++first;
    --count;
  }
  const uint8_t *src = reinterpret_cast<const uint8_t *>(packed) + first / 2;
  size_t pairs = count / 2;
#ifdef ONNX_MLIR_INT4_SSE2
  const __m128i mask = _mm_set1_epi8(0x0F);
  for (; pairs >= 16; pairs -= 16, src += 16, dst += 32) {
    __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
    __m128i low = _mm_and_si128(bytes, mask);
    __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    _mm_storeu_si128(
        reinterpret_cast<__m128i *>(dst), _mm_unpacklo_epi8(low, high));
    _mm_storeu_si128(
        reinterpret_cast<__m128i *>(dst + 16), _mm_unpackhi_epi8(low, high));
  }
#endif
  for (; pairs > 0; --pairs, ++src, dst += 2) {
    dst[0] = static_cast<char>(*src & 0x0F);
    dst[1] = static_cast<char>(*src >> 4);
  }
  if (count % 2)
    *dst = static_cast<char>(*src & 0x0F);
}

} // namespace onnx_mlir

// Enable DenseElementsAttr to operate on int_4 and uint_4 types and enables
// type dependent template programming
template <>
struct std::numeric_limits<onnx_mlir::int_4> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = true;
  static constexpr bool is_signed = true;
};

template <>
struct std::numeric_limits<onnx_mlir::uint_4> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = true;
  static constexpr bool is_signed = false;
};
#endif
