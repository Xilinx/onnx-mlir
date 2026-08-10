// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

//==========================-- TestSubByteUtils.cpp --========================//
//
// Tests unpackSubByteValues.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/Transforms/SubByteUtils.hpp"

#include "mlir/IR/Builders.h"

#include <bitset>
#include <iostream>

using namespace mlir;
using namespace onnx_mlir;

namespace {

class Test {
  MLIRContext ctx;
  Builder builder;

  // Builds a uint8 attribute holding `bytes` as a 1-D tensor, which is how the
  // com.microsoft operators hand us their packed sub-byte data.
  DenseElementsAttr packedAttr(ArrayRef<uint8_t> bytes) {
    auto type = RankedTensorType::get(
        {static_cast<int64_t>(bytes.size())}, builder.getIntegerType(8, false));
    return DenseElementsAttr::get(type, bytes);
  }

  template <typename T>
  int expect(StringRef name, ArrayRef<T> actual, ArrayRef<T> expected) {
    if (actual == expected)
      return 0;
    // Printed as bit patterns to line up with how the cases are written.
    auto print = [](ArrayRef<T> values) {
      for (T v : values)
        std::cerr << " 0b" << std::bitset<8>(uint64_t(v));
    };
    std::cerr << name.str() << ": expected [";
    print(expected);
    std::cerr << " ] but got [";
    print(actual);
    std::cerr << " ]\n";
    return 1;
  }

public:
  Test() : builder(&ctx) { ctx.loadDialect<ONNXDialect>(); }

  // 1 byte 0bAAAABBBB unpacks as 0bBBBB 0bAAAA.
  int test_unpack_4bits() {
    std::cout << "test_unpack_4bits:" << std::endl;
    auto unpacked =
        unpackSubByteValues<uint8_t>(packedAttr({0b0001'0010, 0b1111'0000}), 4);
    return expect<uint8_t>(
        "test_unpack_4bits", unpacked, {0b0010, 0b0001, 0b0000, 0b1111});
  }

  // 1 byte 0bAABBCCDD unpacks as 0bDD 0bCC 0bBB 0bAA.
  int test_unpack_2bits() {
    std::cout << "test_unpack_2bits:" << std::endl;
    auto unpacked =
        unpackSubByteValues<uint8_t>(packedAttr({0b11'10'01'00}), 2);
    return expect<uint8_t>(
        "test_unpack_2bits", unpacked, {0b00, 0b01, 0b10, 0b11});
  }

  // bits = 8 is the identity, and is the case that needs the full 0..255 range
  // the static_assert on T guards.
  int test_unpack_8bits_is_identity() {
    std::cout << "test_unpack_8bits_is_identity:" << std::endl;
    auto unpacked = unpackSubByteValues<uint8_t>(
        packedAttr({0b0000'0000, 0b0111'1111, 0b1000'0000, 0b1111'1111}), 8);
    return expect<uint8_t>("test_unpack_8bits_is_identity", unpacked,
        {0b0000'0000, 0b0111'1111, 0b1000'0000, 0b1111'1111});
  }

  // Bytes with the high bit set must come out zero-extended, not sign-extended,
  // whatever the width of T. This is what the com.microsoft decompositions rely
  // on before they subtract the zero point.
  int test_unpack_is_zero_extended() {
    std::cout << "test_unpack_is_zero_extended:" << std::endl;
    int failures = 0;
    failures += expect<int64_t>("test_unpack_is_zero_extended 4 bits",
        unpackSubByteValues<int64_t>(packedAttr({0b1111'1111, 0b1000'1111}), 4),
        {0b1111, 0b1111, 0b1111, 0b1000});
    failures += expect<int64_t>("test_unpack_is_zero_extended 8 bits",
        unpackSubByteValues<int64_t>(packedAttr({0b1000'0000, 0b1111'1111}), 8),
        {0b1000'0000, 0b1111'1111});
    return failures;
  }

  // The unpacked values are laid out linearly, so a multi-byte buffer keeps
  // byte order while expanding each byte in place.
  int test_unpack_keeps_linear_order() {
    std::cout << "test_unpack_keeps_linear_order:" << std::endl;
    auto unpacked = unpackSubByteValues<int64_t>(
        packedAttr({0b0001'0000, 0b0011'0010, 0b0101'0100}), 4);
    return expect<int64_t>("test_unpack_keeps_linear_order", unpacked,
        {0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101});
  }

  // A DisposableElementsAttr has to unpack the same as the dense attribute it
  // stands in for, since that is what the decompositions actually see.
  int test_unpack_disposable_matches_dense() {
    std::cout << "test_unpack_disposable_matches_dense:" << std::endl;
    DenseElementsAttr dense = packedAttr({0b0001'0010, 0b1111'0000});
    OnnxElementsAttrBuilder elementsBuilder(&ctx);
    Attribute disposable = elementsBuilder.toDisposableElementsAttr(dense);
    return expect<uint8_t>("test_unpack_disposable_matches_dense",
        unpackSubByteValues<uint8_t>(disposable, 4),
        unpackSubByteValues<uint8_t>(dense, 4));
  }
};

} // namespace

int main(int argc, char *argv[]) {
  Test test;
  int failures = 0;
  failures += test.test_unpack_4bits();
  failures += test.test_unpack_2bits();
  failures += test.test_unpack_8bits_is_identity();
  failures += test.test_unpack_is_zero_extended();
  failures += test.test_unpack_keeps_linear_order();
  failures += test.test_unpack_disposable_matches_dense();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
