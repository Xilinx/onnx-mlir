/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===================-- TestDisposableElementsAttr.cpp --=====================//
//
// Tests DisposableElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposablePool.hpp"
#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Support/Arrays.hpp"

#include "mlir/IR/Builders.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace mlir;
using namespace onnx_mlir;

namespace {

bool near(double a, double b) { return fabs(a - b) < 1e-6; }

template <typename CPPTY>
bool eq(CPPTY a, CPPTY b) {
  if constexpr (isSmallFPType<CPPTY>)
    return a.toFloat() == b.toFloat();
  else
    return a == b;
}

bool forAllBTypes(std::function<bool(BType)> predicate) {
  bool result = true;
  // Only iterate up to INT4: dispatchByBType(), used by the predicates below,
  // does not handle the packed markers (see test_packed_int4()) or the unused
  // values between INT4 and them.
  for (BType d = static_cast<BType>(0); d <= BType::INT4;
      d = static_cast<BType>(static_cast<int>(d) + 1)) {
    if (d == BType::UNDEFINED || d == BType::STRING || d == BType::COMPLEX64 ||
        d == BType::COMPLEX128)
      continue;
    result &= predicate(d);
  }
  return result;
}

template <typename T, T... ints>
std::vector<T> nums(std::integer_sequence<T, ints...> int_seq) {
  std::vector<T> v;
  (v.push_back(ints), ...);
  return v;
}

MLIRContext *createCtx() {
  MLIRContext *ctx = new MLIRContext();
  ctx->loadDialect<ONNXDialect>();
  return ctx;
}

template <typename T>
std::unique_ptr<llvm::MemoryBuffer> buffer(ArrayRef<T> data) {
  return llvm::MemoryBuffer::getMemBufferCopy(asStringRef(data));
}

class Test {
  MLIRContext *ctx;
  Location loc;
  OpBuilder builder;
  OnnxElementsAttrBuilder elmsBuilder;
  Type F32;
  Type F16;
  Type U32;
  Type U8;
  Type I32;
  Type I64;
  Type I8;
  Type I1;
  Type I4;
  Type U4;

public:
  Test()
      : ctx(createCtx()), loc(UnknownLoc::get(ctx)), builder(ctx),
        elmsBuilder(ctx) {
    F32 = builder.getF32Type();
    F16 = builder.getF16Type();
    U32 = builder.getIntegerType(32, /*isSigned=*/false);
    U8 = builder.getIntegerType(8, /*isSigned=*/false);
    I32 = builder.getI32Type();
    I64 = builder.getI64Type();
    I8 = builder.getI8Type();
    I1 = builder.getI1Type();
    I4 = builder.getIntegerType(4);
    U4 = builder.getIntegerType(4, /*isSigned=*/false);
  }
  ~Test() { delete ctx; }

  IntegerType getUInt(unsigned width) const {
    return IntegerType::get(ctx, width, IntegerType::Unsigned);
  }

  int test_splat() {
    std::cout << "test_splat:" << std::endl;

    bool all = forAllBTypes([this](BType d) {
      return dispatchByBType(d, [this](auto btype) {
        using cpptype = CppType<btype>;

        Type elementType = mlirTypeOfBType(btype, ctx);
        if (elementType.getIntOrFloatBitWidth() == sizeof(cpptype) * 8) {
          ShapedType type = RankedTensorType::get({2, 1}, elementType);
          cpptype one(1);
          Attribute a = elmsBuilder.toDisposableElementsAttr(
              DenseElementsAttr::get(type, one));
          ElementsAttr e = mlir::cast<ElementsAttr>(a);
          assert(e.isSplat());
          DisposableElementsAttr i = mlir::cast<DisposableElementsAttr>(e);
          assert(i.isSplat());

          assert(eq<cpptype>(i.getSplatValue<cpptype>(), one));

          auto b = i.value_begin<cpptype>();
          assert(eq<cpptype>(*b, one));

          if (isFloatBType(btype)) {
            auto apf = i.getSplatValue<APFloat>();
            assert(near(apf.convertToDouble(), static_cast<double>(one)));
          } else {
            bool isSigned = isSignedIntBType(btype);
            auto api = i.getSplatValue<APInt>();
            auto x =
                WideNum::fromAPInt(api, isSigned).template to<cpptype>(btype);
            assert(eq<cpptype>(x, one));
          }

          auto d = i.toDenseElementsAttr();
          assert(eq<cpptype>(d.getSplatValue<cpptype>(), one));

          return true;
        } else if (!isFloatBType(btype)) {
          ShapedType type = RankedTensorType::get({2, 1}, elementType);
          APInt one(
              elementType.getIntOrFloatBitWidth(), 1, isSignedIntBType(btype));
          Attribute a = elmsBuilder.toDisposableElementsAttr(
              DenseElementsAttr::get(type, one));
          ElementsAttr e = mlir::cast<ElementsAttr>(a);
          assert(e.isSplat());
          DisposableElementsAttr i = mlir::cast<DisposableElementsAttr>(e);
          assert(i.isSplat());

          assert(eq<APInt>(i.getSplatValue<APInt>(), one));

          auto b = i.value_begin<APInt>();
          assert(eq<APInt>(*b, one));

          bool isSigned = isSignedIntBType(btype);
          auto api = i.getSplatValue<APInt>();
          auto x = WideNum::fromAPInt(api, isSigned).toAPInt(btype);
          assert(eq<APInt>(x, one));

          auto d = i.toDenseElementsAttr();
          assert(eq<APInt>(d.getSplatValue<APInt>(), one));

          return true;
        }
        llvm_unreachable("BType size should match CppType size or be an int. "
                         "If not, this test needs to be updated.");
      });
    });
    assert(all);

    return 0;
  }

  int test_transpose() {
    std::cout << "test_transpose:" << std::endl;

    ShapedType type = RankedTensorType::get({2, 3, 5}, getUInt(8));
    auto elms = nums<uint8_t>(std::make_integer_sequence<uint8_t, 30>{});
    auto e = elmsBuilder.fromMemoryBuffer(type, buffer<uint8_t>(elms));
    assert(e.getValues<uint8_t>()[0] == 0);
    assert(e.getValues<uint8_t>()[1] == 1);
    assert(e.getValues<uint8_t>()[28] == 28);
    assert(e.getValues<uint8_t>()[29] == 29);

    auto t = elmsBuilder.transpose(e, {1, 2, 0});
    assert(t.getValues<uint8_t>()[0] == 0);
    assert(t.getValues<uint8_t>()[1] == 15);
    assert(t.getValues<uint8_t>()[28] == 14);
    assert(t.getValues<uint8_t>()[29] == 29);

    return 0;
  }

  // A DisposableElementsAttr can be backed by packed int4/uint4 bytes (two
  // values per byte, as in ONNX's packed data) and unpack them on read.
  int test_packed_int4() {
    std::cout << "test_packed_int4:" << std::endl;

    // byte0=0xE1: nibble0=0x1=1,        nibble1=0xE=-2 (int4) / 14 (uint4)
    // byte1=0x83: nibble0=0x3=3,        nibble1=0x8=-8 (int4) /  8 (uint4)
    // byte2=0x07: nibble0=0x7=7,        nibble1=0x0=0
    std::vector<uint8_t> packedBytes = {0xE1, 0x83, 0x07};
    std::vector<int_4> expectedI4 = {
        int_4(1), int_4(-2), int_4(3), int_4(-8), int_4(7), int_4(0)};
    std::vector<uint_4> expectedU4 = {
        uint_4(1), uint_4(14), uint_4(3), uint_4(8), uint_4(7), uint_4(0)};

    ShapedType typeI4 = RankedTensorType::get({6}, I4);
    ShapedType typeU4 = RankedTensorType::get({6}, U4);

    ElementsAttr packedI4 = elmsBuilder.fromPackedInt4MemoryBuffer(
        typeI4, BType::PACKED_INT4, buffer<uint8_t>(packedBytes));
    ElementsAttr packedU4 = elmsBuilder.fromPackedInt4MemoryBuffer(
        typeU4, BType::PACKED_UINT4, buffer<uint8_t>(packedBytes));

    // getArray<X>() unpacks correctly.
    {
      auto arr = mlir::cast<DisposableElementsAttr>(packedI4).getArray<int_4>();
      for (size_t i = 0; i < 6; ++i)
        assert(eq<int_4>(arr.get()[i], expectedI4[i]));
      auto arrU =
          mlir::cast<DisposableElementsAttr>(packedU4).getArray<uint_4>();
      for (size_t i = 0; i < 6; ++i)
        assert(eq<uint_4>(arrU.get()[i], expectedU4[i]));
    }

    // Iteration (value_begin / getValues) unpacks correctly.
    {
      auto b =
          mlir::cast<DisposableElementsAttr>(packedI4).value_begin<int_4>();
      for (size_t i = 0; i < 6; ++i, ++b)
        assert(eq<int_4>(*b, expectedI4[i]));
    }

    // toDenseElementsAttr() unpacks correctly.
    {
      // Dense's generic getValues<T>() requires sizeof(T)*8 to match the
      // element type's declared bit width exactly, which int_4 (a 1-byte
      // wrapper for a 4-bit value) doesn't -- iterate as APInt instead, like
      // the existing test_splat() does for non-byte-aligned int types.
      DenseElementsAttr dense =
          mlir::cast<DisposableElementsAttr>(packedI4).toDenseElementsAttr();
      size_t i = 0;
      for (const APInt &v : dense.getValues<APInt>())
        assert(v.getSExtValue() == static_cast<int64_t>(expectedI4[i++]));
      assert(i == 6);
    }

    // Bit-for-bit cross-check against the eager fromArray path, built from the
    // already-unpacked equivalent.
    {
      ElementsAttr eagerI4 = elmsBuilder.fromArray<int_4>(
          typeI4, [&expectedI4](MutableArrayRef<int_4> dst) {
            for (size_t i = 0; i < 6; ++i)
              dst[i] = expectedI4[i];
          });
      assert(ElementsAttrBuilder::equal(packedI4, eagerI4));

      ElementsAttr eagerU4 = elmsBuilder.fromArray<uint_4>(
          typeU4, [&expectedU4](MutableArrayRef<uint_4> dst) {
            for (size_t i = 0; i < 6; ++i)
              dst[i] = expectedU4[i];
          });
      assert(ElementsAttrBuilder::equal(packedU4, eagerU4));
    }

    // Non-contiguous access (reshape [6] -> [2,3] -> transpose -> [3,2])
    // exercises atFlatIndex/flatIndexToBufferPos's packed branch outside the
    // pure contiguous fast path.
    {
      auto reshaped = elmsBuilder.reshape(packedI4, {2, 3});
      auto transposed = elmsBuilder.transpose(reshaped, {1, 0});
      // transposed[i][j] == reshaped[j][i] == packedI4[j*3 + i]
      std::vector<int_4> expectedTransposed = {
          int_4(1), int_4(-8), int_4(-2), int_4(7), int_4(3), int_4(0)};
      auto tv =
          mlir::cast<DisposableElementsAttr>(transposed).getValues<int_4>();
      for (size_t i = 0; i < 6; ++i)
        assert(eq<int_4>(tv[i], expectedTransposed[i]));
      // The raw bytes of the view hold one nibble per byte, in view order.
      auto transposedAttr = mlir::cast<DisposableElementsAttr>(transposed);
      ArrayBuffer<char> transposedBytes = transposedAttr.getRawBytes();
      std::vector<char> expectedBytes = {0x1, 0x8, 0xE, 0x7, 0x3, 0x0};
      assert(transposedBytes.get().size() == 6);
      for (size_t i = 0; i < 6; ++i)
        assert(transposedBytes.get()[i] == expectedBytes[i]);
    }

    // Slicing a packed buffer, contiguously and with a stride.
    {
      auto sliced = elmsBuilder.slice(packedI4, {4}, {1}, {1});
      std::vector<int_4> expectedSlice = {
          int_4(-2), int_4(3), int_4(-8), int_4(7)};
      auto sv = mlir::cast<DisposableElementsAttr>(sliced).getValues<int_4>();
      for (size_t i = 0; i < 4; ++i)
        assert(eq<int_4>(sv[i], expectedSlice[i]));

      auto strided = elmsBuilder.slice(packedI4, {3}, {0}, {2});
      std::vector<int_4> expectedStrided = {int_4(1), int_4(3), int_4(7)};
      auto tv2 = mlir::cast<DisposableElementsAttr>(strided).getValues<int_4>();
      for (size_t i = 0; i < 3; ++i)
        assert(eq<int_4>(tv2[i], expectedStrided[i]));
    }

    // A broadcast addresses the buffer through zero strides, so the view has
    // more elements than the one-byte buffer holds.
    {
      ShapedType type2 = RankedTensorType::get({2}, I4);
      ElementsAttr packed2 = elmsBuilder.fromPackedInt4MemoryBuffer(
          type2, BType::PACKED_INT4, buffer<uint8_t>({0xE1}));
      auto expanded = mlir::cast<DisposableElementsAttr>(
          elmsBuilder.expand(packed2, {3, 2}));
      ArrayBuffer<WideNum> wideNums = expanded.getWideNums();
      assert(wideNums.get().size() == 6);
      for (size_t i = 0; i < 6; ++i)
        assert(wideNums.get()[i].i64 == (i % 2 == 0 ? 1 : -2));
      ArrayBuffer<char> rawBytes = expanded.getRawBytes();
      assert(rawBytes.get().size() == 6);
      for (size_t i = 0; i < 6; ++i)
        assert(rawBytes.get()[i] == (i % 2 == 0 ? 0x1 : 0xE));
    }

    // Widening casts must read the packed buffer through its own type and give
    // sign-extended (int4) or zero-extended (uint4) values in the new type.
    {
      for (Type wide : {I8, I32, I64}) {
        auto castI4 = mlir::cast<DisposableElementsAttr>(
            elmsBuilder.castElementType(packedI4, wide));
        ElementsAttr denseI4 = castI4.toDenseElementsAttr();
        size_t i = 0;
        for (const APInt &v : denseI4.getValues<APInt>())
          assert(v.getSExtValue() == static_cast<int64_t>(expectedI4[i++]));
        assert(i == 6);
      }
      auto castI8 = mlir::cast<DisposableElementsAttr>(
          elmsBuilder.castElementType(packedI4, I8));
      ArrayBuffer<char> bytesI8 = castI8.getRawBytes();
      assert(bytesI8.get().size() == 6);
      for (size_t i = 0; i < 6; ++i)
        assert(static_cast<int8_t>(bytesI8.get()[i]) ==
               static_cast<int8_t>(expectedI4[i]));

      for (Type wide : {U8, U32}) {
        auto castU4 = mlir::cast<DisposableElementsAttr>(
            elmsBuilder.castElementType(packedU4, wide));
        ElementsAttr denseU4 = castU4.toDenseElementsAttr();
        size_t i = 0;
        for (const APInt &v : denseU4.getValues<APInt>())
          assert(v.getZExtValue() == static_cast<uint64_t>(expectedU4[i++]));
        assert(i == 6);
      }
    }

    return 0;
  }

  int test_scrub_packed_int4() {
    std::cout << "test_scrub_packed_int4:" << std::endl;

    // Scrubbing makes every constant dense, except packed int4/uint4 ones that
    // are large enough, when asked to preserve them.
    struct Case {
      int64_t minElements;
      bool expectPreserved;
    };
    for (Case c :
        {Case{-1, false}, Case{7, false}, Case{6, true}, Case{0, true}}) {
      ShapedType packedType = RankedTensorType::get({6}, I4);
      ElementsAttr packed = elmsBuilder.fromPackedInt4MemoryBuffer(
          packedType, BType::PACKED_INT4, buffer<uint8_t>({0xE1, 0x83, 0x07}));
      ShapedType plainType = RankedTensorType::get({2}, I8);
      ElementsAttr plain =
          elmsBuilder.fromMemoryBuffer(plainType, buffer<int8_t>({1, 2}));

      OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
      OpBuilder b(ctx);
      b.setInsertionPointToStart(module->getBody());
      auto packedOp = b.create<ONNXConstantOp>(loc, Attribute(), packed);
      auto plainOp = b.create<ONNXConstantOp>(loc, Attribute(), plain);

      DisposablePool::get<ONNXDialect>(ctx)->scrub(*module,
          {{ONNXConstantOp::getOperationName(), "value"}}, c.minElements);

      // Constants other than packed int4/uint4 are always dense afterwards.
      assert(isa<DenseElementsAttr>(plainOp.getValueAttr()));
      assert(isa<DisposableElementsAttr>(packedOp.getValueAttr()) ==
             c.expectPreserved);
      assert(isa<DenseElementsAttr>(packedOp.getValueAttr()) ==
             !c.expectPreserved);
      // Either way the values are still readable.
      auto values = mlir::cast<ElementsAttr>(packedOp.getValueAttr());
      std::vector<int64_t> expected = {1, -2, 3, -8, 7, 0};
      size_t i = 0;
      for (const APInt &v : values.getValues<APInt>())
        assert(v.getSExtValue() == expected[i++]);
      assert(i == 6);
    }

    return 0;
  }

  int test_cast() {
    std::cout << "test_cast:" << std::endl;

    ShapedType type = RankedTensorType::get({1}, I64);
    auto e = elmsBuilder.fromMemoryBuffer(type, buffer<int64_t>({256}));
    auto c = elmsBuilder.castElementType(e, F32);
    assert(c.getSplatValue<float>() == 256.0);

    return 0;
  }

  int test_equal_ints() {
    std::cout << "test_equal_ints:" << std::endl;

    ShapedType type2xi64 = RankedTensorType::get({2}, I64);
    auto e2s_i64 =
        elmsBuilder.fromMemoryBuffer(type2xi64, buffer<int64_t>({-2, 2}));
    auto e3s_i64 =
        elmsBuilder.fromMemoryBuffer(type2xi64, buffer<int64_t>({-3, 3}));

    assert(ElementsAttrBuilder::equal(e2s_i64, e2s_i64));
    assert(!ElementsAttrBuilder::equal(e3s_i64, e2s_i64));

    ShapedType type2xu8 = RankedTensorType::get({2}, U8);
    auto e2s_u8 =
        elmsBuilder.fromMemoryBuffer(type2xu8, buffer<uint8_t>({0xfe, 2}));
    auto e2s_i64_u8 = elmsBuilder.castElementType(e2s_i64, U8);
    auto e3s_i64_u8 = elmsBuilder.castElementType(e3s_i64, U8);

    assert(!ElementsAttrBuilder::equal(e3s_i64_u8, e2s_u8));
    assert(ElementsAttrBuilder::equal(e2s_i64_u8, e2s_u8));

    uint8_t u8_0xfe = 0xfe, u8_2 = 2;
    auto d2s_u8 = DenseElementsAttr::get(type2xu8, {u8_0xfe, u8_2});

    assert(ElementsAttrBuilder::equal(d2s_u8, e2s_u8));
    assert(ElementsAttrBuilder::equal(d2s_u8, e2s_i64_u8));
    assert(!ElementsAttrBuilder::equal(d2s_u8, e3s_i64_u8));

    return 0;
  }

  int test_equal_fps() {
    std::cout << "test_equal_fps:" << std::endl;

    ShapedType type2xf32 = RankedTensorType::get({2}, F32);
    float zero = 0.0f;
    auto d0s_f32_splat = DenseElementsAttr::get(type2xf32, {zero});
    auto d0s_f32 = DenseElementsAttr::get(type2xf32, {zero, -zero});

    assert(d0s_f32_splat != d0s_f32);
    assert(ElementsAttrBuilder::equal(d0s_f32_splat, d0s_f32));

    float nan = std::nanf("");
    assert(std::isnan(nan));
    auto dnans = DenseElementsAttr::get(type2xf32, {nan});

    // float NaN != NaN and the same goes for ElementsAttr::equal
    assert(nan != nan);
    assert(!ElementsAttrBuilder::equal(dnans, dnans));

    // one+delta can be expressed with f32 precision but not f16
    float one = 1.0f, delta = 0.00001f;

    ShapedType type1xf32 = RankedTensorType::get({1}, F32);
    auto d_one_f32 = DenseElementsAttr::get(type1xf32, {one});
    auto d_oneplus_f32 = DenseElementsAttr::get(type1xf32, {one + delta});
    assert(!ElementsAttrBuilder::equal(d_one_f32, d_oneplus_f32));

    auto d_one_f16 = elmsBuilder.castElementType(d_one_f32, F16);
    auto d_oneplus_f16 = elmsBuilder.castElementType(d_oneplus_f32, F16);
    assert(ElementsAttrBuilder::equal(d_one_f16, d_oneplus_f16));

    return 0;
  }

  int test_equal_bools() {
    std::cout << "test_equal_bools:" << std::endl;

    ShapedType type2xu32 = RankedTensorType::get({2}, U32);
    uint32_t u32_0 = 0, u32_2 = 2;
    auto d0_2_u32 = DenseElementsAttr::get(type2xu32, {u32_0, u32_2});
    auto e0_2_i1 = elmsBuilder.castElementType(d0_2_u32, I1);

    ShapedType type2xi1 = RankedTensorType::get({2}, I1);
    auto dF_T_i1 = DenseElementsAttr::get(type2xi1, {false, true});

    assert(e0_2_i1 != dF_T_i1);
    assert(ElementsAttrBuilder::equal(e0_2_i1, dF_T_i1));

    return 0;
  }
};

} // namespace

int main(int argc, char *argv[]) {
  Test test;
  int failures = 0;
  failures += test.test_splat();
  failures += test.test_transpose();
  failures += test.test_packed_int4();
  failures += test.test_scrub_packed_int4();
  failures += test.test_cast();
  failures += test.test_equal_ints();
  failures += test.test_equal_fps();
  failures += test.test_equal_bools();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
