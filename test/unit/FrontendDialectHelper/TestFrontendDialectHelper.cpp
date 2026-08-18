/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====================-- TestFrontendDialectHelper.cpp --=====================//
//
// Copyright 2026 AMD.
//
// Tests for FrontendDialectHelper.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "onnx/onnx_pb.h"
#include "src/Builder/FrontendDialectHelper.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Support/Int4.hpp"

using namespace mlir;
using namespace onnx_mlir;

// The tag onnxruntime uses to indicate the external_data "location" points to
// an in-memory address rather than a file (see FrontendDialectHelper.cpp).
static constexpr const char *kInMemoryLocationTag = "*/_ORT_MEM_ADDR_/*";

class FrontendDialectHelperTest {
private:
  MLIRContext ctx;

  // Shared test vector for float tests.
  static constexpr float kFloatTestData[] = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  static constexpr size_t kFloatTestNumElements = 6;

  // Bool test data: stored as 1 byte per element (NOT bit-packed).
  // ONNX spec (raw_data field): "Boolean type MUST be written one byte per
  // tensor element (00000001 for true, 00000000 for false)."
  // External data uses the same byte layout as raw_data.
  static constexpr bool kBoolTestData[] = {
      true, false, true, true, false, false};
  static constexpr size_t kBoolTestNumElements = 6;
  // Raw bytes: 1 byte per bool (0x01 = true, 0x00 = false), not bit-packed.
  static constexpr uint8_t kBoolRawBytes[] = {
      0x01, 0x00, 0x01, 0x01, 0x00, 0x00};

  // INT4 test data: signed 4-bit integers in range [-8, 7].
  // Expected element values (widened to int8_t for comparison).
  static constexpr int8_t kInt4TestValues[] = {-8, 7, -1, 0, 3, -4};
  static constexpr size_t kInt4TestNumElements = 6;
  // ONNX packs 2 int4 values per byte: first element in 4 LSBs, second in
  // 4 MSBs.
  //   byte 0: -8=0x8 (LSB), 7=0x7 (MSB) → 0x78
  //   byte 1: -1=0xF (LSB), 0=0x0 (MSB) → 0x0F
  //   byte 2:  3=0x3 (LSB), -4=0xC (MSB) → 0xC3
  static constexpr uint8_t kInt4RawBytes[] = {0x78, 0x0F, 0xC3};

  // Int32/Int8 test data for the null-check-only in-memory tests.
  static constexpr int32_t kInt32TestData[] = {10, 20, 30, 40};
  static constexpr int8_t kInt8TestData[] = {-128, -1, 0, 1, 127, 64};

  // Generic value verifier. |StorageT| is the type MLIR yields from
  // attr.getValues<StorageT>() (e.g. float, bool, int_4); |ExpectedT| is the
  // element type of the expected array (e.g. float, bool, int8_t).  The two
  // may differ, e.g. int4 is iterated as int_4 but compared widened to int8_t.
  template <typename StorageT, typename ExpectedT>
  static bool verifyTestData(const char *testName, mlir::ElementsAttr attr,
      llvm::ArrayRef<ExpectedT> expected) {
    if (!attr) {
      std::cerr << "[" << testName << "] attr is null\n";
      return false;
    }
    size_t i = 0;
    for (StorageT v : attr.getValues<StorageT>()) {
      if (i >= expected.size()) {
        std::cerr << "[" << testName << "] too many elements, expected "
                  << expected.size() << "\n";
        return false;
      }
      if (static_cast<ExpectedT>(v) != expected[i]) {
        std::cerr << "[" << testName << "] value[" << i << "] mismatch\n";
        return false;
      }
      ++i;
    }
    if (i != expected.size()) {
      std::cerr << "[" << testName << "] wrong element count: got " << i
                << ", expected " << expected.size() << "\n";
      return false;
    }
    return true;
  }

  // Verifier that only checks the attribute is non-null (used for the tests
  // that assert successful loading without checking element values).
  static bool verifyNonNull(const char *testName, mlir::ElementsAttr attr) {
    if (!attr) {
      std::cerr << "[" << testName << "] attr is null\n";
      return false;
    }
    return true;
  }

  // Builds an EXTERNAL TensorProto of type |dataType| with shape |dims| and a
  // single external_data entry set {location, offset, length}.  Used by both
  // the in-memory and on-file helpers below, which differ only in the values
  // they pass for |location| and |length|.
  static onnx::TensorProto makeExternalTensorProto(
      onnx::TensorProto::DataType dataType,
      std::initializer_list<int64_t> dims, llvm::StringRef location,
      uint64_t offset, llvm::StringRef length) {
    onnx::TensorProto tp;
    tp.set_name("test_tensor");
    tp.set_data_type(dataType);
    for (int64_t d : dims)
      tp.add_dims(d);
    tp.set_data_location(onnx::TensorProto::EXTERNAL);

    auto *locEntry = tp.add_external_data();
    locEntry->set_key("location");
    locEntry->set_value(location.str());

    auto *offEntry = tp.add_external_data();
    offEntry->set_key("offset");
    offEntry->set_value(std::to_string(offset));

    auto *lenEntry = tp.add_external_data();
    lenEntry->set_key("length");
    lenEntry->set_value(length.str());
    return tp;
  }

  using Verifier = std::function<bool(const char *, mlir::ElementsAttr)>;

  // In-memory external data test (location "*/_ORT_MEM_ADDR_/*").  The
  // "offset" carries the address of |rawBytes| and "length" its byte count.
  template <onnx::TensorProto::DataType OnnxDataType>
  bool testInMemoryExternalData(const char *testName, const void *rawBytes,
      size_t byteCount, std::initializer_list<int64_t> dims,
      const Verifier &verifier) {
    onnx::TensorProto tp = makeExternalTensorProto(OnnxDataType, dims,
        kInMemoryLocationTag, reinterpret_cast<uintptr_t>(rawBytes),
        std::to_string(byteCount));
    mlir::ElementsAttr attr = onnx_mlir::onnxTensorProtoToElmAttr(&ctx, "", tp);
    return verifier(testName, attr);
  }

  // "Exporter wrote length=0" zero-length fallback test.  Writes |byteCount|
  // bytes from |rawBytes| to a temp file, builds a TensorProto with length="0"
  // (exercising the recompute-from-shape fallback), loads and verifies.
  template <onnx::TensorProto::DataType OnnxDataType>
  bool testFileExternalDataWithZeroLength(const char *testName,
      const void *rawBytes, size_t byteCount,
      std::initializer_list<int64_t> dims, const Verifier &verifier) {
    llvm::SmallString<128> tmpPath;
    int fd = -1;
    if (llvm::sys::fs::createTemporaryFile(
            "onnx_ext_data_test", "bin", fd, tmpPath)) {
      std::cerr << "[" << testName << "] Could not create temp file\n";
      return false;
    }
    {
      llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
      os.write(static_cast<const char *>(rawBytes), byteCount);
    }

    // Pass only the filename; the directory is given as externalDataDir below.
    // Deliberately write "0" for length to exercise the fallback that
    // recomputes the byte count from the tensor shape and element type.
    onnx::TensorProto tp = makeExternalTensorProto(OnnxDataType, dims,
        llvm::sys::path::filename(tmpPath), /*offset=*/0, /*length=*/"0");

    const std::string tmpDir = llvm::sys::path::parent_path(tmpPath).str();
    mlir::ElementsAttr attr =
        onnx_mlir::onnxTensorProtoToElmAttr(&ctx, tmpDir, tp);

    llvm::sys::fs::remove(tmpPath);

    return verifier(testName, attr);
  }

  // Verifiers bound to the shared expected data.
  static bool verifyFloat(const char *n, mlir::ElementsAttr a) {
    return verifyTestData<float, float>(n, a,
        llvm::ArrayRef<float>(kFloatTestData, kFloatTestNumElements));
  }
  static bool verifyBool(const char *n, mlir::ElementsAttr a) {
    return verifyTestData<bool, bool>(n, a,
        llvm::ArrayRef<bool>(kBoolTestData, kBoolTestNumElements));
  }
  static bool verifyInt4(const char *n, mlir::ElementsAttr a) {
    return verifyTestData<int_4, int8_t>(n, a,
        llvm::ArrayRef<int8_t>(kInt4TestValues, kInt4TestNumElements));
  }

public:
  FrontendDialectHelperTest() { ctx.getOrLoadDialect<ONNXDialect>(); }

  // --- In-memory external data (location "*/_ORT_MEM_ADDR_/*") -------------

  bool testInMemoryExternalDataFloat32() {
    return testInMemoryExternalData<onnx::TensorProto::FLOAT>(
        "testInMemoryExternalDataFloat32", kFloatTestData,
        kFloatTestNumElements * sizeof(float), {2, 3}, verifyFloat);
  }

  // Per the ONNX spec, bools in raw_data / external_data are stored as 1 byte
  // per element (0x00 = false, 0x01 = true); they are NOT bit-packed.
  bool testInMemoryExternalDataBool() {
    return testInMemoryExternalData<onnx::TensorProto::BOOL>(
        "testInMemoryExternalDataBool", kBoolRawBytes,
        kBoolTestNumElements * sizeof(uint8_t), {2, 3}, verifyBool);
  }

  // ONNX packs 2 int4 values per byte (first element in 4 LSBs, second in
  // 4 MSBs), matching the raw_data packing described in the ONNX spec.
  bool testInMemoryExternalDataInt4() {
    return testInMemoryExternalData<onnx::TensorProto::INT4>(
        "testInMemoryExternalDataInt4", kInt4RawBytes, sizeof(kInt4RawBytes),
        {2, 3}, verifyInt4);
  }

  bool testInMemoryExternalDataInt32() {
    return testInMemoryExternalData<onnx::TensorProto::INT32>(
        "testInMemoryExternalDataInt32", kInt32TestData, sizeof(kInt32TestData),
        {2, 2}, verifyNonNull);
  }

  bool testInMemoryExternalDataInt8() {
    return testInMemoryExternalData<onnx::TensorProto::INT8>(
        "testInMemoryExternalDataInt8", kInt8TestData, sizeof(kInt8TestData),
        {2, 3}, verifyNonNull);
  }

  bool testEmptyTensorWithInMemoryExternalData() {
    // Empty tensor: 0 dims, null address, 0 length. Should load gracefully.
    return testInMemoryExternalData<onnx::TensorProto::FLOAT>(
        "testEmptyTensorWithInMemoryExternalData", /*rawBytes=*/nullptr,
        /*byteCount=*/0, {0}, verifyNonNull);
  }

  // --- On-file external data with "length=0" fallback ---------------------

  bool testFileExternalDataWithZeroLengthFloat32() {
    return testFileExternalDataWithZeroLength<onnx::TensorProto::FLOAT>(
        "testFileExternalDataWithZeroLengthFloat32", kFloatTestData,
        kFloatTestNumElements * sizeof(float), {2, 3}, verifyFloat);
  }

  // ONNX stores bools as 1 byte per element (not bit-packed), so the fallback
  // must use 8 bits per element, not i1 = 1 bit.
  bool testFileExternalDataWithZeroLengthBool() {
    return testFileExternalDataWithZeroLength<onnx::TensorProto::BOOL>(
        "testFileExternalDataWithZeroLengthBool", kBoolRawBytes,
        kBoolTestNumElements * sizeof(uint8_t), {2, 3}, verifyBool);
  }

  // The fallback computes ceil(N*4/8) = 3 bytes for 6 elements, matching the
  // packed representation.
  bool testFileExternalDataWithZeroLengthInt4() {
    return testFileExternalDataWithZeroLength<onnx::TensorProto::INT4>(
        "testFileExternalDataWithZeroLengthInt4", kInt4RawBytes,
        sizeof(kInt4RawBytes), {2, 3}, verifyInt4);
  }

  bool runAllTests() {
    bool allPassed = true;

    allPassed = testInMemoryExternalDataFloat32() && allPassed;
    allPassed = testInMemoryExternalDataBool() && allPassed;
    allPassed = testInMemoryExternalDataInt4() && allPassed;
    allPassed = testInMemoryExternalDataInt32() && allPassed;
    allPassed = testInMemoryExternalDataInt8() && allPassed;
    allPassed = testEmptyTensorWithInMemoryExternalData() && allPassed;
    allPassed = testFileExternalDataWithZeroLengthFloat32() && allPassed;
    allPassed = testFileExternalDataWithZeroLengthBool() && allPassed;
    allPassed = testFileExternalDataWithZeroLengthInt4() && allPassed;

    return allPassed;
  }
};

int main(int /*argc*/, char * /*argv*/[]) {
  FrontendDialectHelperTest test;

  if (!test.runAllTests()) {
    return 1;
  }
  return 0;
}
