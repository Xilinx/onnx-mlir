/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- FrontendDialectHelper.cpp ----------------------===//
//
// Copyright 2019 The IBM Research Authors.
// Modifications Copyright 2025-2026 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// Helper methods for handling input ONNX models.
//
//===----------------------------------------------------------------------===//

#include "src/Builder/FrontendDialectHelper.hpp"

#include <memory>
#include <mutex>

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SwapByteOrder.h"

#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Support/Arrays.hpp"
#include "src/Support/SmallFP.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Parses unsigned number.
size_t parseOffsetOrLength(const std::string &value) {
  char *end = nullptr;
  size_t offsetOrLength = strtoull(value.c_str(), &end, 0);
  assert(end != value.c_str() && "failed to parse offset or length");
  return offsetOrLength;
}

struct ExternalDataLoc {
  std::string location;
  uint64_t offset = 0;
  uint64_t length = -1; // MemoryBuffer uses -1 to mean infinity

  ExternalDataLoc(const onnx::TensorProto &tp) {
    for (const onnx::StringStringEntryProto &entry : tp.external_data()) {
      assert(entry.has_key() && "external_data entry must have key");
      assert(entry.has_value() && "external_data entry must have value");
      if (entry.key() == "location") {
        location = entry.value();
      } else if (entry.key() == "offset") {
        offset = parseOffsetOrLength(entry.value());
      } else if (entry.key() == "length") {
        length = parseOffsetOrLength(entry.value());
      }
    }
  }
};

// True if the location of a TensorProto::EXTERNAL is in-memory.
bool isInMemoryExternal(std::string_view location) {
  // onnxruntime#12465 kTensorProtoMemoryAddressTag = "*/_ORT_MEM_ADDR_/*"
  return location == "*/_ORT_MEM_ADDR_/*";
}

// A view of part of a shared, whole-file MemoryBuffer; keeps the whole buffer
// alive for as long as the slice is.
class SlicedMemoryBuffer : public llvm::MemoryBuffer {
public:
  SlicedMemoryBuffer(
      std::shared_ptr<llvm::MemoryBuffer> whole, llvm::StringRef slice)
      : whole(std::move(whole)) {
    init(slice.begin(), slice.end(), /*RequiresNullTerminator=*/false);
  }
  llvm::StringRef getBufferIdentifier() const override {
    return whole->getBufferIdentifier();
  }
  BufferKind getBufferKind() const override { return whole->getBufferKind(); }

private:
  std::shared_ptr<llvm::MemoryBuffer> whole;
};

// Returns a read-only buffer (mmap'ed if large) for a whole external data
// file. A model typically stores thousands of tensors in one data file, and
// opening it once per tensor is costly on network filesystems. The mapping is
// cached only while some tensor slice still references it.
std::shared_ptr<llvm::MemoryBuffer> getWholeExternalDataFile(
    const std::string &pathStr) {
  static std::mutex mutex;
  static llvm::StringMap<std::weak_ptr<llvm::MemoryBuffer>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  std::weak_ptr<llvm::MemoryBuffer> &entry = cache[pathStr];
  if (std::shared_ptr<llvm::MemoryBuffer> whole = entry.lock())
    return whole;
  auto bufferOrError = llvm::MemoryBuffer::getFile(pathStr, /*IsText=*/false,
      /*RequiresNullTerminator=*/false, /*IsVolatile=*/false);
  if (std::error_code ec = bufferOrError.getError()) {
    llvm::errs() << "Error " << ec.message() << " reading from file " << pathStr
                 << "\n";
    llvm::report_fatal_error("Cannot read external data file");
  }
  std::shared_ptr<llvm::MemoryBuffer> whole = std::move(bufferOrError.get());
  entry = whole;
  return whole;
}

// Reads external data from file location specified in tensor proto.
// The data is little endian encoded.
// See https://github.com/onnx/onnx/blob/main/docs/ExternalData.md
std::unique_ptr<llvm::MemoryBuffer> readExternalData_LE(
    const std::string &externalDataDir, const ExternalDataLoc &loc) {
  assert(!loc.location.empty() && "missing external data location");
  // This should only be used for on-file external data.
  assert(!isInMemoryExternal(loc.location));

  SmallVector<char> path(externalDataDir.begin(), externalDataDir.end());
  llvm::sys::path::append(path, loc.location);
  const std::string pathStr(path.data(), path.size());

  std::shared_ptr<llvm::MemoryBuffer> whole = getWholeExternalDataFile(pathStr);
  const uint64_t fileSize = whole->getBufferSize();
  const uint64_t length =
      loc.length == uint64_t(-1) ? fileSize - loc.offset : loc.length;
  if (loc.offset > fileSize || loc.offset + length > fileSize) {
    llvm::errs() << "Error: External data file " << pathStr
                 << " is too small.\n"
                 << "  File size: " << fileSize << " bytes\n"
                 << "  Required:  " << loc.offset + length << " bytes "
                 << "(offset=" << loc.offset << " + length=" << length << ")\n";
    llvm::report_fatal_error("External data file is truncated or corrupted");
  }
  return std::make_unique<SlicedMemoryBuffer>(
      whole, whole->getBuffer().substr(loc.offset, length));
}

template <typename T>
struct TransformValueToONNXData {
  static const google::protobuf::RepeatedField<int32_t> &data(
      const onnx::TensorProto &tp) {
    // int32_data is used for:
    // int32, uint8, int8, uint16, int16, bool, float_16, bfloat_16,
    // float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz
    // int4 and uint4 are packed, int32_data stores 2 int4s or uint4s.
    return tp.int32_data();
  }
};

template <>
struct TransformValueToONNXData<double> {
  static const google::protobuf::RepeatedField<double> &data(
      const onnx::TensorProto &tp) {
    return tp.double_data();
  }
};

template <>
struct TransformValueToONNXData<float> {
  static const google::protobuf::RepeatedField<float> &data(
      const onnx::TensorProto &tp) {
    return tp.float_data();
  }
};

template <>
struct TransformValueToONNXData<int64_t> {
  static const google::protobuf::RepeatedField<int64_t> &data(
      const onnx::TensorProto &tp) {
    return tp.int64_data();
  }
};

template <>
struct TransformValueToONNXData<uint32_t> {
  static const google::protobuf::RepeatedField<uint64_t> &data(
      const onnx::TensorProto &tp) {
    return tp.uint64_data();
  }
};

template <>
struct TransformValueToONNXData<uint64_t> {
  static const google::protobuf::RepeatedField<uint64_t> &data(
      const onnx::TensorProto &tp) {
    return tp.uint64_data();
  }
};

template <typename T, typename Range, typename Transformation>
ElementsAttr createElmAttrFromArray(RankedTensorType tensorType,
    const Range &array, const Transformation &transformation) {
  MLIRContext *ctx = tensorType.getContext();
  assert(tensorType.getElementType() == toMlirType<T>(ctx));
  const int64_t numElements = cast<ShapedType>(tensorType).getNumElements();
  return OnnxElementsAttrBuilder(ctx).fromArray<T>(tensorType,
      [array, &transformation, numElements](MutableArrayRef<T> copy) {
        for (int64_t idx = 0; idx < numElements; ++idx)
          transformation(array, copy, idx);
      });
}

// Perform byte swap if system endianness is BE.
// ONNX tensor content raw data is always in LE.
// Don't byte swap single byte types, because that's unnecessary
// and llvm::sys::getSwappedBytes(bool) also happens to be broken.
template <typename T>
constexpr bool shouldSwapLEBytes =
    sizeof(T) > 1 && llvm::endianness::native != llvm::endianness::little;
// Extension of llvm::sys::getSwappedBytes to also handle float_16, bfloat_16.
template <typename T>
T swappedBytes(T x) {
  if constexpr (isSmallFPType<T>)
    return T::bitcastFromUInt(llvm::sys::getSwappedBytes(x.bitcastToUInt()));
  else
    return llvm::sys::getSwappedBytes(x);
}

template <typename FromContainer, typename To, typename Transform>
auto getRangeTransformer(Transform transform) {
  return [transform](FromContainer data, MutableArrayRef<To> output,
             size_t idx) { output[idx] = transform(data[idx]); };
}

// Unpacks packed int4/uint4 bytes (2 values per byte, low nibble first) in a
// tight byte loop. A per-element indirect call here dominated import time for
// large int4-weight models.
template <typename T>
ElementsAttr createInt4ElmAttrFromPackedBytes(
    RankedTensorType tensorType, ArrayRef<char> packed) {
  const int64_t numElements = tensorType.getNumElements();
  return OnnxElementsAttrBuilder(tensorType.getContext())
      .fromArray<T>(tensorType, [packed, numElements](MutableArrayRef<T> dst) {
        assert(packed.size() >= static_cast<size_t>((numElements + 1) / 2));
        const char *in = packed.data();
        T *out = dst.data();
        const int64_t numPairs = numElements / 2;
        for (int64_t i = 0; i < numPairs; ++i) {
          out[2 * i] = T::extractFromPacked(in[i], /*isFirst=*/true);
          out[2 * i + 1] = T::extractFromPacked(in[i], /*isFirst=*/false);
        }
        if (numElements % 2)
          out[numElements - 1] =
              T::extractFromPacked(in[numPairs], /*isFirst=*/true);
      });
}

template <typename T>
ElementsAttr createElementsAttrFromMemoryBuffer_LE(
    RankedTensorType tensorType, std::unique_ptr<llvm::MemoryBuffer> membuf) {
  MLIRContext *ctx = tensorType.getContext();
  assert(tensorType.getElementType() == toMlirType<T>(ctx));
  if constexpr (isAnyInt4Type<T>) {
    return createInt4ElmAttrFromPackedBytes<T>(tensorType,
        ArrayRef<char>(membuf->getBuffer().begin(), membuf->getBuffer().end()));
  } else if constexpr (shouldSwapLEBytes<T>) {
    ArrayRef<T> array = asArrayRef<T>(membuf->getBuffer());
    return createElmAttrFromArray<T>(tensorType, array,
        getRangeTransformer<ArrayRef<T>, T>(swappedBytes<T>));
  } else {
    return OnnxElementsAttrBuilder(ctx).fromMemoryBuffer(
        tensorType, std::move(membuf));
  }
}

template <typename T>
ElementsAttr createElmAttrFromRawBytes_LE(
    RankedTensorType tensorType, ArrayRef<char> bytes) {
  if constexpr (isAnyInt4Type<T>) {
    return createInt4ElmAttrFromPackedBytes<T>(tensorType, bytes);
  } else {
    ArrayRef<T> array = castArrayRef<T>(bytes);
    return createElmAttrFromArray<T>(
        tensorType, array, getRangeTransformer<ArrayRef<T>, T>([](T x) {
          if constexpr (shouldSwapLEBytes<T>)
            return swappedBytes<T>(x);
          else
            return x;
        }));
  }
}

// Converts to the cpp type 'To' that correspond's to the tensor element type
// (bool, int8, float_16, uint32, etc) from the the proto data field type
// which may be a wider type (int32, uint64). In most cases the conversion is
// just standard C implicit conversion. The exception is float_16 and bfloat_16
// which must be bit-wise converted from uint16_t.
template <typename To, typename From>
To deserializeDatum(const From &from) {
  if constexpr (isSmallFPType<To>)
    return To::bitcastFromUInt(from);
  else
    return from;
}

template <typename To, typename From>
void deserializeDatumRange(const google::protobuf::RepeatedField<From> &data,
    MutableArrayRef<To> output, size_t idx) {
  if constexpr (isAnyInt4Type<To>) {
    static_assert(std::is_same_v<From, int32_t>,
        "int4 and uint4 can only be deserialized from int32_data");
    const bool isEven = (idx % 2) == 0;
    // int4 and uint4 are packed, each int32_data stores 2 int4s or uint4s.
    output[idx] = To::extractFromPacked(data[idx / 2], /*isFirst*/ isEven);
  } else {
    output[idx] = deserializeDatum<To, From>(data[idx]);
  }
}

template <typename T, typename U>
ElementsAttr createElmAttrFromProtoData(RankedTensorType tensorType,
    const google::protobuf::RepeatedField<U> &data) {
  // "Deserialize" the data to the correct bitwidth.
  return createElmAttrFromArray<T>(
      tensorType, data, deserializeDatumRange<T, U>);
}

// Returns ElementsAttr with tp's data.
template <typename T>
ElementsAttr createElmAttr(RankedTensorType tensorType,
    const onnx::TensorProto &tp, const std::string &externalDataDir) {
  if (tp.has_data_location() &&
      tp.data_location() == onnx::TensorProto::EXTERNAL) {
    ExternalDataLoc loc(tp);
    if (isInMemoryExternal(loc.location)) {
      return createElmAttrFromRawBytes_LE<T>(tensorType,
          llvm::ArrayRef(reinterpret_cast<char *>(loc.offset), loc.length));
    }
    return createElementsAttrFromMemoryBuffer_LE<T>(
        tensorType, readExternalData_LE(externalDataDir, loc));
  }
  if (tp.has_raw_data()) {
    return createElmAttrFromRawBytes_LE<T>(
        tensorType, asArrayRef(tp.raw_data()));
  }
  // Not raw, no need to take care of endianness.
  const auto &data = TransformValueToONNXData<T>::data(tp);
  return createElmAttrFromProtoData<T>(tensorType, data);
}

ElementsAttr createStringElmAttr(
    RankedTensorType tensorType, const onnx::TensorProto &tp) {
  // The string type is different from other data types in that it cannot be
  // raw or external data, it cannot be represented as a DisposableElementsAttr,
  // and it needs to be converted to StringRef (or StringAttr) to construct a
  // DenseElementsAttr.
  assert(!(tp.has_data_location() &&
             tp.data_location() == onnx::TensorProto::EXTERNAL) &&
         "string TensorProto cannot be external data");
  assert(!tp.has_raw_data() && "string TensorProto cannot be raw data");
  auto data = tp.string_data();
  SmallVector<StringRef> copy(data.begin(), data.end());
  return DenseElementsAttr::get(tensorType, ArrayRef(copy));
}

} // namespace

ElementsAttr onnxTensorProtoToElmAttr(MLIRContext *ctx,
    const std::string &externalDataDir, const onnx::TensorProto &tp) {
  // Tensor dimensions.
  ArrayRef<int64_t> tensorDims(tp.dims().data(), tp.dims().size());
  if (tp.data_type() == onnx::TensorProto::STRING) {
    Type elmType = ONNXStringType::get(ctx);
    auto tensorType = RankedTensorType::get(tensorDims, elmType);
    return createStringElmAttr(tensorType, tp);
  }
  BType btype = btypeOfOnnxDataType(tp.data_type());
  Type elmType = mlirTypeOfBType(btype, ctx);
  auto tensorType = RankedTensorType::get(tensorDims, elmType);
  return dispatchByBType(btype, [&](auto btype) {
    using cpptype = CppType<btype>;
    return createElmAttr<cpptype>(tensorType, tp, externalDataDir);
  });
}

} // namespace onnx_mlir
