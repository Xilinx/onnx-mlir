/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- DisposableElementsAttr.cpp --------------------===//
//
// DisposableElementsAttr, garbage collectible alternative to DenseElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttributeStorage.hpp"

#include "src/Dialect/ONNX/ElementsAttr/Strides.hpp"
#include "src/Support/TypeUtilities.hpp"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Endian.h"

#include <algorithm>
#include <atomic>
#include <string>

using namespace onnx_mlir;

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)

namespace mlir {

namespace {

template <BType BTYPE>
inline constexpr CppType<BTYPE> narrow(WideNum n) {
  return n.narrow<BTYPE>();
}

// Copies src to dstBytes while narrowing to the given datatype.
void narrowArray(
    BType bt, ArrayRef<WideNum> src, MutableArrayRef<char> dstBytes) {
  dispatchByBType(bt, [src, dstBytes](auto btype) {
    auto dst = castMutableArrayRef<CppType<btype>>(dstBytes);
    assert(src.size() == dst.size() && "narrowArray size mismatch");
    std::transform(src.begin(), src.end(), dst.begin(), narrow<btype>);
  });
}

// Copies srcBytes to dst while widening from the given datatype.
void widenArray(
    BType bt, ArrayRef<char> srcBytes, MutableArrayRef<WideNum> dst) {
  // AIESW-46865: PACKED_INT4/PACKED_UINT4 srcBytes hold two values packed per
  // byte (same layout as onnx TensorProto's packed int4/uint4 data), unlike
  // every other BType where srcBytes has one buffer slot per dst element.
  // This is the lazy-unpack step: called on read (e.g. once, from
  // getBufferAsWideNums()/readWideNums() over the *whole* contiguous buffer,
  // so index i here is the absolute flat element index -- not from a
  // relative byte slice; see atFlatIndex, which bypasses this function for
  // that reason).
  if (bt == BType::PACKED_INT4 || bt == BType::PACKED_UINT4) {
    bool isSigned = bt == BType::PACKED_INT4;
    for (size_t i = 0; i < dst.size(); ++i) {
      char packedByte = srcBytes[i / 2];
      bool isFirst = (i % 2) == 0;
      dst[i] = isSigned ? WideNum::widen<BType::INT4>(
                               int_4::extractFromPacked(packedByte, isFirst))
                         : WideNum::widen<BType::UINT4>(
                               uint_4::extractFromPacked(packedByte, isFirst));
    }
    return;
  }
  dispatchByBType(bt, [srcBytes, dst](auto btype) {
    auto src = castArrayRef<CppType<btype>>(srcBytes);
    assert(src.size() == dst.size() && "widenArray size mismatch");
    std::transform(src.begin(), src.end(), dst.begin(), WideNum::widen<btype>);
  });
}

} // namespace

/*static*/
DisposableElementsAttr DisposableElementsAttr::create(ShapedType type,
    size_t id, BType bufferBType, ArrayRef<int64_t> strides,
    const Buffer &buffer, Transformer transformer) {
  BType btype = btypeOfMlirType(type.getElementType());
  // AIESW-46865: PACKED_INT4/PACKED_UINT4 are storage-only bufferBTypes with
  // no mlir::Type/CppType of their own (two logical elements packed per
  // buffer byte), so wideBTypeOfBType(bufferBType) is not meaningful for them
  // -- check the packed/unpacked pairing explicitly instead.
  assert((transformer != nullptr ||
             (bufferBType == BType::PACKED_INT4 && btype == BType::INT4) ||
             (bufferBType == BType::PACKED_UINT4 && btype == BType::UINT4) ||
             wideBTypeOfBType(bufferBType) == wideBTypeOfBType(btype)) &&
         "buffer wide type mismatch requires transformer");
  bool isContiguous = areStridesContiguous(type.getShape(), strides);
  DisposableElementsAttr a = Base::get(
      type.getContext(), type, strides, bufferBType, btype, isContiguous, id);
  DisposableElementsAttributeStorage &s = *a.getImpl();
  s.buffer = buffer;
  s.transformer = std::move(transformer);
  return a;
}

void DisposableElementsAttr::dispose() {
  getImpl()->buffer.reset();
  getImpl()->transformer = nullptr;
}

bool DisposableElementsAttr::isSplat() const {
  return areStridesSplat(getStrides()) && getBuffer()->getBufferSize() != 0;
}

BType DisposableElementsAttr::getBType() const { return getImpl()->btype; }

ShapedType DisposableElementsAttr::getType() const { return getImpl()->type; }

bool DisposableElementsAttr::isDisposed() const { return !getImpl()->buffer; }

size_t DisposableElementsAttr::getId() const { return getImpl()->id; }

ArrayRef<int64_t> DisposableElementsAttr::getStrides() const {
  return getImpl()->strides;
}

auto DisposableElementsAttr::getBuffer() const -> const Buffer & {
  assert(!isDisposed());
  return getImpl()->buffer;
}

auto DisposableElementsAttr::getTransformer() const -> const Transformer & {
  assert(!isDisposed());
  return getImpl()->transformer;
}

bool DisposableElementsAttr::isContiguous() const {
  return getImpl()->isContiguous;
}

bool DisposableElementsAttr::isTransformed() const {
  return getImpl()->transformer != nullptr;
}

bool DisposableElementsAttr::isTransformedOrCast() const {
  return isTransformed() || getBType() != getBufferBType();
}

BType DisposableElementsAttr::getBufferBType() const {
  return getImpl()->bufferBType;
}

unsigned DisposableElementsAttr::getBufferElementBytewidth() const {
  BType bufferBType = getBufferBType();
  // AIESW-46865: packed buffers have no fixed per-element bytewidth (two
  // elements share one byte) -- 0 is a safe sentinel: callers that still
  // divide by it unconditionally would need fixing anyway, and the one
  // existing caller that compares this against sizeof(WideNum) correctly
  // treats 0 as "not that fast path".
  if (bufferBType == BType::PACKED_INT4 || bufferBType == BType::PACKED_UINT4)
    return 0;
  return bytewidthOfBType(bufferBType);
}

int64_t DisposableElementsAttr::getNumBufferElements() const {
  BType bufferBType = getBufferBType();
  if (bufferBType == BType::PACKED_INT4 || bufferBType == BType::PACKED_UINT4)
    return getBuffer()->getBufferSize() * 2;
  return getBuffer()->getBufferSize() / getBufferElementBytewidth();
}

ArrayBuffer<WideNum> DisposableElementsAttr::getWideNums() const {
  if (isContiguous()) {
    return getBufferAsWideNums();
  }
  ArrayBuffer<WideNum>::Vector dst;
  dst.resize_for_overwrite(getNumElements());
  readWideNums(dst);
  return std::move(dst);
}

void DisposableElementsAttr::readWideNums(MutableArrayRef<WideNum> dst) const {
  if (isContiguous()) {
    readBytesAsWideNums(getBufferBytes(), dst);
    return;
  }
  ArrayBuffer<WideNum> src = getBufferAsWideNums();
  restrideArray<WideNum>(getShape(), getStrides(), src.get(), dst);
}

DenseElementsAttr DisposableElementsAttr::toDenseElementsAttr() const {
  if (isSplat())
    return DenseElementsAttr::get(getType(), {getSplatValue<Attribute>()});
  ArrayBuffer<char> bytes = getRawBytes();
  if (getElementType().isInteger(1))
    // don't use getFromRawBuffer which requires bit packing
    return DenseElementsAttr::get(getType(), castArrayRef<bool>(bytes.get()));
  return DenseElementsAttr::getFromRawBuffer(getType(), bytes.get());
}

namespace {
// Perform byte swap if system endianness is BE and elements are multi-byte.
bool shouldSwapLEBytes(unsigned elementByteWidth) {
  return elementByteWidth > 1 &&
         llvm::endianness::native != llvm::endianness::little;
}
} // namespace

/*static*/
std::unique_ptr<llvm::MemoryBuffer> DisposableElementsAttr::parse(
    AsmParser &parser, ShapedType type) {
  size_t id = 0; // The parsed id is ignored.
  std::string str;
  if (parser.parseLess() || parser.parseInteger(id) || parser.parseColon() ||
      parser.parseString(&str))
    return nullptr;
  StringRef hex = str;
  std::string bytes;
  if (!hex.consume_front("0x") || (hex.size() & 1) ||
      !llvm::tryGetFromHex(hex, bytes)) {
    parser.emitError(parser.getCurrentLocation(), "ill-formed hex string");
    return nullptr;
  }
  if (bytes.size() != static_cast<size_t>(getSizeInBytes(type))) {
    parser.emitError(
        parser.getCurrentLocation(), "data size doesn't match type size");
    return nullptr;
  }
  if (!shouldSwapLEBytes(getIntOrFloatByteWidth(type.getElementType()))) {
    return llvm::MemoryBuffer::getMemBufferCopy(bytes);
  } else {
    // Reorder bytes from little-endian on big-endian platforms:
    std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
        llvm::WritableMemoryBuffer::getNewUninitMemBuffer(bytes.size());
    DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine(
        {bytes.data(), bytes.size()}, writeBuffer->getBuffer(), type);
    return writeBuffer;
  }
}

namespace {
// Threshold set through setPrintElisionThreshold; negative means no override.
std::atomic<int64_t> gPrintElisionThreshold{-1};
} // namespace

void DisposableElementsAttr::setPrintElisionThreshold(
    int64_t elideLargerThanOrNegativeOne) {
  gPrintElisionThreshold.store(elideLargerThanOrNegativeOne);
}

void DisposableElementsAttr::printWithoutType(AsmPrinter &printer) const {
  // It would be ideal if we could read the printer flags from printer instead
  // of constructing them here, because printer may have been constructed with
  // an override of elideLargeElementsAttrs which we cannot see here.
  // Oh well, at least OpPrintingFlags().shouldElideElementsAttr(ElementsAttr)
  // lets us respect the --mlir-elide-elementsattrs-if-larger command line flag.
  // A threshold set through setPrintElisionThreshold takes precedence.
  OpPrintingFlags printerFlags;
  int64_t threshold = gPrintElisionThreshold.load();
  if (threshold >= 0)
    printerFlags.elideLargeElementsAttrs(threshold);
  printer << getMnemonic() << "<" << getImpl()->id << ":";
  if (!printerFlags.shouldElideElementsAttr(*this)) {
    auto rawBytes = getRawBytes();
    SmallVector<char> buffer;
    ArrayRef<char> bytes;
    if (!shouldSwapLEBytes(getIntOrFloatByteWidth(getElementType()))) {
      bytes = rawBytes.get();
    } else {
      // Reorder raw bytes to little-endian on big-endian platforms:
      buffer.resize_for_overwrite(rawBytes.get().size());
      DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine(
          rawBytes.get(), buffer, getType());
      ArrayRef<char> bufferRef(buffer);
      bytes = bufferRef;
    }
    printer << "\"0x" << llvm::toHex(castArrayRef<uint8_t>(bytes)) << "\"";
  } else {
    printer << "__elided__";
  }
  printer << ">";
}

void DisposableElementsAttr::printAsDenseElementsAttr(
    AsmPrinter &printer) const {
  // See printWithoutType.
  OpPrintingFlags printerFlags;
  int64_t threshold = gPrintElisionThreshold.load();
  if (threshold >= 0)
    printerFlags.elideLargeElementsAttrs(threshold);
  if (isSplat() || !printerFlags.shouldElideElementsAttr(*this)) {
    // Take shortcut by first converting to DenseElementsAttr.
    // NOTE: This creates a copy which is never garbage collected. This is not
    // only slow but also defeats the garbage collection benefits of
    // DisposableElementsAttr, depending on when the printing
    // takes place (the print at the end of onnx-mlir-opt in lit tests is ok).
    printer.printAttribute(toDenseElementsAttr());
    // TODO: Do the work to print without constructing DenseElementsAttr.
  } else {
    // In this special case it's easy to avoid conversion to DenseElementsAttr.
    //
    // AIESW-46865: MLIR's own AsmPrinter always emits "dense_resource<...>"
    // (never plain "dense<...>") for an elided ElementsAttr
    // (llvm-project/mlir/lib/IR/AsmPrinter.cpp) -- this hand-rolled shortcut
    // must match that syntax to stay re-parseable. This was a latent,
    // pre-existing bug: before DisposableElementsAttr could survive past
    // ScrubDisposablePass (which used to materialize every attr to Dense
    // immediately, early in the pipeline), no Disposable attr ever reached a
    // dump-and-reparse point large enough to hit this elided branch in
    // practice.
    printer << "dense_resource<__elided__> : " << getType();
  }
}

void DisposableElementsAttr::readBytesAsWideNums(
    ArrayRef<char> srcBytes, llvm::MutableArrayRef<WideNum> dst) const {
  widenArray(getBufferBType(), srcBytes, dst);
  if (const Transformer &transformer = getTransformer())
    transformer(dst);
}

ArrayRef<char> DisposableElementsAttr::getBufferBytes() const {
  return asArrayRef(getBuffer()->getBuffer());
}

ArrayBuffer<WideNum> DisposableElementsAttr::getBufferAsWideNums() const {
  if (!isTransformed() && getBufferElementBytewidth() == sizeof(WideNum)) {
    return castArrayRef<WideNum>(getBufferBytes());
  }
  ArrayBuffer<WideNum>::Vector dst;
  BType bufferBType = getBufferBType();
  if (bufferBType == BType::PACKED_INT4 || bufferBType == BType::PACKED_UINT4) {
    // AIESW-46865: a packed buffer's true element count can't always be
    // derived from its byte size alone (ambiguous by one nibble for an odd
    // total element count, since the last byte is half-used), so use this
    // attr's own logical element count directly instead of
    // getNumBufferElements()'s bufferSize-based estimate. Exact as long as
    // the buffer isn't shared by a broadcast view with more logical elements
    // than physical buffer capacity -- not a case that arises for packed
    // int4/uint4 weights in this compiler (never broadcast-expanded after
    // import).
    dst.resize_for_overwrite(getNumElements());
  } else {
    dst.resize_for_overwrite(getNumBufferElements());
  }
  readBytesAsWideNums(getBufferBytes(), dst);
  return std::move(dst);
}

WideNum DisposableElementsAttr::atFlatIndex(size_t flatIndex) const {
  size_t pos = flatIndexToBufferPos(flatIndex);
  BType bufferBType = getBufferBType();
  if (bufferBType == BType::PACKED_INT4 || bufferBType == BType::PACKED_UINT4) {
    // AIESW-46865: can't go through widenArray here like the generic path
    // below does, because widenArray's packed branch derives which nibble to
    // extract from the *index into the array it's given*, which must be the
    // absolute flat element position -- but this function slices out a
    // single byte first, which would always look like "index 0" (the first
    // nibble) to widenArray. Extract the needed nibble directly instead.
    char packedByte = getBufferBytes()[pos / 2];
    bool isFirst = (pos % 2) == 0;
    WideNum n = bufferBType == BType::PACKED_INT4
                    ? WideNum::widen<BType::INT4>(
                          int_4::extractFromPacked(packedByte, isFirst))
                    : WideNum::widen<BType::UINT4>(
                          uint_4::extractFromPacked(packedByte, isFirst));
    if (const Transformer &transformer = getTransformer())
      transformer(llvm::MutableArrayRef(n));
    return n;
  }
  unsigned bufBytewidth = getBufferElementBytewidth();
  ArrayRef<char> bytes =
      getBufferBytes().slice(pos * bufBytewidth, bufBytewidth);
  WideNum n;
  readBytesAsWideNums(bytes, llvm::MutableArrayRef(n));
  return n;
}

size_t DisposableElementsAttr::flatIndexToBufferPos(size_t flatIndex) const {
  if (flatIndex == 0 || isContiguous())
    return flatIndex;
  if (isSplat())
    return 0;
  auto indices = unflattenIndex(getShape(), flatIndex);
  return getStridesPosition(indices, getStrides());
}

void DisposableElementsAttr::readRawBytes(
    MutableArrayRef<char> dstBytes) const {
  BType btype = getBType();
  unsigned elemBytewidth = bytewidthOfBType(btype);
  BType bufferBType = getBufferBType();
  if (!isTransformed() && isContiguous() &&
      (bufferBType == BType::PACKED_INT4 ||
          bufferBType == BType::PACKED_UINT4)) {
    // AIESW-46865: unpack directly to the final 1-byte-per-element raw
    // layout -- int_4/uint_4's only data member is the nibble value itself
    // (not sign-extended), so no further per-type handling is needed here.
    // This matters because this is the one unavoidable real materialize
    // point for a packed weight (e.g. ScrubDisposablePass, FormatConstants,
    // toDenseElementsAttr): skipping the generic path below's
    // 8-bytes-per-element WideNum detour avoids transiently ballooning
    // memory to 8x the final materialized size while this runs, which with
    // many large weights materializing concurrently (e.g. multi-threaded
    // pass execution) measurably regressed peak memory above even the
    // pre-lazy-import baseline.
    ArrayRef<char> src = getBufferBytes();
    for (int64_t i = 0, n = getNumElements(); i < n; ++i) {
      char packedByte = src[i / 2];
      bool isFirst = (i % 2) == 0;
      dstBytes[i] =
          static_cast<char>((isFirst ? packedByte : (packedByte >> 4)) & 0x0F);
    }
    return;
  }
  if (!isTransformedOrCast()) {
    auto srcBytes = getBufferBytes();
    restrideArray(elemBytewidth, getShape(), getStrides(), srcBytes, dstBytes);
  } else if (elemBytewidth == sizeof(WideNum)) {
    readWideNums(castMutableArrayRef<WideNum>(dstBytes));
  } else {
    SmallVector<WideNum, 1> dst;
    dst.resize_for_overwrite(getNumElements());
    readWideNums(dst);
    narrowArray(btype, dst, dstBytes);
  }
}

ArrayBuffer<char> DisposableElementsAttr::getRawBytes() const {
  if (!isTransformedOrCast() && isContiguous())
    return getBufferBytes();
  unsigned elemBytewidth = bytewidthOfBType(getBType());
  ArrayBuffer<char>::Vector dstBytes;
  dstBytes.resize_for_overwrite(getNumElements() * elemBytewidth);
  readRawBytes(dstBytes);
  return std::move(dstBytes);
}

} // namespace mlir
