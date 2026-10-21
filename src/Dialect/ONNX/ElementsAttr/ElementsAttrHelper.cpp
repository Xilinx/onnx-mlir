/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ElementsAttrHelper.cpp -----------------------===//
//
// Helper functions for accessing ElementsAttr contents.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrHelper.hpp"

#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>

using namespace mlir;

namespace onnx_mlir {

WideNum getElementsSplatWideNum(ElementsAttr elms) {
  if (auto disposable = mlir::dyn_cast<DisposableElementsAttr>(elms))
    return disposable.getSplatValue<WideNum>();
  Type elementType = elms.getElementType();
  if (isa<FloatType>(elementType))
    return WideNum::fromAPFloat(elms.getSplatValue<APFloat>());
  if (auto itype = dyn_cast<IntegerType>(elementType))
    return WideNum::fromAPInt(elms.getSplatValue<APInt>(), !itype.isUnsigned());
  llvm_unreachable("WideNum only supports integer and float types");
}

namespace {
void readDenseElementsWideNums(
    ElementsAttr elms, MutableArrayRef<WideNum> dst) {
  if (elms.isSplat())
    return std::fill(dst.begin(), dst.end(), getElementsSplatWideNum(elms));
  // TODO: Implement the following in a more efficient way.
  Type elementType = elms.getElementType();
  if (isa<FloatType>(elementType)) {
    llvm::transform(elms.getValues<APFloat>(), dst.begin(),
        [](APFloat f) { return WideNum::fromAPFloat(f); });
  } else if (auto itype = dyn_cast<IntegerType>(elementType)) {
    bool isSigned = !itype.isUnsigned();
    llvm::transform(elms.getValues<APInt>(), dst.begin(),
        [isSigned](APInt i) { return WideNum::fromAPInt(i, isSigned); });
  } else {
    llvm_unreachable("WideNum only supports integer and float types");
  }
}
} // namespace

// Returns a pointer to the underlying data as a flat WideNum array, if
// everything aligns, otherwise makes and returns a copy.
// Precondition: elms.getElementType.isIntOrFloat().
ArrayBuffer<WideNum> getElementsWideNums(ElementsAttr elms) {
  if (auto disposable = mlir::dyn_cast<DisposableElementsAttr>(elms))
    return disposable.getWideNums();

  // Return raw data if non-splat DenseElementsAttr and element type is wide.
  if (auto dense = mlir::dyn_cast<DenseElementsAttr>(elms)) {
    auto isWideType = [](Type t) { return t.isInteger(64) || t.isF64(); };
    if (isWideType(dense.getElementType()) && !dense.isSplat())
      return castArrayRef<WideNum>(dense.getRawData());
  }

  ArrayBuffer<WideNum>::Vector dst;
  dst.resize_for_overwrite(elms.size());
  readDenseElementsWideNums(elms, dst);
  return std::move(dst);
}

// Copies out the elements in a flat WideNum array in row-major order.
// Precondition: elms.getElementType.isIntOrFloat().
void readElementsWideNums(ElementsAttr elms, MutableArrayRef<WideNum> dst) {
  if (auto disposable = mlir::dyn_cast<DisposableElementsAttr>(elms))
    return disposable.readWideNums(dst);
  assert(dst.size() == static_cast<size_t>(elms.size()));
  readDenseElementsWideNums(elms, dst);
}

SmallVector<IntegerElementRange> getIntegerElementRanges(
    ElementsAttr elms, size_t groupSize) {
  assert(isa<IntegerType>(elms.getElementType()) &&
         !elms.getElementType().isInteger(1) &&
         "expected a non-boolean integer element type");
  assert(groupSize > 0 && "group size must be positive");
  if (elms.size() == 0)
    return {};
  size_t rangeCount =
      std::min(groupSize, static_cast<size_t>(elms.size()));

  if (elms.isSplat()) {
    APInt value = elms.getSplatValue<APInt>();
    SmallVector<IntegerElementRange> ranges;
    ranges.reserve(rangeCount);
    for (size_t i = 0; i < rangeCount; ++i)
      ranges.push_back({value, value, i, i});
    return ranges;
  }

  BType elementBType = btypeOfMlirType(elms.getElementType());
  return dispatchByBType(elementBType,
      [&](auto btype) -> SmallVector<IntegerElementRange> {
    constexpr BType tag = decltype(btype)::value;
    if constexpr (BTypeTrait<tag>::isInt && tag != BType::BOOL) {
      using cpptype = CppType<tag>;
      using widetype = WideType<tag>;
      ArrayBuffer<cpptype> buffer = getElementsArray<cpptype>(elms);
      ArrayRef<cpptype> values = buffer.get();

      struct NativeRange {
        widetype min;
        widetype max;
        size_t minIndex;
        size_t maxIndex;
      };
      SmallVector<NativeRange> nativeRanges;
      nativeRanges.reserve(rangeCount);
      for (size_t i = 0; i < rangeCount; ++i) {
        widetype value = static_cast<widetype>(values[i]);
        nativeRanges.push_back({value, value, i, i});
      }
      for (size_t i = groupSize; i < values.size(); ++i) {
        widetype value = static_cast<widetype>(values[i]);
        NativeRange &range = nativeRanges[i % groupSize];
        if (value < range.min) {
          range.min = value;
          range.minIndex = i;
        }
        if (value > range.max) {
          range.max = value;
          range.maxIndex = i;
        }
      }

      SmallVector<IntegerElementRange> ranges;
      ranges.reserve(rangeCount);
      for (const NativeRange &range : nativeRanges) {
        constexpr unsigned bitWidth = BTypeTrait<tag>::bitwidth;
        constexpr bool isSigned = BTypeTrait<tag>::isSignedInt;
        ranges.push_back(
            {APInt(bitWidth, static_cast<uint64_t>(range.min), isSigned),
                APInt(bitWidth, static_cast<uint64_t>(range.max), isSigned),
                range.minIndex, range.maxIndex});
      }
      return ranges;
    } else {
      llvm_unreachable("expected a non-boolean integer element type");
    }
  });
}

std::optional<IntegerElementValue> findIntegerRangeViolation(
    const IntegerElementRange &range, int64_t signedMin, int64_t signedMax,
    bool isUnsigned) {
  if (isUnsigned) {
    if (signedMax < 0 ||
        range.max.getZExtValue() > static_cast<uint64_t>(signedMax))
      return IntegerElementValue{range.max, range.maxIndex};
    return std::nullopt;
  }

  bool minIsInvalid = range.min.getSExtValue() < signedMin;
  bool maxIsInvalid = range.max.getSExtValue() > signedMax;
  if (!minIsInvalid && !maxIsInvalid)
    return std::nullopt;
  if (minIsInvalid && (!maxIsInvalid || range.minIndex < range.maxIndex))
    return IntegerElementValue{range.min, range.minIndex};
  return IntegerElementValue{range.max, range.maxIndex};
}

} // namespace onnx_mlir
