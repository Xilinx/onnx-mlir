/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ElementsAttrHelper.hpp -----------------------===//
//
// Helper functions for accessing ElementsAttr contents.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ELEM_ATTR_HELPER_H
#define ONNX_MLIR_ELEM_ATTR_HELPER_H

#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ElementsAttr/WideNum.hpp"
#include "src/Support/Arrays.hpp"

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstddef>
#include <optional>

namespace onnx_mlir {

// Preconditions: elms.isSplat() and elms.getElementType().isIntOrFloat().
WideNum getElementsSplatWideNum(mlir::ElementsAttr elms);

// Returns a pointer to the underlying data as a flat aray, if
// everything aligns, otherwise makes and returns a copy.
// Presents the same view of the data as elms.getValues() but as a
// contiguous array and only for the type X matching the element type.
// Precondition: X must correspond to elms.getElementType().
template <typename X>
ArrayBuffer<X> getElementsArray(mlir::ElementsAttr elms);

// Copies out the elements in a flat array in row-major order.
// Precondition: X must correspond to elms.getElementType().
template <typename X>
void readElementsArray(mlir::ElementsAttr elms, llvm::MutableArrayRef<X> dst);

// Returns a pointer to the underlying data as a flat WideNum array, if
// everything aligns, otherwise makes and returns a copy.
// Precondition: elms.getElementType().isIntOrFloat().
ArrayBuffer<WideNum> getElementsWideNums(mlir::ElementsAttr elms);

// Copies out the elements in a flat WideNum array in row-major order.
// Precondition: elms.getElementType().isIntOrFloat().
void readElementsWideNums(
    mlir::ElementsAttr elms, llvm::MutableArrayRef<WideNum> dst);

// The minimum and maximum integer values at flat indices congruent modulo a
// group size. Tracking their positions lets callers report which element is
// invalid without scanning the attribute again.
struct IntegerElementRange {
  llvm::APInt min;
  llvm::APInt max;
  size_t minIndex;
  size_t maxIndex;
};

// Finds the integer value range in each interleaved group. Group g contains
// flat indices g, g + groupSize, g + 2 * groupSize, and so on. Returns no
// ranges for an empty attribute.
//
// Reads DenseElementsAttr and DisposableElementsAttr through their bulk native
// integer views, avoiding the per-element APInt construction performed by
// ElementsAttr::getValues<APInt>(). A splat is read once.
//
// Preconditions: elms has integer element type other than i1 and groupSize >
// 0. The final set of groups may be incomplete, which permits verification to
// diagnose malformed IR whose constant value and result type disagree.
llvm::SmallVector<IntegerElementRange> getIntegerElementRanges(
    mlir::ElementsAttr elms, size_t groupSize = 1);

struct IntegerElementValue {
  llvm::APInt value;
  size_t index;
};

// Returns an extreme value outside [signedMin, signedMax], if there is one.
// The lower bound is ignored for unsigned elements. If both extremes are out
// of range, returns the one occurring first in the flattened attribute.
std::optional<IntegerElementValue> findIntegerRangeViolation(
    const IntegerElementRange &range, int64_t signedMin, int64_t signedMax,
    bool isUnsigned);

// Include template implementations.
#include "ElementsAttrHelper.hpp.inc"

} // namespace onnx_mlir
#endif
