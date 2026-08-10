// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#ifndef ONNX_MLIR_SUB_BYTE_UTILS_H
#define ONNX_MLIR_SUB_BYTE_UTILS_H

#include <cstdint>
#include <limits>

#include "mlir/IR/BuiltinAttributes.h"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "llvm/ADT/SmallVector.h"

namespace onnx_mlir {

// Unpacks a uint8 attribute where the values are actually `bits`-wide values
// packed as uint8s along the last dimension. This is how the com.microsoft
// operators store sub-byte data; it is not ONNX's native int4/uint4 storage.
// The values are returned zero-extended as `T`, so callers still have to
// subtract the zero point.
template <typename T>
llvm::SmallVector<T> unpackSubByteValues(
    mlir::Attribute packedAttr, int64_t bits) {
  static_assert(std::numeric_limits<T>::max() >= 255,
      "T has to represent every zero-extended value of a full byte");

  mlir::DenseElementsAttr values;
  if (auto disposable =
          mlir::dyn_cast<mlir::DisposableElementsAttr>(packedAttr)) {
    values = disposable.toDenseElementsAttr();
  } else {
    values = mlir::cast<mlir::DenseElementsAttr>(packedAttr);
  }

  // Perform the unpacking, low-order bits first:
  // bits = 2: 1xuint8 0bAABBCCDD => 4 values 0bDD 0bCC 0bBB 0bAA
  // bits = 4: 1xuint8 0bAAAABBBB => 2 values 0bBBBB 0bAAAA
  llvm::SmallVector<T> unpackedValues;
  unpackedValues.reserve(values.getNumElements() * 8 / bits);
  const uint64_t mask = (uint64_t(1) << bits) - 1;
  for (llvm::APInt packed : values.getValues<llvm::APInt>())
    for (int64_t j = 0; j < 8 / bits; j++)
      unpackedValues.push_back(T((packed.getZExtValue() >> (j * bits)) & mask));

  return unpackedValues;
}

} // namespace onnx_mlir
#endif
