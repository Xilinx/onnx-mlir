// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
#ifndef ONNX_MLIR_CALLER_OWNED_DATA_ATTR_HPP
#define ONNX_MLIR_CALLER_OWNED_DATA_ATTR_HPP

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
// The caller owns callerOwnedData and keeps it alive and bytewise immutable
// until this attribute's MLIRContext is destroyed. Cloning preserves the same
// non-owning view. Assembly prints and bytecode stores an empty marker, so
// parsed IR has an empty view.
struct CallerOwnedDataAttributeStorage : AttributeStorage {
  using CallerOwnedData = llvm::ArrayRef<char>;
  using KeyTy = CallerOwnedData;
  CallerOwnedDataAttributeStorage(CallerOwnedData callerOwnedData)
      : callerOwnedData(callerOwnedData) {}
  bool operator==(const KeyTy &key) const {
    return callerOwnedData.data() == key.data() &&
           callerOwnedData.size() == key.size();
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.data(), key.size());
  }
  static CallerOwnedDataAttributeStorage *construct(
      AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<CallerOwnedDataAttributeStorage>())
        CallerOwnedDataAttributeStorage(key);
  }
  CallerOwnedData callerOwnedData;
};

class CallerOwnedDataAttr : public Attribute::AttrBase<CallerOwnedDataAttr,
                                Attribute, CallerOwnedDataAttributeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "onnx.caller_owned_data";
  static CallerOwnedDataAttr get(MLIRContext *context,
      CallerOwnedDataAttributeStorage::CallerOwnedData callerOwnedData = {}) {
    return Base::get(context, callerOwnedData);
  }
  [[nodiscard]] CallerOwnedDataAttributeStorage::CallerOwnedData
  getData() const {
    return getImpl()->callerOwnedData;
  }
};
} // namespace mlir
#endif
