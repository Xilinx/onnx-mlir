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

namespace onnx {
class TensorProto;
}

namespace mlir {
// The caller owns the referenced bytes or TensorProto and keeps them alive and
// bytewise immutable until this attribute's MLIRContext is destroyed. Cloning
// preserves the same non-owning reference. Assembly prints and bytecode stores
// an empty marker, so parsed IR has no process-local reference.
struct CallerOwnedDataAttributeStorage : AttributeStorage {
  struct CallerOwnedData {
    llvm::ArrayRef<char> bytes;
    const onnx::TensorProto *tensor = nullptr;

    bool operator==(const CallerOwnedData &other) const {
      return bytes.data() == other.bytes.data() &&
             bytes.size() == other.bytes.size() && tensor == other.tensor;
    }
  };
  using KeyTy = CallerOwnedData;
  CallerOwnedDataAttributeStorage(CallerOwnedData callerOwnedData)
      : callerOwnedData(callerOwnedData) {}
  bool operator==(const KeyTy &key) const { return callerOwnedData == key; }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.bytes.data(), key.bytes.size(), key.tensor);
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
  static CallerOwnedDataAttr get(
      MLIRContext *context, llvm::ArrayRef<char> callerOwnedData = {}) {
    return Base::get(context,
        CallerOwnedDataAttributeStorage::CallerOwnedData{callerOwnedData});
  }
  static CallerOwnedDataAttr get(
      MLIRContext *context, const onnx::TensorProto *callerOwnedTensor) {
    return Base::get(context, CallerOwnedDataAttributeStorage::CallerOwnedData{
                                  {}, callerOwnedTensor});
  }
  [[nodiscard]] llvm::ArrayRef<char> getData() const {
    return getImpl()->callerOwnedData.bytes;
  }
  [[nodiscard]] const onnx::TensorProto *getTensor() const {
    return getImpl()->callerOwnedData.tensor;
  }
};
} // namespace mlir
#endif
