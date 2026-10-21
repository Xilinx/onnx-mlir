// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
#ifndef ONNX_MLIR_INITIALIZER_SOURCE_ATTR_HPP
#define ONNX_MLIR_INITIALIZER_SOURCE_ATTR_HPP

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
// The caller owns live storage and keeps it alive and bytewise immutable until
// this attribute's MLIRContext is destroyed. It is deliberately absent from
// assembly and bytecode. Cloning IR copies the non-owning view; parsed IR
// contains only the disk dictionary.
struct InitializerSourceAttributeStorage : AttributeStorage {
  using LiveData = llvm::ArrayRef<char>;
  using KeyTy = std::pair<DictionaryAttr, LiveData>;
  InitializerSourceAttributeStorage(DictionaryAttr disk, LiveData live)
      : disk(disk), live(live) {}
  bool operator==(const KeyTy &key) const {
    return disk == key.first && live.data() == key.second.data() &&
           live.size() == key.second.size();
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second.data(), key.second.size());
  }
  static InitializerSourceAttributeStorage *construct(
      AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<InitializerSourceAttributeStorage>())
        InitializerSourceAttributeStorage(key.first, key.second);
  }
  DictionaryAttr disk;
  LiveData live;
};

class InitializerSourceAttr
    : public Attribute::AttrBase<InitializerSourceAttr, Attribute,
          InitializerSourceAttributeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "onnx.initializer_source";
  static InitializerSourceAttr get(DictionaryAttr disk,
      InitializerSourceAttributeStorage::LiveData live = {}) {
    return Base::get(disk.getContext(), disk, live);
  }
  DictionaryAttr getDisk() const { return getImpl()->disk; }
  InitializerSourceAttributeStorage::LiveData getLive() const {
    return getImpl()->live;
  }
};
} // namespace mlir

namespace onnx_mlir {
// Shared by the op verifier and the importer/materializer. Keep source-type
// acceptance and payload sizing in the dialect layer so those users cannot
// drift apart.
bool isSupportedInitializerSourceDataType(int64_t dataType);
mlir::FailureOr<uint64_t> getInitializerSourcePayloadSize(
    mlir::RankedTensorType type);
} // namespace onnx_mlir
#endif
