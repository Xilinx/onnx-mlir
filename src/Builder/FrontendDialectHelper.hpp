/*
 * SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
 */

//===--------------------- FrontendDialectHelper.hpp ----------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// Helper methods for handling input ONNX models.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_FRONTEND_HELPER_H
#define ONNX_MLIR_FRONTEND_HELPER_H

#include "mlir/IR/BuiltinAttributeInterfaces.h"

#include "onnx/onnx_pb.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>
#include <string>

namespace onnx_mlir {

// Shared whole-file resolver, also used by eager import. Opening is explicit;
// callers can report recoverable diagnostics instead of a fatal decoder error.
llvm::ErrorOr<std::shared_ptr<const llvm::MemoryBuffer>> getExternalDataFile(
    llvm::StringRef path);

// Decode an already resolved external byte range using the importer's existing
// numerical decoders. The shared buffer remains owned by the resulting attr.
// Precondition: [offset, offset + length) is within buffer.
mlir::ElementsAttr externalTensorBytesToElmAttr(mlir::RankedTensorType type,
    int dataType, std::shared_ptr<const llvm::MemoryBuffer> buffer,
    uint64_t offset, uint64_t length);

// Decode a caller-owned in-memory byte range. The decoder copies the data into
// the resulting attribute and does not retain the range.
mlir::ElementsAttr externalTensorBytesToElmAttr(
    mlir::RankedTensorType type, int dataType, llvm::ArrayRef<char> bytes);

mlir::ElementsAttr onnxTensorProtoToElmAttr(mlir::MLIRContext *ctx,
    const std::string &externalDataDir, const onnx::TensorProto &initializer);

} // namespace onnx_mlir
#endif
