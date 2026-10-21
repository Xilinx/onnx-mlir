// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
#ifndef ONNX_MLIR_BUILDER_INITIALIZER_SOURCE_HPP
#define ONNX_MLIR_BUILDER_INITIALIZER_SOURCE_HPP
#include "mlir/Pass/Pass.h"
#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Dialect/ONNX/InitializerSourceAttr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {
mlir::FailureOr<mlir::InitializerSourceAttr> createInitializerSource(
    const onnx::TensorProto &tensor, mlir::RankedTensorType type,
    llvm::StringRef scope, const ImportOptions &options, mlir::Location loc);
std::unique_ptr<mlir::Pass> createMaterializeInitializerSourcesPass(
    bool skipAll = false, llvm::ArrayRef<std::string> skipNames = {});
} // namespace onnx_mlir
#endif
