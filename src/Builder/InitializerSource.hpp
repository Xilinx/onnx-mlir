// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
#ifndef ONNX_MLIR_BUILDER_INITIALIZER_SOURCE_HPP
#define ONNX_MLIR_BUILDER_INITIALIZER_SOURCE_HPP
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include <memory>

namespace onnx_mlir {
struct InitializerSourceCache;

// Reads one source and replaces all its uses with a compiler-owned constant.
// Preserves discardable metadata. On failure, leaves the source unchanged.
mlir::FailureOr<mlir::ONNXConstantOp> materializeInitializerSource(
    mlir::ONNXInitializerSourceOp op, mlir::RewriterBase &rewriter,
    InitializerSourceCache &cache);

// Reuse a cache across materializations to parse each backup model only once.
// The returned handle owns the cache without exposing its implementation.
std::shared_ptr<InitializerSourceCache> createInitializerSourceCache();

std::string getInitializerIdentity(
    mlir::MLIRContext *context, const InitializerIdentity &identity);

// Records a durable locator and optional caller-owned data without reading
// tensor contents. Does not open the external data file or backup model.
mlir::FailureOr<mlir::ONNXInitializerSourceOp> createInitializerSource(
    mlir::OpBuilder &builder, const onnx::TensorProto &tensor,
    mlir::RankedTensorType type, llvm::ArrayRef<std::string> scope,
    const ImportOptions &options, mlir::Location loc);
std::unique_ptr<mlir::Pass> createMaterializeInitializerSourcesPass(
    bool skipAll, llvm::ArrayRef<std::string> skipNames);
} // namespace onnx_mlir
#endif
