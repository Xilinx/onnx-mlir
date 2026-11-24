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

// Read `op`'s tensor contents without modifying `op` or its uses.
// May add parsed source models to `cache`.
// Returns an attribute that retains its backing storage.
//
// Note: callers that specialize compiled code using these contents must use
// `materializeInitializerSource` instead, which replaces `op` with a constant.
mlir::FailureOr<mlir::ElementsAttr> loadInitializerSource(
    mlir::ONNXInitializerSourceOp op, InitializerSourceCache &cache);

// Reads one source and replaces all its uses with a compiler-owned constant,
// making the initializer part of the compiled-model semantics rather than a
// replaceable source. Preserves discardable metadata. On failure, leaves the
// source unchanged.
mlir::FailureOr<mlir::ONNXConstantOp> materializeInitializerSource(
    mlir::ONNXInitializerSourceOp op, mlir::RewriterBase &rewriter,
    InitializerSourceCache &cache);

// Reuse a cache across materializations to parse each source model only once.
// The returned handle owns the cache without exposing its implementation.
std::shared_ptr<InitializerSourceCache> createInitializerSourceCache();

std::string getInitializerIdentity(
    mlir::MLIRContext *context, const InitializerIdentity &identity);

// Records a durable locator and optional caller-owned data without reading
// tensor contents. Does not open the external data file or source model.
// isInput records membership in the owning graph's input list.
mlir::FailureOr<mlir::ONNXInitializerSourceOp> createInitializerSource(
    mlir::OpBuilder &builder, const onnx::TensorProto &tensor,
    mlir::RankedTensorType type, llvm::ArrayRef<std::string> scope,
    const ImportOptions &options, mlir::Location loc, bool isInput,
    bool referenceCallerOwnedTensor = false);
std::unique_ptr<mlir::Pass> createMaterializeInitializerSourcesPass(
    bool skipAll, llvm::ArrayRef<std::string> skipNames,
    llvm::ArrayRef<std::string> forceNames = {});
} // namespace onnx_mlir
#endif
