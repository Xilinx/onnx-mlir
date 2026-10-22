// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
#ifndef ONNX_MLIR_BUILDER_INITIALIZER_SOURCE_HPP
#define ONNX_MLIR_BUILDER_INITIALIZER_SOURCE_HPP
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Dialect/ONNX/CallerOwnedDataAttr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {
// Converts individual initializer sources on demand. Reusing one instance
// shares its parsed backup-model cache across materializations.
class InitializerSourceMaterializer {
public:
  InitializerSourceMaterializer();
  ~InitializerSourceMaterializer();

  mlir::FailureOr<mlir::ONNXConstantOp> materialize(
      mlir::ONNXInitializerSourceOp op, mlir::RewriterBase &rewriter);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

std::string getInitializerIdentity(
    mlir::MLIRContext *context, const InitializerIdentity &identity);
mlir::FailureOr<mlir::ONNXInitializerSourceOp> createInitializerSource(
    mlir::OpBuilder &builder, const onnx::TensorProto &tensor,
    mlir::RankedTensorType type, llvm::ArrayRef<std::string> scope,
    const ImportOptions &options, mlir::Location loc,
    bool retainTensorReference = false);
std::unique_ptr<mlir::Pass> createMaterializeInitializerSourcesPass(
    bool skipAll, llvm::ArrayRef<std::string> skipNames,
    llvm::ArrayRef<std::string> forceNames = {});
} // namespace onnx_mlir
#endif
