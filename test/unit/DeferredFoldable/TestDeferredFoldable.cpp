/*
 * SPDX-License-Identifier: Apache-2.0
 * Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
 * affiliates
 */

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace onnx_mlir;

namespace {

OwningOpRef<ModuleOp> parseModule(MLIRContext &context, StringRef body) {
  return parseSourceString<ModuleOp>(body, &context);
}

ONNXDeferredFoldableOp findDeferred(ModuleOp module) {
  ONNXDeferredFoldableOp result;
  module.walk([&](ONNXDeferredFoldableOp op) {
    if (!result)
      result = op;
  });
  return result;
}

int testPeelOneInternalOperand(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> tensor<4x4xf32> {
        %arg0 = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
        %repeats = onnx.Constant dense<[2, 2]> : tensor<2xi64>
        %0 = "onnx.DeferredFoldable"(%arg0, %repeats) ({
        ^bb0(%arg1: tensor<2x2xf32>, %arg2: tensor<2xi64>):
          %1 = "onnx.Transpose"(%arg1) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
          %2 = "onnx.Tile"(%1, %arg2) : (tensor<2x2xf32>, tensor<2xi64>) -> tensor<4x4xf32>
          onnx.DeferredYield %2 : tensor<4x4xf32>
        }) : (tensor<2x2xf32>, tensor<2xi64>) -> tensor<4x4xf32>
        return %0 : tensor<4x4xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  ONNXDeferredFoldableOp deferred = findDeferred(*module);
  IRRewriter rewriter(&context);
  auto peeled = peelDeferredFoldableExpression(deferred, rewriter);
  if (failed(peeled) || !isa<ONNXTileOp>(*peeled) || failed(verify(*module)))
    return 1;
  if (!isa<ONNXTransposeOp>(deferred.getTerminalOperation()))
    return 1;
  return 0;
}

int testRejectMultipleInternalOperands(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> tensor<2x2xf32> {
        %arg0 = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
        %0 = "onnx.DeferredFoldable"(%arg0) ({
        ^bb0(%arg1: tensor<2x2xf32>):
          %1 = "onnx.Transpose"(%arg1) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
          %2 = "onnx.Identity"(%arg1) : (tensor<2x2xf32>) -> tensor<2x2xf32>
          %3 = "onnx.Add"(%1, %2) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
          onnx.DeferredYield %3 : tensor<2x2xf32>
        }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %0 : tensor<2x2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  ONNXDeferredFoldableOp deferred = findDeferred(*module);
  IRRewriter rewriter(&context);
  if (succeeded(peelDeferredFoldableExpression(deferred, rewriter)))
    return 1;
  return failed(verify(*module));
}

} // namespace

int main() {
  MLIRContext context;
  context.loadDialect<func::FuncDialect, ONNXDialect>();
  int failures = testPeelOneInternalOperand(context) +
                 testRejectMultipleInternalOperands(context);
  if (failures)
    llvm::errs() << failures << " DeferredFoldable test(s) failed\n";
  return failures;
}
