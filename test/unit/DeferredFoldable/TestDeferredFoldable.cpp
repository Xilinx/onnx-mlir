/*
 * SPDX-License-Identifier: Apache-2.0
 * Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
 * affiliates
 */

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ConstProp.hpp"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace onnx_mlir;

namespace {

OwningOpRef<ModuleOp> parseModule(MLIRContext &context, StringRef body) {
  return parseSourceString<ModuleOp>(body, &context);
}

OwningOpRef<ModuleOp> parseModuleUnverified(
    MLIRContext &context, StringRef body) {
  ParserConfig config(&context, /*verifyAfterParse=*/false);
  return parseSourceString<ModuleOp>(body, config);
}

ONNXDeferredFoldableOp findDeferred(ModuleOp module) {
  ONNXDeferredFoldableOp result;
  module.walk([&](ONNXDeferredFoldableOp op) {
    if (!result)
      result = op;
  });
  return result;
}

int testDirectFormation(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> tensor<2x2xf32> {
        %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
        %transposed = "onnx.Transpose"(%weight) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %transposed : tensor<2x2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  ONNXTransposeOp transpose;
  module->walk([&](ONNXTransposeOp op) { transpose = op; });
  llvm::DenseMap<Value, bool> knownValues;
  if (isConstantFoldable(transpose.getResult()) ||
      !isSemanticallyConstantFoldable(transpose.getResult(), knownValues))
    return 1;
  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(transpose);
  FailureOr<Value> formed = formDeferredFoldableExpression(transpose, rewriter);
  if (failed(formed) || !formed->getDefiningOp<ONNXDeferredFoldableOp>() ||
      !isConstantFoldable(*formed) ||
      rewriter.getInsertionBlock() != formed->getParentBlock() ||
      failed(verify(*module)))
    return 1;

  SmallString<1024> bytecode;
  llvm::raw_svector_ostream output(bytecode);
  if (failed(writeBytecodeToFile(*module, output)))
    return 1;
  OwningOpRef<ModuleOp> roundTripped =
      parseSourceString<ModuleOp>(bytecode, &context);
  if (!roundTripped || !findDeferred(*roundTripped))
    return 1;
  return 0;
}

int testConstantExpressionFormation(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> tensor<2x2xf32> {
        %weight = onnx.Constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
        %transposed = "onnx.Transpose"(%weight) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %transposed : tensor<2x2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  ONNXTransposeOp transpose;
  module->walk([&](ONNXTransposeOp op) { transpose = op; });
  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(transpose);
  FailureOr<Value> formed = formDeferredFoldableExpression(transpose, rewriter);
  if (failed(formed))
    return 1;
  auto deferred = formed->getDefiningOp<ONNXDeferredFoldableOp>();
  if (!deferred || !isConstantFoldable(*formed) ||
      !isa<ONNXConstantOp>(deferred.getCaptures().front().getDefiningOp()) ||
      failed(verify(*module)))
    return 1;
  return 0;
}

int testNormalizeDuplicateCaptureUses(MLIRContext &context) {
  auto module = parseModuleUnverified(context, R"mlir(
    module {
      func.func @test() -> tensor<2x2xf32> {
        %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
        %cast = "onnx.Cast"(%weight) {saturate = 1 : si64, to = f32} : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %result = "onnx.DeferredFoldable"(%cast, %cast) ({
        ^bb0(%lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>):
          %sum = "onnx.Add"(%lhs, %rhs) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
          onnx.DeferredYield %sum : tensor<2x2xf32>
        }) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        return %result : tensor<2x2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  RewritePatternSet patterns(&context);
  populateFormDeferredFoldableExpressionPatterns(patterns, PatternBenefit(1));
  if (failed(applyPatternsGreedily(*module, std::move(patterns))))
    return 1;

  ONNXDeferredFoldableOp deferred = findDeferred(*module);
  if (!deferred || deferred.getCaptures().size() != 1 ||
      !isa<ONNXInitializerSourceOp>(
          deferred.getCaptures().front().getDefiningOp()) ||
      failed(verify(*module)))
    return 1;
  return 0;
}

int testCrossDialectFormation(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> tensor<?x2xf32> {
        %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
        %cast = tensor.cast %weight : tensor<2x2xf32> to tensor<?x2xf32>
        return %cast : tensor<?x2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  tensor::CastOp cast;
  module->walk([&](tensor::CastOp op) { cast = op; });
  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(cast);
  FailureOr<Value> formed = formDeferredFoldableExpression(cast, rewriter);
  if (failed(formed))
    return 1;
  auto deferred = formed->getDefiningOp<ONNXDeferredFoldableOp>();
  if (!deferred || !isa<tensor::CastOp>(deferred.getTerminalOperation()) ||
      deferred.getCaptures().size() != 1 ||
      !isa<ONNXInitializerSourceOp>(
          deferred.getCaptures().front().getDefiningOp()) ||
      rewriter.getInsertionBlock() != formed->getParentBlock() ||
      failed(verify(*module)))
    return 1;
  return 0;
}

int testFlattenSharedDeferredOperandDAG(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> tensor<2x2xf32> {
        %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
        %transposed = "onnx.DeferredFoldable"(%weight) ({
        ^bb0(%arg0: tensor<2x2xf32>):
          %0 = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
          onnx.DeferredYield %0 : tensor<2x2xf32>
        }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %sum = "onnx.Add"(%transposed, %transposed) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        return %sum : tensor<2x2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  ONNXAddOp add;
  module->walk([&](ONNXAddOp op) { add = op; });
  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(add);
  FailureOr<Value> formed = formDeferredFoldableExpressionDAG(add, rewriter);
  if (failed(formed) || failed(verify(*module)))
    return 1;

  auto deferred = formed->getDefiningOp<ONNXDeferredFoldableOp>();
  int64_t numDeferred = 0;
  module->walk([&](ONNXDeferredFoldableOp) { ++numDeferred; });
  if (!deferred || deferred.getCaptures().size() != 1 || numDeferred != 1)
    return 1;
  auto transposes = deferred.getBody().front().getOps<ONNXTransposeOp>();
  if (std::distance(transposes.begin(), transposes.end()) != 1)
    return 1;
  auto adds = deferred.getBody().front().getOps<ONNXAddOp>();
  if (std::distance(adds.begin(), adds.end()) != 1)
    return 1;
  ONNXAddOp clonedAdd = *adds.begin();
  if (clonedAdd.getA() != clonedAdd.getB() ||
      rewriter.getInsertionBlock() != formed->getParentBlock())
    return 1;
  return 0;
}

int testRejectNondeterministicFormation(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> tensor<2x2xf32> {
        %probabilities = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "probabilities", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
        %random = "onnx.Bernoulli"(%probabilities) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %random : tensor<2x2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  ONNXBernoulliOp random;
  module->walk([&](ONNXBernoulliOp op) { random = op; });
  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(random);
  if (isPure(random) ||
      succeeded(formDeferredFoldableExpression(random, rewriter)) ||
      rewriter.getInsertionPoint() != random->getIterator() ||
      failed(verify(*module)))
    return 1;
  return 0;
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

int testInlineCompleteRecipe(MLIRContext &context) {
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
  auto inlined = inlineDeferredFoldableExpression(deferred, rewriter);
  if (failed(inlined) || inlined->size() != 2 ||
      !isa<ONNXTransposeOp>((*inlined)[0]) || !isa<ONNXTileOp>((*inlined)[1]) ||
      findDeferred(*module) || failed(verify(*module)))
    return 1;
  return 0;
}

} // namespace

int main() {
  MLIRContext context;
  context.loadDialect<func::FuncDialect, ONNXDialect, tensor::TensorDialect>();
  int failures = testDirectFormation(context) +
                 testConstantExpressionFormation(context) +
                 testNormalizeDuplicateCaptureUses(context) +
                 testCrossDialectFormation(context) +
                 testFlattenSharedDeferredOperandDAG(context) +
                 testRejectNondeterministicFormation(context) +
                 testPeelOneInternalOperand(context) +
                 testRejectMultipleInternalOperands(context) +
                 testInlineCompleteRecipe(context);
  if (failures)
    llvm::errs() << failures << " DeferredFoldable test(s) failed\n";
  return failures;
}
