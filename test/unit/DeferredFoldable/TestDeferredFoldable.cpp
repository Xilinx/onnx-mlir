/*
 * SPDX-License-Identifier: Apache-2.0
 * Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
 * affiliates
 */

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/TensorName.hpp"
#include "src/Dialect/ONNX/Transforms/ConstProp.hpp"
#include "src/Interface/DequantizationOpInterface.hpp"

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

static_assert(
    !ONNXQuantizeLinearOp::hasTrait<DequantizationOpInterface::Trait>());
static_assert(
    ONNXDequantizeLinearOp::hasTrait<DequantizationOpInterface::Trait>());
static_assert(
    !ONNXDynamicQuantizeLinearOp::hasTrait<DequantizationOpInterface::Trait>());
static_assert(AMDQuarkBFPQuantizeDequantizeOp::hasTrait<
    DequantizationOpInterface::Trait>());
static_assert(!AMDQuarkExtendedQuantizeLinearOp::hasTrait<
    DequantizationOpInterface::Trait>());
static_assert(AMDQuarkExtendedDequantizeLinearOp::hasTrait<
    DequantizationOpInterface::Trait>());

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
  if (isConstantFoldable(transpose.getResult()))
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

int testDequantFoldability(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    func.func @test(%runtimeScale: tensor<f32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xi8>, tensor<2xf32>) {
      %w = onnx.Constant dense<[4, 8]> : tensor<2xi8>
      %s = onnx.Constant dense<0.5> : tensor<f32>
      %z = onnx.Constant dense<0> : tensor<i8>
      %a = "onnx.DequantizeLinear"(%w, %s, %z) : (tensor<2xi8>, tensor<f32>, tensor<i8>) -> tensor<2xf32>
      %b = "onnx.DequantizeLinear"(%w, %runtimeScale, %z) : (tensor<2xi8>, tensor<f32>, tensor<i8>) -> tensor<2xf32>
      %q, %dq = "onnx.DeferredFoldable"(%w, %s, %z) ({
      ^bb0(%data: tensor<2xi8>, %scale: tensor<f32>, %zero: tensor<i8>):
        %d = "onnx.DequantizeLinear"(%data, %scale, %zero) : (tensor<2xi8>, tensor<f32>, tensor<i8>) -> tensor<2xf32>
        %q = "onnx.QuantizeLinear"(%d, %scale, %zero) : (tensor<2xf32>, tensor<f32>, tensor<i8>) -> tensor<2xi8>
        onnx.Yield %q, %d : tensor<2xi8>, tensor<2xf32>
      }) : (tensor<2xi8>, tensor<f32>, tensor<i8>) -> (tensor<2xi8>, tensor<2xf32>)
      return %a, %b, %q, %dq : tensor<2xf32>, tensor<2xf32>, tensor<2xi8>, tensor<2xf32>
    }
  )mlir");
  if (!module)
    return 1;
  auto function = module->lookupSymbol<func::FuncOp>("test");
  auto values = function.front().getTerminator()->getOperands();
  return !isConstantFoldable(values[0]) || isConstantFoldable(values[1]) ||
         !isConstantFoldable(values[2]) || !isConstantFoldable(values[3]) ||
         isConstantFoldableExceptDequant(values[0]) ||
         isConstantFoldableExceptDequant(values[1]) ||
         !isConstantFoldableExceptDequant(values[2]) ||
         isConstantFoldableExceptDequant(values[3]);
}

int testConstantExpressionFormation(MLIRContext &context, bool disposable) {
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

  if (disposable) {
    OnnxElementsAttrBuilder elementsBuilder(&context);
    module->walk([&](ONNXConstantOp constant) {
      auto dense = cast<DenseElementsAttr>(constant.getValueAttr());
      constant.setValueAttr(elementsBuilder.toDisposableElementsAttr(dense));
    });
  }

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
          onnx.Yield %sum : tensor<2x2xf32>
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

int testBottomUpFormationWithSharedOperand(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> tensor<2x2xf32> {
        %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
        %transposed = "onnx.Transpose"(%weight) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %sum = "onnx.Add"(%transposed, %transposed) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        return %sum : tensor<2x2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  RewritePatternSet patterns(&context);
  populateFormDeferredFoldableExpressionPatterns(patterns, PatternBenefit(1));
  GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  if (failed(applyPatternsGreedily(*module, std::move(patterns), config)) ||
      failed(verify(*module)))
    return 1;

  ONNXDeferredFoldableOp deferred = findDeferred(*module);
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
  if (clonedAdd.getA() != clonedAdd.getB())
    return 1;
  return 0;
}

int testBottomUpFormationWithDenseSibling(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> tensor<2x2xf32> {
        %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
        %bias = onnx.Constant dense<1.0> : tensor<2x2xf32>
        %transposed_bias = "onnx.Transpose"(%bias) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %sum = "onnx.Add"(%weight, %transposed_bias) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        return %sum : tensor<2x2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  RewritePatternSet patterns(&context);
  populateFormDeferredFoldableExpressionPatterns(
      patterns, PatternBenefit(1), /*requireDeferredSource=*/false);
  if (failed(applyPatternsGreedily(*module, std::move(patterns))) ||
      failed(verify(*module)))
    return 1;

  ONNXDeferredFoldableOp deferred = findDeferred(*module);
  int64_t numDeferred = 0;
  module->walk([&](ONNXDeferredFoldableOp) { ++numDeferred; });
  if (!deferred || numDeferred != 1 || deferred.getCaptures().size() != 2)
    return 1;
  auto transposes = deferred.getBody().front().getOps<ONNXTransposeOp>();
  auto adds = deferred.getBody().front().getOps<ONNXAddOp>();
  if (std::distance(transposes.begin(), transposes.end()) != 1 ||
      std::distance(adds.begin(), adds.end()) != 1)
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
          onnx.Yield %2 : tensor<4x4xf32>
        }) : (tensor<2x2xf32>, tensor<2xi64>) -> tensor<4x4xf32>
        return %0 : tensor<4x4xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  ONNXDeferredFoldableOp deferred = findDeferred(*module);
  if (!canPeelDeferredFoldableExpression(deferred))
    return 1;
  IRRewriter rewriter(&context);
  auto peeled = peelDeferredFoldableExpression(deferred, rewriter);
  if (failed(peeled) || !isa<ONNXTileOp>(*peeled) || failed(verify(*module)))
    return 1;
  if (!isa<ONNXTransposeOp>(findDeferred(*module).getTerminalOperation()))
    return 1;
  return 0;
}

int testPeelMultipleInternalOperands(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> tensor<2x2xf32> {
        %arg0 = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
        %0 = "onnx.DeferredFoldable"(%arg0) ({
        ^bb0(%arg1: tensor<2x2xf32>):
          %1 = "onnx.Transpose"(%arg1) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
          %2 = "onnx.Identity"(%arg1) : (tensor<2x2xf32>) -> tensor<2x2xf32>
          %3 = "onnx.Add"(%1, %2) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
          onnx.Yield %3 : tensor<2x2xf32>
        }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %0 : tensor<2x2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  ONNXDeferredFoldableOp deferred = findDeferred(*module);
  if (!canPeelDeferredFoldableExpression(deferred))
    return 1;
  IRRewriter rewriter(&context);
  auto peeled = peelDeferredFoldableExpression(deferred, rewriter);
  if (failed(peeled) || !isa<ONNXAddOp>(*peeled))
    return 1;
  auto prefix = findDeferred(*module);
  if (!prefix || prefix.getNumResults() != 2 ||
      (*peeled)->getOperand(0) != prefix.getResult(0) ||
      (*peeled)->getOperand(1) != prefix.getResult(1))
    return 1;
  return failed(verify(*module));
}

int testPeelSelectedResultAndDeduplicateOperands(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) {
        %w = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "w", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
        %out:3 = "onnx.DeferredFoldable"(%w) ({
        ^bb0(%a: tensor<2x2xf32>):
          %t = "onnx.Transpose"(%a) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
          %other = "onnx.Identity"(%a) : (tensor<2x2xf32>) -> tensor<2x2xf32>
          %sum = "onnx.Add"(%t, %t) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
          onnx.Yield %other, %sum, %other : tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>
        }) : (tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
        return %out#0, %out#1, %out#2 : tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;
  auto deferred = findDeferred(*module);
  Builder attrs(&context);
  deferred->setAttr("onnx_node_name", attrs.getStringAttr("original"));
  deferred->setAttr(
      "ResultNames", attrs.getStrArrayAttr({"other", "selected", "alias"}));
  deferred->setAttr("test.marker", attrs.getUnitAttr());
  TensorName transposed("transposed");
  transposed.push_back(
      std::make_unique<TransposeTransform>(ArrayRef<int64_t>{2, 2},
          ArrayRef<int64_t>{1, 0}, ArrayRef<int64_t>{2, 2}));
  if (failed(transposed.setTo(deferred.getBody().front().front().getResult(0))))
    return 1;
  Value selected = deferred.getResult(1);
  if (!isa<ONNXAddOp>(deferred.getTerminalOperation(1)))
    return 1;
  IRRewriter rewriter(&context);
  auto peeled = peelDeferredFoldableExpression(selected, rewriter);
  if (failed(peeled) || !isa<ONNXAddOp>(*peeled))
    return 1;
  auto prefix = findDeferred(*module);
  auto function = module->lookupSymbol<func::FuncOp>("test");
  auto ret = cast<func::ReturnOp>(function.front().getTerminator());
  unsigned transposes = 0;
  module->walk([&](ONNXTransposeOp) { ++transposes; });
  if (prefix.getNumResults() != 3 || transposes != 1 ||
      (*peeled)->getOperand(0) != (*peeled)->getOperand(1) ||
      ret.getOperand(0) != prefix.getResult(0) ||
      ret.getOperand(2) != prefix.getResult(1) ||
      ret.getOperand(1) != (*peeled)->getResult(0))
    return 1;
  if (prefix->hasAttr("onnx_node_name") || !prefix->hasAttr("test.marker") ||
      prefix->getAttr("ResultNames") !=
          attrs.getArrayAttr({attrs.getStringAttr("other"),
              attrs.getStringAttr("alias"), transposed.toAttr(&context)}) ||
      (*peeled)->getAttr("ResultNames") != attrs.getStrArrayAttr({"selected"}))
    return 1;
  if (failed(verify(*module)))
    return 1;
  SmallString<1024> bytecode;
  llvm::raw_svector_ostream output(bytecode);
  if (failed(writeBytecodeToFile(*module, output)))
    return 1;
  auto roundTripped = parseSourceString<ModuleOp>(bytecode, &context);
  return !roundTripped || findDeferred(*roundTripped).getNumResults() != 3 ||
         failed(verify(*roundTripped));
}

int testPeelDQWithInternalParameters(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> tensor<4x2xf32> {
        %w = "onnx.InitializerSource"() {kind = "external_data", length = 8 : i64, name = "w", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x4xi8>
        %out = "onnx.DeferredFoldable"(%w) ({
        ^bb0(%a: tensor<2x4xi8>):
          %t = "onnx.Transpose"(%a) {perm = [1, 0]} : (tensor<2x4xi8>) -> tensor<4x2xi8>
          %s = onnx.Constant dense<0.125> : tensor<f32>
          %z = onnx.Constant dense<0> : tensor<i8>
          %dq = "onnx.DequantizeLinear"(%t, %s, %z) : (tensor<4x2xi8>, tensor<f32>, tensor<i8>) -> tensor<4x2xf32>
          onnx.Yield %dq : tensor<4x2xf32>
        }) : (tensor<2x4xi8>) -> tensor<4x2xf32>
        return %out : tensor<4x2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;
  IRRewriter rewriter(&context);
  auto peeled = peelDeferredFoldableExpression(findDeferred(*module), rewriter);
  if (failed(peeled) || !isa<ONNXDequantizeLinearOp>(*peeled))
    return 1;
  auto prefix = findDeferred(*module);
  if (prefix.getNumResults() != 1 ||
      (*peeled)->getOperand(0) != prefix.getResult(0))
    return 1;
  for (unsigned index : {1u, 2u})
    if (!isConstLikeValue((*peeled)->getOperand(index)) ||
        (*peeled)->getOperand(index).getDefiningOp()->getBlock() !=
            (*peeled)->getBlock())
      return 1;
  if (failed(verify(*module)))
    return 1;
  SmallString<1024> bytecode;
  llvm::raw_svector_ostream output(bytecode);
  if (failed(writeBytecodeToFile(*module, output)))
    return 1;
  auto roundTripped = parseSourceString<ModuleOp>(bytecode, &context);
  return !roundTripped || findDeferred(*roundTripped).getNumResults() != 1 ||
         failed(verify(*roundTripped));
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
          onnx.Yield %2 : tensor<4x4xf32>
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

int testInlineSelectedMultiResultProvenance(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> tensor<2xf32> {
        %data = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "data", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<4xf32>
        %sizes = onnx.Constant dense<[1, 1, 2]> : tensor<3xi64>
        %out = "onnx.DeferredFoldable"(%data, %sizes) ({
        ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<3xi64>):
          %split:3 = "onnx.Split"(%arg0, %arg1) {ResultNames = ["first"], axis = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> (tensor<1xf32>, tensor<1xf32>, tensor<2xf32>)
          onnx.Yield %split#2 : tensor<2xf32>
        }) {ResultNames = ["selected"], onnx_node_name = "wrapper"} : (tensor<4xf32>, tensor<3xi64>) -> tensor<2xf32>
        return %out : tensor<2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  IRRewriter rewriter(&context);
  auto inlined =
      inlineDeferredFoldableExpression(findDeferred(*module), rewriter);
  ONNXSplitOp split;
  module->walk([&](ONNXSplitOp op) { split = op; });
  auto ret = cast<func::ReturnOp>(
      module->lookupSymbol<func::FuncOp>("test").front().getTerminator());
  if (failed(inlined) || inlined->size() != 1 || (*inlined)[0] != split ||
      ret.getOperand(0) != split->getResult(2) ||
      split->getAttr("onnx_node_name") !=
          StringAttr::get(&context, "wrapper") ||
      split->getAttr("ResultNames") !=
          Builder(&context).getStrArrayAttr({"first", "", "selected"}) ||
      failed(verify(*module)))
    return 1;
  return 0;
}

struct ModificationListener : RewriterBase::Listener {
  explicit ModificationListener(Operation *expected) : expected(expected) {}

  void notifyOperationModified(Operation *op) override {
    if (op == expected)
      ++modifications;
  }

  Operation *expected;
  unsigned modifications = 0;
};

int testInlineNotifiesProvenanceModification(MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> tensor<2x2xf32> {
        %data = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "data", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
        %out = "onnx.DeferredFoldable"(%data) ({
        ^bb0(%arg0: tensor<2x2xf32>):
          %transposed = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
          %sum = "onnx.Add"(%transposed, %transposed) {ResultNames = ["sum"]} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
          onnx.Yield %sum : tensor<2x2xf32>
        }) {ResultNames = ["selected"], onnx_node_name = "wrapper"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %out : tensor<2x2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  ONNXAddOp add;
  module->walk([&](ONNXAddOp op) { add = op; });
  ModificationListener listener(add);
  IRRewriter rewriter(&context, &listener);
  auto inlined =
      inlineDeferredFoldableExpression(findDeferred(*module), rewriter);
  if (failed(inlined) || inlined->size() != 2 || listener.modifications != 1 ||
      add->getAttr("onnx_node_name") != StringAttr::get(&context, "wrapper") ||
      add->getAttr("ResultNames") !=
          Builder(&context).getStrArrayAttr({"selected"}) ||
      failed(verify(*module)))
    return 1;
  return 0;
}

int testInlineForwardedCapturePreservesProducerProvenance(
    MLIRContext &context) {
  auto module = parseModule(context, R"mlir(
    module {
      func.func @test() -> (tensor<2xf32>, tensor<2xf32>) {
        %source = "onnx.Constant"() {ResultNames = ["source"], onnx_node_name = "source_node", value = dense<[1.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
        %other = "onnx.Identity"(%source) : (tensor<2xf32>) -> tensor<2xf32>
        %out = "onnx.DeferredFoldable"(%source) ({
        ^bb0(%arg0: tensor<2xf32>):
          onnx.Yield %arg0 : tensor<2xf32>
        }) {ResultNames = ["wrapper"], onnx_node_name = "wrapper_node"} : (tensor<2xf32>) -> tensor<2xf32>
        return %out, %other : tensor<2xf32>, tensor<2xf32>
      }
    }
  )mlir");
  if (!module)
    return 1;

  ONNXConstantOp source;
  ONNXIdentityOp identity;
  module->walk([&](ONNXConstantOp op) { source = op; });
  module->walk([&](ONNXIdentityOp op) { identity = op; });
  IRRewriter rewriter(&context);
  auto inlined =
      inlineDeferredFoldableExpression(findDeferred(*module), rewriter);
  auto ret = cast<func::ReturnOp>(
      module->lookupSymbol<func::FuncOp>("test").front().getTerminator());
  if (failed(inlined) || !inlined->empty() || ret.getOperand(0) != source ||
      identity.getInput() != source ||
      source->getAttr("onnx_node_name") !=
          StringAttr::get(&context, "source_node") ||
      source->getAttr("ResultNames") !=
          Builder(&context).getStrArrayAttr({"source"}) ||
      failed(verify(*module)))
    return 1;
  return 0;
}

} // namespace

int main() {
  MLIRContext context;
  context.loadDialect<func::FuncDialect, ONNXDialect, tensor::TensorDialect>();
  int failures =
      testDirectFormation(context) + testDequantFoldability(context) +
      testConstantExpressionFormation(context, false) +
      testConstantExpressionFormation(context, true) +
      testNormalizeDuplicateCaptureUses(context) +
      testCrossDialectFormation(context) +
      testBottomUpFormationWithSharedOperand(context) +
      testBottomUpFormationWithDenseSibling(context) +
      testRejectNondeterministicFormation(context) +
      testPeelOneInternalOperand(context) +
      testPeelMultipleInternalOperands(context) +
      testPeelSelectedResultAndDeduplicateOperands(context) +
      testPeelDQWithInternalParameters(context) +
      testInlineCompleteRecipe(context) +
      testInlineSelectedMultiResultProvenance(context) +
      testInlineForwardedCapturePreservesProducerProvenance(context) +
      testInlineNotifiesProvenanceModification(context);
  if (failures)
    llvm::errs() << failures << " DeferredFoldable test(s) failed\n";
  return failures;
}
