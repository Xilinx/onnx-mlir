// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

static void addRedundantOpReductionSubPasses(OpPassManager &pm) {
  // Mirrors xcompiler RedundantOpReductionPass::exec() for QDQ models.
  pm.addPass(createMergeContinuousStridedSlicePass());
  pm.addPass(createRemoveContinuousTransposeWithReshapePass());
  pm.addPass(createRemoveSemanticallyUselessOpsPass());
  pm.addPass(createRemoveUselessQLinearPoolPass());
  pm.addPass(createConvertQDQToRequantizePass());
  pm.addPass(createConvertSCastPairToRequantizePass());
  pm.addPass(createONNXTransposeOptimizationPass());
  pm.addPass(createCombineTransposePairPass());
  pm.addPass(createCanonicalizeWithResultNamesPass());
}

struct RedundantOpReductionPass
    : public PassWrapper<RedundantOpReductionPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "redundant-op-reduction"; }
  StringRef getDescription() const override {
    return "Combined redundant-op reduction (mirrors xcompiler "
           "RedundantOpReductionPass for QDQ models)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpPassManager pm(func.getOperationName());
    addRedundantOpReductionSubPasses(pm);
    if (failed(runPipeline(pm, func)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createRedundantOpReductionPass() {
  return std::make_unique<RedundantOpReductionPass>();
}

} // namespace onnx_mlir
