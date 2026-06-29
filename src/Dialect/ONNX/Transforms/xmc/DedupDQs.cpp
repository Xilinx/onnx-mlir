// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include <memory>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

class DQOpInfo : public DenseMapInfo<ONNXDequantizeLinearOp> {
public:
  static unsigned getHashValue(const ONNXDequantizeLinearOp &dqOpC) {
    auto &dqOp = const_cast<ONNXDequantizeLinearOp &>(dqOpC);
    return llvm::hash_combine(hash_value(dqOp.getX()),
        hash_value(dqOp.getXScale()), hash_value(dqOp.getXZeroPoint()),
        dqOp.getAxis(), dqOp.getBlockSize());
  }

  static bool isEqual(
      const ONNXDequantizeLinearOp &LHS, const ONNXDequantizeLinearOp &RHS) {
    auto &lhs = const_cast<ONNXDequantizeLinearOp &>(LHS);
    auto &rhs = const_cast<ONNXDequantizeLinearOp &>(RHS);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;

    return lhs.getX() == rhs.getX() && lhs.getXScale() == rhs.getXScale() &&
           lhs.getXZeroPoint() == rhs.getXZeroPoint() &&
           lhs.getAxis() == rhs.getAxis() &&
           lhs.getBlockSize() == rhs.getBlockSize();
  }
};

class DedupDQsPass
    : public PassWrapper<DedupDQsPass, OperationPass<func::FuncOp>> {
public:
  [[nodiscard]] StringRef getName() const override { return "dedup-dqs"; }
  [[nodiscard]] StringRef getArgument() const override { return "dedup-dqs"; }
  [[nodiscard]] StringRef getDescription() const override {
    return "Alternative to CSE, targets only DQ ops";
  }

  void runOnOperation() override {
    auto func = getOperation();
    DenseSet<ONNXDequantizeLinearOp, DQOpInfo> uniqDQs;
    DenseSet<ONNXDequantizeLinearOp> opsToErase;

    for (auto dqOp : func.getOps<ONNXDequantizeLinearOp>()) {
      if (auto foundIter = uniqDQs.find(dqOp); foundIter != uniqDQs.end()) {
        dqOp->replaceAllUsesWith((*foundIter)->getResults());
        opsToErase.insert(dqOp);
      } else {
        uniqDQs.insert(dqOp);
      }
    }

    for (auto dqOp : opsToErase)
      dqOp->erase();
  }
};

std::unique_ptr<Pass> createDedupDQsPass() {
  return std::make_unique<DedupDQsPass>();
}

} // namespace onnx_mlir
