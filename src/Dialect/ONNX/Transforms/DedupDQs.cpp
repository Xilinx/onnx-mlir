// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include <memory>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

#define GEN_PASS_DEF_DEDUPDQSPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"

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

class DedupDQsPass : public impl::DedupDQsPassBase<DedupDQsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    auto func = getOperation();
    DenseSet<ONNXDequantizeLinearOp, DQOpInfo> uniqDQs;
    DenseSet<ONNXDequantizeLinearOp> opsToErase;

    for (auto currDQ : func.getOps<ONNXDequantizeLinearOp>()) {
      if (auto foundIter = uniqDQs.find(currDQ); foundIter != uniqDQs.end()) {
        ONNXDequantizeLinearOp &existingDQ = *foundIter;

        if (currDQ == existingDQ) {
          // Can happen, since we're moving the DQ's around
          continue;
        } else if (isOutput(currDQ) && !isOutput(existingDQ)) {
          // Prefer output DQs even if they occur second
          // Replacing DQ should dominate all it's uses, so it's moved
          currDQ->moveAfter(existingDQ);
          existingDQ->replaceAllUsesWith(currDQ->getResults());
          opsToErase.insert(existingDQ);
          uniqDQs.erase(foundIter);
          uniqDQs.insert(currDQ);
        } else {
          // Usual replacement, keep the first-seen DQ
          currDQ->replaceAllUsesWith(existingDQ->getResults());
          opsToErase.insert(currDQ);
        }
      } else {
        uniqDQs.insert(currDQ);
      }
    }

    for (auto dqOp : opsToErase)
      dqOp->erase();
  }

private:
  static bool isOutput(ONNXDequantizeLinearOp dqOp) {
    return any_of(dqOp->getUsers(),
        [](Operation *user) { return isa_and_present<func::ReturnOp>(user); });
  }
};

} // namespace onnx_mlir
