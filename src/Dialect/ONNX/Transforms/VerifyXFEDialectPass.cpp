/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- VerifyXFEDialectPass.cpp - XFE dialect constraint verifier --------===//
//
// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

namespace onnx_mlir {
#define GEN_PASS_DEF_VERIFYXFEDIALECTPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"
} // namespace onnx_mlir

using namespace mlir;

namespace {

struct VerifyXFEDialectPass
    : public onnx_mlir::impl::VerifyXFEDialectPassBase<VerifyXFEDialectPass> {
  using Base::Base;

  void runOnOperation() override {
    llvm::StringSet<> denySet;

    llvm::StringRef list(disallowedOps);
    while (!list.empty()) {
      auto [token, rest] = list.split(',');
      token = token.trim();
      if (!token.empty())
        denySet.insert(("onnx." + token).str());
      list = rest;
    }

    if (denySet.empty())
      return;

    bool anyFailed = false;
    func::FuncOp func = getOperation();

    for (Operation &op : func.getBody().getOps()) {
      // Check deny-list.
      if (!denySet.count(op.getName().getStringRef()))
        continue;

      // Static gate: skip if any tensor-typed operand is not fully static.
      bool allStatic = true;
      for (Value operand : op.getOperands()) {
        Type ty = operand.getType();
        if (isa<NoneType>(ty))
          continue;
        if (!isa<RankedTensorType>(ty) || !onnx_mlir::hasStaticShape(ty)) {
          allStatic = false;
          break;
        }
      }
      if (!allStatic)
        continue;

      op.emitOpError("disallowed in XFE dialect");
      anyFailed = true;
    }

    if (anyFailed)
      signalPassFailure();
  }
};

} // namespace
