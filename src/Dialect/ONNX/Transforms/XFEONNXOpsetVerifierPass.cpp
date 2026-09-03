/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- XFEONNXOpsetVerifierPass.cpp - XFE ONNX opset verifier -----------===//
//
// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

namespace onnx_mlir {
#define GEN_PASS_DEF_XFEONNXOPSETVERIFIERPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"
} // namespace onnx_mlir

using namespace mlir;
using namespace onnx_mlir;

namespace {

// Parses the comma-separated bare ONNX op deny-list into dialect-qualified
// operation names.
llvm::StringSet<> parseDisallowedOps(StringRef disallowedOps) {
  llvm::StringSet<> denySet;
  StringRef list(disallowedOps);
  while (!list.empty()) {
    auto [token, rest] = list.split(',');
    token = token.trim();
    if (!token.empty())
      denySet.insert(("onnx." + token).str());
    list = rest;
  }
  return denySet;
}

bool isStaticallyShapedTensorLike(Type ty) {
  if (isa<NoneType>(ty))
    return true;
  if (auto tensorType = dyn_cast<RankedTensorType>(ty))
    return tensorType.hasStaticShape();
  if (auto seqType = dyn_cast<SeqType>(ty)) {
    if (ShapedType::isDynamic(seqType.getLength()))
      return false;
    if (auto elementType = dyn_cast<RankedTensorType>(seqType.getElementType()))
      return elementType.hasStaticShape();
  }
  return false;
}

bool hasAllStaticTensorValues(Operation *op) {
  for (const Value operand : op->getOperands())
    if (!isStaticallyShapedTensorLike(operand.getType()))
      return false;
  for (const Value result : op->getResults())
    if (!isStaticallyShapedTensorLike(result.getType()))
      return false;
  return true;
}

struct XFEONNXOpsetVerifierPass
    : public onnx_mlir::impl::XFEONNXOpsetVerifierPassBase<
          XFEONNXOpsetVerifierPass> {
  using Base::Base;

  void runOnOperation() override {
    const llvm::StringSet<> denySet = parseDisallowedOps(disallowedOps);

    bool anyFailed = false;
    func::FuncOp func = getOperation();

    for (Operation &op : func.getBody().getOps()) {
      const bool allStatic = hasAllStaticTensorValues(&op);

      // Check deny-list.
      if (denySet.count(op.getName().getStringRef()) && allStatic) {
        op.emitOpError("disallowed in XFE ONNX opset");
        anyFailed = true;
      }

      if (verifyNonNegativeAxis && allStatic && hasNegativeONNXAxisValue(&op)) {
        op.emitOpError("negative axis value is disallowed in XFE ONNX opset");
        anyFailed = true;
      }
    }

    if (anyFailed)
      signalPassFailure();
  }
};

} // namespace
