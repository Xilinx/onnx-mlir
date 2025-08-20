//===- foldDqBinaryQPattern.cpp - Remove DQ-Binary-Q chains -----*- C++ -*-===//
//
// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include <variant>

using namespace mlir;
using namespace onnx_mlir;

namespace {

// Helper to print the ONNX node name (or fallback to NameLoc)
static void printOnnxNodeName(mlir::Operation *op, llvm::StringRef tag = "") {
  if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("onnx_node_name")) {
    llvm::outs() << (tag.empty() ? "" : (tag.str() + ": "))
                 << nameAttr.getValue() << "\n";
    return;
  }
  mlir::Location loc = op->getLoc();
  if (auto nameLoc = loc.dyn_cast<mlir::NameLoc>()) {
    llvm::outs() << (tag.empty() ? "" : (tag.str() + ": "))
                 << nameLoc.getName().str() << "\n";
    return;
  }
  llvm::outs() << (tag.empty() ? "" : (tag.str() + ": "))
               << "<no_onnx_node_name>\n";
}

// Checks if an ONNXConstantOp represents a scalar or a splat.
// Returns an optional containing the value of type T if successful.
template <typename T>
std::optional<T> getScalarOrSplatValue(ONNXConstantOp constOp) {
  // 1. Get the attribute that holds the tensor data.
  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(constOp.getValueAttr());
  if (!elementsAttr || !elementsAttr.isSplat()) {
    return std::nullopt;
  }

  // 2. Safely extract the value based on the element type.
  mlir::Type elementType = elementsAttr.getElementType();

  // Case 1: Floating-point types (f32, f64).
  if (elementType.isF32() || elementType.isF64()) {
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
      // APFloat can handle both f32 and f64.
      APFloat splatValue = elementsAttr.getSplatValue<APFloat>();
      return static_cast<T>(splatValue.convertToDouble());
    }
  }

  // Case 2: Integer types (i8, ui16, etc.).
  if (auto intType = elementType.dyn_cast<IntegerType>()) {
    if constexpr (std::is_integral_v<T>) {
      APInt splatValue = elementsAttr.getSplatValue<APInt>();
      if (intType.isUnsigned()) {
        return static_cast<T>(splatValue.getZExtValue());
      } else {
        return static_cast<T>(splatValue.getSExtValue());
      }
    }
  }
  // Return std::nullopt if the type is not a match.
  return std::nullopt;
}

template <typename T>
std::optional<T> getScalarOrSplatConstant(Value value) {
  if (!value) {
    return std::nullopt;
  }

  auto constOp = value.getDefiningOp<ONNXConstantOp>();
  if (!constOp) {
    return std::nullopt;
  }

  // Call the templated helper function.
  return getScalarOrSplatValue<T>(constOp);
}

static LogicalResult match_qdq(ONNXDequantizeLinearOp dq1,
    ONNXDequantizeLinearOp dq2, double &kValue,
    ONNXDequantizeLinearOp &activationDqOp, mlir::Type &scaleDtype,
    mlir::Type &zpDtype) {

  ONNXDequantizeLinearOp constantDqOp = nullptr;
  ONNXConstantOp constantSourceOp = nullptr;

  // Case 1: Direct ConstantOp as input to the DQ.
  if (auto constOp = dq1.getX().getDefiningOp<ONNXConstantOp>()) {
    constantDqOp = dq1;
    activationDqOp = dq2;
    constantSourceOp = constOp;
  } else if (auto constOp = dq2.getX().getDefiningOp<ONNXConstantOp>()) {
    constantDqOp = dq2;
    activationDqOp = dq1;
    constantSourceOp = constOp;
  }
  // Case 2: The input to the DQ op comes from a chain whose input is a
  // constant.
  else if (auto intermediateOp = dq1.getX().getDefiningOp()) {
    if (auto constOp =
            intermediateOp->getOperand(0).getDefiningOp<ONNXConstantOp>()) {
      constantDqOp = dq1;
      activationDqOp = dq2;
      constantSourceOp = constOp;
    }
  } else if (auto intermediateOp = dq2.getX().getDefiningOp()) {
    if (auto constOp =
            intermediateOp->getOperand(0).getDefiningOp<ONNXConstantOp>()) {
      constantDqOp = dq2;
      activationDqOp = dq1;
      constantSourceOp = constOp;
    }
  }

  if (!constantDqOp) {
    return failure();
  }

  // Use the templated helper to get the scalar value of the constant source.
  auto scalar_value_opt = getScalarOrSplatValue<int64_t>(constantSourceOp);
  if (!scalar_value_opt) {
    return failure();
  }
  int64_t scalar_value = *scalar_value_opt;

  // Use the templated helper to get the scale and zero-point values.
  Value scaleVal = constantDqOp.getXScale();
  Value zpVal = constantDqOp.getXZeroPoint();
  auto scale_value_opt = getScalarOrSplatConstant<double>(scaleVal);
  auto zp_value_opt = getScalarOrSplatConstant<int64_t>(zpVal);
  if (!scale_value_opt || !zp_value_opt) {
    return failure();
  }
  double scale_value = *scale_value_opt;
  int64_t zp_value = *zp_value_opt;

  // Store the data types.
  scaleDtype = scaleVal.getType().cast<TensorType>().getElementType();
  zpDtype = zpVal.getType().cast<TensorType>().getElementType();

  // Calculate kValue.
  kValue = (scalar_value - zp_value) * scale_value;

  return success();
}

template <typename BinOp>
static LogicalResult match_binary_op(BinOp binaryOp,
    ONNXDequantizeLinearOp &dequantActivationOp, ONNXConstantOp &constantOp,
    double &kValue, mlir::Type &scaleDtype, mlir::Type &zpDtype) {

  Value lhs = binaryOp.getOperand(0);
  Value rhs = binaryOp.getOperand(1);

  // -------- Case A: lhs is DQ, rhs is Constant --------
  if (auto dqOp = lhs.getDefiningOp<ONNXDequantizeLinearOp>()) {
    if (auto constOp = rhs.getDefiningOp<ONNXConstantOp>()) {
      dequantActivationOp = dqOp;
      constantOp = constOp;
    }
  }
  // -------- Case A reversed --------
  else if (auto dqOp = rhs.getDefiningOp<ONNXDequantizeLinearOp>()) {
    if (auto constOp = lhs.getDefiningOp<ONNXConstantOp>()) {
      dequantActivationOp = dqOp;
      constantOp = constOp;
    }
  }

  // -------- Fill kValue --------
  if (dequantActivationOp && constantOp) {
    // if (!isScalarOrSplat(constantOp, kValue))
    //   return failure();
    auto kValueOpt = getScalarOrSplatValue<double>(constantOp);
    if (!kValueOpt.has_value()) {
      return failure();
    }
    double kValue = kValueOpt.value();

    // Debug - To be removed
    llvm::outs() << "A - SUCCESS\n";
    printOnnxNodeName(binaryOp, "[RemoveBinary] matched");
    return success();
  }

  // -------- Case B: both inputs are DQ --------
  auto dqOp1 = lhs.getDefiningOp<ONNXDequantizeLinearOp>();
  auto dqOp2 = rhs.getDefiningOp<ONNXDequantizeLinearOp>();

  if (dqOp1 && dqOp2) {
    if (failed(match_qdq(
            dqOp1, dqOp2, kValue, dequantActivationOp, scaleDtype, zpDtype)))
      return failure();

    // Debug - To be removed
    llvm::outs() << "B. SUCCESS\n";
    llvm::outs() << "kValue = " << kValue << "\n";
    printOnnxNodeName(binaryOp, "[RemoveBinary] matched");
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Pattern to (eventually) fold DQ->Binary->Q when quantization parameters
//===----------------------------------------------------------------------===//

template <typename BinOp>
struct FoldBinaryThroughQDQ : public OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      BinOp op, PatternRewriter &rewriter) const override {

    // STEP 1: Find the Quantize op after the binary op.
    ONNXQuantizeLinearOp quantOutputOp = nullptr;
    for (Value res : op->getResults()) {
      for (Operation *user : res.getUsers()) {
        if (auto q = dyn_cast<ONNXQuantizeLinearOp>(user)) {
          quantOutputOp = q;
          break;
        }
      }
      if (quantOutputOp)
        break;
    }
    if (!quantOutputOp)
      return failure();

    // STEP 2: Match the binary op's inputs.
    ONNXDequantizeLinearOp dequantActivationOp = nullptr;
    ONNXConstantOp constantOp = nullptr;
    double kValue = 0.0;
    mlir::Type scaleDtype, zeroPointDtype;

    if (failed(match_binary_op<BinOp>(op, dequantActivationOp, constantOp,
            kValue, scaleDtype, zeroPointDtype)))
      return failure();

    llvm::outs() << "kValue OUT " << kValue << "\n";
    llvm::outs() << "scaleDtype OUT " << scaleDtype << "\n";
    llvm::outs() << "zeroPointDtype OUT " << zeroPointDtype << "\n";
    llvm::outs() << "dequantActivationOp " << *dequantActivationOp << "\n";
    llvm::outs() << "\n";

    return failure(); // TODO: replace with SUCCESS LATER
  }
};

struct FoldDQBinaryQPass
    : public PassWrapper<FoldDQBinaryQPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldDQBinaryQPass)

  StringRef getArgument() const final { return "fold-dq-binary-q"; }
  StringRef getDescription() const final {
    return "Fold Add/Sub/Mul/Div through Q/DQ by updating scale/zero_point, "
           "then remove trivial Q->DQ chains when safe.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns
        .add<FoldBinaryThroughQDQ<ONNXDivOp>, FoldBinaryThroughQDQ<ONNXSubOp>,
            FoldBinaryThroughQDQ<ONNXMulOp>, FoldBinaryThroughQDQ<ONNXAddOp>>(
            &getContext());
    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createFoldDQBinaryQPass() {
  return std::make_unique<FoldDQBinaryQPass>();
}
} // namespace onnx_mlir