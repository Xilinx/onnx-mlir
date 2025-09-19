
// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/STLExtras.h"
#include <cmath>
#include <optional>
#include <variant>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {

struct AddQDQAroundOp : public PassWrapper<AddQDQAroundOp, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddQDQAroundOp)

  StringRef getArgument() const final { return "add-qdq-around-op"; }
  StringRef getDescription() const final {
    return "Add Q ,DQ around nodes which are missing them";
  }
  
  Type extractZeroPointType(OpBuilder &builder,func::FuncOp &func){
    Type zpElemType = builder.getIntegerType(8); // default int8

    func.walk([&](Operation *op) -> WalkResult {
        if (auto q = mlir::dyn_cast<ONNXQuantizeLinearOp>(op)) {
            Value zp = q.getYZeroPoint();
            if (auto zpShaped = llvm::dyn_cast<ShapedType>(zp.getType())) {
                zpElemType = zpShaped.getElementType();
                return WalkResult::interrupt();
            }
        } else if (auto dq = mlir::dyn_cast<ONNXDequantizeLinearOp>(op)) {
            Value zp = dq.getXZeroPoint();
            if (auto zpShaped = llvm::dyn_cast<ShapedType>(zp.getType())) {
                zpElemType = zpShaped.getElementType();
                return WalkResult::interrupt();
            }
        }
        return WalkResult::advance();
    });
    return zpElemType;

  }
  
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    Block &entryBlock = func.getBody().front();
    Location entryLoc = func.getLoc();


    RankedTensorType scaleType = RankedTensorType::get({}, builder.getF32Type());
    DenseElementsAttr scaleAttr = DenseElementsAttr::get(scaleType, {1.0f});
    OperationState scaleState(entryLoc, "onnx.Constant");
    scaleState.addAttribute("value", scaleAttr);
    scaleState.addTypes(scaleAttr.getType());
    Operation *scaleOp = Operation::create(scaleState);
    entryBlock.getOperations().insert(entryBlock.begin(), scaleOp);
    Value scaleVal = scaleOp->getResult(0);

    Type zpEleType = extractZeroPointType(builder, func);
    RankedTensorType zpType = RankedTensorType::get({}, zpEleType);
    DenseElementsAttr zpAttr;
    if (isa<IntegerType>(zpEleType)){
        auto intType = mlir::dyn_cast<IntegerType>(zpEleType);
        unsigned width = intType.getWidth();
        bool isSigned = intType.isSignedInteger();

         if (width == 8) {
            if (isSigned) {
                zpAttr = DenseElementsAttr::get(
                    zpType,
                    {static_cast<int8_t>(0)});
            } else {
                zpAttr = DenseElementsAttr::get(
                    zpType,
                    {static_cast<uint8_t>(0)});
            }
        } else if (width == 16) {
            if (isSigned) {
                zpAttr = DenseElementsAttr::get(
                    zpType,
                    {static_cast<int16_t>(0)});
            } else {
                zpAttr = DenseElementsAttr::get(
                    zpType,
                    {static_cast<uint16_t>(0)});
            }
        } else if (width == 32) {
            if (isSigned) {
                zpAttr = DenseElementsAttr::get(
                    zpType,
                    {static_cast<int32_t>(0)});
            } else {
                zpAttr = DenseElementsAttr::get(
                    zpType,
                    {static_cast<uint32_t>(0)});
            }
        } else {
            // fallback: default int8 zero-point
            zpAttr = DenseElementsAttr::get(
                zpType,
                {static_cast<int8_t>(0)});
        }
    } else {
        // fallback if not integer
        zpAttr = DenseElementsAttr::get(
            zpType,
            {static_cast<int8_t>(0)});
    }
    
    
    OperationState zpState(entryLoc, "onnx.Constant");
    zpState.addAttribute("value", zpAttr);
    zpState.addTypes(zpAttr.getType());
    Operation *zpOp = Operation::create(zpState);
    entryBlock.getOperations().insert(entryBlock.begin(), zpOp);
    Value zpVal = zpOp->getResult(0);

    llvm::SmallDenseMap<Value, Value> producerToDQ;

    for (Operation &opRef : llvm::make_early_inc_range(func.getOps())) {
        Operation *op = &opRef;

        if (isa<ONNXConstantOp, ONNXQuantizeLinearOp, ONNXDequantizeLinearOp>(op))
        continue;

        Location loc = op->getLoc();

        for (Value operand : op->getOperands()) {
            Operation *def = operand.getDefiningOp();
            if (def && isa<ONNXQuantizeLinearOp, ONNXDequantizeLinearOp>(def)) continue;
            
            auto it = producerToDQ.find(operand);
            if (it != producerToDQ.end()) {
                op->replaceUsesOfWith(operand, it->second);
                continue;
            }

            builder.setInsertionPoint(op);

            ShapedType inShaped = llvm::dyn_cast<ShapedType>(operand.getType());

            Type qResultType = operand.getType();
            if (isa<FloatType>(operand.getType()) || (inShaped && isa<FloatType>(inShaped.getElementType()))){
                if (inShaped)
                    qResultType = RankedTensorType::get(inShaped.getShape(), zpEleType);
                else
                    qResultType = RankedTensorType::get({}, zpEleType);     

                auto q = builder.create<ONNXQuantizeLinearOp>(loc, qResultType, operand, scaleVal, zpVal);
                auto dq = builder.create<ONNXDequantizeLinearOp>(loc, operand.getType(), q.getResult(), scaleVal, zpVal);
                
                producerToDQ.try_emplace(operand, dq.getResult());
                op->replaceUsesOfWith(operand, dq.getResult());  

            }
            
        }
    } 
    }
};
} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createMissingQDQAroundOpOptONNXToONNXPass() {
  return std::make_unique<AddQDQAroundOp>();
}
} // namespace onnx_mlir