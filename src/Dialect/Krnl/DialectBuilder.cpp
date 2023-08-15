/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-------------- DialectBuilder.cpp - Krnl Dialect Builder ------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file declares helper methods to build Krnl Dialect Ops.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"

#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

static StringRef getFormat(const Type &inputType) {
  StringRef format;
  TypeSwitch<Type>(inputType)
      .Case<Float16Type>([&](Float16Type) { format = "%g"; })
      .Case<Float32Type>([&](Float32Type) { format = "%f"; })
      .Case<Float64Type>([&](Float64Type) { format = "%f"; })
      .Case<IntegerType>([&](IntegerType type) {
        switch (type.getWidth()) {
        case 1:
        case 8:
        case 16:
        case 32:
          format = type.isUnsigned() ? "%u" : "%d";
          break;
        case 64:
          format = type.isUnsigned() ? "%llu" : "%lld";
          break;
        }
      })
      .Case<IndexType>([&](IndexType) { format = "%lld"; })
      .Case<onnx_mlir::krnl::StringType>(
          [&](onnx_mlir::krnl::StringType) { format = "%s"; })
      .Case<LLVM::LLVMPointerType>(
          [&](LLVM::LLVMPointerType) { format = "%s"; })
      .Default([&](Type type) {
        llvm::errs() << "type: " << type << "\n";
        llvm_unreachable("Unhandled type");
      });

  return format;
}

//====---------------- Support for Krnl Builder ----------------------===//

Value KrnlBuilder::load(Value memref, ValueRange indices) {
  return create<KrnlLoadOp>(memref, indices);
}

mlir::Value KrnlBuilder::load(
    mlir::Value memref, mlir::ValueRange indices, mlir::ValueRange offsets) {
  SmallVector<Value, 4> computedIndices;
  MathBuilder createMath(*this);
  createMath.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  return load(memref, computedIndices);
}

Value KrnlBuilder::loadIE(Value memref, ArrayRef<IndexExpr> indices) {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  return create<KrnlLoadOp>(memref, indexValues);
}

void KrnlBuilder::store(Value val, Value memref, ValueRange indices) {
  create<KrnlStoreOp>(val, memref, indices);
}

void KrnlBuilder::store(mlir::Value val, mlir::Value memref,
    mlir::ValueRange indices, mlir::ValueRange offsets) {
  SmallVector<Value, 4> computedIndices;
  MathBuilder createMath(*this);
  createMath.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  store(val, memref, computedIndices);
}

void KrnlBuilder::storeIE(
    Value val, Value memref, ArrayRef<IndexExpr> indices) {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  create<KrnlStoreOp>(val, memref, indexValues);
}

void KrnlBuilder::seqstore(
    mlir::Value element, mlir::Value seq, mlir::Value index) {
  create<KrnlSeqStoreOp>(element, seq, index);
}

void KrnlBuilder::seqstore(
    mlir::Value element, mlir::Value seq, IndexExpr index) {
  create<KrnlSeqStoreOp>(element, seq, index.getValue());
}

Value KrnlBuilder::vectorTypeCast(Value sourceMemref, int64_t vectorLen) {
  return create<KrnlVectorTypeCastOp>(sourceMemref, vectorLen);
}

ValueRange KrnlBuilder::defineLoops(int64_t originalLoopNum) {
  return this->create<KrnlDefineLoopsOp>(originalLoopNum).getResults();
}

ValueRange KrnlBuilder::block(Value loop, int64_t blockSize) {
  return create<KrnlBlockOp>(loop, blockSize).getResults();
}

void KrnlBuilder::permute(ValueRange loops, ArrayRef<int64_t> map) {
  create<KrnlPermuteOp>(loops, map);
}

ValueRange KrnlBuilder::getInductionVarValue(ValueRange loops) {
  return this->create<KrnlGetInductionVariableValueOp>(loops).getResults();
}

void KrnlBuilder::iterate(ValueRange originalLoops, ValueRange optimizedLoops,
    ValueRange lbs, ValueRange ubs,
    function_ref<void(KrnlBuilder &createKrnl, ValueRange indices)>
        bodyBuilderFn) {
  // Check that originalLoops, lbs, and ubs have the same rank.
  assert(originalLoops.size() == lbs.size() && "expected same rank");
  assert(originalLoops.size() == ubs.size() && "expected same rank");
  ValueRange empty;
  create<KrnlIterateOp>(originalLoops, optimizedLoops, lbs, ubs, empty,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        KrnlBuilder createKrnl(builder, loc);
        ValueRange indices = createKrnl.getInductionVarValue(optimizedLoops);
        bodyBuilderFn(createKrnl, indices);
      });
}

KrnlIterateOp KrnlBuilder::iterate(
    const krnl::KrnlIterateOperandPack &operands) {
  return create<KrnlIterateOp>(operands);
}

void KrnlBuilder::iterateIE(ValueRange originalLoops, ValueRange optimizedLoops,
    ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs,
    function_ref<void(KrnlBuilder &createKrnl, ValueRange indices)>
        bodyBuilderFn) {
  // Check that originalLoops, lbs, and ubs have the same rank.
  assert(originalLoops.size() == lbs.size() && "expected same rank");
  assert(originalLoops.size() == ubs.size() && "expected same rank");
  ValueRange empty;
  create<KrnlIterateOp>(originalLoops, optimizedLoops, lbs, ubs, empty,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        KrnlBuilder createKrnl(builder, loc);
        ValueRange indices = createKrnl.getInductionVarValue(optimizedLoops);
        bodyBuilderFn(createKrnl, indices);
      });
}

void KrnlBuilder::copyToBuffer(Value bufferMemref, Value sourceMemref,
    ValueRange starts, Value padValue, ArrayRef<int64_t> tileSize,
    ArrayRef<int64_t> padToNext, bool transpose) {
  create<KrnlCopyToBufferOp>(bufferMemref, sourceMemref, starts, padValue,
      tileSize, padToNext, transpose);
}

void KrnlBuilder::copyToBuffer(Value bufferMemref, Value sourceMemref,
    ValueRange starts, Value padValue, bool transpose) {
  create<KrnlCopyToBufferOp>(
      bufferMemref, sourceMemref, starts, padValue, transpose);
}

void KrnlBuilder::copyFromBuffer(Value bufferMemref, Value memref,
    ValueRange starts, ArrayRef<int64_t> tileSize) {
  create<KrnlCopyFromBufferOp>(bufferMemref, memref, starts, tileSize);
}

void KrnlBuilder::copyFromBuffer(
    Value bufferMemref, Value memref, ValueRange starts) {
  create<KrnlCopyFromBufferOp>(bufferMemref, memref, starts);
}

void KrnlBuilder::matmul(Value A, ValueRange aStart, Value B, ValueRange bStart,
    Value C, ValueRange cStart, ValueRange loops, ValueRange computeStarts,
    ValueRange globalUBs, ArrayRef<int64_t> computeTileSize,
    ArrayRef<int64_t> aTileSize, ArrayRef<int64_t> bTileSize,
    ArrayRef<int64_t> cTileSize, bool simdize, bool unroll, bool overCompute) {
  create<KrnlMatMulOp>(A, aStart, B, bStart, C, cStart, loops, computeStarts[0],
      computeStarts[1], computeStarts[2], globalUBs[0], globalUBs[1],
      globalUBs[2], computeTileSize, aTileSize, bTileSize, cTileSize, simdize,
      unroll, overCompute);
}

void KrnlBuilder::matmul(Value A, ValueRange aStart, Value B, ValueRange bStart,
    Value C, ValueRange cStart, ValueRange loops, ValueRange computeStarts,
    ValueRange globalUBs, bool simdize, bool unroll, bool overCompute) {
  create<KrnlMatMulOp>(A, aStart, B, bStart, C, cStart, loops, computeStarts[0],
      computeStarts[1], computeStarts[2], globalUBs[0], globalUBs[1],
      globalUBs[2], simdize, unroll, overCompute);
}

Value KrnlBuilder::dim(Type type, Value alloc, Value index) {
  return create<KrnlDimOp>(type, alloc, index);
}

KrnlMovableOp KrnlBuilder::movable() { return create<KrnlMovableOp>(); }

KrnlGetRefOp KrnlBuilder::getRef(
    Type type, Value memref, Value offset, ValueRange indices) {
  return create<KrnlGetRefOp>(type, memref, offset, indices);
}

Value KrnlBuilder::constant(MemRefType type, StringRef name,
    std::optional<Attribute> value, std::optional<IntegerAttr> offset,
    std::optional<IntegerAttr> alignment) {
  static int32_t constantID = 0;
  return create<KrnlGlobalOp>(type, getI64ArrayAttr(type.getShape()),
      getStringAttr(name + std::to_string(constantID++)),
      value.value_or(nullptr), offset.value_or(nullptr),
      alignment.value_or(nullptr));
}

void KrnlBuilder::memcpy(Value dest, Value src, Value numElems) {
  MultiDialectBuilder<MathBuilder> create(*this, getLoc());
  Value zero = create.math.constantIndex(0);
  this->create<KrnlMemcpyOp>(
      dest, src, numElems, /*dest_offset=*/zero, /*src_offset=*/zero);
}

void KrnlBuilder::memcpy(
    Value dest, Value src, Value numElems, Value destOffset, Value srcOffset) {
  create<KrnlMemcpyOp>(dest, src, numElems, destOffset, srcOffset);
}

void KrnlBuilder::memset(Value dest, Value val, bool delayed) {
  create<KrnlMemsetOp>(dest, val, getBoolAttr(delayed));
}

Value KrnlBuilder::strncmp(Value str1, Value str2, Value len) {
  return create<KrnlStrncmpOp>(getI32Type(), str1, str2, len);
}

Value KrnlBuilder::strlen(Value str) {
  return create<KrnlStrlenOp>(getI64Type(), str);
}

void KrnlBuilder::randomNormal(Value alloc, Value numberOfRandomValues,
    Value mean, Value scale, Value seed) {
  create<KrnlRandomNormalOp>(alloc, numberOfRandomValues, mean, scale, seed);
}

Value KrnlBuilder::findIndex(Value input, Value G, Value V, Value len) {
  return create<KrnlFindIndexOp>(getIndexType(), input, G, V, len);
}

void KrnlBuilder::printTensor(StringRef msg, Value input) {
  create<KrnlPrintTensorOp>(msg, input);
}

void KrnlBuilder::printf(StringRef msg) {
  Value noneValue;
  create<KrnlPrintOp>(msg, noneValue);
}

void KrnlBuilder::printf(
    StringRef msg, Value input, Type inputType, bool endsWithNewLine) {
  StringRef format = getFormat(inputType);
  std::string concat(msg.str() + format.str() + (endsWithNewLine ? "\n" : ""));
  StringRef newFormat(concat);
  create<KrnlPrintOp>(newFormat, input);
}

void KrnlBuilder::printf(Value input, Type inputType) {
  StringRef format = getFormat(inputType);
  create<KrnlPrintOp>(format, input);
}

// =============================================================================
// IndexExpr Builder for Analysis
// =============================================================================

// Return null if none is found.
ElementsAttr IndexExprBuilderForKrnl::getConst(mlir::Value value) {
  auto definingOp = value.getDefiningOp();
  if (auto globalOp = dyn_cast_or_null<mlir::KrnlGlobalOp>(definingOp)) {
    if (globalOp.getValue().has_value())
      return globalOp.getValueAttr().dyn_cast<ElementsAttr>();
  } else if (auto globalOp =
                 dyn_cast_or_null<mlir::ONNXConstantOp>(definingOp)) {
    if (globalOp.getValue().has_value())
      return globalOp.getValueAttr().dyn_cast<ElementsAttr>();
  }
  return nullptr;
}

Value IndexExprBuilderForKrnl::getVal(Value intArrayVal, uint64_t i) {
  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
      this->getBuilder(), getLoc());
  uint64_t rank = getShapedTypeRank(intArrayVal);
  if (rank == 0)
    return create.krnl.load(intArrayVal, {});
  uint64_t size = getArraySize(intArrayVal);
  assert(i < size && "out of bound reference");
  Value iVal = create.math.constantIndex(i);
  return create.krnl.load(intArrayVal, {iVal});
}

Value IndexExprBuilderForKrnl::getShapeVal(
    Value tensorOrMemrefValue, uint64_t i) {
  MemRefBuilder createMemRef(this->getBuilder(), getLoc());
  return createMemRef.dim(tensorOrMemrefValue, i);
}

} // namespace onnx_mlir
