/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- DialectBuilder.hpp - Helper functions for MLIR dialects -----===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for building MLIR operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

// Please do not add dependences on ONNX or KRNL dialects.
#include "src/Dialect/Mlir/IndexExpr.hpp"

namespace onnx_mlir {

struct DialectBuilder {
  // Constructor for analysis (no code generation, get builder disabled).
  DialectBuilder(mlir::Location loc) : builder(nullptr), location(loc) {}
  // Constructors for code generation.
  DialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : builder(&b), location(loc) {}
  DialectBuilder(const DialectBuilder &db)
      : builder(db.builder), location(db.location) {}
  virtual ~DialectBuilder() {}
  DialectBuilder(DialectBuilder &&) = delete;
  DialectBuilder &operator=(const DialectBuilder &) = delete;
  DialectBuilder &&operator=(const DialectBuilder &&) = delete;

  // Public getters of builder and location.
  mlir::OpBuilder &getBuilder() const { return b(); }
  mlir::OpBuilder *getBuilderPtr() const { return builder; } // Possibly null.
  mlir::Location getLoc() const { return loc(); }

protected:
  // Private getters of builder and location (concise version).
  mlir::OpBuilder &b() const {
    assert(builder && "builder is null");
    return *builder;
  }
  mlir::Location loc() const { return location; }

private:
  mlir::OpBuilder *builder;
  mlir::Location location;
};

template <typename GenericBuilder>
struct WithLoc : public GenericBuilder {
  using GenericBuilder::create;
  template <typename OldBuilder>
  WithLoc(OldBuilder &b, mlir::Location loc)
      : GenericBuilder(b.getContext()), loc(loc) {
    this->setInsertionPoint(b.getBlock(), b.getInsertionPoint());
  };
  /// Initialize the builder.
  template <typename... Args>
  explicit WithLoc(mlir::Location loc, Args... args)
      : GenericBuilder(args...), loc(loc){};
  explicit WithLoc(mlir::Location loc) : loc(loc){};
  explicit WithLoc(mlir::Operation *op)
      : GenericBuilder(op->getContext()), loc(op->getLoc()) {
    this->setInsertionPoint(op);
  }
  explicit WithLoc(WithLoc<mlir::OpBuilder> &b)
      : GenericBuilder(b.getContext()), loc(b.getLoc()) {
    this->setInsertionPoint(b.getBlock(), b.getInsertionPoint());
  }
  explicit WithLoc(mlir::OpBuilder &b)
      : GenericBuilder(b.getContext()), loc(mlir::UnknownLoc()) {
    this->setInsertionPoint(b.getBlock(), b.getInsertionPoint());
  }
  explicit WithLoc(const DialectBuilder &db)
      : GenericBuilder(db.getBuilder().getContext()), loc(db.getLoc()) {
    this->setInsertionPoint(
        db.getBuilder().getBlock(), db.getBuilder().getInsertionPoint());
  }

  template <typename OpTy, typename... Args>
  OpTy create(Args &&...args) {
    return this->create<OpTy>(getLoc(), std::forward<Args>(args)...);
  }

  ~WithLoc() = default;

  void operator=(const WithLoc &) = delete;
  WithLoc(const WithLoc &) = delete;

  mlir::Location getLoc() { return this->loc; }

private:
  mlir::Location loc;
};

//===----------------------------------------------------------------------===//
// Math Builder
//===----------------------------------------------------------------------===//

/// Helper struct to build simple arithmetic quantities with minimal type
/// inference support. Code is adapted to support the DialectBuilder
/// super-class that facilitate the building of other dialect builders using
/// another dialect builder.

//===----------------------------------------------------------------------===//
// Original code for MathBuilder is copied from LLVM MLIR Utils.cpp
// Modified here to add operations, add super class.
// License added here for this class for completeness.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

struct MathBuilder final : WithLoc<mlir::OpBuilder> {
  using WithLoc<mlir::OpBuilder>::WithLoc;
  MathBuilder(WithLoc<mlir::OpBuilder> &b) : WithLoc<mlir::OpBuilder>(b){};
  virtual ~MathBuilder() {}

  // Support for vectors: we provide queries that work regardless of if we
  // have (1) a scalar or (2) a vector of a basic element type.

  // The method belows ignore the vectors part of the type to provide answer
  // on the basic element types alone.
  static bool isIntegerWithVector(mlir::Type elementOrVectorType);
  static bool isUnsignedIntegerWithVector(mlir::Type elementOrVectorType);
  static bool isFloatWithVector(mlir::Type elementOrVectorType);
  // Return the basic element type regardless of if we are given (1) a scalar
  // or (2) a vector of a basic element type.
  static mlir::Type elementTypeWithVector(mlir::Type elementOrVectorType);
  // Return a type of the same vector shape as vectorType with a basic element
  // type of elementType. When vectorType is null, then the returned type is
  // simply a scalar of elementType.
  static mlir::Type getTypeWithVector(
      mlir::VectorType vectorType, mlir::Type elementType);

  mlir::Value abs(mlir::Value val);
  mlir::Value add(mlir::Value lhs, mlir::Value rhs);
  mlir::Value andi(mlir::Value lhs, mlir::Value rhs);     // Int only.
  mlir::Value ceil(mlir::Value val);                      // Float only.
  mlir::Value ceilDiv(mlir::Value lhs, mlir::Value rhs);  // Int only.
  mlir::Value copySign(mlir::Value rem, mlir::Value div); // Float only.
  mlir::Value div(mlir::Value lhs, mlir::Value rhs);
  mlir::Value exp(mlir::Value val);                       // Float only.
  mlir::Value exp2(mlir::Value val);                      // Float only.
  mlir::Value floor(mlir::Value val);                     // Float only.
  mlir::Value floorDiv(mlir::Value lhs, mlir::Value rhs); // Int only.
  mlir::Value fma(mlir::Value lhs, mlir::Value rhs, mlir::Value acc);
  mlir::Value log(mlir::Value val);  // Float only.
  mlir::Value log2(mlir::Value val); // Float only.
  mlir::Value mul(mlir::Value lhs, mlir::Value rhs);
  mlir::Value neg(mlir::Value val);
  mlir::Value ori(mlir::Value lhs, mlir::Value rhs);  // Int only.
  mlir::Value pow(mlir::Value base, mlir::Value exp); // Float only.
  mlir::Value rem(mlir::Value lhs, mlir::Value rhs);
  mlir::Value sqrt(mlir::Value val); // Float only.
  mlir::Value sub(mlir::Value lhs, mlir::Value rhs);
  mlir::Value xori(mlir::Value lhs, mlir::Value rhs); // Int only.

  mlir::Value select(mlir::Value cmp, mlir::Value lhs, mlir::Value rhs);
  mlir::Value gt(mlir::Value lhs, mlir::Value rhs);
  mlir::Value ge(mlir::Value lhs, mlir::Value rhs);
  mlir::Value lt(mlir::Value lhs, mlir::Value rhs);
  mlir::Value le(mlir::Value lhs, mlir::Value rhs);
  mlir::Value eq(mlir::Value lhs, mlir::Value rhs);
  mlir::Value neq(mlir::Value lhs, mlir::Value rhs);
  // Signed versions (index/signless/signed int or float)
  mlir::Value sgt(mlir::Value lhs, mlir::Value rhs); // No unsigned.
  mlir::Value sge(mlir::Value lhs, mlir::Value rhs); // No unsigned.
  mlir::Value slt(mlir::Value lhs, mlir::Value rhs); // No unsigned.
  mlir::Value sle(mlir::Value lhs, mlir::Value rhs); // No unsigned.
  // Unsigned versions
  mlir::Value ugt(mlir::Value lhs, mlir::Value rhs); // Unsigned int only
  mlir::Value uge(mlir::Value lhs, mlir::Value rhs); // Unsigned int only
  mlir::Value ult(mlir::Value lhs, mlir::Value rhs); // Unsigned int only
  mlir::Value ule(mlir::Value lhs, mlir::Value rhs); // Unsigned int only

  mlir::Value min(mlir::Value lhs, mlir::Value rhs);
  mlir::Value max(mlir::Value lhs, mlir::Value rhs);

  mlir::Value constant(mlir::Type type, double val);
  mlir::Value constantIndex(int64_t val);

  mlir::TypedAttr negativeInfAttr(mlir::Type type);
  mlir::TypedAttr positiveInfAttr(mlir::Type type);

  /// Emit a negative infinity constant of a specific type. Supported types:
  /// F16, F32, F64, Int8, Int16, Int32, Int64. In case of Float, emit the
  /// negative of the positive infinity. In case of Integer, emit the minimum
  /// mlir::Value.
  mlir::Value negativeInf(mlir::Type type);

  /// Emit a positive infinity constant of a specific type. Supported types:
  /// F16, F32, F64, Int8, Int16, Int32, Int64. In case of Integer, emit the
  /// maximum mlir::Value.
  mlir::Value positiveInf(mlir::Type type);

  // Cast handle bool/int/float/index elementary types. Do not convert
  // signed/index to unsigned.
  mlir::Value cast(mlir::Type destType, mlir::Value val);
  mlir::Value castToIndex(mlir::Value val);

  // Add indexOffsets to the least significant indices. So if indices are (i,
  // j, k, l) and offsets are (K, L), the results will be (i, j, k+K, l+L).
  void addOffsetToLeastSignificant(mlir::ValueRange indices,
      mlir::ValueRange offsets,
      llvm::SmallVectorImpl<mlir::Value> &computedIndices);
  void addOffsetToLeastSignificant(mlir::ArrayRef<IndexExpr> indices,
      mlir::ValueRange offsets,
      llvm::SmallVectorImpl<mlir::Value> &computedIndices);

private:
  mlir::Value createArithCmp(
      mlir::Value lhs, mlir::Value rhs, mlir::arith::CmpIPredicate pred);
  mlir::Value createArithCmp(
      mlir::Value lhs, mlir::Value rhs, mlir::arith::CmpFPredicate pred);
  mlir::Value castToSignless(mlir::Value source, int64_t width);
  mlir::Value castToUnsigned(mlir::Value source, int64_t width);
};

//===----------------------------------------------------------------------===//
// Shape Builder
//===----------------------------------------------------------------------===//

struct ShapeBuilder final : WithLoc<mlir::OpBuilder> {
  using WithLoc<mlir::OpBuilder>::WithLoc;
  ShapeBuilder(WithLoc<mlir::OpBuilder> &b) : WithLoc<mlir::OpBuilder>(b){};
  //   ShapeBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  //   ShapeBuilder(mlir::OpBuilder &b, mlir::Location loc)
  //       : DialectBuilder(b, loc) {}
  //   ShapeBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~ShapeBuilder() {}

  mlir::Value dim(mlir::Value val, int64_t index);
  mlir::Value shapeOf(mlir::Value val);
  mlir::Value getExtent(mlir::Value val, int64_t index);
};

//===----------------------------------------------------------------------===//
// MemRef Builder with added support for aligned memory
//===----------------------------------------------------------------------===//

// Default alignment attribute for all allocation of memory. On most system,
// it numElems is 16 bytes.
static constexpr int64_t gDefaultAllocAlign = 16;

struct MemRefBuilder final : WithLoc<mlir::OpBuilder> {
  using WithLoc<mlir::OpBuilder>::WithLoc;
  MemRefBuilder(WithLoc<mlir::OpBuilder> &b) : WithLoc<mlir::OpBuilder>(b){};
  //   MemRefBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  //   MemRefBuilder(mlir::OpBuilder &b, mlir::Location loc)
  //       : DialectBuilder(b, loc) {}
  //   MemRefBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~MemRefBuilder() {}

  // Constants
  static const int64_t defaultAlign;

  // Info: get static and dynamic size of memory. Return true if static only.
  bool getStaticAndDynamicMemSize(mlir::MemRefType type,
      mlir::ValueRange dynSymbols, int64_t &staticSize, IndexExpr &dynSize);
  bool getStaticAndDynamicMemSize(mlir::MemRefType type,
      llvm::SmallVectorImpl<IndexExpr> &dims, int64_t &staticSize,
      IndexExpr &dynSize);

  // Alloc for static shapes without alignment.
  mlir::memref::AllocOp alloc(mlir::MemRefType type);
  // Alloc for static/dynamic shapes without alignment.
  mlir::memref::AllocOp alloc(
      mlir::MemRefType type, mlir::ValueRange dynSymbols);
  mlir::memref::AllocOp alloc(
      mlir::Value operandOfSameType, mlir::MemRefType type);
  mlir::memref::AllocOp alloc(
      mlir::MemRefType type, llvm::SmallVectorImpl<IndexExpr> &dims);

  // Alloc for static shapes with alignment.
  // Minimum alignment is gDefaultAllocAlign.
  mlir::memref::AllocOp alignedAlloc(
      mlir::MemRefType type, int64_t align = defaultAlign);
  // Alloc for static/dynamic shapes with alignment.
  mlir::memref::AllocOp alignedAlloc(mlir::MemRefType type,
      mlir::ValueRange dynSymbols, int64_t align = defaultAlign);
  mlir::memref::AllocOp alignedAlloc(mlir::Value operandOfSameType,
      mlir::MemRefType type, int64_t align = defaultAlign);
  mlir::memref::AllocOp alignedAlloc(mlir::MemRefType type,
      llvm::SmallVectorImpl<IndexExpr> &dims, int64_t align = defaultAlign);

  // Alloc for shapes with alignment and padding for safe full SIMD
  // operations. Padding may be added so that every values in the shape may
  // safely be computed by a SIMD operation (or possibly multiple ones when
  // simdUnroll>1). Minimum alignment is gDefaultAllocAlign. Operation does
  // not support layouts at this time.
  //
  // Alloc for static shapes with alignment and SIMD padding.
  mlir::Value alignedAllocWithSimdPadding(mlir::MemRefType type,
      int64_t simdUnroll = 1, int64_t align = defaultAlign);
  // Alloc for static/dynamic shapes with alignment and SIMD padding.
  mlir::Value alignedAllocWithSimdPadding(mlir::MemRefType type,
      mlir::ValueRange dynSymbols, int64_t simdUnroll = 1,
      int64_t align = defaultAlign);
  mlir::Value alignedAllocWithSimdPadding(mlir::Value operandOfSameType,
      mlir::MemRefType type, int64_t simdUnroll = 1,
      int64_t align = defaultAlign);
  mlir::Value alignedAllocWithSimdPadding(mlir::MemRefType type,
      llvm::SmallVectorImpl<IndexExpr> &dims, int64_t simdUnroll = 1,
      int64_t align = defaultAlign);

  // The alloca instruction allocates memory on the stack frame of the
  // currently executing function, to be automatically released when this
  // function returns to its caller. It is strongly suggested to place alloca
  // instructions outside of a loop.
  mlir::memref::AllocaOp alloca(mlir::MemRefType type);
  mlir::memref::AllocaOp alignedAlloca(
      mlir::MemRefType type, int64_t align = defaultAlign);

  mlir::memref::DeallocOp dealloc(mlir::Value val);

  // Reshapes.
  mlir::memref::ReshapeOp reshape(mlir::MemRefType destType,
      mlir::Value valToReshape, mlir::Value destShapeStoredInMem);
  // Flatten dimsToFlatten innermost dimensions, -1 means all.
  mlir::Value reshapeToFlat(mlir::Value valToReshape,
      llvm::SmallVectorImpl<IndexExpr> &nDims, mlir::Value &flattenedSize,
      int64_t dimsToFlatten = -1);
  mlir::memref::ReshapeOp reshapeFromFlat(mlir::Value valToReshape,
      llvm::SmallVectorImpl<IndexExpr> &nDims, mlir::MemRefType outputType);

  // Casts.
  mlir::memref::CastOp cast(mlir::Value input, mlir::MemRefType outputType);
  mlir::Value reinterpretCast(
      mlir::Value input, llvm::SmallVectorImpl<IndexExpr> &outputDims);

  // Does not support layouts at this time. Does only work for values that are
  // then loaded with affine or memref scalar load/store (MLIR limitations).
  mlir::Value collapseShape(mlir::Value input,
      llvm::ArrayRef<mlir::ReassociationIndices> reassociation);

  // Create a view of input value (<byte size>xi8) starting at byteOffset and
  // shaped by outputType.
  mlir::memref::ViewOp view(mlir::Value input, int64_t byteOffset,
      mlir::MemRefType outputType, mlir::ValueRange outputDynSymbols);

  // Create a subview of val. Size of 1 => remove that dim.
  mlir::memref::SubViewOp subView(mlir::Value val,
      llvm::SmallVectorImpl<IndexExpr> &offsets, // Offset for each val dims.
      llvm::SmallVectorImpl<IndexExpr> &sizes,   // Sizes for each val dims.
      llvm::SmallVectorImpl<IndexExpr> &strides) // Stride for each val dims.
      ;

  mlir::Value dim(mlir::Value val, int64_t index);
  mlir::Value dim(mlir::Value val, mlir::Value index);

private:
  mlir::IntegerAttr computeAlignment(int64_t alignment);
  void computeDynSymbols(
      mlir::MemRefType type, // Use type to determine dynamic dimensions.
      llvm::SmallVectorImpl<IndexExpr> &dims, // Get dyn syms from index expr.
      llvm::SmallVectorImpl<mlir::Value> &dynSymbols) // Output dim symbols.
      ;
  void computeDynSymbols(
      mlir::Value operandOfSameType, // Extract dyn symbols from this value.
      mlir::MemRefType type, // Use type to determine dynamic dimensions.
      llvm::SmallVectorImpl<mlir::Value> &dynSymbols) // Output dim symbols.
      ;
};

//===----------------------------------------------------------------------===//
// Structured Control Flow (SCF) Builder
//===----------------------------------------------------------------------===//

struct SCFBuilder final : WithLoc<mlir::OpBuilder> {
  using WithLoc<mlir::OpBuilder>::WithLoc;
  SCFBuilder(WithLoc<mlir::OpBuilder> &b) : WithLoc<mlir::OpBuilder>(b){};
  virtual ~SCFBuilder() {}

  /// Create an if then with optional else. Construct does not generate a
  /// result (unlike some scf::if) and introduces the yields automatically.
  void ifThenElse(mlir::Value cond,
      mlir::function_ref<void(SCFBuilder &createSCF)> thenFn,
      mlir::function_ref<void(SCFBuilder &createSCF)> elseFn = nullptr);

  void parallelLoop(mlir::ValueRange lowerBounds, mlir::ValueRange upperBounds,
      mlir::ValueRange steps,
      mlir::function_ref<void(SCFBuilder &, mlir::ValueRange)> bodyFn);
  void yield();
};

//===----------------------------------------------------------------------===//
// Vector Builder
//===----------------------------------------------------------------------===//

struct VectorBuilder final : WithLoc<mlir::OpBuilder> {
  using WithLoc<mlir::OpBuilder>::WithLoc;
  VectorBuilder(WithLoc<mlir::OpBuilder> &b) : WithLoc<mlir::OpBuilder>(b){};
  virtual ~VectorBuilder() {}

  // Get the machine SIMD vector length for the given elementary type.
  // This can help guide certain optimizations.
  int64_t getMachineVectorLength(const mlir::Type &elementType);
  int64_t getMachineVectorLength(const mlir::VectorType &vecType);
  int64_t getMachineVectorLength(mlir::Value vecValue);

  mlir::Value load(mlir::VectorType vecType, mlir::Value memref,
      mlir::ValueRange indices = {});
  // When ranks of offsets<indices, add offsets to the least significant dims.
  mlir::Value load(mlir::VectorType vecType, mlir::Value memref,
      mlir::ValueRange indices, mlir::ValueRange offsets);
  mlir::Value loadIE(mlir::VectorType vecType, mlir::Value memref,
      llvm::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets);
  void store(
      mlir::Value val, mlir::Value memref, mlir::ValueRange indices = {});
  // When ranks of offsets<indices, add offsets to the least significant dims.
  void store(mlir::Value val, mlir::Value memref, mlir::ValueRange indices,
      mlir::ValueRange offsets);
  void storeIE(mlir::Value val, mlir::Value memref,
      llvm::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets);

  // Splat: a single value is copied.
  mlir::Value splat(mlir::VectorType vecType, mlir::Value val);
  // Broadcast: possibly a N dim vector is copied to M>N dim vector.
  mlir::Value broadcast(mlir::VectorType vecType, mlir::Value val);
  // Shuffle: use mask to determine which value to write to the output.
  mlir::Value shuffle(
      mlir::Value lhs, mlir::Value rhs, llvm::SmallVectorImpl<int64_t> &mask);
  mlir::Value fma(mlir::Value lhs, mlir::Value rhs, mlir::Value acc);

  // Composite functions.
  mlir::Value mergeHigh(mlir::Value lhs, mlir::Value rhs, int64_t step);
  mlir::Value mergeLow(mlir::Value lhs, mlir::Value rhs, int64_t step);
  void multiReduction(llvm::SmallVectorImpl<mlir::Value> &inputVecArray,
      llvm::SmallVectorImpl<mlir::Value> &outputVecArray);

private:
  bool isPowerOf2(uint64_t num);
  uint64_t getLengthOf1DVector(mlir::Value vec);
};

//===----------------------------------------------------------------------===//
// Affine Builder
//===----------------------------------------------------------------------===//

template <class LOAD_OP, class STORE_OP>
struct GenericAffineBuilder final : WithLoc<mlir::OpBuilder> {
  using WithLoc<mlir::OpBuilder>::WithLoc;
  GenericAffineBuilder(WithLoc<mlir::OpBuilder> &b)
      : WithLoc<mlir::OpBuilder>(b){};
  virtual ~GenericAffineBuilder() {}

  mlir::Value load(mlir::Value memref, mlir::ValueRange indices = {});
  // When ranks of offsets<indices, add offsets to the least significant dims.
  mlir::Value load(
      mlir::Value memref, mlir::ValueRange indices, mlir::ValueRange offsets);
  mlir::Value loadIE(mlir::Value memref, llvm::ArrayRef<IndexExpr> indices,
      mlir::ValueRange offsets);

  void store(
      mlir::Value val, mlir::Value memref, mlir::ValueRange indices = {});
  // When ranks of offsets<indices, add offsets to the least significant dims.
  void store(mlir::Value val, mlir::Value memref, mlir::ValueRange indices,
      mlir::ValueRange offsets);
  void storeIE(mlir::Value val, mlir::Value memref,
      llvm::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets);

  void forIE(IndexExpr lb, IndexExpr ub, int64_t step,
      mlir::function_ref<void(GenericAffineBuilder &, mlir::Value)> builderFn);
  void forIE(llvm::SmallVectorImpl<IndexExpr> &lbs,
      llvm::SmallVectorImpl<IndexExpr> &ubs,
      llvm::SmallVectorImpl<int64_t> &steps,
      mlir::function_ref<void(GenericAffineBuilder &, mlir::ValueRange)>
          builderFn);

  // This if then else construct has no arguments to the blocks.
  void ifThenElse(IndexExprScope &scope,
      llvm::SmallVectorImpl<IndexExpr> &conditions,
      mlir::function_ref<void(GenericAffineBuilder &createAffine)> thenFn,
      mlir::function_ref<void(GenericAffineBuilder &createAffine)> elseFn);

  // AffineApplyOp
  mlir::Value apply(mlir::AffineMap map, mlir::ValueRange operands);

  void yield();

private:
  // Support for multiple forIE loops.
  void recursionForIE(llvm::SmallVectorImpl<IndexExpr> &lbs,
      llvm::SmallVectorImpl<IndexExpr> &ubs,
      llvm::SmallVectorImpl<int64_t> &steps,
      llvm::SmallVectorImpl<mlir::Value> &loopIndices,
      mlir::function_ref<void(GenericAffineBuilder &, mlir::ValueRange)>
          builderFn);

  // Support for adding blocks.
  void appendToBlock(
      mlir::Block *block, mlir::function_ref<void(mlir::ValueRange)> builderFn);
};

// Affine builder uses affine load and store for memory operations. A later
// definition of AffineBuilderKrnlMem will use Krnl load and store for memory
// operations. We recommend to use AffineBuilderKrnlMem when converting the
// Krnl dialect into the affine dialect.
using AffineBuilder = GenericAffineBuilder<mlir::affine::AffineLoadOp,
    mlir::affine::AffineStoreOp>;

//===----------------------------------------------------------------------===//
// LLVM Builder
//===----------------------------------------------------------------------===//

struct LLVMBuilder final : WithLoc<mlir::OpBuilder> {
  using WithLoc<mlir::OpBuilder>::WithLoc;
  using voidFuncRef = mlir::function_ref<void(LLVMBuilder &createLLVM)>;
  using valueFuncRef = mlir::function_ref<mlir::Value(LLVMBuilder &createLLVM)>;
  LLVMBuilder(WithLoc<mlir::OpBuilder> &b) : WithLoc<mlir::OpBuilder>(b){};

  virtual ~LLVMBuilder() {}

  // AddOp
  mlir::Value add(mlir::Value lhs, mlir::Value rhs);

  // AddressOfOp
  mlir::Value addressOf(mlir::LLVM::GlobalOp op);

  // AllocaOp
  mlir::Value _alloca(mlir::Type resultType, mlir::Type elementType,
      mlir::Value size, int64_t alignment);

  // BitcastOp
  mlir::Value bitcast(mlir::Type type, mlir::Value val);

  // BrOp
  void br(llvm::ArrayRef<mlir::Value> destOperands, mlir::Block *destBlock);

  // CallOp
  mlir::Value call(mlir::ArrayRef<mlir::Type> resultTypes,
      llvm::StringRef funcName, mlir::ArrayRef<mlir::Value> inputs);
  mlir::Value call(mlir::ArrayRef<mlir::Type> resultTypes,
      mlir::FlatSymbolRefAttr funcSymbol, mlir::ArrayRef<mlir::Value> inputs);

  // CondBrOp
  void condBr(mlir::Value cond, mlir::Block *trueBlock,
      llvm::ArrayRef<mlir::Value> trueOperands, mlir::Block *falseBlock,
      llvm::ArrayRef<mlir::Value> falseOperands);

  // ConstantOp
  mlir::Value constant(mlir::Type type, int64_t val);
  mlir::Value constant(mlir::Type type, double val);

  // ExtractValueOp
  mlir::Value extractValue(mlir::Type resultType, mlir::Value container,
      llvm::ArrayRef<int64_t> position);

  // FuncOp
  mlir::LLVM::LLVMFuncOp func(llvm::StringRef name, mlir::Type type);

  // GEPOp
  mlir::Value getElemPtr(mlir::Type resultType, mlir::Type elemType,
      mlir::Value base, llvm::ArrayRef<mlir::LLVM::GEPArg> indices);

  // GlobalOp
  mlir::LLVM::GlobalOp globalOp(mlir::Type resultType, bool isConstant,
      mlir::LLVM::Linkage, llvm::StringRef name, mlir::Attribute attr,
      uint64_t alignment = 0);

  // ICmpOp
  mlir::Value icmp(
      mlir::LLVM::ICmpPredicate cond, mlir::Value lhs, mlir::Value rhs);

  // InsertValueOp
  mlir::Value insertValue(mlir::Type resultType, mlir::Value container,
      mlir::Value val, llvm::ArrayRef<int64_t> position);

  // Inttoptr
  mlir::Value inttoptr(mlir::Type type, mlir::Value val);

  // LoadOp
  mlir::Value load(mlir::Type elementType, mlir::Value addr);

  // MulOp
  mlir::Value mul(mlir::Value lhs, mlir::Value rhs);

  // NullOp
  mlir::Value null(mlir::Type type);

  // Ptrtoint
  mlir::Value ptrtoint(mlir::Type type, mlir::Value val);

  // ReturnOp
  void _return(mlir::Value val);

  // SExtOp
  mlir::Value sext(mlir::Type type, mlir::Value val);

  // StoreOp
  void store(mlir::Value val, mlir::Value addr);

  //===--------------------------------------------------------------------===//
  // Helper functions
  //===--------------------------------------------------------------------===//

  // Get or insert a function declaration at the beginning of the module.
  mlir::FlatSymbolRefAttr getOrInsertSymbolRef(mlir::ModuleOp module,
      llvm::StringRef symName, mlir::Type resultType,
      llvm::ArrayRef<mlir::Type> operandTypes, bool isVarArg = false);

  /// Generate code that looks like "if then with optional else" at LLVM.
  /// The following prototype code will be generated:
  /// ```
  /// llvm.condBr cond, ^thenBlock, ^elseBlock
  /// ^thenBlock:
  ///   thenBody
  /// ^elseBlock:
  ///   elseBody
  /// ^mainBlock
  ///   ...
  /// ```
  void ifThenElse(
      valueFuncRef cond, voidFuncRef thenFn, voidFuncRef elseFn = nullptr);
};

//===----------------------------------------------------------------------===//
// Multi Dialect Builder
//===----------------------------------------------------------------------===//

/*
  Instead of creating multiple builders, e.g.

  KrnlBuilder createKrnl(rewriter, loc);
  MathBuilder createMath(createKrnl);
  MemRefBuilder createMemRef(createKrnl);

  createKrnl.defineLoop(1);
  createMath.add(i1, i2);
  createMemRef.alloca(type);

  We can create a single builder composed of multiple types

  MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder>
    create(rewriter, loc);

  create.krnl.defineLoop(1);
  create.math.add(i1, i2);
  create.mem.alloca(type);

  Types that can be used here are
  *  AffineBuilder, access field with affine
  *  AffineBuilderKrnlMem, access field with affineKMem
  *  KrnlBuilder, access field with krnl
  *  MathBuilder, access field with math
  *  MemRefBuilder, access field with mem
  *  ONNXBuilder, access field with onnx
  *  SCFBuilder, access field with scf
  *  VectorBuilder, access field with vec
*/

// Anchor class.
template <class... Ts>
struct MultiDialectBuilder {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : builder(&b), location(loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : builder(db.getBuilderPtr()), location(db.getLoc()) {}
  template <typename Builder>
  MultiDialectBuilder(WithLoc<Builder> &b)
      : builder(&b), location(b.getLoc()) {}
  MultiDialectBuilder(mlir::OpBuilder &b)
      : builder(&b), location(mlir::UnknownLoc()) {}

  // Public getters of builder and location.
  mlir::OpBuilder &getBuilder() const {
    assert(builder);
    return *builder;
  }
  mlir::OpBuilder *getBuilderPtr() const { return builder; }
  mlir::Location getLoc() const { return location; }

private:
  mlir::OpBuilder *builder;
  mlir::Location location;
};

// Recursive class specialized for MathBuilder refereed to as math.
template <class... Ts>
struct MultiDialectBuilder<MathBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), math(b, loc) {}
  MultiDialectBuilder(mlir::OpBuilder &b)
      : MultiDialectBuilder<Ts...>(b), math(b){};
  template <typename Builder>
  MultiDialectBuilder(WithLoc<Builder> &b)
      : MultiDialectBuilder<Ts...>(b), math(b){};
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), math(db) {}
  MathBuilder math;
};

// Recursive class specialized for ShapeBuilder refereed to as shape.
template <class... Ts>
struct MultiDialectBuilder<ShapeBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), shape(b, loc) {}
  MultiDialectBuilder(mlir::OpBuilder &b)
      : MultiDialectBuilder<Ts...>(b), shape(b){};
  template <typename Builder>
  MultiDialectBuilder(WithLoc<Builder> &b)
      : MultiDialectBuilder<Ts...>(b), shape(b){};
  ShapeBuilder shape;
};

// Recursive class specialized for MemRefBuilder refereed to as mem.
template <class... Ts>
struct MultiDialectBuilder<MemRefBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), mem(b, loc) {}
  MultiDialectBuilder(mlir::OpBuilder &b)
      : MultiDialectBuilder<Ts...>(b), mem(b){};
  template <typename Builder>
  MultiDialectBuilder(WithLoc<Builder> &b)
      : MultiDialectBuilder<Ts...>(b), mem(b){};
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), mem(db.getBuilder()) {}
  MemRefBuilder mem;
};

// Recursive class specialized for AffineBuilder refereed to as affine.
template <class... Ts>
struct MultiDialectBuilder<AffineBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), affine(b, loc) {}
  MultiDialectBuilder(mlir::OpBuilder &b)
      : MultiDialectBuilder<Ts...>(b), affine(b){};
  template <typename Builder>
  MultiDialectBuilder(WithLoc<Builder> &b)
      : MultiDialectBuilder<Ts...>(b), affine(b){};
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), affine(db) {}
  AffineBuilder affine;
};

// Recursive class specialized for SCFBuilder refereed to as scf.
template <class... Ts>
struct MultiDialectBuilder<SCFBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), scf(b, loc) {}
  MultiDialectBuilder(mlir::OpBuilder &b)
      : MultiDialectBuilder<Ts...>(b), scf(b){};
  template <typename Builder>
  MultiDialectBuilder(WithLoc<Builder> &b)
      : MultiDialectBuilder<Ts...>(b), scf(b){};
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), scf(db) {}
  SCFBuilder scf;
};

// Recursive class specialized for VectorBuilder refereed to as vec.
template <class... Ts>
struct MultiDialectBuilder<VectorBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), vec(b, loc) {}
  MultiDialectBuilder(mlir::OpBuilder &b)
      : MultiDialectBuilder<Ts...>(b), vec(b){};
  template <typename Builder>
  MultiDialectBuilder(WithLoc<Builder> &b)
      : MultiDialectBuilder<Ts...>(b), vec(b){};
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), vec(db) {}
  VectorBuilder vec;
};

// Recursive class specialized for LLVMBuilder refereed to as llvm.
template <class... Ts>
struct MultiDialectBuilder<LLVMBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), llvm(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), llvm(db) {}
  template <typename Builder>
  MultiDialectBuilder(WithLoc<Builder> &b)
      : MultiDialectBuilder<Ts...>(b), llvm(b){};
  MultiDialectBuilder(mlir::OpBuilder &b)
      : MultiDialectBuilder<Ts...>(b), llvm(b){};
  LLVMBuilder llvm;
};

// Include template implementations.
#include "DialectBuilder.hpp.inc"

} // namespace onnx_mlir
