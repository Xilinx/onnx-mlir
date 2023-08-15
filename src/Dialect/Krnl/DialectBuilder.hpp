/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------- DialectBuilder.hpp - Krnl Dialect Builder -----------------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file declares the Krnl Dialect Builder.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"

namespace onnx_mlir {

//====-------------------- Support for Krnl Builder ----------------------===//

struct KrnlBuilder : WithLoc<mlir::OpBuilder> {
  using WithLoc<mlir::OpBuilder>::WithLoc;
  KrnlBuilder(WithLoc<mlir::OpBuilder> &b) : WithLoc<mlir::OpBuilder>(b){};
  virtual ~KrnlBuilder() {}

  mlir::Value load(mlir::Value memref, mlir::ValueRange indices = {});
  // When ranks of offsets<indices, add offsets to the least significant dims.
  mlir::Value load(
      mlir::Value memref, mlir::ValueRange indices, mlir::ValueRange offsets);
  mlir::Value loadIE(mlir::Value memref, mlir::ArrayRef<IndexExpr> indices);
  void store(
      mlir::Value val, mlir::Value memref, mlir::ValueRange indices = {});
  // When ranks of offsets<indices, add offsets to the least significant dims.
  void store(mlir::Value val, mlir::Value memref, mlir::ValueRange indices,
      mlir::ValueRange offsets);
  void storeIE(
      mlir::Value val, mlir::Value memref, mlir::ArrayRef<IndexExpr> indices);

  void seqstore(mlir::Value element, mlir::Value seq, mlir::Value index);
  void seqstore(mlir::Value element, mlir::Value seq, IndexExpr index);

  mlir::Value vectorTypeCast(mlir::Value sourceMemref, int64_t vectorLen);

  mlir::ValueRange defineLoops(int64_t originalLoopNum);
  mlir::ValueRange block(mlir::Value loop, int64_t blockSize);
  void permute(mlir::ValueRange loops, mlir::ArrayRef<int64_t> map);
  mlir::ValueRange getInductionVarValue(mlir::ValueRange loops);

  // Lambda passes loop indices as 2nd parameter.
  void iterate(mlir::ValueRange originalLoops, mlir::ValueRange optimizedLoops,
      mlir::ValueRange lbs, mlir::ValueRange ubs,
      mlir::function_ref<void(
          KrnlBuilder &createKrnl, mlir::ValueRange indices)>
          bodyBuilderFn);
  mlir::KrnlIterateOp iterate(const krnl::KrnlIterateOperandPack &operands);

  // Lambda passes loop indices as 2nd parameter.
  void iterateIE(mlir::ValueRange originalLoops,
      mlir::ValueRange optimizedLoops, mlir::ArrayRef<IndexExpr> lbs,
      mlir::ArrayRef<IndexExpr> ubs,
      mlir::function_ref<void(
          KrnlBuilder &createKrnl, mlir::ValueRange indices)>
          bodyBuilderFn);

  void copyToBuffer(
      // Buffer and source memory. Source memref may have a higher rank than
      // buffer.
      mlir::Value bufferMemref, mlir::Value sourceMemref,
      // Indices that points to the first data to be copied from source.
      // Starts has the same rank as sourceMemref.
      mlir::ValueRange starts,
      // If padding is needed, value to pad.
      mlir::Value padValue,
      // Now the bufferMemref may be larger than the actual data to be stored
      // in the buffer, if the user want to pad the data to a higher size.
      // TileSize enables the user to
      mlir::ArrayRef<int64_t> tileSize, mlir::ArrayRef<int64_t> padToNext,
      bool transpose = false);
  void copyToBuffer(mlir::Value bufferMemref, mlir::Value sourceMemref,
      mlir::ValueRange starts, mlir::Value padValue, bool transpose = false);

  void copyFromBuffer(mlir::Value bufferMemref, mlir::Value memref,
      mlir::ValueRange starts, mlir::ArrayRef<int64_t> tileSize);
  void copyFromBuffer(
      mlir::Value bufferMemref, mlir::Value memref, mlir::ValueRange starts);

  void matmul(
      // The a/b/cStart are the indices at the beginning of the buffer/mem
      // A/B/C.
      mlir::Value A, mlir::ValueRange aStart, mlir::Value B,
      mlir::ValueRange bStart, mlir::Value C, mlir::ValueRange cStart,
      // Loops are the krnl loop indices that this matmul replaces
      mlir::ValueRange loops,
      // the computeStarts indicate the i/j/k indices pointing to the beginning
      // of the matmul computation.
      mlir::ValueRange computeStarts,
      // The globalUBs are the global bounds on the original I, J, K
      // dimensions.
      mlir::ValueRange globalUBs,
      // If not the full A, B, C buffers are used by this matmul, meaning the
      // matmul uses a subtile of the buffers, this compute tile size
      // specifies the actual size of the i/j/k computations. Empty means
      // compute tiles encompass the entire buffer A, B, and C as defined by
      // their tile sizes.
      mlir::ArrayRef<int64_t> computeTileSize,
      // If buffers A, B, or C were padded, then the tile sizes give the size
      // of the non-padded data, basically the size of the data when the tile
      // is full. Partial tiles (due to computation on the edges of the
      // matrices) are handled differently (using the UBs), so no need to
      // worry about this. Empty means no padding was used.
      mlir::ArrayRef<int64_t> aTileSize, mlir::ArrayRef<int64_t> bTileSize,
      mlir::ArrayRef<int64_t> cTileSize,
      // Optimizations for code gen.
      bool simdize, bool unroll, bool overCompute);
  void matmul(mlir::Value A, mlir::ValueRange aStart, mlir::Value B,
      mlir::ValueRange bStart, mlir::Value C, mlir::ValueRange cStart,
      mlir::ValueRange loops, mlir::ValueRange computeStarts,
      mlir::ValueRange globalUBs, bool simdize, bool unroll, bool overCompute);

  mlir::Value dim(mlir::Type type, mlir::Value alloc, mlir::Value index);

  mlir::KrnlMovableOp movable();

  mlir::KrnlGetRefOp getRef(mlir::Type type, mlir::Value memref,
      mlir::Value offset, mlir::ValueRange indices = {});

  mlir::Value constant(mlir::MemRefType type, mlir::StringRef name,
      std::optional<mlir::Attribute> value,
      std::optional<mlir::IntegerAttr> offset = std::nullopt,
      std::optional<mlir::IntegerAttr> alignment = std::nullopt);

  // C library functions.
  void memcpy(mlir::Value dest, mlir::Value src, mlir::Value numElems);
  void memcpy(mlir::Value dest, mlir::Value src, mlir::Value numElems,
      mlir::Value destOffset, mlir::Value srcOffset);
  void memset(mlir::Value dest, mlir::Value val, bool delayed = false);
  mlir::Value strncmp(mlir::Value str1, mlir::Value str2, mlir::Value len);
  mlir::Value strlen(mlir::Value str);
  void printf(mlir::StringRef msg);
  void printf(mlir::StringRef msg, mlir::Value input, mlir::Type inputType,
      bool endsWithNewLine = false);
  void printf(mlir::Value input, mlir::Type inputType);

  // Onnx-mlir runtime functions.
  void randomNormal(mlir::Value alloc, mlir::Value numberOfRandomValues,
      mlir::Value mean, mlir::Value scale, mlir::Value seed);
  mlir::Value findIndex(
      mlir::Value input, mlir::Value G, mlir::Value V, mlir::Value len);
  void printTensor(mlir::StringRef msg, mlir::Value input);
};

//====--- Support for Affine Builder with Krnl Mem Ops ------------------===//

// We use here a Affine builder that generates Krnl Load and Store ops instead
// of the affine memory ops directly. This is because we can still generate
// Krnl Ops while lowering the dialect, and the big advantage of the Krnl memory
// operations is that they distinguish themselves if they are affine or not.
using AffineBuilderKrnlMem =
    GenericAffineBuilder<mlir::KrnlLoadOp, mlir::KrnlStoreOp>;

// =============================================================================
// IndexExpr Builder for building
// =============================================================================

struct IndexExprBuilderForKrnl : IndexExprBuilder {
  IndexExprBuilderForKrnl(mlir::Location loc) : IndexExprBuilder(loc) {}
  IndexExprBuilderForKrnl(mlir::OpBuilder &b, mlir::Location loc)
      : IndexExprBuilder(b, loc) {}
  IndexExprBuilderForKrnl(const DialectBuilder &db) : IndexExprBuilder(db) {}
  IndexExprBuilderForKrnl(WithLoc<mlir::OpBuilder> &b) : IndexExprBuilder(b) {}
  virtual ~IndexExprBuilderForKrnl() {}

protected:
  mlir::ElementsAttr getConst(mlir::Value value) final;
  mlir::Value getVal(mlir::Value intArrayVal, uint64_t i) final;
  mlir::Value getShapeVal(mlir::Value tensorOrMemrefValue, uint64_t i) final;
};

// =============================================================================
// MultiDialectBuilder for Krnl
// =============================================================================

// Recursive class specialized for AffineBuilderKrnlMem refereed to as
// affineKMem.
template <class... Ts>
struct MultiDialectBuilder<AffineBuilderKrnlMem, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), affineKMem(loc, b) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), affineKMem(db) {}
  MultiDialectBuilder(mlir::OpBuilder &b)
      : MultiDialectBuilder<Ts...>(b), affineKMem(b){};
  template <typename Builder>
  MultiDialectBuilder(WithLoc<Builder> &b)
      : MultiDialectBuilder<Ts...>(b), affineKMem(b){};
  AffineBuilderKrnlMem affineKMem;
};

// Recursive class specialized for KrnlBuilder referred to as krnl.
template <class... Ts>
struct MultiDialectBuilder<KrnlBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), krnl(loc, b) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), krnl(db) {}
  MultiDialectBuilder(mlir::OpBuilder &b)
      : MultiDialectBuilder<Ts...>(b), krnl(b){};
  template <typename Builder>
  MultiDialectBuilder(WithLoc<Builder> &b)
      : MultiDialectBuilder<Ts...>(b), krnl(b){};
  KrnlBuilder krnl;
};

// Recursive class specialized for IndexExprBuilderForKrnl referred to as
// krnlIE.
template <class... Ts>
struct MultiDialectBuilder<IndexExprBuilderForKrnl, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), krnlIE(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), krnlIE(db) {}
  template <typename Builder>
  MultiDialectBuilder(WithLoc<Builder> &b)
      : MultiDialectBuilder<Ts...>(b), krnlIE(b){};
  IndexExprBuilderForKrnl krnlIE;
};

} // namespace onnx_mlir
