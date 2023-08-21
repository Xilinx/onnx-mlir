/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ DialectBuilder.cpp - Helper functions for MLIR dialects -------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for building MLIR operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

// Please do not add dependences on ONNX or KRNL dialects.
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/VectorMachineSupport.hpp"

#include <algorithm>

#define DEBUG_TYPE "dialect_builder"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Original code for MathBuilder is copied from LLVM MLIR Utils.cpp
// Modified here to add operations, add super class.
// License added here for this class for completeness.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

// Test for unsigned as signless are treated as signed. For reference, check in
// MLIR AffineToStandard where comparison of indices are done with slt and sgt,
// for example. Indices are signless. Also, in ONNX, we currently treat all
// ONNX Integers as MLIR signless, and only flag the ONNX Unsigned Integer as
// MLIR unsigned integer.

/* static */ Type MathBuilder::elementTypeWithVector(Type elementOrVectorType) {
  VectorType vectorType = elementOrVectorType.dyn_cast<VectorType>();
  if (vectorType)
    return vectorType.getElementType();
  return elementOrVectorType;
}

/* static */ Type MathBuilder::getTypeWithVector(
    VectorType vectorType, Type elementType) {
  if (vectorType)
    return VectorType::get(vectorType.getShape(), elementType);
  return elementType;
}

/* static */ bool MathBuilder::isIntegerWithVector(Type elementOrVectorType) {
  Type elementType = elementTypeWithVector(elementOrVectorType);
  return elementType.isa<IntegerType>() || elementType.isa<IndexType>();
}

/* static */ bool MathBuilder::isUnsignedIntegerWithVector(
    Type elementOrVectorType) {
  Type elementType = elementTypeWithVector(elementOrVectorType);
  return elementType.isUnsignedInteger();
}

/* static */ bool MathBuilder::isFloatWithVector(Type elementOrVectorType) {
  Type elementType = elementTypeWithVector(elementOrVectorType);
  return elementType.isa<FloatType>();
}

Value MathBuilder::abs(Value val) {
  if (isIntegerWithVector(val.getType()))
    return create<math::AbsIOp>(val);
  if (isFloatWithVector(val.getType()))
    return create<math::AbsFOp>(val);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::andi(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return create<arith::AndIOp>(lhs, rhs);
  llvm_unreachable("expected int");
}

Value MathBuilder::ori(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return create<arith::OrIOp>(lhs, rhs);
  llvm_unreachable("expected int");
}

Value MathBuilder::xori(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return create<arith::XOrIOp>(lhs, rhs);
  llvm_unreachable("expected int");
}

Value MathBuilder::add(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType())) {
    Type elemType = elementTypeWithVector(lhs.getType());
    if (elemType.isUnsignedInteger()) {
      unsigned elemWidth = elemType.cast<IntegerType>().getWidth();
      Value castLhs = castToSignless(lhs, elemWidth);
      Value castRhs = castToSignless(rhs, elemWidth);
      Value castAdd = create<arith::AddUIExtendedOp>(castLhs, castRhs).getSum();
      return castToUnsigned(castAdd, elemWidth);
    } else
      return create<arith::AddIOp>(lhs, rhs);
  }
  if (isFloatWithVector(lhs.getType()))
    return create<arith::AddFOp>(lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::sub(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return create<arith::SubIOp>(lhs, rhs);
  if (isFloatWithVector(lhs.getType()))
    return create<arith::SubFOp>(lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::mul(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType())) {
    Type elemType = elementTypeWithVector(lhs.getType());
    if (elemType.isUnsignedInteger()) {
      unsigned elemWidth = elemType.cast<IntegerType>().getWidth();
      Value castLhs = castToSignless(lhs, elemWidth);
      Value castRhs = castToSignless(rhs, elemWidth);
      Value castMul = create<arith::MulUIExtendedOp>(castLhs, castRhs).getLow();
      return castToUnsigned(castMul, elemWidth);
    } else
      return create<arith::MulIOp>(lhs, rhs);
  }
  if (isFloatWithVector(lhs.getType()))
    return create<arith::MulFOp>(lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::div(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isFloatWithVector(lhs.getType()))
    return create<arith::DivFOp>(lhs, rhs);
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return create<arith::DivUIOp>(lhs, rhs);
  if (isIntegerWithVector(lhs.getType()))
    return create<arith::DivSIOp>(lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::rem(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isFloatWithVector(lhs.getType()))
    return create<arith::RemFOp>(lhs, rhs);
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return create<arith::RemUIOp>(lhs, rhs);
  if (isIntegerWithVector(lhs.getType()))
    return create<arith::RemSIOp>(lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::copySign(mlir::Value rem, mlir::Value dividend) {
  assert(rem.getType() == dividend.getType() && "expected same type");
  if (isFloatWithVector(rem.getType()))
    return create<math::CopySignOp>(rem, dividend);
  llvm_unreachable("expected float");
}

Value MathBuilder::ceilDiv(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return create<arith::CeilDivUIOp>(lhs, rhs);
  if (isIntegerWithVector(lhs.getType()))
    return create<arith::CeilDivSIOp>(lhs, rhs);
  llvm_unreachable("expected int");
}

Value MathBuilder::floorDiv(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isUnsignedIntegerWithVector(lhs.getType()))
    // Using regular unsigned div is ok as it rounds toward zero.
    return this->create<arith::DivUIOp>(lhs, rhs);
  if (isIntegerWithVector(lhs.getType()))
    return this->create<arith::FloorDivSIOp>(lhs, rhs);
  llvm_unreachable("expected int");
}

// return (lhs * rhs) + acc
Value MathBuilder::fma(Value lhs, Value rhs, Value acc) {
  assert((lhs.getType() == rhs.getType()) && (rhs.getType() == acc.getType()) &&
         "expected same type");
  if (isFloatWithVector(lhs.getType()) && !isa<FloatType>(lhs.getType()))
    return this->create<vector::FMAOp>(lhs, rhs, acc);
  return add(mul(lhs, rhs), acc);
}

Value MathBuilder::exp(Value val) {
  if (isFloatWithVector(val.getType()))
    return this->create<math::ExpOp>(val);
  llvm_unreachable("expected float");
}

Value MathBuilder::exp2(Value val) {
  if (isFloatWithVector(val.getType()))
    return this->create<math::Exp2Op>(val);
  llvm_unreachable("expected float");
}

Value MathBuilder::log(Value val) {
  if (isFloatWithVector(val.getType()))
    return this->create<math::LogOp>(val);
  llvm_unreachable("expected float");
}

Value MathBuilder::log2(Value val) {
  if (isFloatWithVector(val.getType()))
    return this->create<math::Log2Op>(val);
  llvm_unreachable("expected float");
}

Value MathBuilder::sqrt(Value val) {
  if (isFloatWithVector(val.getType()))
    return this->create<math::SqrtOp>(val);
  llvm_unreachable("expected float");
}

Value MathBuilder::pow(Value base, Value exp) {
  if (isFloatWithVector(base.getType()))
    return this->create<math::PowFOp>(base, exp);
  llvm_unreachable("expected base float");
}

Value MathBuilder::neg(Value val) {
  if (isIntegerWithVector(val.getType()))
    // Returns 0 - val.
    return sub(constant(val.getType(), 0), val);
  if (isFloatWithVector(val.getType()))
    return this->create<arith::NegFOp>(val);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::ceil(Value val) {
  if (isFloatWithVector(val.getType()))
    return this->create<math::CeilOp>(val);
  llvm_unreachable("expected float");
}

Value MathBuilder::floor(Value val) {
  if (isFloatWithVector(val.getType()))
    return this->create<math::FloorOp>(val);
  llvm_unreachable("expected float");
}

Value MathBuilder::min(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isFloatWithVector(lhs.getType()))
    return this->create<arith::MinFOp>(lhs, rhs);
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return this->create<arith::MinUIOp>(lhs, rhs);
  if (isIntegerWithVector(lhs.getType()))
    return this->create<arith::MinSIOp>(lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::max(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isFloatWithVector(lhs.getType()))
    return this->create<arith::MaxFOp>(lhs, rhs);
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return this->create<arith::MaxUIOp>(lhs, rhs);
  if (isIntegerWithVector(lhs.getType()))
    return this->create<arith::MaxSIOp>(lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::sgt(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::sgt);
  if (isFloatWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OGT);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::sge(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::sge);
  if (isFloatWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OGE);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::slt(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::slt);
  if (isFloatWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OLT);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::sle(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::sle);
  if (isFloatWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OLE);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::ugt(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::ugt);
  llvm_unreachable("expected unsigned int");
}

Value MathBuilder::uge(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::uge);
  llvm_unreachable("expected unsigned int");
}

Value MathBuilder::ult(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::ult);
  llvm_unreachable("expected unsigned int");
}

Value MathBuilder::ule(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::ule);
  llvm_unreachable("expected unsigned int");
}

Value MathBuilder::gt(Value lhs, Value rhs) {
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return ugt(lhs, rhs);
  return sgt(lhs, rhs);
}

Value MathBuilder::ge(Value lhs, Value rhs) {
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return uge(lhs, rhs);
  return sge(lhs, rhs);
}

Value MathBuilder::lt(Value lhs, Value rhs) {
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return ult(lhs, rhs);
  return slt(lhs, rhs);
}

Value MathBuilder::le(Value lhs, Value rhs) {
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return ule(lhs, rhs);
  return sle(lhs, rhs);
}

Value MathBuilder::eq(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::eq);
  if (isFloatWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OEQ);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::neq(Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::ne);
  if (isFloatWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::ONE);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::select(Value cmp, Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return this->create<arith::SelectOp>(cmp, lhs, rhs);
}

Value MathBuilder::constant(Type type, double val) {
  Value constant = nullptr;
  // Could be a vector type; look at the element type.
  Type elementType = elementTypeWithVector(type);
  TypeSwitch<Type>(elementType)
      .Case<Float16Type>([&](Type) {
        constant = this->create<arith::ConstantOp>(this->getF16FloatAttr(val));
      })
      .Case<Float32Type>([&](Type) {
        constant = this->create<arith::ConstantOp>(this->getF32FloatAttr(val));
      })
      .Case<Float64Type>([&](Type) {
        constant = this->create<arith::ConstantOp>(this->getF64FloatAttr(val));
      })
      .Case<IntegerType>([&](IntegerType elementType) {
        assert(val == (int64_t)val && "value is ambiguous");
        unsigned width = elementType.getWidth();

        if (width == 1)
          constant =
              this->create<arith::ConstantOp>(this->getBoolAttr(val != 0));
        else {
          // If unsigned, create a signless constant, then cast it to unsigned.
          if (elementType.isUnsignedInteger()) {
            Type signlessTy = this->getIntegerType(width);
            constant = this->create<arith::ConstantOp>(
                this->getIntegerAttr(signlessTy, APInt(width, (int64_t)val)));
            constant = castToUnsigned(constant, width);
          } else {
            constant = this->create<arith::ConstantOp>(
                this->getIntegerAttr(elementType, APInt(width, (int64_t)val)));
          }
        }
      })
      .Case<IndexType>([&](Type elementType) {
        constant = this->create<arith::ConstantOp>(
            this->getIntegerAttr(elementType, val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  if (type.isa<VectorType>()) {
    // For vectors, need to splat the constant.
    MultiDialectBuilder<VectorBuilder> create(*this, getLoc());
    VectorType vecType = type.dyn_cast<VectorType>();
    constant = create.vec.splat(vecType, constant);
  }
  return constant;
}

Value MathBuilder::constantIndex(int64_t val) {
  IntegerAttr constantAttr = this->getIntegerAttr(this->getIndexType(), val);
  return this->create<arith::ConstantOp>(constantAttr);
}

TypedAttr MathBuilder::negativeInfAttr(mlir::Type type) {
  TypedAttr attr;
  TypeSwitch<Type>(type)
      .Case<Float32Type>([&](Type) {
        attr = this->getF32FloatAttr(-std::numeric_limits<float>::infinity());
      })
      .Case<Float64Type>([&](Type) {
        attr = this->getF64FloatAttr(-std::numeric_limits<double>::infinity());
      })
      .Case<IntegerType>([&](IntegerType type) {
        unsigned width = type.getWidth();
        bool isSignless = type.isSignless();
        bool isSigned = type.isSigned();
        int64_t value;
        switch (width) {
        case 8:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int8_t>::min()
                      : std::numeric_limits<uint8_t>::min();
          break;
        case 16:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int16_t>::min()
                      : std::numeric_limits<uint16_t>::min();
          break;
        case 32:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int32_t>::min()
                      : std::numeric_limits<uint32_t>::min();
          break;
        case 64:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int64_t>::min()
                      : std::numeric_limits<uint64_t>::min();
          break;
        default:
          llvm_unreachable("unsupported element type");
        }
        attr = this->getIntegerAttr(type, APInt(width, value));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });
  assert(attr != nullptr && "Expecting valid attribute");
  return attr;
}

TypedAttr MathBuilder::positiveInfAttr(mlir::Type type) {
  TypedAttr attr;
  TypeSwitch<Type>(type)
      .Case<Float32Type>([&](Type) {
        attr = this->getF32FloatAttr(std::numeric_limits<float>::infinity());
      })
      .Case<Float64Type>([&](Type) {
        attr = this->getF64FloatAttr(std::numeric_limits<double>::infinity());
      })
      .Case<IntegerType>([&](IntegerType type) {
        unsigned width = type.getWidth();
        bool isSignless = type.isSignless();
        bool isSigned = type.isSigned();
        int64_t value;
        switch (width) {
        case 8:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int8_t>::max()
                      : std::numeric_limits<uint8_t>::max();
          break;
        case 16:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int16_t>::max()
                      : std::numeric_limits<uint16_t>::max();
          break;
        case 32:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int32_t>::max()
                      : std::numeric_limits<uint32_t>::max();
          break;
        case 64:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int64_t>::max()
                      : std::numeric_limits<uint64_t>::max();
          break;
        default:
          llvm_unreachable("unsupported element type");
        }
        attr = this->getIntegerAttr(type, APInt(width, value));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });
  assert(attr != nullptr && "Expecting valid attribute");
  return attr;
}

Value MathBuilder::negativeInf(Type type) {
  TypedAttr attr = negativeInfAttr(type);
  Value constant = this->create<arith::ConstantOp>(attr);
  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value MathBuilder::positiveInf(Type type) {
  TypedAttr attr = positiveInfAttr(type);
  Value constant = this->create<arith::ConstantOp>(attr);
  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value MathBuilder::createArithCmp(
    Value lhs, Value rhs, arith::CmpIPredicate pred) {
  Type type = lhs.getType();
  assert(type == rhs.getType() && "Operands should have the same type");
  assert(isIntegerWithVector(type) && "expected int");
  return this->create<arith::CmpIOp>(pred, lhs, rhs);
}

Value MathBuilder::createArithCmp(
    Value lhs, Value rhs, arith::CmpFPredicate pred) {
  Type type = lhs.getType();
  assert(type == rhs.getType() && "Operands should have the same type");
  assert(isFloatWithVector(type) && "expected float");
  return this->create<arith::CmpFOp>(pred, lhs, rhs);
}

// Several operations in the arith dialect require signless integers. This
// cast remove the sign of integer types for successful processing, to the
// best of my understanding.
Value MathBuilder::castToSignless(Value val, int64_t width) {
  Type valType = val.getType();
  VectorType vecType = valType.dyn_cast<VectorType>();
  Type valElemType = elementTypeWithVector(valType);
  assert(valElemType.isa<IntegerType>() && !valElemType.isSignlessInteger() &&
         "Expecting signed integer type");
  Type destType = getTypeWithVector(vecType, this->getIntegerType(width));
  return create<UnrealizedConversionCastOp>(destType, val).getResult(0);
}

Value MathBuilder::castToUnsigned(Value val, int64_t width) {
  Type valType = val.getType();
  VectorType vecType = valType.dyn_cast<VectorType>();
  Type valElemType = elementTypeWithVector(valType);
  assert(valElemType.isa<IntegerType>() && "Expecting integer type");
  Type destType =
      getTypeWithVector(vecType, this->getIntegerType(width, false /*signed*/));
  return create<UnrealizedConversionCastOp>(destType, val).getResult(0);
}

// Methods inspired from MLIR TosaToLinalg CastOp.
Value MathBuilder::cast(Type destType, Value src) {
  // Get element type and vector types (if any, i.e. possibly nullptr).
  Type srcType = src.getType();
  VectorType srcVecType = srcType.dyn_cast<VectorType>();
  VectorType destVecType = destType.dyn_cast<VectorType>();
  Type srcElemType = elementTypeWithVector(srcType);
  Type destElemType = elementTypeWithVector(destType);
  // Make sure we don't mix vector and scalars.
  assert(((srcVecType && destVecType) || (!srcVecType && !destVecType)) &&
         "expect both to be scalars or vectors");
  // Check if we even need a cast.
  if (srcType == destType)
    return src;

  // Process index types first.
  if (srcElemType.isa<IndexType>()) {
    // If the source is an index type, first convert it into a signless int of
    // size 64.
    srcElemType = this->getIntegerType(64);
    srcType = getTypeWithVector(srcVecType, srcElemType);
    src = this->create<arith::IndexCastOp>(srcType, src);
  }
  bool destIsIndex = false;
  Type savedDestType = destType; // Used when destIsIndex is true.
  if (destElemType.isa<IndexType>()) {
    // If the dest is an index type, pretend for now that we want it to be
    // converted to signless int of size 64.
    destElemType = this->getIntegerType(64);
    destType = getTypeWithVector(destVecType, destElemType);
    destIsIndex = true;
  }

  // Only support Integer or Float type at this stage. Index were transformed
  // to signless int.
  // TODO: add support for shaped tensor (MemRef, Vector, Tensor?) if needed.
  assert((srcElemType.isa<IntegerType>() || srcElemType.isa<FloatType>()) &&
         "support only float or int");
  assert((destElemType.isa<IntegerType>() || destElemType.isa<FloatType>()) &&
         "support only float or int");
  // Get source and dest type width.
  int64_t srcElemWidth = srcElemType.getIntOrFloatBitWidth();
  int64_t destElemWidth = destElemType.getIntOrFloatBitWidth();
  bool bitExtend = srcElemWidth < destElemWidth;
  bool bitTrunc = srcElemWidth > destElemWidth;

  LLVM_DEBUG(llvm::dbgs() << "srcType: " << srcType << "\n";
             llvm::dbgs() << "destType: " << destType << "\n";);

  // Handle boolean first because they need special handling.
  // Boolean to int/float conversions. Boolean are unsigned.
  if (srcElemType.isInteger(1)) {
    if (destElemType.isa<FloatType>()) {
      return this->create<arith::UIToFPOp>(destType, src);
    } else {
      Value dest = this->create<arith::ExtUIOp>(destType, src);
      if (destIsIndex)
        dest = this->create<arith::IndexCastOp>(savedDestType, dest);
      return dest;
    }
  }

  // Int/Float to booleans, just compare value to be unequal zero.
  if (destElemType.isInteger(1)) {
    Type constantType = srcType;
    if (srcElemType.isa<IntegerType>() && !srcElemType.isSignlessInteger()) {
      // An integer constant must be signless.
      unsigned srcElemWidth = srcElemType.cast<IntegerType>().getWidth();
      constantType = getTypeWithVector(
          srcVecType, IntegerType::get(srcElemType.getContext(), srcElemWidth));
      src = castToSignless(src, srcElemWidth);
    }
    Value zero = constant(constantType, 0);
    return neq(src, zero);
  }

  // Float to float conversions.
  if (srcElemType.isa<FloatType>() && destElemType.isa<FloatType>()) {
    assert((bitExtend || bitTrunc) && "expected extend or trunc");
    if (bitExtend)
      return this->create<arith::ExtFOp>(destType, src);
    else
      return this->create<arith::TruncFOp>(destType, src);
  }

  // Float to int conversions.
  if (srcElemType.isa<FloatType>() && destElemType.isa<IntegerType>()) {
    // TosaToLinalg in MLIR uses a fancier algorithm that clamps values to
    // min/max signed/unsigned integer values.
    if (destType.isUnsignedInteger()) {
      Type castType = this->getIntegerType(destElemWidth);
      Value cast = this->create<arith::FPToUIOp>(castType, src);
      return castToUnsigned(cast, destElemWidth);
    } else {
      // Handle signed int.
      Value dest = this->create<arith::FPToSIOp>(destType, src);
      if (destIsIndex)
        dest = this->create<arith::IndexCastOp>(savedDestType, dest);
      return dest;
    }
  }

  // Int to float conversion.
  if (srcElemType.isa<IntegerType>() && destElemType.isa<FloatType>()) {
    if (srcElemType.isUnsignedInteger()) {
      Value cast = castToSignless(src, srcElemWidth);
      return this->create<arith::UIToFPOp>(destType, cast);
    } else {
      // Handle signed int.
      return this->create<arith::SIToFPOp>(destType, src);
    }
  }

  // Int to int conversion.
  if (srcType.isa<IntegerType>() && destType.isa<IntegerType>()) {
    if (srcType.isUnsignedInteger()) {
      // Unsigned to unsigned/signed conversion.
      // Same bit width for unsigned to signed conversion.
      if ((srcElemWidth == destElemWidth) && destType.isSignlessInteger())
        return castToSignless(src, srcElemWidth);
      // Different bit width.
      assert((bitExtend || bitTrunc) && "expected extend or trunc");
      // Has to convert to signless first, and reconvert output to unsigned.
      Value cast = castToSignless(src, srcElemWidth);
      Type castType = this->getIntegerType(destElemWidth);
      if (bitExtend) {
        cast = this->create<arith::ExtUIOp>(castType, cast);
      } else {
        // TosaToLinalg use a clipping algo, not sure if needed.
        cast = this->create<arith::TruncIOp>(castType, cast);
      }
      if (destType.isUnsignedInteger()) {
        // Unsigned to unsigned conversion.
        return castToUnsigned(cast, destElemWidth);
      } else {
        // Unsigned to signed conversion.
        return cast;
      }
    } else {
      // Signed to unsigned/signed conversion.
      // Handle signed integer
      // Same bit width for signed to unsigned conversion.
      if ((srcElemWidth == destElemWidth) && destType.isUnsignedInteger())
        return castToUnsigned(src, srcElemWidth);
      // Different bit width.
      Value dest = src;
      if (bitExtend)
        dest = this->create<arith::ExtSIOp>(destType, src);
      if (bitTrunc)
        // TosaToLinalg use a clipping algo
        dest = this->create<arith::TruncIOp>(destType, src);
      if (destIsIndex)
        return this->create<arith::IndexCastOp>(this->getIndexType(), dest);
      if (destType.isUnsignedInteger()) {
        return castToUnsigned(dest, destElemWidth);
      } else {
        return dest;
      }
    }
  }

  // Handled all the cases supported so far.
  llvm_unreachable("unsupported element type");
  return nullptr;
}

Value MathBuilder::castToIndex(Value src) {
  return cast(this->getIndexType(), src);
}

// Add offsets to least significant values in indices. So if indices has 4
// values, (i, j, k, l) and offsets has 2 values (K, L), the results will be (i,
// j, k+K, l+L).
void MathBuilder::addOffsetToLeastSignificant(mlir::ValueRange indices,
    mlir::ValueRange offsets,
    llvm::SmallVectorImpl<mlir::Value> &computedIndices) {
  int64_t indexRank = indices.size();
  int64_t offsetRank = offsets.size();
  int64_t firstOffset = indexRank - offsetRank;
  assert(firstOffset >= 0 && "indexOffset should not have a higher rank than "
                             "the indices in the memref");
  computedIndices.clear();
  for (int64_t i = 0; i < indexRank; i++) {
    if (i < firstOffset) {
      computedIndices.emplace_back(indices[i]);
    } else {
      computedIndices.emplace_back(add(offsets[i - firstOffset], indices[i]));
    }
  }
}

void MathBuilder::addOffsetToLeastSignificant(mlir::ArrayRef<IndexExpr> indices,
    ValueRange offsets, llvm::SmallVectorImpl<Value> &computedIndices) {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  addOffsetToLeastSignificant(indexValues, offsets, computedIndices);
}

//===----------------------------------------------------------------------===//
// Shape support.
//===----------------------------------------------------------------------===//

Value ShapeBuilder::dim(Value val, int64_t index) {
  Value inputShape = shapeOf(val);
  return getExtent(inputShape, index);
}

Value ShapeBuilder::shapeOf(Value val) {
  return this->create<shape::ShapeOfOp>(getLoc(), val);
}

Value ShapeBuilder::getExtent(Value val, int64_t index) {
  return this->create<shape::GetExtentOp>(getLoc(), val, index);
}

//===----------------------------------------------------------------------===//
// Memref support, including inserting default alignment.
//===----------------------------------------------------------------------===//

const int64_t MemRefBuilder::defaultAlign = -1;

//===----------------------------------------------------------------------===//
// Helper private functions.

// Compute alignment, which is at least gDefaultAllocAlign.
IntegerAttr MemRefBuilder::computeAlignment(int64_t alignment) {
  alignment = (alignment > gDefaultAllocAlign ? alignment : gDefaultAllocAlign);
  return this->getI64IntegerAttr(alignment);
}

// Alloc calls need a list of values, only for the dynamic shapes. Extract these
// values from the list of index expressions that represent the shape of the
// memref.
void MemRefBuilder::computeDynSymbols(MemRefType type,
    llvm::SmallVectorImpl<IndexExpr> &dims,
    llvm::SmallVectorImpl<Value> &dynSymbols) {
  dynSymbols.clear();
  int64_t rank = type.getRank();
  ArrayRef<int64_t> shape = type.getShape();
  for (int64_t i = 0; i < rank; ++i)
    if (shape[i] == ShapedType::kDynamic)
      dynSymbols.emplace_back(dims[i].getValue());
}

// Alloc calls need a list of values, only for the dynamic shapes. Extract these
// values from an existing operands that has the same shape. Use dim ops for
// each dynamic dimension.
void MemRefBuilder::computeDynSymbols(Value operandOfSameType, MemRefType type,
    llvm::SmallVectorImpl<Value> &dynSymbols) {
  dynSymbols.clear();
  if (operandOfSameType == nullptr)
    return;
  int64_t rank = type.getRank();
  ArrayRef<int64_t> shape = type.getShape();
  for (int64_t i = 0; i < rank; ++i)
    if (shape[i] == ShapedType::kDynamic)
      dynSymbols.emplace_back(dim(operandOfSameType, i));
}

//===----------------------------------------------------------------------===//
// Alloc functions without alignment.

memref::AllocOp MemRefBuilder::alloc(MemRefType type) {
  llvm::SmallVector<Value, 4> dynSymbols;
  return alloc(type, dynSymbols);
}

memref::AllocOp MemRefBuilder::alloc(MemRefType type, ValueRange dynSymbols) {
  // Constant, ignore the dynamic symbols.
  if (dynSymbols.size() == 0)
    return this->create<memref::AllocOp>(type);
  return this->create<memref::AllocOp>(type, dynSymbols);
}

memref::AllocOp MemRefBuilder::alloc(Value operandOfSameType, MemRefType type) {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(operandOfSameType, type, dynSymbols);
  return alloc(type, dynSymbols);
}

memref::AllocOp MemRefBuilder::alloc(
    MemRefType type, llvm::SmallVectorImpl<IndexExpr> &dims) {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(type, dims, dynSymbols);
  return alloc(type, dynSymbols);
}

//===----------------------------------------------------------------------===//
// Alloc functions with alignment.

memref::AllocOp MemRefBuilder::alignedAlloc(
    MemRefType type, int64_t alignment) {
  llvm::SmallVector<Value, 4> dynSymbols;
  return alignedAlloc(type, dynSymbols, alignment);
}

memref::AllocOp MemRefBuilder::alignedAlloc(
    MemRefType type, ValueRange dynSymbols, int64_t alignment) {
  // Drop align for scalars.
  if (type.getShape().size() == 0)
    return alloc(type, dynSymbols);
  // Has array, use alignment.
  IntegerAttr alignmentAttr = computeAlignment(alignment);
  // Constant, ignore the dynamic symbols.
  if (dynSymbols.size() == 0)
    return create<memref::AllocOp>(type, alignmentAttr);
  return create<memref::AllocOp>(type, dynSymbols, alignmentAttr);
}

memref::AllocOp MemRefBuilder::alignedAlloc(
    Value operandOfSameType, MemRefType type, int64_t alignment) {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(operandOfSameType, type, dynSymbols);
  return alignedAlloc(type, dynSymbols, alignment);
}

memref::AllocOp MemRefBuilder::alignedAlloc(MemRefType type,
    llvm::SmallVectorImpl<IndexExpr> &dims, int64_t alignment) {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(type, dims, dynSymbols);
  return alignedAlloc(type, dynSymbols, alignment);
}

//===----------------------------------------------------------------------===//
// Info about memory size.

// Compute static and dynamic size of memref. Return true if has static size.
bool MemRefBuilder::getStaticAndDynamicMemSize(MemRefType type,
    ValueRange dynSymbols, int64_t &staticSize, IndexExpr &dynSize) {
  Type elementType = type.getElementType();
  assert(!(elementType.isa<VectorType>()) && "unsupported vector type");
  ArrayRef<int64_t> shape = type.getShape();
  staticSize = 1;                // Multiplication of static sizes.
  dynSize = LiteralIndexExpr(1); // Multiplication of dyn sizes.
  bool staticShape = (dynSymbols.size() == 0);
  int64_t rank = type.getRank();
  int64_t iDim = 0;
  for (int64_t i = 0; i < rank; ++i) {
    if (shape[i] == ShapedType::kDynamic) {
      assert(!staticShape && "expected static shape");
      assert(iDim < (int64_t)dynSymbols.size() && "not enough dynamic symbols");
      dynSize = dynSize * SymbolIndexExpr(dynSymbols[iDim++]);
    } else {
      // Has constant shape.
      staticSize *= shape[i];
    }
  }
  return staticShape;
}

bool MemRefBuilder::getStaticAndDynamicMemSize(MemRefType type,
    llvm::SmallVectorImpl<IndexExpr> &dims, int64_t &staticSize,
    IndexExpr &dynSize) {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(type, dims, dynSymbols);
  return getStaticAndDynamicMemSize(type, dynSymbols, staticSize, dynSize);
}

//===----------------------------------------------------------------------===//
// Alloc functions with alignment and padding for SIMD

Value MemRefBuilder::alignedAllocWithSimdPadding(
    mlir::MemRefType type, int64_t simdUnroll, int64_t alignment) {
  llvm::SmallVector<Value, 4> dynSymbols;
  return alignedAllocWithSimdPadding(type, dynSymbols, simdUnroll, alignment);
}

Value MemRefBuilder::alignedAllocWithSimdPadding(MemRefType type,
    ValueRange dynSymbols, int64_t simdUnroll, int64_t alignment) {
  Type elementType = type.getElementType();
  assert(!hasNonIdentityLayout(type) && "unsupported layout");
  assert(!(elementType.isa<VectorType>()) && "unsupported vector type");
  assert(simdUnroll >= 1 && "expected positive simd unroll factor");
  // Compute total size of memref (in unit of element type).
  int64_t staticSize;
  IndexExpr dynSize;
  bool staticShape =
      getStaticAndDynamicMemSize(type, dynSymbols, staticSize, dynSize);
  // Get vector length for this element type, multiplied by the unroll factor.
  MultiDialectBuilder<VectorBuilder> create(*this, getLoc());
  int64_t VL = create.vec.getMachineVectorLength(elementType) * simdUnroll;
  // If the static size component is already a multiple of VL, no matter the
  // values of the dynamic shapes, the last value is part of a full SIMD. No
  // need for extra padding then.
  if (staticSize % VL == 0)
    return alignedAlloc(type, dynSymbols, alignment);

  // We now need some padding. VL as this is an upper bound on padding. Padding
  // in element size.
  int64_t paddingSize = VL;
  if (staticShape)
    // Static shape: we can pad by the exact right amount.
    paddingSize = VL - staticSize % VL;

  // Allocate data as byte.
  int64_t bitWidth = elementType.getIntOrFloatBitWidth();
  IndexExpr totPaddedByteSize;
  if (bitWidth % 8 == 0) {
    // We have elements that have sizes of 1 or more bytes.
    int64_t byteWidth = bitWidth / 8;
    IndexExpr totByteSize = LiteralIndexExpr(staticSize * byteWidth) * dynSize;
    totPaddedByteSize = totByteSize + LiteralIndexExpr(paddingSize * byteWidth);
  } else {
    // We have sub-byte element sizes. Need to do precise computations. Namely
    // first compute tot total number of bits (including static/dynamic
    // and padding bit sizes), and then doing a ceil division by
    // 8 (number of bits in a byte).
    IndexExpr totBitSize = LiteralIndexExpr(staticSize * bitWidth) * dynSize;
    IndexExpr totPaddedBitSize =
        totBitSize + LiteralIndexExpr(paddingSize * bitWidth);
    totPaddedByteSize = totPaddedBitSize.ceilDiv(LiteralIndexExpr(8));
  }
  if (staticShape)
    assert(totPaddedByteSize.isLiteral() && "expected literal padded tot size");
  // Construct memref for padded array of bytes.
  memref::AllocOp paddedAlloc;
  if (totPaddedByteSize.isLiteral()) {
    MemRefType paddedType =
        MemRefType::get({totPaddedByteSize.getLiteral()}, this->getI8Type());
    paddedAlloc = alignedAlloc(paddedType, alignment);
  } else {
    MemRefType paddedType =
        MemRefType::get({ShapedType::kDynamic}, this->getI8Type());
    paddedAlloc =
        alignedAlloc(paddedType, {totPaddedByteSize.getValue()}, alignment);
  }
  // Used to create a subview, it does not appear that the view cares about
  // whether the entire input data participates in the viewed data or not.
  return view(paddedAlloc, /*offset*/ 0, type, dynSymbols);
}

Value MemRefBuilder::alignedAllocWithSimdPadding(Value operandOfSameType,
    MemRefType type, int64_t simdUnroll, int64_t alignment) {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(operandOfSameType, type, dynSymbols);
  return alignedAllocWithSimdPadding(type, dynSymbols, simdUnroll, alignment);
}

Value MemRefBuilder::alignedAllocWithSimdPadding(MemRefType type,
    llvm::SmallVectorImpl<IndexExpr> &dims, int64_t simdUnroll,
    int64_t alignment) {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(type, dims, dynSymbols);
  return alignedAllocWithSimdPadding(type, dynSymbols, simdUnroll, alignment);
}

//===----------------------------------------------------------------------===//
// Alloca

memref::AllocaOp MemRefBuilder::alloca(MemRefType type) {
  return create<memref::AllocaOp>(type);
}

memref::AllocaOp MemRefBuilder::alignedAlloca(
    MemRefType type, int64_t alignment) {
  // Drop align for scalars.
  if (type.getShape().size() == 0)
    return create<memref::AllocaOp>(type);
  // Has array, use alignment.
  IntegerAttr alignmentAttr = computeAlignment(alignment);
  return create<memref::AllocaOp>(type, alignmentAttr);
}

//===----------------------------------------------------------------------===//
// Dealloc.

memref::DeallocOp MemRefBuilder::dealloc(Value val) {
  return create<memref::DeallocOp>(val);
}

//===----------------------------------------------------------------------===//
// Reshape.

memref::ReshapeOp MemRefBuilder::reshape(
    MemRefType destType, Value valToReshape, Value destShapeStoredInMem) {
  return create<memref::ReshapeOp>(
      destType, valToReshape, destShapeStoredInMem);
}

// Flatten the innermost dimsToFlatten of the value valToReshape. Return in
// flattenSize the cumulative size of the flattened dimensions. If flattenSize
// is -1, flatten them all. Expect to flatten at least 1 dim (which is a noop).
// Output rank is Rank(input) - dimsToFlatten + 1.
Value MemRefBuilder::reshapeToFlat(Value valToReshape,
    llvm::SmallVectorImpl<IndexExpr> &dims, Value &flattenedSize,
    int64_t dimsToFlatten) {
  // Parse input.
  MemRefType inputType = valToReshape.getType().cast<MemRefType>();
  int64_t inputRank = inputType.getRank();
  assert(inputRank == (int64_t)dims.size() && "rank mismatch");
  Type elementType = inputType.getElementType();
  assert(!hasNonIdentityLayout(inputType) && "MemRef is not normalized");
  // Set/check dimsToFlatten.
  if (dimsToFlatten == -1)
    dimsToFlatten = inputRank;
  assert(dimsToFlatten > 0 && dimsToFlatten <= inputRank &&
         "out of range dimsToFlatten");
  // Create scope to avoid issues.
  IndexExprScope innerScope(this, getLoc());
  MultiDialectBuilder<AffineBuilder, MathBuilder> create(*this, getLoc());
  // Compute total number of flattened elements in new scope.
  IndexExpr numOfFlattenedElements = LiteralIndexExpr(1);
  for (int64_t d = inputRank - dimsToFlatten; d < inputRank; ++d) {
    numOfFlattenedElements = numOfFlattenedElements * SymbolIndexExpr(dims[d]);
  }
  // flattenedSize is an output value corresponding to the total number of
  // elements that were flattened.
  flattenedSize = numOfFlattenedElements.getValue();
  if (dimsToFlatten == 1)
    // Flattening of the last dim is really no flattening at all. Return
    // original value before doing the actual reshaping, which is unnecessary.
    // Waited until here as we need to return a valid flattenedSize,
    return valToReshape;
  // Shape for reshaping from N-D to M-D saved into memory.
  int64_t outputRank = (inputRank - dimsToFlatten) + 1;
  Type indexType = this->getIndexType();
  Value outputShapeInMem =
      alignedAlloc(MemRefType::get({outputRank}, indexType));
  llvm::SmallVector<int64_t, 4> outputShape;
  // Compute shape and store it in memory.
  for (int64_t d = 0; d < outputRank; ++d) {
    Value dd = create.math.constantIndex(d);
    IndexExpr shapeIE =
        (d == outputRank - 1) ? numOfFlattenedElements : dims[d];
    create.affine.store(shapeIE.getValue(), outputShapeInMem, {dd});
    outputShape.emplace_back(shapeIE.getShape());
  }
  // Reshape the input N-D MemRef into a M-D MemRef.
  MemRefType outputType = MemRefType::get(outputShape, elementType);
  return reshape(outputType, valToReshape, outputShapeInMem);
}

memref::ReshapeOp MemRefBuilder::reshapeFromFlat(Value valToReshape,
    llvm::SmallVectorImpl<IndexExpr> &dims, MemRefType outputType) {
  assert(!hasNonIdentityLayout(outputType) && "MemRef is not normalized");
  MultiDialectBuilder<AffineBuilder, MathBuilder> create(*this, getLoc());
  Type indexType = this->getIndexType();
  int64_t rank = outputType.getRank();
  // Shape for reshaping from N1D to N-D saved into memory.
  Value shapeND = alignedAlloc(MemRefType::get({rank}, indexType));
  for (int64_t i = 0; i < rank; ++i) {
    Value index = create.math.constantIndex(i);
    create.affine.store(dims[i].getValue(), shapeND, {index});
  }
  // Reshape the 1-D MemRef into a N-D MemRef.
  return reshape(outputType, valToReshape, shapeND);
}

//===----------------------------------------------------------------------===//
// Casts and views.

memref::CastOp MemRefBuilder::cast(Value input, MemRefType outputType) {
  return create<memref::CastOp>(outputType, input);
}

Value MemRefBuilder::reinterpretCast(
    Value input, SmallVectorImpl<IndexExpr> &outputDims) {
  // Compute new sizes and strides.
  int64_t rank = outputDims.size();
  SmallVector<IndexExpr, 4> sizesIE, stridesIE;
  sizesIE.resize(rank);
  stridesIE.resize(rank);
  IndexExpr strideIE = LiteralIndexExpr(1);
  for (int i = rank - 1; i >= 0; --i) {
    sizesIE[i] = outputDims[i];
    stridesIE[i] = strideIE;
    if (i > 0)
      strideIE = strideIE * sizesIE[i];
  }
  // Compute output type
  SmallVector<int64_t, 4> outputShape;
  SmallVector<OpFoldResult, 4> sizes, strides;
  IndexExpr::getShape(outputDims, outputShape);
  IndexExpr::getOpOrFoldResults(sizesIE, sizes);
  IndexExpr::getOpOrFoldResults(stridesIE, strides);
  Type elementType = input.getType().cast<ShapedType>().getElementType();
  MemRefType outputMemRefType = MemRefType::get(outputShape, elementType);

  return create<memref::ReinterpretCastOp>(outputMemRefType, input,
      /*offset=*/this->getIndexAttr(0), sizes, strides);
}

Value MemRefBuilder::collapseShape(
    Value input, ArrayRef<ReassociationIndices> reassociation) {
  // Extract input info.
  MemRefType inputType = input.getType().cast<MemRefType>();
  assert(inputType && "expected input with memref type");
  assert(!hasNonIdentityLayout(inputType) &&
         "collapse only for identity layout at this time");
  int64_t inputRank = inputType.getRank();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  // Compute shape of output.
  int64_t outputRank = reassociation.size();
  SmallVector<int64_t, 4> outputShape;
  for (int64_t r = 0; r < outputRank; ++r) {
    int64_t indexNum = reassociation[r].size();
    assert(indexNum > 0 && "expect one or more index in reassociation indices");
    // Compute the cumulative size of the output dim as the product of all dim
    // of the sizes in the input being re-associated with this output.
    int64_t currShape = 1;
    for (int64_t i = 0; i < indexNum; i++) {
      int64_t ii = reassociation[r][i];
      assert(ii >= 0 && ii < inputRank && "out of bound reassociation index");
      int64_t ss = inputShape[ii];
      if (ss == ShapedType::kDynamic) {
        // If a re-associated shapes is dynamic, output is dynamic.
        currShape = ShapedType::kDynamic;
        break;
      }
      currShape *= ss;
    }
    outputShape.emplace_back(currShape);
  }
  // Compute type of output.
  MemRefType outputType =
      MemRefType::get(outputShape, inputType.getElementType());
  // Create collapse shape op.
  return create<memref::CollapseShapeOp>(outputType, input, reassociation);
}

memref::ViewOp MemRefBuilder::view(Value input, int64_t byteOffset,
    MemRefType outputType, ValueRange outputDynSymbols) {
  MultiDialectBuilder<MathBuilder> create(*this, getLoc());
  Value offset = create.math.constantIndex(byteOffset);
  // auto offset = this->createOrFold<arith::ConstantIndexOp>(byteOffset);
  return this->create<memref::ViewOp>(
      outputType, input, offset, outputDynSymbols);
}

memref::SubViewOp MemRefBuilder::subView(Value input,
    llvm::SmallVectorImpl<IndexExpr> &offsetsIE,
    llvm::SmallVectorImpl<IndexExpr> &sizesIE,
    llvm::SmallVectorImpl<IndexExpr> &stridesIE) {
  SmallVector<OpFoldResult, 4> offsets, sizes, strides;
  IndexExpr::getOpOrFoldResults(offsetsIE, offsets);
  IndexExpr::getOpOrFoldResults(sizesIE, sizes);
  IndexExpr::getOpOrFoldResults(stridesIE, strides);
  SmallVector<int64_t, 4> outputShape;
  IndexExpr::getShape(sizesIE, outputShape);
  MemRefType inputType = input.getType().dyn_cast<MemRefType>();
  MemRefLayoutAttrInterface layout;
  MemRefType outputType = MemRefType::get(outputShape,
      inputType.getElementType(), layout, inputType.getMemorySpace());
  return this->create<memref::SubViewOp>(
      outputType, input, offsets, sizes, strides);
}

//===----------------------------------------------------------------------===//
// Dims.

Value MemRefBuilder::dim(Value val, int64_t index) {
  assert(index >= 0 && "Expecting a valid index");
  return dim(val, this->create<arith::ConstantIndexOp>(index));
}

Value MemRefBuilder::dim(Value val, Value index) {
  // assert((val.getType().isa<MemRefType>() ||
  //           val.getType().isa<UnrankedMemRefType>()) &&
  //       "memref::DimOp expects input operand to have MemRefType or "
  //       "UnrankedMemRefType");
  return Value(createOrFold<memref::DimOp>(getLoc(), val, index));
}

//===----------------------------------------------------------------------===//
// Structured Control Flow (SCF).
//===----------------------------------------------------------------------===//

void SCFBuilder::ifThenElse(Value cond,
    function_ref<void(SCFBuilder &createSCF)> thenFn,
    function_ref<void(SCFBuilder &createSCF)> elseFn) {
  if (!elseFn) {
    this->create<scf::IfOp>(cond,
        /* then */
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childLoc, childBuilder);
          thenFn(scfBuilder);
          yield();
        });
  } else {
    this->create<scf::IfOp>(
        cond,
        /* then */
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childLoc, childBuilder);
          thenFn(scfBuilder);
          this->create<scf::YieldOp>();
        },
        /*else*/
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childLoc, childBuilder);
          elseFn(scfBuilder);
          yield();
        });
  }
}

void SCFBuilder::parallelLoop(ValueRange lowerBounds, ValueRange upperBounds,
    ValueRange steps,
    function_ref<void(SCFBuilder &createSCF, ValueRange)> bodyFn) {
  // SmallVectorImpl<Value> ivStorage;
  this->create<scf::ParallelOp>(lowerBounds, upperBounds, steps,
      [&](OpBuilder &childBuilder, Location childLoc,
          ValueRange inductionVars) {
        SCFBuilder builder(childLoc, childBuilder);
        bodyFn(builder, inductionVars);
        yield();
      });
}

void SCFBuilder::yield() { this->create<scf::YieldOp>(); }

//===----------------------------------------------------------------------===//
// Vector Builder
//===----------------------------------------------------------------------===//

int64_t VectorBuilder::getMachineVectorLength(const Type &elementType) {
  VectorMachineSupport *vms =
      VectorMachineSupport::getGlobalVectorMachineSupport();
  // Even if unsupported, we can always compute one result per vector.
  return std::max((int64_t)1, vms->getVectorLength(elementType));
}

int64_t VectorBuilder::getMachineVectorLength(const VectorType &vecType) {
  return getMachineVectorLength(vecType.getElementType());
}

int64_t VectorBuilder::getMachineVectorLength(Value vecValue) {
  VectorType vecType = vecValue.getType().dyn_cast_or_null<VectorType>();
  assert(vecType && "expected vector type");
  return getMachineVectorLength(vecType.getElementType());
}

Value VectorBuilder::load(
    VectorType vecType, Value memref, ValueRange indices) {
  return this->create<vector::LoadOp>(vecType, memref, indices);
}
mlir::Value VectorBuilder::load(mlir::VectorType vecType, mlir::Value memref,
    mlir::ValueRange indices, mlir::ValueRange offsets) {
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MultiDialectBuilder<MathBuilder> create(*this, getLoc());
  create.math.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  return load(vecType, memref, computedIndices);
}

mlir::Value VectorBuilder::loadIE(mlir::VectorType vecType, mlir::Value memref,
    llvm::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets) {
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MultiDialectBuilder<MathBuilder> create(*this, getLoc());
  create.math.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  return load(vecType, memref, computedIndices);
}

void VectorBuilder::store(Value val, Value memref, ValueRange indices) {
  this->create<vector::StoreOp>(val, memref, indices);
}

void VectorBuilder::store(mlir::Value val, mlir::Value memref,
    mlir::ValueRange indices, mlir::ValueRange offsets) {
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MultiDialectBuilder<MathBuilder> create(*this, getLoc());
  create.math.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  store(val, memref, computedIndices);
}

void VectorBuilder::storeIE(mlir::Value val, mlir::Value memref,
    llvm::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets) {
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MultiDialectBuilder<MathBuilder> create(*this, getLoc());
  create.math.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  store(val, memref, computedIndices);
}

Value VectorBuilder::fma(Value lhs, Value rhs, Value acc) {
  return this->create<vector::FMAOp>(lhs, rhs, acc);
}

// Val is required to be a index/integer/float.
Value VectorBuilder::splat(VectorType vecType, Value val) {
  return this->create<vector::SplatOp>(vecType, val);
}

Value VectorBuilder::broadcast(VectorType vecType, Value val) {
  return this->create<vector::BroadcastOp>(vecType, val);
}

Value VectorBuilder::shuffle(
    Value lhs, Value rhs, SmallVectorImpl<int64_t> &mask) {
  return this->create<vector::ShuffleOp>(lhs, rhs, mask);
}

// Private vector utilities.
bool VectorBuilder::isPowerOf2(uint64_t num) { return (num & (num - 1)) == 0; }

uint64_t VectorBuilder::getLengthOf1DVector(Value vec) {
  VectorType vecType = vec.getType().dyn_cast_or_null<VectorType>();
  assert(vecType && "expected a vector type");
  auto vecShape = vecType.getShape();
  assert(vecShape.size() == 1 && "expected a 1D vector");
  return vecShape[0];
}

Value VectorBuilder::mergeHigh(Value lhs, Value rhs, int64_t step) {
  // Inputs: lrs <l0, l1, l2, l3, l4, l5, l6, l7>;
  //         rhs <r0, r1, r2, r3, r4, r5, r6, r7>.
  // Merge alternatively the low (least significant) values of lrs and rhs
  // Step 1:     <(l0), (r0), (l1), (r1), (l2), (r2), (l3), (r3)> (1x sizes)
  // Step 2:     <(l0, l1),   (r0, r1),   (l2, l3),   (r2, r3)>   (2x sizes)
  // Step 4:     <(l0, l1, l2, l3),       (r0, r1, r2, r3)>       (4x sizes)
  uint64_t VL = getLengthOf1DVector(lhs);
  assert(getLengthOf1DVector(rhs) == VL && "expected same sized vectors");
  assert(isPowerOf2(VL) && "expected power of 2 vector length");
  SmallVector<int64_t, 8> mask(VL, 0);
  int i = 0;
  int64_t pairsOfLhsRhs = VL / (2 * step);
  int64_t firstHalf = 0;
  for (int64_t p = 0; p < pairsOfLhsRhs; ++p) {
    // One step-sized item from the LHS
    for (int64_t e = 0; e < step; ++e)
      mask[i++] = firstHalf + p * step + e;
    // One step-sized item from the RHS (RHS offset is VL for the shuffle op).
    for (int64_t e = 0; e < step; ++e)
      mask[i++] = firstHalf + VL + p * step + e;
  }
  return shuffle(lhs, rhs, mask);
}

Value VectorBuilder::mergeLow(Value lhs, Value rhs, int64_t step) {
  // Inputs: lrs <l0, l1, l2, l3, l4, l5, l6, l7>;
  //         rhs <r0, r1, r2, r3, r4, r5, r6, r7>.
  // Merge alternatively the low (least significant) values of lrs and rhs
  // Step 1:     <(l4), (r4), (l5), (r5), (l6), (r6), (l7), (r7)> (1x sizes)
  // Step 2:     <(l4, l5),   (r4, r5),   (l6, l7),   (r6, r7)>   (2x sizes)
  // Step 4:     <(l4, l5, l6, l7),       (r4, r5, r6, r7)>       (4x sizes)
  uint64_t VL = getLengthOf1DVector(lhs);
  assert(getLengthOf1DVector(rhs) == VL && "expected same sized vectors");
  assert(isPowerOf2(VL) && "expected power of 2 vector length");
  SmallVector<int64_t, 8> mask(VL, 0);
  int i = 0;
  int64_t pairsOfLhsRhs = VL / (2 * step);
  int64_t secondHalf = VL / 2;
  for (int64_t p = 0; p < pairsOfLhsRhs; ++p) {
    // One step-sized item from the LHS
    for (int64_t e = 0; e < step; ++e)
      mask[i++] = secondHalf + p * step + e;
    // One step-sized item from the RHS (RHS offset is VL for the shuffle op).
    for (int64_t e = 0; e < step; ++e)
      mask[i++] = secondHalf + VL + p * step + e;
  }
  return shuffle(lhs, rhs, mask);
}

// Do a parallel-simd reduction of N vectors of SIMD length VL.
// Restrictions:
// *  VL is the vector length of the machine SIMD vectors.
// *  N is a multiple of VL as we can perform consecutive VL x VL
//    reductions.
void VectorBuilder::multiReduction(SmallVectorImpl<Value> &inputVecArray,
    SmallVectorImpl<Value> &outputVecArray) {
  uint64_t N = inputVecArray.size();
  assert(N > 0 && "expected at least one value to reduce");
  uint64_t VL = getLengthOf1DVector(inputVecArray[0]);
  uint64_t machineVL = getMachineVectorLength(inputVecArray[0]);
  assert(VL == machineVL && "only natural sizes supported at this time");
  assert(N % machineVL == 0 &&
         "can only reduces multiple of VL vectors at this time");
  LLVM_DEBUG(llvm::dbgs() << "reduction with N " << N << ", VL " << VL
                          << ", mVL " << machineVL << "\n";);

  // Emplace all input vectors in a temporary array.
  SmallVector<Value, 8> tmpArray;
  for (uint64_t i = 0; i < N; ++i) {
    tmpArray.emplace_back(inputVecArray[i]);
    // Also verify that all have the same vector length.
    assert(getLengthOf1DVector(inputVecArray[i]) == VL &&
           "different vector length");
  }

  // Reductions of full physical vectors.
  outputVecArray.clear();
  MultiDialectBuilder<MathBuilder> create(*this, getLoc());
  for (uint64_t r = 0; r < N; r += machineVL) {
    // Algorithm for the set of input arrays from tmp[r] to
    // tmp[r+machineVL-1].
    uint64_t numPairs = machineVL / 2; // Pair number decrease by power of 2.
    for (uint64_t step = 1; step < machineVL; step = step * 2) {
      for (uint64_t p = 0; p < numPairs; ++p) {
        Value highVal =
            mergeHigh(tmpArray[r + 2 * p], tmpArray[r + 2 * p + 1], step);
        Value lowVal =
            mergeLow(tmpArray[r + 2 * p], tmpArray[r + 2 * p + 1], step);
        Value red = create.math.add(highVal, lowVal);
        tmpArray[r + p] = red;
      }
      numPairs = numPairs / 2; // Pair number decrease by power of 2.
    }
    // Completed the machineVL x machineVL reduction, save it in the output.
    outputVecArray.emplace_back(tmpArray[r]);
  }
}

//===----------------------------------------------------------------------===//
// LLVM Builder
//===----------------------------------------------------------------------===//

Value LLVMBuilder::add(Value lhs, Value rhs) {
  return this->create<LLVM::AddOp>(lhs, rhs);
}

Value LLVMBuilder::addressOf(LLVM::GlobalOp op) {
  return this->create<LLVM::AddressOfOp>(op);
}

Value LLVMBuilder::_alloca(
    Type resultType, Type elementType, Value size, int64_t alignment) {
  return this->create<LLVM::AllocaOp>(resultType, elementType, size, alignment);
}

Value LLVMBuilder::bitcast(Type type, Value val) {
  return this->create<LLVM::BitcastOp>(type, val);
}

void LLVMBuilder::br(ArrayRef<Value> destOperands, Block *destBlock) {
  this->create<LLVM::BrOp>(destOperands, destBlock);
}

Value LLVMBuilder::call(
    ArrayRef<Type> resultTypes, StringRef funcName, ArrayRef<Value> inputs) {
  assert((resultTypes.size() == 0 || resultTypes.size() == 1) &&
         "LLVM:CallOp must return either 0 or 1 value");
  LLVM::CallOp callOp =
      this->create<LLVM::CallOp>(resultTypes, funcName, inputs);
  // CallOp may return either 0 or 1 value.
  if (resultTypes.empty())
    return nullptr;
  return callOp.getResult();
}

Value LLVMBuilder::call(ArrayRef<Type> resultTypes,
    FlatSymbolRefAttr funcSymbol, ArrayRef<Value> inputs) {
  assert((resultTypes.size() == 0 || resultTypes.size() == 1) &&
         "LLVM:CallOp must return either 0 or 1 value");
  LLVM::CallOp callOp =
      this->create<LLVM::CallOp>(resultTypes, funcSymbol, inputs);
  // CallOp may return either 0 or 1 value.
  if (resultTypes.empty())
    return nullptr;
  return callOp.getResult();
}

void LLVMBuilder::condBr(Value cond, Block *trueBlock,
    llvm::ArrayRef<Value> trueOperands, Block *falseBlock,
    llvm::ArrayRef<Value> falseOperands) {
  this->create<LLVM::CondBrOp>(
      cond, trueBlock, trueOperands, falseBlock, falseOperands);
}

Value LLVMBuilder::constant(Type type, int64_t val) {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<IntegerType>([&](IntegerType type) {
        unsigned width = type.getWidth();
        if (width == 1)
          constant =
              this->create<LLVM::ConstantOp>(type, this->getBoolAttr(val != 0));
        else {
          assert(type.isSignless() &&
                 "LLVM::ConstantOp requires a signless type.");
          constant = this->create<LLVM::ConstantOp>(
              type, this->getIntegerAttr(type, APInt(width, (int64_t)val)));
        }
      })
      .Case<IndexType>([&](Type) {
        constant = this->create<LLVM::ConstantOp>(
            type, this->getIntegerAttr(type, val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value LLVMBuilder::constant(Type type, double val) {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<Float16Type>([&](Type) {
        constant =
            this->create<LLVM::ConstantOp>(type, this->getF16FloatAttr(val));
      })
      .Case<Float32Type>([&](Type) {
        constant =
            this->create<LLVM::ConstantOp>(type, this->getF32FloatAttr(val));
      })
      .Case<Float64Type>([&](Type) {
        constant =
            this->create<LLVM::ConstantOp>(type, this->getF64FloatAttr(val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value LLVMBuilder::extractValue(
    Type resultType, Value container, ArrayRef<int64_t> position) {
  return this->create<LLVM::ExtractValueOp>(resultType, container, position);
}

LLVM::LLVMFuncOp LLVMBuilder::func(StringRef name, Type type) {
  return this->create<LLVM::LLVMFuncOp>(name, type);
}

Value LLVMBuilder::getElemPtr(Type resultType, Type elemType, Value base,
    ArrayRef<LLVM::GEPArg> indices) {
  return this->create<LLVM::GEPOp>(resultType, elemType, base, indices);
}

LLVM::GlobalOp LLVMBuilder::globalOp(Type resultType, bool isConstant,
    LLVM::Linkage linkage, StringRef name, Attribute valueAttr,
    uint64_t alignment) {
  return this->create<LLVM::GlobalOp>(resultType,
      /*isConstant=*/isConstant, linkage, name, valueAttr);
}

Value LLVMBuilder::icmp(LLVM::ICmpPredicate cond, Value lhs, Value rhs) {
  return this->create<LLVM::ICmpOp>(cond, lhs, rhs);
}

Value LLVMBuilder::insertValue(Type resultType, Value container, Value val,
    llvm::ArrayRef<int64_t> position) {
  return this->create<LLVM::InsertValueOp>(
      resultType, container, val, position);
}

Value LLVMBuilder::inttoptr(Type type, Value val) {
  return this->create<LLVM::IntToPtrOp>(type, val);
}

Value LLVMBuilder::load(Type elementType, Value addr) {
  return this->create<LLVM::LoadOp>(elementType, addr);
}

Value LLVMBuilder::mul(Value lhs, Value rhs) {
  return this->create<LLVM::MulOp>(lhs, rhs);
}

Value LLVMBuilder::null(Type type) { return this->create<LLVM::NullOp>(type); }

Value LLVMBuilder::ptrtoint(Type type, Value val) {
  return this->create<LLVM::PtrToIntOp>(type, val);
}

void LLVMBuilder::_return(Value val) {
  this->create<LLVM::ReturnOp>(ArrayRef<Value>({val}));
}

Value LLVMBuilder::sext(Type type, Value val) {
  return this->create<LLVM::SExtOp>(type, val);
}

void LLVMBuilder::store(Value val, Value addr) {
  this->create<LLVM::StoreOp>(val, addr);
}

FlatSymbolRefAttr LLVMBuilder::getOrInsertSymbolRef(ModuleOp module,
    StringRef funcName, Type resultType, ArrayRef<Type> operandTypes,
    bool isVarArg) {
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    OpBuilder::InsertionGuard guard(*this);
    this->setInsertionPointToStart(module.getBody());
    LLVM::LLVMFunctionType funcType =
        LLVM::LLVMFunctionType::get(resultType, operandTypes, isVarArg);
    this->create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);
  }
  return SymbolRefAttr::get(this->getContext(), funcName);
}

void LLVMBuilder::ifThenElse(
    valueFuncRef cond, voidFuncRef thenFn, voidFuncRef elseFn) {
  LLVMBuilder createLLVM(*this, getLoc());

  // Split the current block into IF, THEN, ELSE and END blocks.
  Block *ifBlock, *thenBlock, *elseBlock, *endBlock;
  ifBlock = this->getInsertionBlock();
  thenBlock = ifBlock->splitBlock(this->getInsertionPoint());
  elseBlock = this->createBlock(
      thenBlock->getParent(), std::next(Region::iterator(thenBlock)));
  if (elseFn)
    endBlock = this->createBlock(
        elseBlock->getParent(), std::next(Region::iterator(elseBlock)));
  else
    endBlock = elseBlock;

  // Emit code for the IF block.
  createLLVM.setInsertionPointToEnd(ifBlock);
  Value condVal = cond(createLLVM);

  // Branch the block into the THEN and ELSE blocks.
  createLLVM.condBr(condVal, thenBlock, {}, elseBlock, {});

  // Emit code for the THEN block.
  createLLVM.setInsertionPointToStart(thenBlock);
  thenFn(createLLVM);
  if (thenBlock->hasNoSuccessors() && !isa<LLVM::ReturnOp>(thenBlock->back()))
    br({}, endBlock);

  // Emit code for the ELSE block if required.
  createLLVM.setInsertionPointToStart(elseBlock);
  if (elseFn) {
    elseFn(createLLVM);
    if (elseBlock->hasNoSuccessors() && !isa<LLVM::ReturnOp>(elseBlock->back()))
      br({}, endBlock);
  }

  // End if-then-else and return to the main body.
  createLLVM.setInsertionPointToStart(endBlock);
}

} // namespace onnx_mlir
