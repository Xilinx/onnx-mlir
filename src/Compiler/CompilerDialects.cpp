/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ CompilerDialects.cpp ------------------------===//

#include "CompilerDialects.hpp"

#include "src/Compiler/CompilerOptions.hpp"
#ifdef ONNX_MLIR_ENABLE_KRNL
#include "src/Dialect/Krnl/KrnlOps.hpp"
#endif
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/TensorName.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"

using namespace mlir;

namespace onnx_mlir {

void registerDialects(
    DialectRegistry &registry, ArrayRef<accel::Accelerator::Kind> accels) {
  // Note that we cannot consult command line options because they have not yet
  // been parsed when registerDialects() is called.

  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<affine::AffineDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<vector::VectorDialect>();
  registry.insert<shape::ShapeDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<ONNXDialect>();
#ifdef ONNX_MLIR_ENABLE_KRNL
  registry.insert<KrnlDialect>();
#endif
  registry.insert<cf::ControlFlowDialect>();
  registerOpenMPDialectTranslation(registry);
  mlir::memref::registerRuntimeVerifiableOpInterfaceExternalModels(registry);

  // Initialize accelerator(s) if required.
  accel::initAccelerators(accels);

  // Register dialects for accelerators.
  for (auto *accel : accel::Accelerator::getAccelerators())
    accel->registerDialects(registry);

  // Register interface needed by both old and new buffer deallocation pass.
  memref::registerAllocationOpInterfaceExternalModels(registry);
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);

  // Register TensorName inference
  registerTensorNameInferenceExternalModels(registry);
}

} // namespace onnx_mlir
