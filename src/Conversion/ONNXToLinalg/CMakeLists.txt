# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc.

add_onnx_mlir_library(OMONNXToLinalg
  Resize.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRTensorDialect
  MLIRTransforms
  )
