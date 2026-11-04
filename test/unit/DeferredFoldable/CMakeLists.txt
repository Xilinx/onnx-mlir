# SPDX-License-Identifier: Apache-2.0
# Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
# affiliates

add_onnx_mlir_unittest(TestDeferredFoldable
  TestDeferredFoldable.cpp

  LINK_LIBS PRIVATE
  OMONNXOps
  MLIRParser
  )
