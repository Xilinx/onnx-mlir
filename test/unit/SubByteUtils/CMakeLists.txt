# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

add_onnx_mlir_unittest(
  TestSubByteUtils TestSubByteUtils.cpp LINK_LIBS PRIVATE OMONNXOps
)
