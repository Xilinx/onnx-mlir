# SPDX-License-Identifier: Apache-2.0
# Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
# affiliates

add_subdirectory(ONNXConversionCommon)
if (ONNX_MLIR_ENABLE_KRNL)
  add_subdirectory(ONNXToKrnl)
  add_subdirectory(KrnlToLLVM)
  add_subdirectory(KrnlToAffine)
  add_subdirectory(KrnlSeqToMemref)
endif()
add_subdirectory(ONNXToTOSA)
add_subdirectory(ONNXToLinalg)

if (ONNX_MLIR_ENABLE_STABLEHLO)
  add_subdirectory(ONNXToStablehlo)
endif()
