# SPDX-License-Identifier: Apache-2.0
# Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
# affiliates
add_onnx_mlir_executable(InitializerSourceDriver NO_INSTALL
  InitializerSourceDriver.cpp
  LINK_LIBS PRIVATE
  OMBuilder
  MLIRBytecodeWriter
  MLIRParser
  MLIRPass
  )
add_dependencies(unittest InitializerSourceDriver)
add_test(NAME TestInitializerSources
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_initializer_sources.py
          $<TARGET_FILE:InitializerSourceDriver>)
set_tests_properties(TestInitializerSources PROPERTIES
  LABELS unittest
  SKIP_RETURN_CODE 77)
