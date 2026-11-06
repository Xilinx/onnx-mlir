// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt --canonicalize --mlir-print-op-generic --mlir-print-elementsattrs-with-hex-if-larger=0 %s | FileCheck %s

func.func @fold_xfe_cast() -> (tensor<24xbf16>, tensor<24xbf16>) {
  %input = onnx.Constant dense<[0x00000000, 0x80000000, 0x00000001, 0x00008000,
      0x00008001, 0x80008001, 0x007FFFFF, 0x807FFFFF, 0x3F807FFF, 0x3F808000,
      0x3F808001, 0x3F80FFFF, 0x3F818000, 0xBF808000, 0xBF80FFFF, 0xBF818000,
      0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7F800001, 0x7FC00000,
      0xFFC00000, 0xFFFFFFFF]> : tensor<24xf32>
  %legacy = onnx.XFECast %input policy = "legacy_bf16_truncate" : tensor<24xf32> to tensor<24xbf16>
  %rne = onnx.XFECast %input policy = "runtime_bf16_rne" : tensor<24xf32> to tensor<24xbf16>
  return %legacy, %rne : tensor<24xbf16>, tensor<24xbf16>
}

// CHECK-LABEL: sym_name = "fold_xfe_cast"
// CHECK-NOT: "onnx.XFECast"
// CHECK-DAG: value = dense<"0x0000008000000000000000807F007F80803F803F803F803F813F80BF80BF81BF7F7F7FFF807F80FF807FC07FC0FFFFFF"> : tensor<24xbf16>
// CHECK-DAG: value = dense<"0x00000080000000000100018080008080803F803F813F813F823F80BF81BF82BF807F7FFF807F7FFF807FC07F7FFF0000"> : tensor<24xbf16>
