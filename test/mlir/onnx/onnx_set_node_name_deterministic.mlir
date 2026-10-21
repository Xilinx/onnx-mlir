// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
// RUN: onnx-mlir-opt --set-onnx-node-name %s > %t.1
// RUN: onnx-mlir-opt --set-onnx-node-name %s > %t.2
// RUN: diff -u %t.1 %t.2
// RUN: FileCheck %s < %t.1

// More than 32 entries exercises the former SmallSet spill to pointer order.
func.func @more_than_small_set_capacity(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "onnx.Identity"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %1 = "onnx.Identity"(%0) : (tensor<1xf32>) -> tensor<1xf32>
  %2 = "onnx.Identity"(%1) : (tensor<1xf32>) -> tensor<1xf32>
  %3 = "onnx.Identity"(%2) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "onnx.Identity"(%3) : (tensor<1xf32>) -> tensor<1xf32>
  %5 = "onnx.Identity"(%4) : (tensor<1xf32>) -> tensor<1xf32>
  %6 = "onnx.Identity"(%5) : (tensor<1xf32>) -> tensor<1xf32>
  %7 = "onnx.Identity"(%6) : (tensor<1xf32>) -> tensor<1xf32>
  %8 = "onnx.Identity"(%7) : (tensor<1xf32>) -> tensor<1xf32>
  %9 = "onnx.Identity"(%8) : (tensor<1xf32>) -> tensor<1xf32>
  %10 = "onnx.Identity"(%9) : (tensor<1xf32>) -> tensor<1xf32>
  %11 = "onnx.Identity"(%10) : (tensor<1xf32>) -> tensor<1xf32>
  %12 = "onnx.Identity"(%11) : (tensor<1xf32>) -> tensor<1xf32>
  %13 = "onnx.Identity"(%12) : (tensor<1xf32>) -> tensor<1xf32>
  %14 = "onnx.Identity"(%13) : (tensor<1xf32>) -> tensor<1xf32>
  %15 = "onnx.Identity"(%14) : (tensor<1xf32>) -> tensor<1xf32>
  %16 = "onnx.Identity"(%15) : (tensor<1xf32>) -> tensor<1xf32>
  %17 = "onnx.Identity"(%16) : (tensor<1xf32>) -> tensor<1xf32>
  %18 = "onnx.Identity"(%17) : (tensor<1xf32>) -> tensor<1xf32>
  %19 = "onnx.Identity"(%18) : (tensor<1xf32>) -> tensor<1xf32>
  %20 = "onnx.Identity"(%19) : (tensor<1xf32>) -> tensor<1xf32>
  %21 = "onnx.Identity"(%20) : (tensor<1xf32>) -> tensor<1xf32>
  %22 = "onnx.Identity"(%21) : (tensor<1xf32>) -> tensor<1xf32>
  %23 = "onnx.Identity"(%22) : (tensor<1xf32>) -> tensor<1xf32>
  %24 = "onnx.Identity"(%23) : (tensor<1xf32>) -> tensor<1xf32>
  %25 = "onnx.Identity"(%24) : (tensor<1xf32>) -> tensor<1xf32>
  %26 = "onnx.Identity"(%25) : (tensor<1xf32>) -> tensor<1xf32>
  %27 = "onnx.Identity"(%26) : (tensor<1xf32>) -> tensor<1xf32>
  %28 = "onnx.Identity"(%27) : (tensor<1xf32>) -> tensor<1xf32>
  %29 = "onnx.Identity"(%28) : (tensor<1xf32>) -> tensor<1xf32>
  %30 = "onnx.Identity"(%29) : (tensor<1xf32>) -> tensor<1xf32>
  %31 = "onnx.Identity"(%30) : (tensor<1xf32>) -> tensor<1xf32>
  %32 = "onnx.Identity"(%31) : (tensor<1xf32>) -> tensor<1xf32>
  %33 = "onnx.Identity"(%32) : (tensor<1xf32>) -> tensor<1xf32>
  %34 = "onnx.Identity"(%33) : (tensor<1xf32>) -> tensor<1xf32>
  %35 = "onnx.Identity"(%34) : (tensor<1xf32>) -> tensor<1xf32>
  %36 = "onnx.Identity"(%35) : (tensor<1xf32>) -> tensor<1xf32>
  %37 = "onnx.Identity"(%36) : (tensor<1xf32>) -> tensor<1xf32>
  %38 = "onnx.Identity"(%37) : (tensor<1xf32>) -> tensor<1xf32>
  %39 = "onnx.Identity"(%38) : (tensor<1xf32>) -> tensor<1xf32>
  return %39 : tensor<1xf32>
}

// CHECK-LABEL: func.func @more_than_small_set_capacity
// CHECK: onnx_node_name = "onnx.Identity_0"
// CHECK: onnx_node_name = "onnx.Identity_1"
// CHECK: onnx_node_name = "onnx.Identity_2"
// CHECK: onnx_node_name = "onnx.Identity_3"
// CHECK: onnx_node_name = "onnx.Identity_4"
// CHECK: onnx_node_name = "onnx.Identity_5"
// CHECK: onnx_node_name = "onnx.Identity_6"
// CHECK: onnx_node_name = "onnx.Identity_7"
// CHECK: onnx_node_name = "onnx.Identity_8"
// CHECK: onnx_node_name = "onnx.Identity_9"
// CHECK: onnx_node_name = "onnx.Identity_10"
// CHECK: onnx_node_name = "onnx.Identity_11"
// CHECK: onnx_node_name = "onnx.Identity_12"
// CHECK: onnx_node_name = "onnx.Identity_13"
// CHECK: onnx_node_name = "onnx.Identity_14"
// CHECK: onnx_node_name = "onnx.Identity_15"
// CHECK: onnx_node_name = "onnx.Identity_16"
// CHECK: onnx_node_name = "onnx.Identity_17"
// CHECK: onnx_node_name = "onnx.Identity_18"
// CHECK: onnx_node_name = "onnx.Identity_19"
// CHECK: onnx_node_name = "onnx.Identity_20"
// CHECK: onnx_node_name = "onnx.Identity_21"
// CHECK: onnx_node_name = "onnx.Identity_22"
// CHECK: onnx_node_name = "onnx.Identity_23"
// CHECK: onnx_node_name = "onnx.Identity_24"
// CHECK: onnx_node_name = "onnx.Identity_25"
// CHECK: onnx_node_name = "onnx.Identity_26"
// CHECK: onnx_node_name = "onnx.Identity_27"
// CHECK: onnx_node_name = "onnx.Identity_28"
// CHECK: onnx_node_name = "onnx.Identity_29"
// CHECK: onnx_node_name = "onnx.Identity_30"
// CHECK: onnx_node_name = "onnx.Identity_31"
// CHECK: onnx_node_name = "onnx.Identity_32"
// CHECK: onnx_node_name = "onnx.Identity_33"
// CHECK: onnx_node_name = "onnx.Identity_34"
// CHECK: onnx_node_name = "onnx.Identity_35"
// CHECK: onnx_node_name = "onnx.Identity_36"
// CHECK: onnx_node_name = "onnx.Identity_37"
// CHECK: onnx_node_name = "onnx.Identity_38"
// CHECK: onnx_node_name = "onnx.Identity_39"
