// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// RUN: onnx-mlir-opt --decompose-onnx=enable-gatherblockquantized-decompose --mlir-elide-elementsattrs-if-larger=128 %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck --check-prefix=DISABLED %s
// RUN: onnx-mlir-opt --onnx-hybrid-transform="shape-inference=false constant-propagation=false enable-gatherblockquantized-decompose" --mlir-elide-elementsattrs-if-larger=128 %s -split-input-file | FileCheck --check-prefix=NORMALIZED %s

func.func @gather_block_quantized_large(%indices: tensor<1x2xi32>) -> tensor<1x2x896xf32> {
  %packed = "onnx.Constant"() {value = dense<170> : tensor<151936x28x16xui8>} : () -> tensor<151936x28x16xui8>
  %shape = "onnx.Constant"() {value = dense<[151936, 448]> : tensor<2xi64>} : () -> tensor<2xi64>
  %data = "onnx.Reshape"(%packed, %shape) : (tensor<151936x28x16xui8>, tensor<2xi64>) -> tensor<151936x448xui8>
  %scales = "onnx.Constant"() {value = dense<0.125> : tensor<151936x28xf32>} : () -> tensor<151936x28xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 4 : si64,
    block_size = 32 : si64,
    gather_axis = 0 : si64,
    quantize_axis = 1 : si64
  } : (tensor<151936x448xui8>, tensor<1x2xi32>, tensor<151936x28xf32>) -> tensor<1x2x896xf32>
  return %out : tensor<1x2x896xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_large
// CHECK-SAME:   ([[INDICES:%.+]]: tensor<1x2xi32>) -> tensor<1x2x896xf32> {
// CHECK-DAG:      [[GATHERED_SHAPE:%.+]] = onnx.Constant dense<[1, 2, 896]> : tensor<3xi64>
// CHECK-DAG:      [[BLOCK_SHAPE:%.+]] = onnx.Constant dense<[1, 2, 28, 32]> : tensor<4xi64>
// CHECK-DAG:      [[AXES:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:      [[SCALES:%.+]] = onnx.Constant dense<1.250000e-01> : tensor<151936x28xf32>
// CHECK-DAG:      [[TABLE:%.+]] = onnx.Constant dense<2> : tensor<151936x896xi8>
// CHECK:          [[Q:%.+]] = "onnx.Gather"([[TABLE]], [[INDICES]]) {axis = 0 : si64} : (tensor<151936x896xi8>, tensor<1x2xi32>) -> tensor<1x2x896xi8>
// CHECK:          [[QF:%.+]] = "onnx.Cast"([[Q]]) {saturate = 1 : si64, to = f32} : (tensor<1x2x896xi8>) -> tensor<1x2x896xf32>
// CHECK:          [[S:%.+]] = "onnx.Gather"([[SCALES]], [[INDICES]]) {axis = 0 : si64} : (tensor<151936x28xf32>, tensor<1x2xi32>) -> tensor<1x2x28xf32>
// CHECK:          [[SU:%.+]] = "onnx.Unsqueeze"([[S]], [[AXES]]) : (tensor<1x2x28xf32>, tensor<1xi64>) -> tensor<1x2x28x1xf32>
// CHECK:          [[SE:%.+]] = "onnx.Expand"([[SU]], [[BLOCK_SHAPE]]) : (tensor<1x2x28x1xf32>, tensor<4xi64>) -> tensor<1x2x28x32xf32>
// CHECK:          [[SR:%.+]] = "onnx.Reshape"([[SE]], [[GATHERED_SHAPE]]) {allowzero = 0 : si64} : (tensor<1x2x28x32xf32>, tensor<3xi64>) -> tensor<1x2x896xf32>
// CHECK:          [[RES:%.+]] = "onnx.Mul"([[QF]], [[SR]]) : (tensor<1x2x896xf32>, tensor<1x2x896xf32>) -> tensor<1x2x896xf32>
// CHECK:          return [[RES]] : tensor<1x2x896xf32>

// DISABLED-LABEL:  func.func @gather_block_quantized_large
// DISABLED:          "onnx.Custom"({{.*}}) {{{.*}}function_name = "GatherBlockQuantized"{{.*}}} : (tensor<151936x448xui8>, tensor<1x2xi32>, tensor<151936x28xf32>) -> tensor<1x2x896xf32>

// -----

func.func @gather_block_quantized_no_reshape(%indices: tensor<4xi64>) -> tensor<4x64xf32> {
  %data = "onnx.Constant"() {value = dense<170> : tensor<8x32xui8>} : () -> tensor<8x32xui8>
  %scales = "onnx.Constant"() {value = dense<0.25> : tensor<8x2xf32>} : () -> tensor<8x2xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 4 : si64,
    block_size = 32 : si64,
    gather_axis = 0 : si64,
    quantize_axis = 1 : si64
  } : (tensor<8x32xui8>, tensor<4xi64>, tensor<8x2xf32>) -> tensor<4x64xf32>
  return %out : tensor<4x64xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_no_reshape
// CHECK-NOT:      onnx.Custom
// CHECK-DAG:      [[TABLE:%.+]] = onnx.Constant dense<2> : tensor<8x64xi8>
// CHECK:          [[Q:%.+]] = "onnx.Gather"([[TABLE]], {{.*}}) {axis = 0 : si64} : (tensor<8x64xi8>, tensor<4xi64>) -> tensor<4x64xi8>
// CHECK:          "onnx.Unsqueeze"({{.*}}) : (tensor<4x2xf32>, tensor<1xi64>) -> tensor<4x2x1xf32>

// -----

// Positive: bits = 2 packs four values per byte and folds a zero point of 2.
func.func @gather_block_quantized_bits2(%indices: tensor<2xi64>) -> tensor<2x32xf32> {
  %data = "onnx.Constant"() {value = dense<[[27, 27, 27, 27, 27, 27, 27, 27], [228, 228, 228, 228, 228, 228, 228, 228]]> : tensor<2x8xui8>} : () -> tensor<2x8xui8>
  %scales = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 2 : si64,
    block_size = 16 : si64,
    gather_axis = 0 : si64,
    quantize_axis = 1 : si64
  } : (tensor<2x8xui8>, tensor<2xi64>, tensor<2x2xf32>) -> tensor<2x32xf32>
  return %out : tensor<2x32xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_bits2
// CHECK-NOT:      onnx.Custom
// CHECK:          onnx.Constant dense<{{.}}[1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2], [-2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1]{{.}}> : tensor<2x32xi8>

// -----

// Positive: int16 table
func.func @gather_block_quantized_bits8(%indices: tensor<2xi64>) -> tensor<2x32xf32> {
  %data = "onnx.Constant"() {value = dense<200> : tensor<2x32xui8>} : () -> tensor<2x32xui8>
  %scales = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 8 : si64,
    block_size = 16 : si64,
    gather_axis = 0 : si64,
    quantize_axis = 1 : si64
  } : (tensor<2x32xui8>, tensor<2xi64>, tensor<2x2xf32>) -> tensor<2x32xf32>
  return %out : tensor<2x32xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_bits8
// CHECK-NOT:      onnx.Custom
// CHECK:          [[TABLE:%.+]] = onnx.Constant dense<72> : tensor<2x32xi16>
// CHECK:          [[Q:%.+]] = "onnx.Gather"([[TABLE]], {{.*}}) {axis = 0 : si64} : (tensor<2x32xi16>, tensor<2xi64>) -> tensor<2x32xi16>
// CHECK:          "onnx.Cast"([[Q]]) {saturate = 1 : si64, to = f32} : (tensor<2x32xi16>) -> tensor<2x32xf32>

// -----

// Positive: explicit zero_points are gathered and block-expanded like the scales
// instead of being folded into the table.
func.func @gather_block_quantized_explicit_zero_points(%indices: tensor<2xi64>) -> tensor<2x64xf32> {
  %data = "onnx.Constant"() {value = dense<170> : tensor<4x32xui8>} : () -> tensor<4x32xui8>
  %scales = "onnx.Constant"() {value = dense<0.5> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
  %zero_points = "onnx.Constant"() {value = dense<51> : tensor<4x1xui8>} : () -> tensor<4x1xui8>
  %out = "onnx.Custom"(%data, %indices, %scales, %zero_points) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 4 : si64,
    block_size = 32 : si64,
    gather_axis = 0 : si64,
    quantize_axis = 1 : si64
  } : (tensor<4x32xui8>, tensor<2xi64>, tensor<4x2xf32>, tensor<4x1xui8>) -> tensor<2x64xf32>
  return %out : tensor<2x64xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_explicit_zero_points
// CHECK-NOT:      onnx.Custom
// CHECK-DAG:      [[ZP:%.+]] = onnx.Constant dense<3> : tensor<4x2xi8>
// CHECK-DAG:      [[TABLE:%.+]] = onnx.Constant dense<10> : tensor<4x64xi8>
// CHECK:          [[Q:%.+]] = "onnx.Gather"([[TABLE]], {{.*}}) {axis = 0 : si64} : (tensor<4x64xi8>, tensor<2xi64>) -> tensor<2x64xi8>
// CHECK:          [[QF:%.+]] = "onnx.Cast"([[Q]]) {saturate = 1 : si64, to = f32} : (tensor<2x64xi8>) -> tensor<2x64xf32>
// CHECK:          [[ZPG:%.+]] = "onnx.Gather"([[ZP]], {{.*}}) {axis = 0 : si64} : (tensor<4x2xi8>, tensor<2xi64>) -> tensor<2x2xi8>
// CHECK:          [[ZPF:%.+]] = "onnx.Cast"([[ZPG]]) {saturate = 1 : si64, to = f32} : (tensor<2x2xi8>) -> tensor<2x2xf32>
// CHECK:          [[ZPR:%.+]] = "onnx.Reshape"({{.*}}) {allowzero = 0 : si64} : (tensor<2x2x32xf32>, tensor<2xi64>) -> tensor<2x64xf32>
// CHECK:          [[SUB:%.+]] = "onnx.Sub"([[QF]], [[ZPR]]) : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xf32>
// CHECK:          "onnx.Gather"({{.*}}) {axis = 0 : si64} : (tensor<4x2xf32>, tensor<2xi64>) -> tensor<2x2xf32>
// CHECK:          [[SR:%.+]] = "onnx.Reshape"({{.*}}) {allowzero = 0 : si64} : (tensor<2x2x32xf32>, tensor<2xi64>) -> tensor<2x64xf32>
// CHECK:          "onnx.Mul"([[SUB]], [[SR]]) : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xf32>

// -----

// Positive: rank-1 indices, where the block axis does not shift at all.
func.func @gather_block_quantized_rank1_indices(%indices: tensor<3xi64>) -> tensor<3x64xf32> {
  %data = "onnx.Constant"() {value = dense<170> : tensor<8x32xui8>} : () -> tensor<8x32xui8>
  %scales = "onnx.Constant"() {value = dense<0.5> : tensor<8x2xf32>} : () -> tensor<8x2xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 4 : si64,
    block_size = 32 : si64,
    gather_axis = 0 : si64,
    quantize_axis = 1 : si64
  } : (tensor<8x32xui8>, tensor<3xi64>, tensor<8x2xf32>) -> tensor<3x64xf32>
  return %out : tensor<3x64xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_rank1_indices
// CHECK-NOT:      onnx.Custom
// CHECK-DAG:      [[TABLE:%.+]] = onnx.Constant dense<2> : tensor<8x64xi8>
// CHECK:          "onnx.Gather"([[TABLE]], {{.*}}) {axis = 0 : si64} : (tensor<8x64xi8>, tensor<3xi64>) -> tensor<3x64xi8>
// CHECK:          "onnx.Unsqueeze"({{.*}}) : (tensor<3x2xf32>, tensor<1xi64>) -> tensor<3x2x1xf32>
// CHECK:          "onnx.Expand"({{.*}}) : (tensor<3x2x1xf32>, tensor<3xi64>) -> tensor<3x2x32xf32>

// -----

// Positive: rank-3 indices, where the block axis shifts by two.
func.func @gather_block_quantized_rank3_indices(%indices: tensor<1x2x3xi64>) -> tensor<1x2x3x64xf32> {
  %data = "onnx.Constant"() {value = dense<170> : tensor<8x32xui8>} : () -> tensor<8x32xui8>
  %scales = "onnx.Constant"() {value = dense<0.5> : tensor<8x2xf32>} : () -> tensor<8x2xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 4 : si64,
    block_size = 32 : si64,
    gather_axis = 0 : si64,
    quantize_axis = 1 : si64
  } : (tensor<8x32xui8>, tensor<1x2x3xi64>, tensor<8x2xf32>) -> tensor<1x2x3x64xf32>
  return %out : tensor<1x2x3x64xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_rank3_indices
// CHECK-NOT:      onnx.Custom
// CHECK-DAG:      [[AXES:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:      [[TABLE:%.+]] = onnx.Constant dense<2> : tensor<8x64xi8>
// CHECK:          "onnx.Gather"([[TABLE]], {{.*}}) {axis = 0 : si64} : (tensor<8x64xi8>, tensor<1x2x3xi64>) -> tensor<1x2x3x64xi8>
// CHECK:          "onnx.Unsqueeze"({{.*}}, [[AXES]]) : (tensor<1x2x3x2xf32>, tensor<1xi64>) -> tensor<1x2x3x2x1xf32>
// CHECK:          "onnx.Reshape"({{.*}}) {allowzero = 0 : si64} : (tensor<1x2x3x2x32xf32>, tensor<4xi64>) -> tensor<1x2x3x64xf32>

// -----

// Positive: a rank-3 table gathered on the middle axis
func.func @gather_block_quantized_rank3_table(%indices: tensor<2xi64>) -> tensor<3x2x64xf32> {
  %data = "onnx.Constant"() {value = dense<170> : tensor<3x8x32xui8>} : () -> tensor<3x8x32xui8>
  %scales = "onnx.Constant"() {value = dense<0.5> : tensor<3x8x2xf32>} : () -> tensor<3x8x2xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 4 : si64,
    block_size = 32 : si64,
    gather_axis = 1 : si64,
    quantize_axis = 2 : si64
  } : (tensor<3x8x32xui8>, tensor<2xi64>, tensor<3x8x2xf32>) -> tensor<3x2x64xf32>
  return %out : tensor<3x2x64xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_rank3_table
// CHECK-NOT:      onnx.Custom
// CHECK-DAG:      [[TABLE:%.+]] = onnx.Constant dense<2> : tensor<3x8x64xi8>
// CHECK:          "onnx.Gather"([[TABLE]], {{.*}}) {axis = 1 : si64} : (tensor<3x8x64xi8>, tensor<2xi64>) -> tensor<3x2x64xi8>
// CHECK:          "onnx.Unsqueeze"({{.*}}) : (tensor<3x2x2xf32>, tensor<1xi64>) -> tensor<3x2x2x1xf32>

// -----

func.func @gather_block_quantized_negative_axes(%indices: tensor<2xi64>) -> tensor<2x64xf32> {
  %data = "onnx.Constant"() {value = dense<170> : tensor<8x32xui8>} : () -> tensor<8x32xui8>
  %scales = "onnx.Constant"() {value = dense<0.5> : tensor<8x2xf32>} : () -> tensor<8x2xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 4 : si64,
    block_size = 32 : si64,
    gather_axis = -2 : si64,
    quantize_axis = -1 : si64
  } : (tensor<8x32xui8>, tensor<2xi64>, tensor<8x2xf32>) -> tensor<2x64xf32>
  return %out : tensor<2x64xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_negative_axes
// CHECK:          "onnx.Custom"
// NORMALIZED-LABEL: func.func @gather_block_quantized_negative_axes
// NORMALIZED-NOT:     onnx.Custom
// NORMALIZED:         "onnx.Gather"({{.*}}) {axis = 0 : si64} : (tensor<8x64xi8>, tensor<2xi64>) -> tensor<2x64xi8>

// -----

// Negative:
func.func @gather_block_quantized_gather_axis_is_quantize_axis(%indices: tensor<2xi64>) -> tensor<8x2xf32> {
  %data = "onnx.Constant"() {value = dense<170> : tensor<8x32xui8>} : () -> tensor<8x32xui8>
  %scales = "onnx.Constant"() {value = dense<0.5> : tensor<8x2xf32>} : () -> tensor<8x2xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 4 : si64,
    block_size = 32 : si64,
    gather_axis = 1 : si64,
    quantize_axis = 1 : si64
  } : (tensor<8x32xui8>, tensor<2xi64>, tensor<8x2xf32>) -> tensor<8x2xf32>
  return %out : tensor<8x2xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_gather_axis_is_quantize_axis
// CHECK:          "onnx.Custom"

// -----

// Negative: a non-constant table cannot be unpacked at compile time.
func.func @gather_block_quantized_dynamic_table(%data: tensor<8x32xui8>, %indices: tensor<2xi64>) -> tensor<2x64xf32> {
  %scales = "onnx.Constant"() {value = dense<0.5> : tensor<8x2xf32>} : () -> tensor<8x2xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 4 : si64,
    block_size = 32 : si64,
    gather_axis = 0 : si64,
    quantize_axis = 1 : si64
  } : (tensor<8x32xui8>, tensor<2xi64>, tensor<8x2xf32>) -> tensor<2x64xf32>
  return %out : tensor<2x64xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_dynamic_table
// CHECK:          "onnx.Custom"

// -----

// Negative:
func.func @gather_block_quantized_native_int4(%indices: tensor<2xi64>) -> tensor<2x64xf32> {
  %data = "onnx.Constant"() {value = dense<3> : tensor<8x64xi4>} : () -> tensor<8x64xi4>
  %scales = "onnx.Constant"() {value = dense<0.5> : tensor<8x2xf32>} : () -> tensor<8x2xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    block_size = 32 : si64,
    gather_axis = 0 : si64,
    quantize_axis = 1 : si64
  } : (tensor<8x64xi4>, tensor<2xi64>, tensor<8x2xf32>) -> tensor<2x64xf32>
  return %out : tensor<2x64xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_native_int4
// CHECK:          "onnx.Custom"

// -----

// Negative:
func.func @gather_block_quantized_quantize_axis_not_last(%indices: tensor<2xi64>) -> tensor<3x2x64xf32> {
  %data = "onnx.Constant"() {value = dense<170> : tensor<3x8x32xui8>} : () -> tensor<3x8x32xui8>
  %scales = "onnx.Constant"() {value = dense<0.5> : tensor<3x1x32xf32>} : () -> tensor<3x1x32xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 4 : si64,
    block_size = 32 : si64,
    gather_axis = 2 : si64,
    quantize_axis = 1 : si64
  } : (tensor<3x8x32xui8>, tensor<2xi64>, tensor<3x1x32xf32>) -> tensor<3x2x64xf32>
  return %out : tensor<3x2x64xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_quantize_axis_not_last
// CHECK:          "onnx.Custom"

// -----

// Negative:
func.func @gather_block_quantized_dynamic_indices(%indices: tensor<?xi64>) -> tensor<?x64xf32> {
  %data = "onnx.Constant"() {value = dense<170> : tensor<8x32xui8>} : () -> tensor<8x32xui8>
  %scales = "onnx.Constant"() {value = dense<0.5> : tensor<8x2xf32>} : () -> tensor<8x2xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 4 : si64,
    block_size = 32 : si64,
    gather_axis = 0 : si64,
    quantize_axis = 1 : si64
  } : (tensor<8x32xui8>, tensor<?xi64>, tensor<8x2xf32>) -> tensor<?x64xf32>
  return %out : tensor<?x64xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_dynamic_indices
// CHECK:          "onnx.Custom"

// -----

// Negative:
func.func @gather_block_quantized_scales_block_mismatch(%indices: tensor<2xi64>) -> tensor<2x64xf32> {
  %data = "onnx.Constant"() {value = dense<170> : tensor<8x32xui8>} : () -> tensor<8x32xui8>
  %scales = "onnx.Constant"() {value = dense<0.5> : tensor<8x3xf32>} : () -> tensor<8x3xf32>
  %out = "onnx.Custom"(%data, %indices, %scales) {
    domain_name = "com.microsoft",
    function_name = "GatherBlockQuantized",
    bits = 4 : si64,
    block_size = 32 : si64,
    gather_axis = 0 : si64,
    quantize_axis = 1 : si64
  } : (tensor<8x32xui8>, tensor<2xi64>, tensor<8x3xf32>) -> tensor<2x64xf32>
  return %out : tensor<2x64xf32>
}
// CHECK-LABEL:  func.func @gather_block_quantized_scales_block_mismatch
// CHECK:          "onnx.Custom"