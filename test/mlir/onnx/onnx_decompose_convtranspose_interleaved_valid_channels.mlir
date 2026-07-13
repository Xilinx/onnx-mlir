// RUN: onnx-mlir-opt --shape-inference --decompose-onnx="enable-convtranspose-phased enable-interleaved-valid-channels-for-convtranspose" %s -split-input-file | FileCheck %s

// Test that with enable-interleaved-valid-channels-for-convtranspose, a 4-phase
// kernel 6x6 ConvTranspose with a single output channel (C_out == 1) combines the
// 4 phased weights into a single Conv that OVER-produces the output channels: each
// phase gets its own group of 4 channels (1 valid + 3 garbage), so the combined
// Conv emits 2(H-phase) x 2(W-phase) x 4 = 16 channels. The pixel merge is then a
// Reshape -> Transpose -> Reshape (free in the C:8 layout) plus a single channel
// Slice (4 -> 1) that drops the garbage lanes. Shapes taken from the asura
// asura_convt_outc1 model.

func.func @test_interleaved_4phase_kernel_66_oc1(%arg0: tensor<1x32x80x336xf32>, %arg1: tensor<32x1x6x6xf32>) -> tensor<1x1x160x672xf32> {
  %0 = "onnx.Constant" () { value = dense<0.02> : tensor<1xf32> } : () -> tensor<1xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x32x80x336xf32>, tensor<32x1x6x6xf32>, tensor<1xf32>) -> tensor<1x1x160x672xf32>
  onnx.Return %1 : tensor<1x1x160x672xf32>

// CHECK-LABEL:  func.func @test_interleaved_4phase_kernel_66_oc1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x80x336xf32>, [[PARAM_1_:%.+]]: tensor<32x1x6x6xf32>) -> tensor<1x1x160x672xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 4, 160, 672]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[2, 2, 4, 80, 336]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<3x32x3x3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<6> : tensor<2xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<6> : tensor<6xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<32x1x6x6xf32>) -> tensor<6x6x32x1xf32>
// CHECK:           [[VAR_16_:%.+]] = "onnx.ReverseSequence"([[VAR_15_]], [[VAR_13_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x32x1xf32>, tensor<6xi64>) -> tensor<6x6x32x1xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.ReverseSequence"([[VAR_16_]], [[VAR_13_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x32x1xf32>, tensor<6xi64>) -> tensor<6x6x32x1xf32>
// CHECK:           [[VAR_18_:%.+]] = "onnx.Transpose"([[VAR_17_]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x32x1xf32>) -> tensor<32x1x6x6xf32>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Transpose"([[VAR_18_]]) {perm = [1, 0, 2, 3]} : (tensor<32x1x6x6xf32>) -> tensor<1x32x6x6xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_10_]], [[VAR_9_]], [[VAR_12_]], [[VAR_11_]]) : (tensor<1x32x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x3x3xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_8_]], [[VAR_9_]], [[VAR_12_]], [[VAR_11_]]) : (tensor<1x32x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x3x3xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_7_]], [[VAR_9_]], [[VAR_12_]], [[VAR_11_]]) : (tensor<1x32x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x3x3xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_6_]], [[VAR_9_]], [[VAR_12_]], [[VAR_11_]]) : (tensor<1x32x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x3x3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Concat"([[VAR_23_]], [[VAR_5_]], [[VAR_21_]], [[VAR_5_]], [[VAR_22_]], [[VAR_5_]], [[VAR_20_]], [[VAR_5_]]) {axis = 0 : si64} : (tensor<1x32x3x3xf32>, tensor<3x32x3x3xf32>, tensor<1x32x3x3xf32>, tensor<3x32x3x3xf32>, tensor<1x32x3x3xf32>, tensor<3x32x3x3xf32>, tensor<1x32x3x3xf32>, tensor<3x32x3x3xf32>) -> tensor<16x32x3x3xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Concat"([[VAR_14_]], [[VAR_4_]], [[VAR_14_]], [[VAR_4_]], [[VAR_14_]], [[VAR_4_]], [[VAR_14_]], [[VAR_4_]]) {axis = 0 : si64} : (tensor<1xf32>, tensor<3xf32>, tensor<1xf32>, tensor<3xf32>, tensor<1xf32>, tensor<3xf32>, tensor<1xf32>, tensor<3xf32>) -> tensor<16xf32>
// CHECK:           [[VAR_26_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_24_]], [[VAR_25_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x32x80x336xf32>, tensor<16x32x3x3xf32>, tensor<16xf32>) -> tensor<1x16x80x336xf32>
// CHECK:           [[VAR_27_:%.+]] = "onnx.Reshape"([[VAR_26_]], [[VAR_3_]]) {allowzero = 0 : si64} : (tensor<1x16x80x336xf32>, tensor<5xi64>) -> tensor<2x2x4x80x336xf32>
// CHECK:           [[VAR_28_:%.+]] = "onnx.Transpose"([[VAR_27_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x4x80x336xf32>) -> tensor<4x80x2x336x2xf32>
// CHECK:           [[VAR_29_:%.+]] = "onnx.Reshape"([[VAR_28_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<4x80x2x336x2xf32>, tensor<4xi64>) -> tensor<1x4x160x672xf32>
// CHECK:           [[VAR_30_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_1_]], [[VAR_0_]], [[VAR_0_]], [[VAR_0_]]) : (tensor<1x4x160x672xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x160x672xf32>
// CHECK:           onnx.Return [[VAR_30_]] : tensor<1x1x160x672xf32>
// CHECK:         }
}

// -----

// Test that the interleaved path is GATED on C_out == 1: a 4-phase kernel 6x6
// ConvTranspose with 256 output channels (C_out != 1) must NOT over-produce into
// groups of 4. It falls back to the normal combined-weights path, i.e. a single
// Conv with 4*256 = 1024 channels and no garbage padding / group-of-4 reshape.

func.func @test_interleaved_gate_oc_gt1(%arg0: tensor<1x512x10x16xf32>, %arg1: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {
  %0 = "onnx.Constant" () { value = dense<0.02> : tensor<256xf32> } : () -> tensor<256xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x10x16xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x20x32xf32>
  onnx.Return %1 : tensor<1x256x20x32xf32>

// CHECK-LABEL:  func.func @test_interleaved_gate_oc_gt1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x512x10x16xf32>, [[PARAM_1_:%.+]]: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {
// CHECK:           [[VAR_20_:%.+]] = "onnx.Concat"({{.*}}) {axis = 0 : si64} : (tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>) -> tensor<1024x512x3x3xf32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Conv"([[PARAM_0_]], {{.*}}) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<1024x512x3x3xf32>, tensor<1024xf32>) -> tensor<1x1024x10x16xf32>
// CHECK-NOT:       tensor<16x512x3x3xf32>
// CHECK-NOT:       tensor<1x2x2x4x10x16xf32>
// CHECK:           onnx.Return
// CHECK:         }
}
