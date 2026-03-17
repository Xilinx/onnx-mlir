// RUN: onnx-mlir-opt -onnx-hybrid-transform %s -split-input-file | FileCheck %s

// Test BN(Add(Conv, Conv)): BN decomposes, Mul distributes over Add, both
// Muls fuse into Conv weights/biases, BN bias constants fold into Conv biases.
// Final result: Add(Conv1'(x), Conv2'(x)) with no remaining BN or Mul.
func.func @test_bn_add_conv_conv(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x4x4x4xf32> {
    %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
    %b1 = onnx.Constant dense<0.5> : tensor<4xf32>
    %w2 = onnx.Constant dense<2.0> : tensor<4x3x1x1xf32>
    %b2 = onnx.Constant dense<0.25> : tensor<4xf32>
    %scale = onnx.Constant dense<2.0> : tensor<4xf32>
    %bias = onnx.Constant dense<1.0> : tensor<4xf32>
    %mean = onnx.Constant dense<0.0> : tensor<4xf32>
    %var = onnx.Constant dense<1.0> : tensor<4xf32>
    %conv1 = "onnx.Conv"(%arg0, %w1, %b1) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x4x4xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x4x4xf32>
    %conv2 = "onnx.Conv"(%arg0, %w2, %b2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x4x4xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x4x4xf32>
    %add = "onnx.Add"(%conv1, %conv2) : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %bn = "onnx.BatchNormalizationInferenceMode"(%add, %scale, %bias, %mean, %var) {epsilon = 1.0E-5 : f32} : (tensor<1x4x4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x4x4xf32>
    return %bn : tensor<1x4x4x4xf32>

    // CHECK-LABEL:  func.func @test_bn_add_conv_conv
    // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x4x4xf32>) -> tensor<1x4x4x4xf32> {
    // CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.4999975> : tensor<4xf32>
    // CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<3.999980e+00> : tensor<4x3x1x1xf32>
    // CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1.999990e+00> : tensor<4x3x1x1xf32>
    // CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<0.999994993> : tensor<4xf32>
    // CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_2_]], [[VAR_3_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x4x4xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x4x4xf32>
    // CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_1_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x4x4xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x4x4xf32>
    // CHECK:           [[VAR_6_:%.+]] = "onnx.Add"([[VAR_4_]], [[VAR_5_]]) : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    // CHECK:           return [[VAR_6_]] : tensor<1x4x4x4xf32>
    // CHECK:         }
}

// -----

// Test BN(Add(Conv, arg)): BN decomposes, Mul distributes, Conv branch fuses.
// Non-const branch keeps a Mul. BN bias folds into Conv bias.
func.func @test_bn_add_conv_nonconst(%arg0: tensor<1x3x4x4xf32>, %arg1: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
    %w = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
    %b = onnx.Constant dense<0.5> : tensor<4xf32>
    %scale = onnx.Constant dense<2.0> : tensor<4xf32>
    %bias = onnx.Constant dense<1.0> : tensor<4xf32>
    %mean = onnx.Constant dense<0.0> : tensor<4xf32>
    %var = onnx.Constant dense<1.0> : tensor<4xf32>
    %conv = "onnx.Conv"(%arg0, %w, %b) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x4x4xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x4x4xf32>
    %add = "onnx.Add"(%conv, %arg1) : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %bn = "onnx.BatchNormalizationInferenceMode"(%add, %scale, %bias, %mean, %var) {epsilon = 1.0E-5 : f32} : (tensor<1x4x4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x4x4xf32>
    return %bn : tensor<1x4x4x4xf32>

    // CHECK-LABEL:  func.func @test_bn_add_conv_nonconst
    // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x4x4xf32>, [[PARAM_1_:%.+]]: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
    // CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.99999499> : tensor<4xf32>
    // CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.999990e+00> : tensor<4x1x1xf32>
    // CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1.999990e+00> : tensor<4x3x1x1xf32>
    // CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[VAR_1_]]) : (tensor<1x4x4x4xf32>, tensor<4x1x1xf32>) -> tensor<1x4x4x4xf32>
    // CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_2_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x4x4xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x4x4xf32>
    // CHECK:           [[VAR_5_:%.+]] = "onnx.Add"([[VAR_4_]], [[VAR_3_]]) : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    // CHECK:           return [[VAR_5_]] : tensor<1x4x4x4xf32>
    // CHECK:         }
}

// -----

// Test BN(Add(arg0, arg1)): no Conv, BN decomposes into Mul+Add but nothing
// further fuses. Result: Add(Mul(Add(arg0, arg1), a_unsqueezed), b_unsqueezed).
func.func @test_bn_add_nonconst_nonconst(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
    %scale = onnx.Constant dense<2.0> : tensor<4xf32>
    %bias = onnx.Constant dense<1.0> : tensor<4xf32>
    %mean = onnx.Constant dense<0.0> : tensor<4xf32>
    %var = onnx.Constant dense<1.0> : tensor<4xf32>
    %add = "onnx.Add"(%arg0, %arg1) : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %bn = "onnx.BatchNormalizationInferenceMode"(%add, %scale, %bias, %mean, %var) {epsilon = 1.0E-5 : f32} : (tensor<1x4x4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x4x4xf32>
    return %bn : tensor<1x4x4x4xf32>

    // CHECK-LABEL:  func.func @test_bn_add_nonconst_nonconst
    // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4x4x4xf32>, [[PARAM_1_:%.+]]: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
    // CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<4x1x1xf32>
    // CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.999990e+00> : tensor<4x1x1xf32>
    // CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    // CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_1_]]) : (tensor<1x4x4x4xf32>, tensor<4x1x1xf32>) -> tensor<1x4x4x4xf32>
    // CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[VAR_3_]], [[VAR_0_]]) : (tensor<1x4x4x4xf32>, tensor<4x1x1xf32>) -> tensor<1x4x4x4xf32>
    // CHECK:           return [[VAR_4_]] : tensor<1x4x4x4xf32>
    // CHECK:         }
}
