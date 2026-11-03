// RUN: onnx-mlir-opt --shape-inference %s -split-input-file -verify-diagnostics

// -----

func.func @block_count(%input: tensor<1x12x5xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{input dimension 2 must equal the product of sliding block counts}}
  %output = "onnx.Col2Im"(%input, %image, %block) {} : (tensor<1x12x5xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
}

// -----

func.func @channels(%input: tensor<1x13x4xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{input dimension 1 must be divisible by the product of block_shape}}
  %output = "onnx.Col2Im"(%input, %image, %block) {} : (tensor<1x13x4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
}

// -----

func.func @zero_block(%input: tensor<1x12x4xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{image_shape and block_shape must be positive}}
  %output = "onnx.Col2Im"(%input, %image, %block) {} : (tensor<1x12x4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
}

// -----

func.func @stride(%input: tensor<1x12x4xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{dilations and strides must be positive and pads nonnegative}}
  %output = "onnx.Col2Im"(%input, %image, %block) {strides = [0, 1]} : (tensor<1x12x4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
}

// -----

func.func @dilation(%input: tensor<1x12x4xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{dilations and strides must be positive and pads nonnegative}}
  %output = "onnx.Col2Im"(%input, %image, %block) {dilations = [1, 0]} : (tensor<1x12x4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
}

// -----

func.func @negative_pad(%input: tensor<1x12x4xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{dilations and strides must be positive and pads nonnegative}}
  %output = "onnx.Col2Im"(%input, %image, %block) {pads = [0, 0, 0, -1]} : (tensor<1x12x4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
}

// -----

func.func @shape_rank(%input: tensor<1x12x4xf32>, %image: tensor<1x2xi64>) -> tensor<*xf32> {
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{image_shape and block_shape must have rank 1}}
  %output = "onnx.Col2Im"(%input, %image, %block) {} : (tensor<1x12x4xf32>, tensor<1x2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
}

// -----

func.func @attr_length(%input: tensor<1x12x4xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{attribute lengths must match the spatial rank}}
  %output = "onnx.Col2Im"(%input, %image, %block) {pads = [0, 0]} : (tensor<1x12x4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
}

// -----

func.func @shape_length(%input: tensor<1x12x4xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{image_shape and block_shape must have the same nonzero length}}
  %output = "onnx.Col2Im"(%input, %image, %block) {} : (tensor<1x12x4xf32>, tensor<2xi64>, tensor<1xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
}

// -----

func.func @no_blocks(%input: tensor<1x12x4xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{sliding block count must be positive}}
  %output = "onnx.Col2Im"(%input, %image, %block) {} : (tensor<1x12x4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
}

// -----

func.func @input_rank(%input: tensor<1x12xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{input must have rank 3}}
  %output = "onnx.Col2Im"(%input, %image, %block) {} : (tensor<1x12xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
}
