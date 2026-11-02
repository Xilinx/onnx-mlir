// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

// -----

func.func @default(%input: tensor<1x12x4xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %output = "onnx.Col2Im"(%input, %image, %block) {} : (tensor<1x12x4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
  // CHECK-LABEL: func.func @default(
  // CHECK: "onnx.Col2Im"{{.*}} -> tensor<1x3x3x3xf32>
}

// -----

func.func @one_dimensional(%input: tensor<2x6x3xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[5]> : tensor<1xi64>} : () -> tensor<1xi64>
  %block = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
  %output = "onnx.Col2Im"(%input, %image, %block) {} : (tensor<2x6x3xf32>, tensor<1xi64>, tensor<1xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
  // CHECK-LABEL: func.func @one_dimensional(
  // CHECK: "onnx.Col2Im"{{.*}} -> tensor<2x2x5xf32>
}

// -----

func.func @three_dimensional(%input: tensor<2x24x12xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 4, 3]> : tensor<3xi64>} : () -> tensor<3xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
  %output = "onnx.Col2Im"(%input, %image, %block) {} : (tensor<2x24x12xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
  // CHECK-LABEL: func.func @three_dimensional(
  // CHECK: "onnx.Col2Im"{{.*}} -> tensor<2x3x3x4x3xf32>
}

// -----

func.func @strides(%input: tensor<1x12x4xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[5, 5]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %output = "onnx.Col2Im"(%input, %image, %block) {strides = [2, 2]} : (tensor<1x12x4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
  // CHECK-LABEL: func.func @strides(
  // CHECK: "onnx.Col2Im"{{.*}} -> tensor<1x3x5x5xf32>
}

// -----

func.func @pads(%input: tensor<1x12x16xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %output = "onnx.Col2Im"(%input, %image, %block) {pads = [1, 1, 1, 1]} : (tensor<1x12x16xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
  // CHECK-LABEL: func.func @pads(
  // CHECK: "onnx.Col2Im"{{.*}} -> tensor<1x3x3x3xf32>
}

// -----

func.func @dilations(%input: tensor<1x12x4xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[4, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %output = "onnx.Col2Im"(%input, %image, %block) {dilations = [2, 2]} : (tensor<1x12x4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
  // CHECK-LABEL: func.func @dilations(
  // CHECK: "onnx.Col2Im"{{.*}} -> tensor<1x3x4x4xf32>
}

// -----

func.func @combined(%input: tensor<2x18x12xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[6, 7]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %output = "onnx.Col2Im"(%input, %image, %block) {pads = [1, 2, 0, 1], dilations = [2, 1], strides = [2, 2]} : (tensor<2x18x12xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
  // CHECK-LABEL: func.func @combined(
  // CHECK: "onnx.Col2Im"{{.*}} -> tensor<2x3x6x7xf32>
}

// -----

func.func @dynamic_input(%input: tensor<?x12x?xf32>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %output = "onnx.Col2Im"(%input, %image, %block) {} : (tensor<?x12x?xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
  // CHECK-LABEL: func.func @dynamic_input(
  // CHECK: "onnx.Col2Im"{{.*}} -> tensor<?x3x3x3xf32>
}

// -----

func.func @dynamic_shapes(%input: tensor<2x12x?xf32>, %image: tensor<2xi64>, %block: tensor<2xi64>) -> tensor<*xf32> {
  %output = "onnx.Col2Im"(%input, %image, %block) : (tensor<2x12x?xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
  // CHECK-LABEL: func.func @dynamic_shapes(
  // CHECK: "onnx.Col2Im"{{.*}} -> tensor<2x?x?x?xf32>
}

// -----

func.func @dynamic_image(%input: tensor<2x12x?xf32>, %image: tensor<2xi64>) -> tensor<*xf32> {
  %block = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %output = "onnx.Col2Im"(%input, %image, %block) : (tensor<2x12x?xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
  // CHECK-LABEL: func.func @dynamic_image(
  // CHECK: "onnx.Col2Im"{{.*}} -> tensor<2x3x?x?xf32>
}

// -----

func.func @unknown_rank(%input: tensor<2x12x?xf32>, %image: tensor<?xi64>, %block: tensor<?xi64>) -> tensor<*xf32> {
  %output = "onnx.Col2Im"(%input, %image, %block) : (tensor<2x12x?xf32>, tensor<?xi64>, tensor<?xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
  // CHECK-LABEL: func.func @unknown_rank(
  // CHECK: "onnx.Col2Im"{{.*}} -> tensor<*xf32>
}

// -----

func.func @dynamic_block(%input: tensor<2x12x?xf32>, %block: tensor<2xi64>) -> tensor<*xf32> {
  %image = "onnx.Constant"() {value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %output = "onnx.Col2Im"(%input, %image, %block) : (tensor<2x12x?xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %output : tensor<*xf32>
  // CHECK-LABEL: func.func @dynamic_block(
  // CHECK: "onnx.Col2Im"{{.*}} -> tensor<2x?x3x3xf32>
}
