// RUN: onnx-mlir-opt --recompose-onnx --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @constant_epsilon_control(
// CHECK: "onnx.LayerNormalization"
// CHECK-SAME: epsilon = 0.00512695313 : f32
func.func @constant_epsilon_control(%x: tensor<2x4xf32>) -> (tensor<2x4xf32>) {
  %eps = "onnx.Constant"() {value = dense<0.005126953125> : tensor<f32>} : () -> tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  %square = "onnx.Mul"(%d, %d) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %var = "onnx.ReduceMeanV13"(%square) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %sd = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %norm = "onnx.Div"(%d, %sd) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  return %norm : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func.func @quantized_epsilon_exact_value(
// CHECK: "onnx.LayerNormalization"
// CHECK-SAME: epsilon = 0.00512695313 : f32
func.func @quantized_epsilon_exact_value(%x: tensor<2x4xf32>) -> (tensor<2x4xf32>) {
  %qeps = "onnx.Constant"() {value = dense<84> : tensor<i8>} : () -> tensor<i8>
  %zp = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
  %scale = "onnx.Constant"() {value = dense<0.00006103515625> : tensor<f32>} : () -> tensor<f32>
  %eps = "onnx.DequantizeLinear"(%qeps, %scale, %zp) : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  %square = "onnx.Mul"(%d, %d) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %var = "onnx.ReduceMeanV13"(%square) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %sd = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %norm = "onnx.Div"(%d, %sd) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  return %norm : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func.func @shared_quantized_epsilon(
// CHECK: %[[EPS:[a-zA-Z0-9_]+]] = "onnx.DequantizeLinear"
// CHECK: "onnx.LayerNormalization"
// CHECK-SAME: epsilon = 0.00512695313 : f32
// CHECK: return {{.*}}, %[[EPS]]
func.func @shared_quantized_epsilon(%x: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<f32>) {
  %qeps = "onnx.Constant"() {value = dense<84> : tensor<i8>} : () -> tensor<i8>
  %zp = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
  %scale = "onnx.Constant"() {value = dense<0.00006103515625> : tensor<f32>} : () -> tensor<f32>
  %eps = "onnx.DequantizeLinear"(%qeps, %scale, %zp) : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  %square = "onnx.Mul"(%d, %d) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %var = "onnx.ReduceMeanV13"(%square) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %sd = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %norm = "onnx.Div"(%d, %sd) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  return %norm, %eps : tensor<2x4xf32>, tensor<f32>
}

// -----

// CHECK-LABEL: func.func @dynamic_epsilon_scale(
// CHECK-NOT: "onnx.LayerNormalization"
// CHECK: "onnx.Div"
func.func @dynamic_epsilon_scale(%x: tensor<2x4xf32>, %scale: tensor<f32>) -> (tensor<2x4xf32>) {
  %qeps = "onnx.Constant"() {value = dense<84> : tensor<i8>} : () -> tensor<i8>
  %zp = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
  %eps = "onnx.DequantizeLinear"(%qeps, %scale, %zp) : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  %square = "onnx.Mul"(%d, %d) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %var = "onnx.ReduceMeanV13"(%square) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %sd = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %norm = "onnx.Div"(%d, %sd) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  return %norm : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func.func @quantized_epsilon_f16_requires_rounding(
// CHECK-NOT: "onnx.LayerNormalization"
// CHECK: "onnx.Div"
func.func @quantized_epsilon_f16_requires_rounding(%x: tensor<2x4xf16>) -> (tensor<2x4xf16>) {
  %qeps = "onnx.Constant"() {value = dense<84> : tensor<i8>} : () -> tensor<i8>
  %zp = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
  %scale = "onnx.Constant"() {value = dense<0.00006103515625> : tensor<f16>} : () -> tensor<f16>
  %eps = "onnx.DequantizeLinear"(%qeps, %scale, %zp) : (tensor<i8>, tensor<f16>, tensor<i8>) -> tensor<f16>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf16>) -> tensor<2x1xf16>
  %d = "onnx.Sub"(%x, %mean) : (tensor<2x4xf16>, tensor<2x1xf16>) -> tensor<2x4xf16>
  %square = "onnx.Mul"(%d, %d) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>
  %var = "onnx.ReduceMeanV13"(%square) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf16>) -> tensor<2x1xf16>
  %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf16>, tensor<f16>) -> tensor<2x1xf16>
  %sd = "onnx.Sqrt"(%ve) : (tensor<2x1xf16>) -> tensor<2x1xf16>
  %norm = "onnx.Div"(%d, %sd) : (tensor<2x4xf16>, tensor<2x1xf16>) -> tensor<2x4xf16>
  return %norm : tensor<2x4xf16>
}

// -----

// CHECK-LABEL: func.func @quantized_epsilon_i32_requires_exact_evaluation(
// CHECK-NOT: "onnx.LayerNormalization"
// CHECK: "onnx.Div"
func.func @quantized_epsilon_i32_requires_exact_evaluation(%x: tensor<2x4xf32>) -> (tensor<2x4xf32>) {
  %qeps = "onnx.Constant"() {value = dense<84> : tensor<i32>} : () -> tensor<i32>
  %zp = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %scale = "onnx.Constant"() {value = dense<0.00006103515625> : tensor<f32>} : () -> tensor<f32>
  %eps = "onnx.DequantizeLinear"(%qeps, %scale, %zp) : (tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  %square = "onnx.Mul"(%d, %d) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %var = "onnx.ReduceMeanV13"(%square) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %sd = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %norm = "onnx.Div"(%d, %sd) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  return %norm : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func.func @intermediate_quantization_preserved(
// CHECK-NOT: "onnx.LayerNormalization"
// CHECK: "onnx.Div"
func.func @intermediate_quantization_preserved(%x: tensor<2x4xf32>) -> (tensor<2x4xf32>) {
  %qeps = "onnx.Constant"() {value = dense<84> : tensor<i8>} : () -> tensor<i8>
  %zp = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
  %scale = "onnx.Constant"() {value = dense<0.00006103515625> : tensor<f32>} : () -> tensor<f32>
  %eps = "onnx.DequantizeLinear"(%qeps, %scale, %zp) : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  %square = "onnx.Mul"(%d, %d) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %qs = "onnx.QuantizeLinear"(%square, %scale, %zp) : (tensor<2x4xf32>, tensor<f32>, tensor<i8>) -> tensor<2x4xi8>
  %dqs = "onnx.DequantizeLinear"(%qs, %scale, %zp) : (tensor<2x4xi8>, tensor<f32>, tensor<i8>) -> tensor<2x4xf32>
  %var = "onnx.ReduceMeanV13"(%dqs) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %sd = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %norm = "onnx.Div"(%d, %sd) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  return %norm : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func.func @per_axis_epsilon_is_not_scalar(
// CHECK-NOT: "onnx.LayerNormalization"
// CHECK: "onnx.Div"
func.func @per_axis_epsilon_is_not_scalar(%x: tensor<2x4xf32>, %scale: tensor<4xf32>) -> (tensor<2x4xf32>) {
  %qeps = "onnx.Constant"() {value = dense<84> : tensor<4xi8>} : () -> tensor<4xi8>
  %zp = "onnx.Constant"() {value = dense<0> : tensor<4xi8>} : () -> tensor<4xi8>
  %eps = "onnx.DequantizeLinear"(%qeps, %scale, %zp) {axis = 0 : si64} : (tensor<4xi8>, tensor<4xf32>, tensor<4xi8>) -> tensor<4xf32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  %square = "onnx.Mul"(%d, %d) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %var = "onnx.ReduceMeanV13"(%square) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  %sd = "onnx.Sqrt"(%ve) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %norm = "onnx.Div"(%d, %sd) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %norm : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func.func @int16_epsilon_with_zero_point_and_rounding(
// CHECK: "onnx.LayerNormalization"
// CHECK-SAME: epsilon = 6.553500e+03 : f32
func.func @int16_epsilon_with_zero_point_and_rounding(%x: tensor<2x4xf32>) -> (tensor<2x4xf32>) {
  %qeps = "onnx.Constant"() {value = dense<32767> : tensor<i16>} : () -> tensor<i16>
  %zp = "onnx.Constant"() {value = dense<-32768> : tensor<i16>} : () -> tensor<i16>
  %scale = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
  %eps = "onnx.DequantizeLinear"(%qeps, %scale, %zp) : (tensor<i16>, tensor<f32>, tensor<i16>) -> tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  %square = "onnx.Mul"(%d, %d) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %var = "onnx.ReduceMeanV13"(%square) {axes = [1], keepdims = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %sd = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %norm = "onnx.Div"(%d, %sd) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  return %norm : tensor<2x4xf32>
}
