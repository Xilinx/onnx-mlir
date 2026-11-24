// RUN: onnx-mlir-opt %s --canonicalize-elementwise-operand-ranks --canonicalize-elementwise-operand-ranks --shape-inference | FileCheck %s --implicit-check-not=onnx.Reshape
// RUN: onnx-mlir-opt --canonicalize-elementwise-operand-ranks --xfe-onnx-opset-verifier="verify-equal-elementwise-operand-ranks=true" --verify-diagnostics %s

// CHECK-LABEL: func.func @sub
func.func @sub(%x: tensor<4xf32>, %y: tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>, tensor<4xf32>) {
  // CHECK: [[SHAPE:%.*]] = onnx.Constant dense<[1, 1, 4]> : tensor<3xi64>
  // CHECK: [[RESHAPED:%.*]] = "onnx.Reshape"(%arg0, [[SHAPE]])
  // CHECK-SAME: -> tensor<1x1x4xf32>
  // CHECK: [[SUB:%.*]] = "onnx.Sub"([[RESHAPED]], %arg1) {{.*}}onnx_node_name = "sub"
  // CHECK: return [[SUB]], %arg0
  %sub = "onnx.Sub"(%x, %y) {onnx_node_name = "sub"} : (tensor<4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %sub, %x : tensor<2x3x4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @where
func.func @where(%cond: tensor<i1>, %x: tensor<2x3xf32>, %y: tensor<3xf32>) -> tensor<2x3xf32> {
  // CHECK-DAG: [[COND:%.*]] = "onnx.Reshape"(%arg0, {{%.*}}) {{.*}} -> tensor<1x1xi1>
  // CHECK-DAG: [[Y:%.*]] = "onnx.Reshape"(%arg2, {{%.*}}) {{.*}} -> tensor<1x3xf32>
  // CHECK: "onnx.Where"([[COND]], %arg1, [[Y]])
  %where = "onnx.Where"(%cond, %x, %y) : (tensor<i1>, tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return %where : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @sum
func.func @sum(%x: tensor<2x3x4xf32>, %y: tensor<3x4xf32>, %z: tensor<4xf32>) -> tensor<2x3x4xf32> {
  // CHECK-DAG: [[Y:%.*]] = "onnx.Reshape"(%arg1, {{%.*}}) {{.*}} -> tensor<1x3x4xf32>
  // CHECK-DAG: [[Z:%.*]] = "onnx.Reshape"(%arg2, {{%.*}}) {{.*}} -> tensor<1x1x4xf32>
  // CHECK: "onnx.Sum"(%arg0, [[Y]], [[Z]])
  %sum = "onnx.Sum"(%x, %y, %z) : (tensor<2x3x4xf32>, tensor<3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
  return %sum : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @dynamic
func.func @dynamic(%x: tensor<2x?x?xf32>, %y: tensor<?x?xf32>) -> tensor<2x?x?xf32> {
  // CHECK-DAG: [[PREFIX:%.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK-DAG: [[INPUT_SHAPE:%.*]] = "onnx.Shape"(%arg1)
  // CHECK: [[SHAPE:%.*]] = "onnx.Concat"([[PREFIX]], [[INPUT_SHAPE]])
  // CHECK: [[Y:%.*]] = "onnx.Reshape"(%arg1, [[SHAPE]]) {{.*}}allowzero = 1 : si64
  // CHECK-SAME: -> tensor<1x?x?xf32>
  // CHECK: "onnx.Add"(%arg0, [[Y]])
  %add = "onnx.Add"(%x, %y) : (tensor<2x?x?xf32>, tensor<?x?xf32>) -> tensor<2x?x?xf32>
  return %add : tensor<2x?x?xf32>
}

// CHECK-LABEL: func.func @zero_extent
func.func @zero_extent(%x: tensor<2x0x3xf32>, %y: tensor<0x3xf32>) -> tensor<2x0x3xf32> {
  // CHECK: [[SHAPE:%.*]] = onnx.Constant dense<[1, 0, 3]> : tensor<3xi64>
  // CHECK: "onnx.Reshape"(%arg1, [[SHAPE]]) {{.*}}allowzero = 1 : si64
  // CHECK-SAME: -> tensor<1x0x3xf32>
  %add = "onnx.Add"(%x, %y) : (tensor<2x0x3xf32>, tensor<0x3xf32>) -> tensor<2x0x3xf32>
  return %add : tensor<2x0x3xf32>
}

// CHECK-LABEL: func.func @non_broadcast_parameters
func.func @non_broadcast_parameters(%x: tensor<2x3xf32>, %type: tensor<i32>, %bound: tensor<f32>) -> (tensor<2x3xi32>, tensor<2x3xf32>) {
  // CHECK: "onnx.CastLike"(%arg0, %arg1)
  // CHECK: "onnx.Clip"(%arg0, %arg2, %arg2)
  %cast = "onnx.CastLike"(%x, %type) : (tensor<2x3xf32>, tensor<i32>) -> tensor<2x3xi32>
  %clip = "onnx.Clip"(%x, %bound, %bound) : (tensor<2x3xf32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  return %cast, %clip : tensor<2x3xi32>, tensor<2x3xf32>
}

// CHECK-LABEL: func.func @unranked
func.func @unranked(%x: tensor<2x3xf32>, %y: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "onnx.Add"(%arg0, %arg1)
  %add = "onnx.Add"(%x, %y) : (tensor<2x3xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %add : tensor<*xf32>
}

// CHECK-LABEL: func.func @mx6_static_f32
func.func @mx6_static_f32(%x: tensor<2x16xf32>, %y: tensor<16xf32>) -> tensor<2x16xf32> {
  // CHECK: [[QDQ:%.*]] = "onnx.AMDQuarkBFPQuantizeDequantizeOp"(%arg1)
  // CHECK-SAME: axis = 0 : si64
  // CHECK-SAME: bfp_method = "to_bfp_prime", bit_width = 13 : si64, block_size = 16 : si64
  // CHECK-SAME: -> tensor<16xf32>
  // CHECK: [[Y:%.*]] = "onnx.Reshape"([[QDQ]], {{%.*}}) {{.*}} -> tensor<1x16xf32>
  // CHECK: [[MIN:%.*]] = "onnx.Min"(%arg0, [[Y]])
  // CHECK: return [[MIN]]
  %qdq = "onnx.AMDQuarkBFPQuantizeDequantizeOp"(%y) {axis = 0 : si64, bfp_method = "to_bfp_prime", bit_width = 13 : si64, block_size = 16 : si64, rounding_mode = 2 : si64, sub_block_shift_bits = 1 : si64, sub_block_size = 2 : si64} : (tensor<16xf32>) -> tensor<16xf32>
  %min = "onnx.Min"(%x, %qdq) : (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
  return %min : tensor<2x16xf32>
}

// CHECK-LABEL: func.func @mx6_dynamic_bf16
func.func @mx6_dynamic_bf16(%x: tensor<2x?xbf16>, %y: tensor<?xbf16>) -> tensor<2x?xbf16> {
  // CHECK-DAG: [[PREFIX:%.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK-DAG: [[QDQ:%.*]] = "onnx.AMDQuarkBFPQuantizeDequantizeOp"(%arg1)
  // CHECK-SAME: axis = 0 : si64
  // CHECK-SAME: bfp_method = "to_bfp_prime", bit_width = 13 : si64, block_size = 16 : si64
  // CHECK-SAME: -> tensor<?xbf16>
  // CHECK-DAG: [[INPUT_SHAPE:%.*]] = "onnx.Shape"([[QDQ]])
  // CHECK: [[SHAPE:%.*]] = "onnx.Concat"([[PREFIX]], [[INPUT_SHAPE]])
  // CHECK: [[Y:%.*]] = "onnx.Reshape"([[QDQ]], [[SHAPE]]) {{.*}}allowzero = 1 : si64
  // CHECK-SAME: -> tensor<1x?xbf16>
  // CHECK: "onnx.Max"(%arg0, [[Y]])
  %qdq = "onnx.AMDQuarkBFPQuantizeDequantizeOp"(%y) {axis = 0 : si64, bfp_method = "to_bfp_prime", bit_width = 13 : si64, block_size = 16 : si64, rounding_mode = 2 : si64, sub_block_shift_bits = 1 : si64, sub_block_size = 2 : si64} : (tensor<?xbf16>) -> tensor<?xbf16>
  %max = "onnx.Max"(%x, %qdq) : (tensor<2x?xbf16>, tensor<?xbf16>) -> tensor<2x?xbf16>
  return %max : tensor<2x?xbf16>
}
