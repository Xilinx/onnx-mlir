// RUN: onnx-mlir-opt --replace-qdq-where %s | FileCheck %s
// NOTE: This pass runs after quant-types, so ops use native
// `!quant.uniform` types (no explicit Q/DQ ops).

// -----
// Pattern 1: Float X constant with quantized Y and quantized output.
// The float constant X should be quantized using the output's scale/zp,
// wrapped with quant.scast.
// CHECK-LABEL: func.func @test_where_quantize_float_x
func.func @test_where_quantize_float_x(
    %cond: tensor<1x4xi1>,
    %y: tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x!quant.uniform<u8:f32, 0.1:128>> {
  %x_float = "onnx.Constant"() {value = dense<1.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
  %where = "onnx.Where"(%cond, %x_float, %y) :
      (tensor<1x4xi1>, tensor<1x4xf32>, tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>
  return %where : tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK: %[[QCONST:.*]] = onnx.Constant {value = dense<{{.*}}> : tensor<1x4xui8>}
  // CHECK: %[[SCAST:.*]] = quant.scast %[[QCONST]] : tensor<1x4xui8> -> tensor<1x4x!quant.uniform<u8:f32, {{.*}}>>
  // CHECK: %[[WHERE:.*]] = "onnx.Where"(%arg0, %[[SCAST]], %arg1)
  // CHECK: return %[[WHERE]]
}

// -----
// Pattern 1: Float Y constant with quantized X and quantized output.
// The float constant Y should be quantized.
// CHECK-LABEL: func.func @test_where_quantize_float_y
func.func @test_where_quantize_float_y(
    %cond: tensor<1x4xi1>,
    %x: tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x!quant.uniform<u8:f32, 0.1:128>> {
  %y_float = "onnx.Constant"() {value = dense<0.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
  %where = "onnx.Where"(%cond, %x, %y_float) :
      (tensor<1x4xi1>, tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>, tensor<1x4xf32>)
      -> tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>
  return %where : tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK: %[[QCONST:.*]] = onnx.Constant {value = dense<{{.*}}> : tensor<1x4xui8>}
  // CHECK: %[[SCAST:.*]] = quant.scast %[[QCONST]] : tensor<1x4xui8> -> tensor<1x4x!quant.uniform<u8:f32, {{.*}}>>
  // CHECK: %[[WHERE:.*]] = "onnx.Where"(%arg0, %arg1, %[[SCAST]])
  // CHECK: return %[[WHERE]]
}

// -----
// Pattern 1: Both X and Y are float constants with quantized output.
// Both should be quantized and wrapped with quant.scast.
// CHECK-LABEL: func.func @test_where_quantize_both_floats
func.func @test_where_quantize_both_floats(
    %cond: tensor<1x4xi1>)
    -> tensor<1x4x!quant.uniform<u8:f32, 0.1:128>> {
  %x_float = "onnx.Constant"() {value = dense<1.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
  %y_float = "onnx.Constant"() {value = dense<0.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
  %where = "onnx.Where"(%cond, %x_float, %y_float) :
      (tensor<1x4xi1>, tensor<1x4xf32>, tensor<1x4xf32>)
      -> tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>
  return %where : tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK-DAG: %[[QX:.*]] = onnx.Constant {value = dense<{{.*}}> : tensor<1x4xui8>}
  // CHECK-DAG: %[[SX:.*]] = quant.scast %[[QX]]
  // CHECK-DAG: %[[QY:.*]] = onnx.Constant {value = dense<{{.*}}> : tensor<1x4xui8>}
  // CHECK-DAG: %[[SY:.*]] = quant.scast %[[QY]]
  // CHECK: "onnx.Where"(%arg0, %[[SX]], %[[SY]])
}

// -----
// Negative: Both X and Y already quantized, no change needed.
// CHECK-LABEL: func.func @test_where_already_quantized
func.func @test_where_already_quantized(
    %cond: tensor<1x4xi1>,
    %x: tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>,
    %y: tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x!quant.uniform<u8:f32, 0.1:128>> {
  %where = "onnx.Where"(%cond, %x, %y) :
      (tensor<1x4xi1>, tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>,
       tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>
  return %where : tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK: "onnx.Where"(%arg0, %arg1, %arg2)
  // CHECK-NOT: quant.scast
}

// -----
// Negative: Float result with no downstream quant consumers → no change.
// CHECK-LABEL: func.func @test_where_float_result_no_quant
func.func @test_where_float_result_no_quant(
    %cond: tensor<1x4xi1>,
    %x: tensor<1x4xf32>,
    %y: tensor<1x4xf32>)
    -> tensor<1x4xf32> {
  %where = "onnx.Where"(%cond, %x, %y) :
      (tensor<1x4xi1>, tensor<1x4xf32>, tensor<1x4xf32>)
      -> tensor<1x4xf32>
  return %where : tensor<1x4xf32>

  // CHECK: "onnx.Where"(%arg0, %arg1, %arg2)
  // CHECK-SAME: -> tensor<1x4xf32>
  // CHECK-NOT: quant.scast
}

// -----
// Pattern 1: u16 quantized types (matching PSO3 model).
// Float constant X quantized to u16.
// CHECK-LABEL: func.func @test_where_quantize_u16
func.func @test_where_quantize_u16(
    %cond: tensor<1x151xi1>,
    %y: tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>)
    -> tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>> {
  %x_float = "onnx.Constant"() {value = dense<-1.0e+09> : tensor<1x151xf32>} : () -> tensor<1x151xf32>
  %where = "onnx.Where"(%cond, %x_float, %y) :
      (tensor<1x151xi1>, tensor<1x151xf32>, tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>)
      -> tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>
  return %where : tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>

  // CHECK: %[[QCONST:.*]] = onnx.Constant {value = dense<0> : tensor<1x151xui16>}
  // CHECK: %[[SCAST:.*]] = quant.scast %[[QCONST]] : tensor<1x151xui16> -> tensor<1x151x!quant.uniform<u16:f32, {{.*}}>>
  // CHECK: "onnx.Where"(%arg0, %[[SCAST]], %arg1)
}

// -----
// Pattern 2: Cast(i32→i1) condition on Where with quantized result.
// Should replace Cast with XCOMPILERFusedEltwise(REQUANTIZE) preserving i1 output.
// CHECK-LABEL: func.func @test_where_cast_cond_requantize
func.func @test_where_cast_cond_requantize(
    %input: tensor<1x151xi32>,
    %x: tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>,
    %y: tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>)
    -> tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>> {
  %cond = "onnx.Cast"(%input) {to = 9 : si64} :
      (tensor<1x151xi32>) -> tensor<1x151xi1>
  %where = "onnx.Where"(%cond, %x, %y) :
      (tensor<1x151xi1>, tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>,
       tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>)
      -> tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>
  return %where : tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>

  // CHECK-NOT: "onnx.Cast"
  // CHECK: %[[NONE:.*]] = "onnx.NoValue"
  // CHECK: %[[REQ:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NONE]])
  // CHECK-SAME: a_scale = [1.000000e+00 : f32]
  // CHECK-SAME: a_zero_point = [0]
  // CHECK-SAME: b_scale = [1.000000e+00 : f32]
  // CHECK-SAME: b_zero_point = [0]
  // CHECK-SAME: type = "REQUANTIZE"
  // CHECK-SAME: y_scale = [1.000000e+00 : f32]
  // CHECK-SAME: y_zero_point = [0]
  // CHECK-SAME: -> tensor<1x151xi1>
  // CHECK: "onnx.Where"(%[[REQ]], %arg1, %arg2)
}

// -----
// Pattern 2: Negative — non-Cast condition should not be replaced.
// CHECK-LABEL: func.func @test_where_no_cast_cond
func.func @test_where_no_cast_cond(
    %cond: tensor<1x4xi1>,
    %x: tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>,
    %y: tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x!quant.uniform<u8:f32, 0.1:128>> {
  %where = "onnx.Where"(%cond, %x, %y) :
      (tensor<1x4xi1>, tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>,
       tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>
  return %where : tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK-NOT: "onnx.XCOMPILERFusedEltwise"
  // CHECK: "onnx.Where"(%arg0, %arg1, %arg2)
}

// -----
// Pattern 3: Greater(i64,i64→i1) condition on Where with quantized result.
// Should replace Greater with XCOMPILERFusedEltwise(GREATER) preserving i1 output.
// CHECK-LABEL: func.func @test_where_greater_cond_eltwise
func.func @test_where_greater_cond_eltwise(
    %lhs: tensor<1x151xi64>,
    %rhs: tensor<1x151xi64>,
    %x: tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>,
    %y: tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>)
    -> tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>> {
  %cond = "onnx.Greater"(%lhs, %rhs) :
      (tensor<1x151xi64>, tensor<1x151xi64>) -> tensor<1x151xi1>
  %where = "onnx.Where"(%cond, %x, %y) :
      (tensor<1x151xi1>, tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>,
       tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>)
      -> tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>
  return %where : tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>

  // CHECK-NOT: "onnx.Greater"
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1)
  // CHECK-SAME: a_scale = [1.000000e+00 : f32]
  // CHECK-SAME: a_zero_point = [0]
  // CHECK-SAME: b_scale = [1.000000e+00 : f32]
  // CHECK-SAME: b_zero_point = [0]
  // CHECK-SAME: type = "GREATER"
  // CHECK-SAME: y_scale = [1.000000e+00 : f32]
  // CHECK-SAME: y_zero_point = [0]
  // CHECK-SAME: -> tensor<1x151xi1>
  // CHECK: "onnx.Where"(%[[FUSED]], %arg2, %arg3)
}

// -----
// Pattern 3: Negative — Greater condition on non-quantized Where should not fire.
// CHECK-LABEL: func.func @test_where_greater_float_result
func.func @test_where_greater_float_result(
    %lhs: tensor<1x4xi64>,
    %rhs: tensor<1x4xi64>,
    %x: tensor<1x4xf32>,
    %y: tensor<1x4xf32>)
    -> tensor<1x4xf32> {
  %cond = "onnx.Greater"(%lhs, %rhs) :
      (tensor<1x4xi64>, tensor<1x4xi64>) -> tensor<1x4xi1>
  %where = "onnx.Where"(%cond, %x, %y) :
      (tensor<1x4xi1>, tensor<1x4xf32>, tensor<1x4xf32>)
      -> tensor<1x4xf32>
  return %where : tensor<1x4xf32>

  // CHECK: "onnx.Greater"(%arg0, %arg1)
  // CHECK: "onnx.Where"
  // CHECK-NOT: "onnx.XCOMPILERFusedEltwise"
}

// -----
// Combined: Cast condition + float constant X on quantized Where.
// Both patterns should fire: Cast→REQUANTIZE and float X→quantized.
// CHECK-LABEL: func.func @test_where_cast_cond_and_float_x
func.func @test_where_cast_cond_and_float_x(
    %input: tensor<1x151xi32>,
    %y: tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>)
    -> tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>> {
  %cond = "onnx.Cast"(%input) {to = 9 : si64} :
      (tensor<1x151xi32>) -> tensor<1x151xi1>
  %x_float = "onnx.Constant"() {value = dense<-1.0e+09> : tensor<1x151xf32>} : () -> tensor<1x151xf32>
  %where = "onnx.Where"(%cond, %x_float, %y) :
      (tensor<1x151xi1>, tensor<1x151xf32>, tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>)
      -> tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>
  return %where : tensor<1x151x!quant.uniform<u16:f32, 0.015259:65535>>

  // CHECK-NOT: "onnx.Cast"
  // CHECK-DAG: %[[NONE:.*]] = "onnx.NoValue"
  // CHECK-DAG: %[[REQ:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NONE]]){{.*}}type = "REQUANTIZE"{{.*}} -> tensor<1x151xi1>
  // CHECK-DAG: %[[QCONST:.*]] = onnx.Constant {value = dense<0> : tensor<1x151xui16>}
  // CHECK-DAG: %[[SCAST:.*]] = quant.scast %[[QCONST]]
  // CHECK: "onnx.Where"(%[[REQ]], %[[SCAST]], %arg1)
}
