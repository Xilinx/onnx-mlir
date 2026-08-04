// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa="excluded-ops=Cast" -cse %s -split-input-file | FileCheck %s --check-prefix=EXCLUDE-CAST

func.func @test_quantizeLinear(%arg0 : tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xi8> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>                       
  %1 = onnx.Constant dense<0> : tensor<i8>                                   
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<32x3x224x224xf32>, tensor<f32>, tensor<i8>) -> tensor<32x3x224x224xi8>
  "func.return"(%2) : (tensor<32x3x224x224xi8>) -> ()
}
// CHECK-LABEL:  @test_quantizeLinear
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xi8>
// CHECK-DAG:    %[[ZP:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1x1x1xi8>}> : () -> tensor<1x1x1x1xi8>
// CHECK-DAG:    %[[SCALE:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:    %[[REC:.*]] = tosa.reciprocal %[[SCALE]] : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:    %[[MUL:.*]] = tosa.mul %[[ARG_0]], %[[REC]], {{.*}}: (tensor<32x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    %[[MUL_CAST:.*]] = tosa.cast %[[MUL]] : (tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xi32>
// CHECK-DAG:    %[[ZPCAST:.*]] = tosa.cast %[[ZP]] : (tensor<1x1x1x1xi8>) -> tensor<1x1x1x1xi32>
// CHECK-DAG:    %[[ADD:.*]] = tosa.add %[[MUL_CAST]], %[[ZPCAST]] : (tensor<32x3x224x224xi32>, tensor<1x1x1x1xi32>) -> tensor<32x3x224x224xi32>
// CHECK-DAG:    %[[CLAMP:.*]] = tosa.clamp %[[ADD]] {max_fp = 1.270000e+02 : f32, max_int = 127 : i64, min_fp = -1.280000e+02 : f32, min_int = -128 : i64} : (tensor<32x3x224x224xi32>) -> tensor<32x3x224x224xi32>
// CHECK-DAG:    %[[CAST:.*]]  = tosa.cast %[[CLAMP]] : (tensor<32x3x224x224xi32>) -> tensor<32x3x224x224xi8>
// CHECK-DAG:    return %[[CAST]] : tensor<32x3x224x224xi8>
// EXCLUDE-CAST-LABEL: @test_quantizeLinear
// EXCLUDE-CAST: onnx.Cast
// EXCLUDE-CAST-NOT: tosa.cast
// EXCLUDE-CAST: return

// -----

func.func @test_quantizeLinear_none(%arg0 : tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xi8> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>                       
  %1 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none                              
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<32x3x224x224xf32>, tensor<f32>, none) -> tensor<32x3x224x224xi8>
  "func.return"(%2) : (tensor<32x3x224x224xi8>) -> ()
}

// CHECK-LABEL: @test_quantizeLinear_none
// CHECK-SAME:    (%[[ARG_0:.*]]: tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xui8>
// CHECK-DAG:   %[[SCALE:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:   %[[REC:.*]] = tosa.reciprocal %[[SCALE]] : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:   %[[MUL:.*]] = tosa.mul %[[ARG_0]], %[[REC]], {{.*}}: (tensor<32x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:   %[[MUL_CAST:.*]] = tosa.cast %[[MUL]] : (tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xi32>
// CHECK-DAG:   %[[CAST:.*]] = tosa.cast %[[MUL_CAST]] : (tensor<32x3x224x224xi32>) -> tensor<32x3x224x224xui8>
// CHECK-DAG:   return %[[CAST]] : tensor<32x3x224x224xui8>

// -----

func.func @test_quantizeLinear_per_axis(%arg0: tensor<8x2xf32>) -> tensor<8x2xi8> {
  %0 = onnx.Constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %1 = onnx.Constant dense<[0, 1]> : tensor<2xi8>
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1)
    {axis = 1 : si64,
     saturate = 1 : si64} : (tensor<8x2xf32>, tensor<2xf32>, tensor<2xi8>) -> tensor<8x2xi8>
  return %2 : tensor<8x2xi8>
}
// CHECK-LABEL:   func.func @test_quantizeLinear_per_axis(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<8x2xf32>) -> tensor<8x2xi8> {
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<{{\[\[}}1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32>}> : () -> tensor<1x2xf32>
// CHECK:           %[[REC:.*]] = tosa.reciprocal %[[VAL_2]] : (tensor<1x2xf32>) -> tensor<1x2xf32>
// CHECK:           %[[MUL:.*]] = tosa.mul %[[VAL_0]], %[[REC]], {{.*}}: (tensor<8x2xf32>, tensor<1x2xf32>, tensor<1xi8>) -> tensor<8x2xf32>
// CHECK:           %[[MUL_CAST:.*]] = tosa.cast %[[MUL]] : (tensor<8x2xf32>) -> tensor<8x2xi32>
// CHECK:           %[[ZP:.*]] = "tosa.const"() <{value = dense<{{\[\[}}0, 1]]> : tensor<1x2xi8>}> : () -> tensor<1x2xi8>
// CHECK:           %[[ZPCAST:.*]] = tosa.cast %[[ZP]] : (tensor<1x2xi8>) -> tensor<1x2xi32>
// CHECK:           %[[ADD:.*]] = tosa.add %[[MUL_CAST]], %[[ZPCAST]] : (tensor<8x2xi32>, tensor<1x2xi32>) -> tensor<8x2xi32>
// CHECK:           %[[CLAMP:.*]] = tosa.clamp %[[ADD]] {max_fp = 1.270000e+02 : f32, max_int = 127 : i64, min_fp = -1.280000e+02 : f32, min_int = -128 : i64} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK:           %[[CAST:.*]] = tosa.cast %[[CLAMP]] : (tensor<8x2xi32>) -> tensor<8x2xi8>
// CHECK:           return %[[CAST]] : tensor<8x2xi8>
// CHECK:         }

// -----

func.func @test_quantizeLinear_negative_axis(%arg0: tensor<8x2xf32>) -> tensor<8x2xi8> {
  %0 = onnx.Constant dense<2.000000e+00> : tensor<8xf32>
  %1 = onnx.Constant dense<1> : tensor<8xi8>
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1)
    {axis = -2 : si64,
     saturate = 1 : si64} : (tensor<8x2xf32>, tensor<8xf32>, tensor<8xi8>) -> tensor<8x2xi8>
  return %2 : tensor<8x2xi8>
}
// CHECK-LABEL: test_quantizeLinear_negative_axis
// CHECK: "tosa.const"() {{.*}} : tensor<8x1xi8>

// -----

func.func @test_quantizeLinear_ui8(%arg0 : tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xui8> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>                       
  %1 = onnx.Constant dense<0> : tensor<ui8>                                   
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<32x3x224x224xf32>, tensor<f32>, tensor<ui8>) -> tensor<32x3x224x224xui8>
  "func.return"(%2) : (tensor<32x3x224x224xui8>) -> ()
}
// CHECK-LABEL:  @test_quantizeLinear_ui8
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xui8>
// CHECK-DAG:    %[[ZP:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1x1x1xui8>}> : () -> tensor<1x1x1x1xui8>
// CHECK-DAG:    %[[SCALE:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:    %[[REC:.*]] = tosa.reciprocal %[[SCALE]] : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:    %[[MUL:.*]] = tosa.mul %[[ARG_0]], %[[REC]], {{.*}}: (tensor<32x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    %[[MUL_CAST:.*]] = tosa.cast %[[MUL]] : (tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xi32>
// CHECK-DAG:    %[[ZPCAST:.*]] = tosa.cast %[[ZP]] : (tensor<1x1x1x1xui8>) -> tensor<1x1x1x1xi32>
// CHECK-DAG:    %[[ADD:.*]] = tosa.add %[[MUL_CAST]], %[[ZPCAST]] : (tensor<32x3x224x224xi32>, tensor<1x1x1x1xi32>) -> tensor<32x3x224x224xi32>
// CHECK-DAG:    %[[CLAMP:.*]] = tosa.clamp %[[ADD]] {max_fp = 2.550000e+02 : f32, max_int = 255 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<32x3x224x224xi32>) -> tensor<32x3x224x224xi32>
// CHECK-DAG:    %[[CAST:.*]]  = tosa.cast %[[CLAMP]] : (tensor<32x3x224x224xi32>) -> tensor<32x3x224x224xui8>
// CHECK-DAG:    return %[[CAST]] : tensor<32x3x224x224xui8>

// -----

// The default `axis` is `1` when it's absent in ONNX, which conflicts
// with the allowed range of `axis` when the input has rank 1.
// See https://github.com/onnx/onnx/issues/6067
func.func @default_axis(%arg0 : tensor<32xf32>) -> tensor<32xi8> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %1 = onnx.Constant dense<0> : tensor<i8>
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<32xf32>, tensor<f32>, tensor<i8>) -> tensor<32xi8>
  return %2 : tensor<32xi8>
}

// CHECK-LABEL: default_axis
// CHECK-NOT: onnx.QuantizeLinear

// -----


func.func @all_scalar(%arg0 : tensor<f32>) -> tensor<i8> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %1 = onnx.Constant dense<0> : tensor<i8>
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<f32>, tensor<f32>, tensor<i8>) -> tensor<i8>
  return %2 : tensor<i8>
}

// CHECK-LABEL: all_scalar
// CHECK-NOT: onnx.QuantizeLinear

// -----

func.func @dynamic_static(%arg0 : tensor<?xf32>, %arg1 : tensor<f32>, %arg2 : tensor<i8>) -> tensor<1xi8> {
  %0 = "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<?xf32>, tensor<f32>, tensor<i8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// CHECK-LABEL: dynamic_static
// CHECK-SAME:    (%[[ARG_0:.*]]: tensor<?xf32>, %[[ARG_1:.*]]: tensor<f32>, %[[ARG_2:.*]]: tensor<i8>) -> tensor<1xi8>
// CHECK:         %[[REC:.*]] = tosa.reciprocal
// CHECK:         %[[CLAMP:.*]] = tosa.clamp
// CHECK:         %[[OUT:.*]] = tosa.cast %[[CLAMP]] : (tensor<1xi32>) -> tensor<1xi8>
// CHECK:         return %[[OUT]] : tensor<1xi8>

// -----

// ONNX requires (x / y_scale) to be rounded to the nearest even before the
// (potentially odd) zero point is added: 2.7 -> 3, 8.5 -> 8, 9.5 -> 10.
// Truncating towards zero instead would bias every element by up to one
// quantization step, so the two lowerings below have to stay numerically equal.
func.func @test_quantizeLinear_round_half_even(%arg0: tensor<8x2xf32>) -> tensor<8x2xi8> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %1 = onnx.Constant dense<1> : tensor<i8>
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<8x2xf32>, tensor<f32>, tensor<i8>) -> tensor<8x2xi8>
  return %2 : tensor<8x2xi8>
}

// tosa.cast float-to-int already rounds half-to-even, so it must do the
// rounding and the widening to i32 in one operation. Any floor/select rounding
// appearing here means the rounding was re-expressed, which is only correct if
// it reproduces round-half-to-even exactly.
// CHECK-LABEL:  @test_quantizeLinear_round_half_even
// CHECK:          %[[MUL:.*]] = tosa.mul %arg0
// CHECK-NEXT:     %[[ROUNDED:.*]] = tosa.cast %[[MUL]] : (tensor<8x2xf32>) -> tensor<8x2xi32>
// CHECK-NOT:      tosa.floor
// CHECK-NOT:      tosa.select
// CHECK:          return

// onnx.Cast truncates towards zero, so with cast lowering isolated the rounding
// is spelled out first. On a tie, 2 * floor(0.5 * x + 0.5) is the even
// neighbour. Truncating the already integral result is exact.
// EXCLUDE-CAST-LABEL:  @test_quantizeLinear_round_half_even
// EXCLUDE-CAST:          %[[MUL:.*]] = tosa.mul %arg0
// EXCLUDE-CAST-DAG:      %[[ONE:.*]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1xf32>}>
// EXCLUDE-CAST-DAG:      %[[TWO:.*]] = "tosa.const"() <{value = dense<2.000000e+00> : tensor<1x1xf32>}>
// EXCLUDE-CAST-DAG:      %[[HALF:.*]] = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x1xf32>}>
// EXCLUDE-CAST:          %[[Y:.*]] = tosa.floor %[[MUL]] : (tensor<8x2xf32>) -> tensor<8x2xf32>
// EXCLUDE-CAST:          %[[R:.*]] = tosa.sub %[[MUL]], %[[Y]]
// EXCLUDE-CAST:          %[[YP1:.*]] = tosa.add %[[Y]], %[[ONE]]
// EXCLUDE-CAST:          %[[GT:.*]] = tosa.greater %[[R]], %[[HALF]]
// EXCLUDE-CAST:          %[[NEAREST:.*]] = tosa.select %[[GT]], %[[YP1]], %[[Y]]
// EXCLUDE-CAST:          %[[HALFX:.*]] = tosa.mul %[[MUL]], %[[HALF]]
// EXCLUDE-CAST:          %[[SHIFTED:.*]] = tosa.add %[[HALFX]], %[[HALF]]
// EXCLUDE-CAST:          %[[FLOORSHIFTED:.*]] = tosa.floor %[[SHIFTED]]
// EXCLUDE-CAST:          %[[EVENNB:.*]] = tosa.mul %[[FLOORSHIFTED]], %[[TWO]]
// EXCLUDE-CAST:          %[[TIE:.*]] = tosa.equal %[[R]], %[[HALF]]
// EXCLUDE-CAST:          %[[ROUNDED:.*]] = tosa.select %[[TIE]], %[[EVENNB]], %[[NEAREST]]
// EXCLUDE-CAST:          %[[NARROWED:.*]] = "onnx.Cast"(%[[ROUNDED]]) {{.*}}to = i32{{.*}} : (tensor<8x2xf32>) -> tensor<8x2xi32>
// EXCLUDE-CAST-NOT:      tosa.cast
// EXCLUDE-CAST:          return
