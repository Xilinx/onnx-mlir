// RUN: onnx-mlir-opt --enable-qdq-data-movement-canonicalization --canonicalize-with-rn %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --canonicalize-with-rn %s -split-input-file | FileCheck %s --check-prefix=NOOPT

// -----

func.func @transpose_moves_dq_and_q(%arg0: tensor<1x2x3xui8>, %arg1: tensor<1x2x3xf32>) -> (tensor<1x3x2xf32>, tensor<1x3x2xui8>) {
  %s = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %z = onnx.Constant dense<128> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%arg0, %s, %z) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x2x3xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2x3xf32>
  %t0 = "onnx.Transpose"(%dq) {perm = [0, 2, 1]} : (tensor<1x2x3xf32>) -> tensor<1x3x2xf32>
  %t1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 1]} : (tensor<1x2x3xf32>) -> tensor<1x3x2xf32>
  %q = "onnx.QuantizeLinear"(%t1, %s, %z) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x3x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x3x2xui8>
  return %t0, %q : tensor<1x3x2xf32>, tensor<1x3x2xui8>
}

// NOOPT-LABEL: func.func @transpose_moves_dq_and_q
// NOOPT:       "onnx.DequantizeLinear"
// NOOPT:       "onnx.Transpose"
// NOOPT:       "onnx.Transpose"
// NOOPT:       "onnx.QuantizeLinear"

// CHECK-LABEL: func.func @transpose_moves_dq_and_q
// CHECK-SAME:  (%[[QI:.*]]: tensor<1x2x3xui8>, %[[FI:.*]]: tensor<1x2x3xf32>)
// CHECK-DAG:   %[[S:.*]] = onnx.Constant dense<2.500000e-01> : tensor<f32>
// CHECK-DAG:   %[[Z:.*]] = onnx.Constant dense<128> : tensor<ui8>
// CHECK-DAG:   %[[TQ:.*]] = "onnx.Transpose"(%[[QI]]) {perm = [0, 2, 1]} : (tensor<1x2x3xui8>) -> tensor<1x3x2xui8>
// CHECK-DAG:   %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[TQ]], %[[S]], %[[Z]]) {{.*}} : (tensor<1x3x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x3x2xf32>
// CHECK-DAG:   %[[QF:.*]] = "onnx.QuantizeLinear"(%[[FI]], %[[S]], %[[Z]]) {{.*}} : (tensor<1x2x3xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x2x3xui8>
// CHECK-DAG:   %[[TQI:.*]] = "onnx.Transpose"(%[[QF]]) {perm = [0, 2, 1]} : (tensor<1x2x3xui8>) -> tensor<1x3x2xui8>
// CHECK:       return %[[DQ]], %[[TQI]]

// -----

func.func @quant_result_multi_use_moved(%arg0: tensor<1x2x3xf32>) -> (tensor<1x3x2xui8>, tensor<1x3x2xui8>) {
  %s = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %z = onnx.Constant dense<128> : tensor<ui8>
  %t = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} : (tensor<1x2x3xf32>) -> tensor<1x3x2xf32>
  %q = "onnx.QuantizeLinear"(%t, %s, %z) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x3x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x3x2xui8>
  return %q, %q : tensor<1x3x2xui8>, tensor<1x3x2xui8>
}

// CHECK-LABEL: func.func @quant_result_multi_use_moved
// CHECK-SAME:  (%[[FI:.*]]: tensor<1x2x3xf32>)
// CHECK-DAG:   %[[S:.*]] = onnx.Constant dense<2.500000e-01> : tensor<f32>
// CHECK-DAG:   %[[Z:.*]] = onnx.Constant dense<128> : tensor<ui8>
// CHECK-DAG:   %[[QF:.*]] = "onnx.QuantizeLinear"(%[[FI]], %[[S]], %[[Z]]) {{.*}} : (tensor<1x2x3xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x2x3xui8>
// CHECK-DAG:   %[[TQI:.*]] = "onnx.Transpose"(%[[QF]]) {perm = [0, 2, 1]} : (tensor<1x2x3xui8>) -> tensor<1x3x2xui8>
// CHECK:       return %[[TQI]], %[[TQI]]

// -----

func.func @quant_input_multi_use_not_moved(%arg0: tensor<1x2x3xf32>) -> (tensor<1x3x2xf32>, tensor<1x3x2xui8>) {
  %s = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %z = onnx.Constant dense<128> : tensor<ui8>
  %t = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} : (tensor<1x2x3xf32>) -> tensor<1x3x2xf32>
  %q = "onnx.QuantizeLinear"(%t, %s, %z) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x3x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x3x2xui8>
  return %t, %q : tensor<1x3x2xf32>, tensor<1x3x2xui8>
}

// CHECK-LABEL: func.func @quant_input_multi_use_not_moved
// CHECK-SAME:  (%[[FI:.*]]: tensor<1x2x3xf32>)
// CHECK-DAG:   %[[S:.*]] = onnx.Constant dense<2.500000e-01> : tensor<f32>
// CHECK-DAG:   %[[Z:.*]] = onnx.Constant dense<128> : tensor<ui8>
// CHECK:       %[[T:.*]] = "onnx.Transpose"(%[[FI]]) {perm = [0, 2, 1]} : (tensor<1x2x3xf32>) -> tensor<1x3x2xf32>
// CHECK-NEXT:  %[[Q:.*]] = "onnx.QuantizeLinear"(%[[T]], %[[S]], %[[Z]]) {{.*}} : (tensor<1x3x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x3x2xui8>
// CHECK-NEXT:  return %[[T]], %[[Q]]

// -----

func.func @reshape_moves_dq_and_q(%arg0: tensor<1x4xui8>, %arg1: tensor<1x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xui8>) {
  %s = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<42> : tensor<ui8>
  %shape = onnx.Constant dense<[2, 2]> : tensor<2xi64>
  %dq = "onnx.DequantizeLinear"(%arg0, %s, %z) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %r0 = "onnx.Reshape"(%dq, %shape) {allowzero = 0 : si64} : (tensor<1x4xf32>, tensor<2xi64>) -> tensor<2x2xf32>
  %r1 = "onnx.Reshape"(%arg1, %shape) {allowzero = 0 : si64} : (tensor<1x4xf32>, tensor<2xi64>) -> tensor<2x2xf32>
  %q = "onnx.QuantizeLinear"(%r1, %s, %z) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<2x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x2xui8>
  return %r0, %q : tensor<2x2xf32>, tensor<2x2xui8>
}

// CHECK-LABEL: func.func @reshape_moves_dq_and_q
// CHECK-SAME:  (%[[QI:.*]]: tensor<1x4xui8>, %[[FI:.*]]: tensor<1x4xf32>)
// CHECK-DAG:   %[[S:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK-DAG:   %[[Z:.*]] = onnx.Constant dense<42> : tensor<ui8>
// CHECK-DAG:   %[[SHAPE:.*]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:   %[[RQ:.*]] = "onnx.Reshape"(%[[QI]], %[[SHAPE]]) {allowzero = 0 : si64} : (tensor<1x4xui8>, tensor<2xi64>) -> tensor<2x2xui8>
// CHECK-DAG:   %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[RQ]], %[[S]], %[[Z]]) {{.*}} : (tensor<2x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x2xf32>
// CHECK-DAG:   %[[QF:.*]] = "onnx.QuantizeLinear"(%[[FI]], %[[S]], %[[Z]]) {{.*}} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
// CHECK-DAG:   %[[RQI:.*]] = "onnx.Reshape"(%[[QF]], %[[SHAPE]]) {allowzero = 0 : si64} : (tensor<1x4xui8>, tensor<2xi64>) -> tensor<2x2xui8>
// CHECK:       return %[[DQ]], %[[RQI]]

// -----

func.func @slice_moves_dq_and_q(%arg0: tensor<1x4xui8>, %arg1: tensor<1x4xf32>) -> (tensor<1x2xf32>, tensor<1x2xui8>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %s = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<128> : tensor<ui8>
  %starts = onnx.Constant dense<1> : tensor<1xi64>
  %ends = onnx.Constant dense<3> : tensor<1xi64>
  %dq = "onnx.DequantizeLinear"(%arg0, %s, %z) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %sl0 = "onnx.Slice"(%dq, %starts, %ends, %none, %none) : (tensor<1x4xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<1x2xf32>
  %sl1 = "onnx.Slice"(%arg1, %starts, %ends, %none, %none) : (tensor<1x4xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<1x2xf32>
  %q = "onnx.QuantizeLinear"(%sl1, %s, %z) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x2xui8>
  return %sl0, %q : tensor<1x2xf32>, tensor<1x2xui8>
}

// CHECK-LABEL: func.func @slice_moves_dq_and_q
// CHECK-SAME:  (%[[QI:.*]]: tensor<1x4xui8>, %[[FI:.*]]: tensor<1x4xf32>)
// CHECK-DAG:   %[[S:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK-DAG:   %[[Z:.*]] = onnx.Constant dense<128> : tensor<ui8>
// CHECK-DAG:   %[[NONE:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:   %[[STARTS:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:   %[[ENDS:.*]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:   %[[SLQ:.*]] = "onnx.Slice"(%[[QI]], %[[STARTS]], %[[ENDS]], %[[NONE]], %[[NONE]]) : (tensor<1x4xui8>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<1x2xui8>
// CHECK-DAG:   %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[SLQ]], %[[S]], %[[Z]]) {{.*}} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>
// CHECK-DAG:   %[[QF:.*]] = "onnx.QuantizeLinear"(%[[FI]], %[[S]], %[[Z]]) {{.*}} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
// CHECK-DAG:   %[[SLQI:.*]] = "onnx.Slice"(%[[QF]], %[[STARTS]], %[[ENDS]], %[[NONE]], %[[NONE]]) : (tensor<1x4xui8>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<1x2xui8>
// CHECK:       return %[[DQ]], %[[SLQI]]

// -----

func.func @pad_reflect_moves_dq_and_q(%arg0: tensor<1x4xui8>, %arg1: tensor<1x4xf32>) -> (tensor<1x6xf32>, tensor<1x6xui8>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<128> : tensor<ui8>
  %pads = onnx.Constant dense<[0, 1, 0, 1]> : tensor<4xi64>
  %dq = "onnx.DequantizeLinear"(%arg0, %s, %z) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %p0 = "onnx.Pad"(%dq, %pads, %none, %none) {mode = "reflect"} : (tensor<1x4xf32>, tensor<4xi64>, none, none) -> tensor<1x6xf32>
  %p1 = "onnx.Pad"(%arg1, %pads, %none, %none) {mode = "reflect"} : (tensor<1x4xf32>, tensor<4xi64>, none, none) -> tensor<1x6xf32>
  %q = "onnx.QuantizeLinear"(%p1, %s, %z) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x6xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x6xui8>
  return %p0, %q : tensor<1x6xf32>, tensor<1x6xui8>
}

// CHECK-LABEL: func.func @pad_reflect_moves_dq_and_q
// CHECK-SAME:  (%[[QI:.*]]: tensor<1x4xui8>, %[[FI:.*]]: tensor<1x4xf32>)
// CHECK-DAG:   %[[S:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG:   %[[Z:.*]] = onnx.Constant dense<128> : tensor<ui8>
// CHECK-DAG:   %[[NONE:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:   %[[PADS:.*]] = onnx.Constant dense<[0, 1, 0, 1]> : tensor<4xi64>
// CHECK-DAG:   %[[PQ:.*]] = "onnx.Pad"(%[[QI]], %[[PADS]], %[[NONE]], %[[NONE]]) {mode = "reflect"} : (tensor<1x4xui8>, tensor<4xi64>, none, none) -> tensor<1x6xui8>
// CHECK-DAG:   %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[PQ]], %[[S]], %[[Z]]) {{.*}} : (tensor<1x6xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x6xf32>
// CHECK-DAG:   %[[QF:.*]] = "onnx.QuantizeLinear"(%[[FI]], %[[S]], %[[Z]]) {{.*}} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
// CHECK-DAG:   %[[PQI:.*]] = "onnx.Pad"(%[[QF]], %[[PADS]], %[[NONE]], %[[NONE]]) {mode = "reflect"} : (tensor<1x4xui8>, tensor<4xi64>, none, none) -> tensor<1x6xui8>
// CHECK:       return %[[DQ]], %[[PQI]]

// -----

func.func @pad_constant_not_moved(%arg0: tensor<1x2xui8>, %arg1: tensor<1x2xf32>) -> (tensor<1x4xf32>, tensor<1x4xui8>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<128> : tensor<ui8>
  %pads = onnx.Constant dense<[0, 1, 0, 1]> : tensor<4xi64>
  %dq = "onnx.DequantizeLinear"(%arg0, %s, %z) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>
  %p0 = "onnx.Pad"(%dq, %pads, %none, %none) {mode = "constant"} : (tensor<1x2xf32>, tensor<4xi64>, none, none) -> tensor<1x4xf32>
  %p1 = "onnx.Pad"(%arg1, %pads, %none, %none) {mode = "constant"} : (tensor<1x2xf32>, tensor<4xi64>, none, none) -> tensor<1x4xf32>
  %q = "onnx.QuantizeLinear"(%p1, %s, %z) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %p0, %q : tensor<1x4xf32>, tensor<1x4xui8>
}

// CHECK-LABEL: func.func @pad_constant_not_moved
// CHECK:       "onnx.DequantizeLinear"
// CHECK:       "onnx.Pad"
// CHECK:       "onnx.Pad"
// CHECK:       "onnx.QuantizeLinear"

// -----

func.func @concat_moves_dq_and_q(%arg0: tensor<1x2xui8>, %arg1: tensor<1x3xui8>, %arg2: tensor<1x2xf32>, %arg3: tensor<1x3xf32>) -> (tensor<1x5xf32>, tensor<1x5xui8>) {
  %s0 = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %z0 = onnx.Constant dense<128> : tensor<ui8>
  %s1 = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %z1 = onnx.Constant dense<128> : tensor<ui8>
  %dq0 = "onnx.DequantizeLinear"(%arg0, %s0, %z0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>
  %dq1 = "onnx.DequantizeLinear"(%arg1, %s1, %z1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x3xf32>
  %c0 = "onnx.Concat"(%dq0, %dq1) {axis = 1 : si64} : (tensor<1x2xf32>, tensor<1x3xf32>) -> tensor<1x5xf32>
  %c1 = "onnx.Concat"(%arg2, %arg3) {axis = 1 : si64} : (tensor<1x2xf32>, tensor<1x3xf32>) -> tensor<1x5xf32>
  %q = "onnx.QuantizeLinear"(%c1, %s0, %z0) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x5xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x5xui8>
  return %c0, %q : tensor<1x5xf32>, tensor<1x5xui8>
}

// CHECK-LABEL: func.func @concat_moves_dq_and_q
// CHECK-SAME:  (%[[QI0:.*]]: tensor<1x2xui8>, %[[QI1:.*]]: tensor<1x3xui8>, %[[FI0:.*]]: tensor<1x2xf32>, %[[FI1:.*]]: tensor<1x3xf32>)
// CHECK-DAG:   %[[S:.*]] = onnx.Constant dense<2.500000e-01> : tensor<f32>
// CHECK-DAG:   %[[Z:.*]] = onnx.Constant dense<128> : tensor<ui8>
// CHECK-DAG:   %[[CQ:.*]] = "onnx.Concat"(%[[QI0]], %[[QI1]]) {axis = 1 : si64} : (tensor<1x2xui8>, tensor<1x3xui8>) -> tensor<1x5xui8>
// CHECK-DAG:   %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[CQ]], %[[S]], %[[Z]]) {{.*}} : (tensor<1x5xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x5xf32>
// CHECK-DAG:   %[[QF0:.*]] = "onnx.QuantizeLinear"(%[[FI0]], %[[S]], %[[Z]]) {{.*}} : (tensor<1x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x2xui8>
// CHECK-DAG:   %[[QF1:.*]] = "onnx.QuantizeLinear"(%[[FI1]], %[[S]], %[[Z]]) {{.*}} : (tensor<1x3xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x3xui8>
// CHECK-DAG:   %[[CQI:.*]] = "onnx.Concat"(%[[QF0]], %[[QF1]]) {axis = 1 : si64} : (tensor<1x2xui8>, tensor<1x3xui8>) -> tensor<1x5xui8>
// CHECK:       return %[[DQ]], %[[CQI]]

// -----

func.func @tile_moves_dq_and_q(%arg0: tensor<2xui8>, %arg1: tensor<2xf32>) -> (tensor<4xf32>, tensor<4xui8>) {
  %s = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<128> : tensor<ui8>
  %repeats = onnx.Constant dense<2> : tensor<1xi64>
  %dq = "onnx.DequantizeLinear"(%arg0, %s, %z) {axis = 1 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %t0 = "onnx.Tile"(%dq, %repeats) : (tensor<2xf32>, tensor<1xi64>) -> tensor<4xf32>
  %t1 = "onnx.Tile"(%arg1, %repeats) : (tensor<2xf32>, tensor<1xi64>) -> tensor<4xf32>
  %q = "onnx.QuantizeLinear"(%t1, %s, %z) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<4xf32>, tensor<f32>, tensor<ui8>) -> tensor<4xui8>
  return %t0, %q : tensor<4xf32>, tensor<4xui8>
}

// CHECK-LABEL: func.func @tile_moves_dq_and_q
// CHECK-SAME:  (%[[QI:.*]]: tensor<2xui8>, %[[FI:.*]]: tensor<2xf32>)
// CHECK-DAG:   %[[S:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK-DAG:   %[[Z:.*]] = onnx.Constant dense<128> : tensor<ui8>
// CHECK-DAG:   %[[REPEATS:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:   %[[TQ:.*]] = "onnx.Tile"(%[[QI]], %[[REPEATS]]) : (tensor<2xui8>, tensor<1xi64>) -> tensor<4xui8>
// CHECK-DAG:   %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[TQ]], %[[S]], %[[Z]]) {{.*}} : (tensor<4xui8>, tensor<f32>, tensor<ui8>) -> tensor<4xf32>
// CHECK-DAG:   %[[QF:.*]] = "onnx.QuantizeLinear"(%[[FI]], %[[S]], %[[Z]]) {{.*}} : (tensor<2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2xui8>
// CHECK-DAG:   %[[TQI:.*]] = "onnx.Tile"(%[[QF]], %[[REPEATS]]) : (tensor<2xui8>, tensor<1xi64>) -> tensor<4xui8>
// CHECK:       return %[[DQ]], %[[TQI]]

// -----

func.func @expand_moves_dq_and_q(%arg0: tensor<1x2xui8>, %arg1: tensor<1x2xf32>) -> (tensor<3x2xf32>, tensor<3x2xui8>) {
  %s = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<128> : tensor<ui8>
  %shape = onnx.Constant dense<[3, 2]> : tensor<2xi64>
  %dq = "onnx.DequantizeLinear"(%arg0, %s, %z) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>
  %e0 = "onnx.Expand"(%dq, %shape) : (tensor<1x2xf32>, tensor<2xi64>) -> tensor<3x2xf32>
  %e1 = "onnx.Expand"(%arg1, %shape) : (tensor<1x2xf32>, tensor<2xi64>) -> tensor<3x2xf32>
  %q = "onnx.QuantizeLinear"(%e1, %s, %z) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<3x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<3x2xui8>
  return %e0, %q : tensor<3x2xf32>, tensor<3x2xui8>
}

// CHECK-LABEL: func.func @expand_moves_dq_and_q
// CHECK-SAME:  (%[[QI:.*]]: tensor<1x2xui8>, %[[FI:.*]]: tensor<1x2xf32>)
// CHECK-DAG:   %[[S:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK-DAG:   %[[Z:.*]] = onnx.Constant dense<128> : tensor<ui8>
// CHECK-DAG:   %[[SHAPE:.*]] = onnx.Constant dense<[3, 2]> : tensor<2xi64>
// CHECK-DAG:   %[[EQ:.*]] = "onnx.Expand"(%[[QI]], %[[SHAPE]]) : (tensor<1x2xui8>, tensor<2xi64>) -> tensor<3x2xui8>
// CHECK-DAG:   %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[EQ]], %[[S]], %[[Z]]) {{.*}} : (tensor<3x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<3x2xf32>
// CHECK-DAG:   %[[QF:.*]] = "onnx.QuantizeLinear"(%[[FI]], %[[S]], %[[Z]]) {{.*}} : (tensor<1x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x2xui8>
// CHECK-DAG:   %[[EQI:.*]] = "onnx.Expand"(%[[QF]], %[[SHAPE]]) : (tensor<1x2xui8>, tensor<2xi64>) -> tensor<3x2xui8>
// CHECK:       return %[[DQ]], %[[EQI]]

// -----

func.func @concat_different_param_values_not_moved(%arg0: tensor<1x2xui8>, %arg1: tensor<1x3xui8>) -> tensor<1x5xf32> {
  %s0 = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %z0 = onnx.Constant dense<128> : tensor<ui8>
  %s1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z1 = onnx.Constant dense<128> : tensor<ui8>
  %dq0 = "onnx.DequantizeLinear"(%arg0, %s0, %z0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>
  %dq1 = "onnx.DequantizeLinear"(%arg1, %s1, %z1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x3xf32>
  %concat = "onnx.Concat"(%dq0, %dq1) {axis = 1 : si64} : (tensor<1x2xf32>, tensor<1x3xf32>) -> tensor<1x5xf32>
  return %concat : tensor<1x5xf32>
}

// CHECK-LABEL: func.func @concat_different_param_values_not_moved
// CHECK:       "onnx.DequantizeLinear"
// CHECK:       "onnx.DequantizeLinear"
// CHECK:       "onnx.Concat"
