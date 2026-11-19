// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Cast,Add,Sub,Mul,Equal,Greater,GreaterOrEqual" -cse %s | FileCheck %s --implicit-check-not=tosa.cast --implicit-check-not=tosa.mul --implicit-check-not=tosa.add --implicit-check-not=tosa.sub --implicit-check-not=tosa.equal --implicit-check-not=tosa.greater --implicit-check-not=onnx.QuantizeLinear --implicit-check-not='dense<0> : tensor<1xi8>'
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Cast,Add,Sub,Mul,Equal,Greater,GreaterOrEqual" --convert-onnx-to-tosa -cse %s | FileCheck %s --check-prefix=LOWERED --implicit-check-not=onnx.

func.func @quantize(%input: tensor<2x3xf32>, %scale: tensor<f32>, %zero: tensor<i8>) -> tensor<2x3xi8> {
  %r = "onnx.QuantizeLinear"(%input, %scale, %zero) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<f32>, tensor<i8>) -> tensor<2x3xi8>
  return %r : tensor<2x3xi8>
}

// CHECK-LABEL: func.func @quantize(
// CHECK-SAME: %[[INPUT:.*]]: tensor<2x3xf32>
// CHECK: %[[REC:.*]] = tosa.reciprocal
// CHECK: %[[SCALED:.*]] = "onnx.Mul"(%[[INPUT]], %[[REC]])
// CHECK: %[[FLOOR:.*]] = tosa.floor %[[SCALED]]
// CHECK: %[[REMAINDER:.*]] = "onnx.Sub"(%[[SCALED]], %[[FLOOR]])
// CHECK-DAG: %[[UP:.*]] = "onnx.Add"(%[[FLOOR]],
// CHECK-DAG: %[[ABOVE_HALF:.*]] = "onnx.Greater"(%[[REMAINDER]],
// CHECK: %[[NEAREST:.*]] = tosa.select %[[ABOVE_HALF]], %[[UP]], %[[FLOOR]]
// CHECK: %[[HALF_INPUT:.*]] = "onnx.Mul"(%[[SCALED]],
// CHECK: %[[SHIFTED:.*]] = "onnx.Add"(%[[HALF_INPUT]],
// CHECK: %[[HALF_FLOOR:.*]] = tosa.floor %[[SHIFTED]]
// CHECK: %[[EVEN:.*]] = "onnx.Mul"(%[[HALF_FLOOR]],
// CHECK: %[[TIE:.*]] = "onnx.Equal"(%[[REMAINDER]],
// CHECK: %[[ROUNDED:.*]] = tosa.select %[[TIE]], %[[EVEN]], %[[NEAREST]]
// CHECK: %[[ROUNDED_INT:.*]] = "onnx.Cast"(%[[ROUNDED]]) {{.*}} -> tensor<2x3xi32>
// CHECK: %[[ZERO_INT:.*]] = "onnx.Cast"{{.*}} -> tensor<1x1xi32>
// CHECK: %[[OFFSET:.*]] = "onnx.Add"(%[[ROUNDED_INT]], %[[ZERO_INT]])
// CHECK: %[[SATURATED:.*]] = tosa.clamp %[[OFFSET]]
// CHECK: %[[RESULT:.*]] = "onnx.Cast"(%[[SATURATED]]) {{.*}} -> tensor<2x3xi8>
// CHECK: return %[[RESULT]]

// LOWERED-LABEL: func.func @quantize(
// LOWERED: tosa.mul
// LOWERED: tosa.equal
// LOWERED: tosa.clamp
// LOWERED: %[[RESULT:.*]] = tosa.cast {{.*}} -> tensor<2x3xi8>
// LOWERED: return %[[RESULT]]
