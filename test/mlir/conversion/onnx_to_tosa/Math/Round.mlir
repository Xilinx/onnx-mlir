// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --implicit-check-not=onnx.Round
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa="excluded-ops=Round" %s -split-input-file | FileCheck %s --check-prefix=KEEP
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa="excluded-ops=Add,Mul,Sub,Greater,Equal" %s -split-input-file | FileCheck %s --check-prefix=BINARY

// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa="excluded-ops=Constant,Floor,Where,Add,Mul,Sub,Greater,Equal" %s -split-input-file | FileCheck %s --check-prefix=ONNX --implicit-check-not=tosa. --implicit-check-not=onnx.Round

func.func @round(%input: tensor<8x2xf32>) -> tensor<8x2xf32> {
  %0 = "onnx.Round"(%input) : (tensor<8x2xf32>) -> tensor<8x2xf32>
  return %0 : tensor<8x2xf32>
}

// Floor and the fractional part select the nearest integer. At a tie,
// 2 * floor(0.5 * x + 0.5) selects the even neighbour for either sign.
// CHECK-LABEL: func.func @round(
// CHECK-SAME: %[[INPUT:.*]]: tensor<8x2xf32>
// CHECK-DAG:      %[[ONE:.*]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1xf32>}>
// CHECK-DAG:      %[[TWO:.*]] = "tosa.const"() <{value = dense<2.000000e+00> : tensor<1x1xf32>}>
// CHECK-DAG:      %[[HALF:.*]] = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x1xf32>}>
// CHECK:          %[[Y:.*]] = tosa.floor %[[INPUT]] : (tensor<8x2xf32>) -> tensor<8x2xf32>
// CHECK:          %[[R:.*]] = tosa.sub %[[INPUT]], %[[Y]]
// CHECK:          %[[YP1:.*]] = tosa.add %[[Y]], %[[ONE]]
// CHECK:          %[[GT:.*]] = tosa.greater %[[R]], %[[HALF]]
// CHECK:          %[[NEAREST:.*]] = tosa.select %[[GT]], %[[YP1]], %[[Y]]
// CHECK:          %[[HALFX:.*]] = tosa.mul %[[INPUT]], %[[HALF]]
// CHECK:          %[[SHIFTED:.*]] = tosa.add %[[HALFX]], %[[HALF]]
// CHECK:          %[[FLOORSHIFTED:.*]] = tosa.floor %[[SHIFTED]]
// CHECK:          %[[EVENNB:.*]] = tosa.mul %[[FLOORSHIFTED]], %[[TWO]]
// CHECK:          %[[TIE:.*]] = tosa.equal %[[R]], %[[HALF]]
// CHECK:          %[[ROUNDED:.*]] = tosa.select %[[TIE]], %[[EVENNB]], %[[NEAREST]]
// CHECK: return %[[ROUNDED]]
// KEEP-LABEL: func.func @round(
// KEEP: %[[ROUND:.*]] = "onnx.Round"
// KEEP: return %[[ROUND]]
// BINARY-LABEL: func.func @round(
// BINARY: onnx.Sub
// BINARY: onnx.Add
// BINARY: onnx.Greater
// BINARY: tosa.select
// BINARY: onnx.Mul
// BINARY: onnx.Add
// BINARY: tosa.floor
// BINARY: onnx.Mul
// BINARY: onnx.Equal
// BINARY: tosa.select
// BINARY: return

// -----

func.func @round_scalar(%input: tensor<f16>) -> tensor<f16> {
  %0 = "onnx.Round"(%input) : (tensor<f16>) -> tensor<f16>
  return %0 : tensor<f16>
}
// CHECK-LABEL: func.func @round_scalar(
// CHECK: tosa.floor {{.*}} : (tensor<f16>) -> tensor<f16>
// CHECK: %[[RESULT:.*]] = tosa.select {{.*}} : (tensor<i1>, tensor<f16>, tensor<f16>) -> tensor<f16>
// CHECK: %[[FINAL:.*]] = tosa.select {{.*}} : (tensor<i1>, tensor<f16>, tensor<f16>) -> tensor<f16>
// CHECK: return %[[FINAL]]

// ONNX-LABEL: func.func @round(
// ONNX-SAME: %[[INPUT:.*]]: tensor<8x2xf32>
// ONNX: onnx.Constant
// ONNX: onnx.Constant
// ONNX: onnx.Constant
// ONNX: %[[Y:.*]] = "onnx.Floor"(%[[INPUT]])
// ONNX: %[[R:.*]] = "onnx.Sub"(%[[INPUT]], %[[Y]])
// ONNX: %[[YP1:.*]] = "onnx.Add"(%[[Y]],
// ONNX: %[[GT:.*]] = "onnx.Greater"(%[[R]],
// ONNX: %[[NEAREST:.*]] = "onnx.Where"(%[[GT]], %[[YP1]], %[[Y]])
// ONNX: %[[HALFX:.*]] = "onnx.Mul"(%[[INPUT]],
// ONNX: %[[SHIFTED:.*]] = "onnx.Add"(%[[HALFX]],
// ONNX: %[[FLOORED:.*]] = "onnx.Floor"(%[[SHIFTED]])
// ONNX: %[[EVEN:.*]] = "onnx.Mul"(%[[FLOORED]],
// ONNX: %[[TIE:.*]] = "onnx.Equal"(%[[R]],
// ONNX: %[[ROUND:.*]] = "onnx.Where"(%[[TIE]], %[[EVEN]], %[[NEAREST]])
// ONNX: return %[[ROUND]]
// ONNX-LABEL: func.func @round_scalar(
// ONNX: onnx.Floor
// ONNX: onnx.Where
// ONNX: onnx.Floor
// ONNX: %[[ROUND:.*]] = "onnx.Where"
// ONNX: return %[[ROUND]]
