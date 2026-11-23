// RUN: onnx-mlir --EmitONNXIR --printIR %s | FileCheck %s --check-prefix=DEFAULT --implicit-check-not=onnx.Reshape
// RUN: onnx-mlir --EmitONNXIR --printIR --enable-xmc-passes %s | FileCheck %s --check-prefix=DEFAULT --implicit-check-not=onnx.Reshape
// RUN: onnx-mlir --EmitONNXIR --printIR --enable-elementwise-operand-rank-canonicalization %s | FileCheck %s --check-prefix=ENABLED
// RUN: onnx-mlir --EmitONNXIR --printIR --enable-xmc-passes --enable-elementwise-operand-rank-canonicalization %s | FileCheck %s --check-prefix=ENABLED

// DEFAULT-LABEL: func.func @main
// DEFAULT: "onnx.Add"(%arg0, %arg1) {{.*}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
// ENABLED-LABEL: func.func @main
// ENABLED: [[RESHAPED:%.*]] = "onnx.Reshape"(%arg1, {{%.*}}) {{.*}} -> tensor<1x3xf32>
// ENABLED: "onnx.Add"(%arg0, [[RESHAPED]]) {{.*}} : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
func.func @main(%x: tensor<2x3xf32>, %y: tensor<3xf32>) -> tensor<2x3xf32> {
  %add = "onnx.Add"(%x, %y) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return %add : tensor<2x3xf32>
}
