// RUN: onnx-mlir-opt --onnx-hybrid-transform="shape-inference=false canonicalization=false constant-propagation=false" %s | FileCheck %s

//Decomposing CastLike must retain that shape for the return.
func.func @castlike_static_output(%input: tensor<*xf32>, %like: tensor<1xi32>) -> tensor<1x16x32x32xi32> {
  %result = "onnx.CastLike"(%input, %like) {saturate = 1 : si64} : (tensor<*xf32>, tensor<1xi32>) -> tensor<1x16x32x32xi32>
  return %result : tensor<1x16x32x32xi32>
}
// CHECK-LABEL: func.func @castlike_static_output
// CHECK: %[[CAST:.*]] = "onnx.Cast"(%arg0) {{.*}}to = i32{{.*}} : (tensor<*xf32>) -> tensor<1x16x32x32xi32>
// CHECK-NEXT: return %[[CAST]] : tensor<1x16x32x32xi32>
