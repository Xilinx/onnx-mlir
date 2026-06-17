// RUN: onnx-mlir-opt --shape-inference --canonicalize="test-convergence=true" --shape-inference --cse %s -split-input-file | FileCheck %s

// -----
// CHECK-LABEL: func @flatten_static_to_reshape
// Static Flatten (axis=1) becomes onnx.Reshape with the same 2-D shape.
func.func @flatten_static_to_reshape(%arg0: tensor<2x3x4xf32>) -> tensor<6x4xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<6x4xf32>
  onnx.Return %0 : tensor<6x4xf32>
  // CHECK: [[CST:%.+]] = onnx.Constant dense<[6, 4]> : tensor<2xi64>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"(%arg0, [[CST]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<6x4xf32>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: "onnx.Flatten"
}

// -----
// CHECK-LABEL: func @flatten_axis0_static_to_reshape
// axis=0 => first dim is 1, second is the full element count.
func.func @flatten_axis0_static_to_reshape(%arg0: tensor<2x3x4xf32>) -> tensor<1x24xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<1x24xf32>
  onnx.Return %0 : tensor<1x24xf32>
  // CHECK: [[CST:%.+]] = onnx.Constant dense<[1, 24]> : tensor<2xi64>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"(%arg0, [[CST]]) {allowzero = 0 : si64}
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: "onnx.Flatten"
}

// -----
// CHECK-LABEL: func @flatten_dynamic_no_reshape
// Dynamic input: gate prevents the rewrite.
func.func @flatten_dynamic_no_reshape(%arg0: tensor<?x3x4xf32>) -> tensor<?x4xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = 1 : si64} : (tensor<?x3x4xf32>) -> tensor<?x4xf32>
  onnx.Return %0 : tensor<?x4xf32>
  // CHECK: "onnx.Flatten"
  // CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func @squeeze_static_to_reshape
// Static Squeeze on a single axis becomes onnx.Reshape.
func.func @squeeze_static_to_reshape(%arg0: tensor<2x1x4xf32>) -> tensor<2x4xf32> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.Squeeze"(%arg0, %axes) : (tensor<2x1x4xf32>, tensor<1xi64>) -> tensor<2x4xf32>
  onnx.Return %0 : tensor<2x4xf32>
  // CHECK: [[CST:%.+]] = onnx.Constant dense<[2, 4]> : tensor<2xi64>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"(%arg0, [[CST]]) {allowzero = 0 : si64}
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: "onnx.Squeeze"
}

// -----
// CHECK-LABEL: func @squeeze_dynamic_data_no_reshape
// Dynamic primary operand: rewrite must not fire.
func.func @squeeze_dynamic_data_no_reshape(%arg0: tensor<?x1x4xf32>) -> tensor<?x4xf32> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.Squeeze"(%arg0, %axes) : (tensor<?x1x4xf32>, tensor<1xi64>) -> tensor<?x4xf32>
  onnx.Return %0 : tensor<?x4xf32>
  // CHECK: "onnx.Squeeze"
  // CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func @unsqueeze_static_to_reshape
// Static Unsqueeze becomes onnx.Reshape.
func.func @unsqueeze_static_to_reshape(%arg0: tensor<2x4xf32>) -> tensor<2x1x4xf32> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.Unsqueeze"(%arg0, %axes) : (tensor<2x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
  onnx.Return %0 : tensor<2x1x4xf32>
  // CHECK: [[CST:%.+]] = onnx.Constant dense<[2, 1, 4]> : tensor<3xi64>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"(%arg0, [[CST]]) {allowzero = 0 : si64}
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: "onnx.Unsqueeze"
}

// -----
// CHECK-LABEL: func @unsqueeze_dynamic_data_no_reshape
// Dynamic primary operand: rewrite must not fire.
func.func @unsqueeze_dynamic_data_no_reshape(%arg0: tensor<?x4xf32>) -> tensor<?x1x4xf32> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.Unsqueeze"(%arg0, %axes) : (tensor<?x4xf32>, tensor<1xi64>) -> tensor<?x1x4xf32>
  onnx.Return %0 : tensor<?x1x4xf32>
  // CHECK: "onnx.Unsqueeze"
  // CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func @squeeze_unsqueeze_different_axes_to_single_reshape
// Squeeze + Unsqueeze with non-matching axes both become Reshape and then fuse.
func.func @squeeze_unsqueeze_different_axes_to_single_reshape(%arg0: tensor<10x1x10xf32>) -> tensor<10x10x1xf32> {
  %s_axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %u_axes = onnx.Constant dense<[2]> : tensor<1xi64>
  %2 = "onnx.Squeeze"(%arg0, %s_axes) : (tensor<10x1x10xf32>, tensor<1xi64>) -> tensor<10x10xf32>
  %3 = "onnx.Unsqueeze"(%2, %u_axes) : (tensor<10x10xf32>, tensor<1xi64>) -> tensor<10x10x1xf32>
  onnx.Return %3 : tensor<10x10x1xf32>
  // CHECK: [[CST:%.+]] = onnx.Constant dense<[10, 10, 1]> : tensor<3xi64>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"(%arg0, [[CST]]) {allowzero = 0 : si64}
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: "onnx.Squeeze"
  // CHECK-NOT: "onnx.Unsqueeze"
}

// -----
// CHECK-LABEL: func @unsqueeze_cast_squeeze_to_cast
// Unsqueeze + Cast + Squeeze with the same net shape restores to plain Cast
// via the FuseCastBetweenReshapes pattern.
func.func @unsqueeze_cast_squeeze_to_cast(%arg0: tensor<10x10xf32>) -> tensor<10x10xi64> {
  %u_axes = onnx.Constant dense<[0, 2]> : tensor<2xi64>
  %s_axes = onnx.Constant dense<[0, -2]> : tensor<2xi64>
  %2 = "onnx.Unsqueeze"(%arg0, %u_axes) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<1x10x1x10xf32>
  %3 = "onnx.Cast"(%2) {to = i64} : (tensor<1x10x1x10xf32>) -> tensor<1x10x1x10xi64>
  %4 = "onnx.Squeeze"(%3, %s_axes) : (tensor<1x10x1x10xi64>, tensor<2xi64>) -> tensor<10x10xi64>
  onnx.Return %4 : tensor<10x10xi64>
  // CHECK: [[RES:%.+]] = "onnx.Cast"(%arg0) {{.*}} : (tensor<10x10xf32>) -> tensor<10x10xi64>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: "onnx.Unsqueeze"
  // CHECK-NOT: "onnx.Squeeze"
}

// -----
// CHECK-LABEL: func @v11_ops_not_touched
func.func @v11_ops_not_touched(%arg0: tensor<10x10xf32>) -> tensor<10x1x10xf32> {
  %0 = "onnx.UnsqueezeV11"(%arg0) {axes=[0, 2]} : (tensor<10x10xf32>) -> tensor<1x10x1x10xf32>
  %1 = "onnx.SqueezeV11"(%0) {axes=[0]} : (tensor<1x10x1x10xf32>) -> tensor<10x1x10xf32>
  onnx.Return %1 : tensor<10x1x10xf32>
  // CHECK: "onnx.UnsqueezeV11"
  // CHECK: "onnx.SqueezeV11"
  // CHECK-NOT: "onnx.Reshape"
}
