func.func @test_pad_const_pads(%arg0 : tensor<16x13xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[0, 2, 2, 4]> : tensor<4xi64>
  %1 = onnx.Constant dense<0.000000e+00> : tensor<1xf32>
  %cst = "onnx.NoValue"() {value} : () -> none
  %2 = "onnx.Pad"(%arg0, %0, %1, %cst) {mode = "constant"} : (tensor<16x13xf32>, tensor<4xi64>, tensor<1xf32>, none) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_pad_const_pads
  // CHECK-SAME:     ([[VAR_arg0:%.+]]: tensor<16x13xf32>) -> tensor<18x19xf32> {
  // CHECK: [[VAR_0:%.+]] = onnx.Constant dense<[0, 2, 2, 4]> : tensor<4xi64>
  // CHECK: [[VAR_1:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<1xf32>
  // CHECK: [[VAR_2:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK: [[VAR_3:%.+]] = "onnx.Pad"([[VAR_arg0]], [[VAR_0]], [[VAR_1]], [[VAR_2]]) {mode = "constant"} : (tensor<16x13xf32>, tensor<4xi64>, tensor<1xf32>, none) -> tensor<18x19xf32>
}