//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
// XFAIL: *
// We expect this to fail, due to the use of an optional `C` argument (implemented as a NoneType in onnx-mlir)
module attributes {}  {
  func @main_graph(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b"], output_names = ["y"]} {
//CHECK: %[[CONST:.*]] = torch.constant.int 1
//CHECK: [[ALPHA:%[^ ]*]] = torch.constant.float 5.000000e-01 
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Gemm"(%arg0, %arg1, %none) {alpha = 5.000000e-01 : f32} : (tensor<3x5xf32>, tensor<5x4xf32>, none) -> tensor<3x4xf32>
//CHECK: [[RES1:%.]] = torch.aten.mul.Scalar %arg0, [[ALPHA]] : !torch.vtensor<[3,5],f32>, !torch.float -> !torch.vtensor<[3,5],f32>
//CHECK: [[RES2:%.]] = torch.aten.mm [[RES1]], %arg1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
//CHECK: [[RES3:%.]] = torch.aten.add.Tensor [[RES2]], %arg2, %[[CONST]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>   
return %0 : tensor<3x4xf32>
  }
}