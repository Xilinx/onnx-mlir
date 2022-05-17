//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<5x5xf32>) -> tensor<5x10xf32> attributes {input_names = ["0"], output_names = ["1"]} {
//CHECK: %[[DIM:.*]] = torch.constant.int 1
//CHECK: [[RES:%.]] = torch.prim.ListConstruct %arg0, %arg0 : (!torch.vtensor<[5,5],f32>, !torch.vtensor<[5,5],f32>) -> !torch.list<vtensor<[5,5],f32>>
    %0 = "onnx.Concat"(%arg0, %arg0) {axis = 1 : si64, onnx_node_name = "Concat_0"} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x10xf32>
//CHECK: torch.aten.cat [[RES]], %[[DIM]] : !torch.list<vtensor<[5,5],f32>>, !torch.int -> !torch.vtensor<[5,10],f32>
    return %0 : tensor<5x10xf32>
  }
}
