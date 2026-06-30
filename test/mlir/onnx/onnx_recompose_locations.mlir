// RUN: onnx-mlir-opt --recompose-onnx --canonicalize %s --mlir-print-debuginfo -split-input-file | FileCheck %s

// CHECK-LABEL:  func.func @layernorm_without_bias
func.func @layernorm_without_bias(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32> loc("mReduce")
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32> loc("sub")
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32> loc("ddMul")
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32> loc("vReduce")
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32> loc("add")
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32> loc("sqrt")
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32> loc("div")
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32> loc("lnMul")
  return %Y : tensor<1x384x768xf32> loc("return")
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"(%arg0, %arg1, %0) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none) loc([[LOC_FUSED:#.+]])
// CHECK:           return [[VAR_Y_]] : tensor<1x384x768xf32>
// CHECK-DAG:       [[LOC_M_REDUCE:#.+]] = loc("mReduce")
// CHECK-DAG:       [[LOC_SUB:#.+]] = loc("sub")
// CHECK-DAG:       [[LOC_DD_MUL:#.+]] = loc("ddMul")
// CHECK-DAG:       [[LOC_V_REDUCE:#.+]] = loc("vReduce")
// CHECK-DAG:       [[LOC_ADD:#.+]] = loc("add")
// CHECK-DAG:       [[LOC_SQRT:#.+]] = loc("sqrt")
// CHECK-DAG:       [[LOC_DIV:#.+]] = loc("div")
// CHECK-DAG:       [[LOC_LN_MUL:#.+]] = loc("lnMul")
// CHECK:           [[LOC_FUSED]] = loc(fused[[[LOC_M_REDUCE]], [[LOC_SUB]], [[LOC_DD_MUL]], [[LOC_V_REDUCE]], [[LOC_ADD]], [[LOC_SQRT]], [[LOC_DIV]], [[LOC_LN_MUL]]]) 
}


// -----

func.func @test_combine_conv_split(%arg0: tensor<1x1x512x512xf32>) -> tensor<1x96x512x512xf32> {
  %0 = onnx.Constant dense<0.00999999976> : tensor<32x1x3x3xf32>
  %1 = onnx.Constant dense<0.00999999976> : tensor<32xf32>
  %2 = onnx.Constant dense<0.00999999976> : tensor<32x1x3x3xf32>
  %3 = onnx.Constant dense<0.00999999976> : tensor<32xf32>
  %4 = onnx.Constant dense<0.00999999976> : tensor<32x1x3x3xf32>
  %5 = onnx.Constant dense<0.00999999976> : tensor<32xf32> 
  %6 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32> loc("conv1")
  %7 = "onnx.Conv"(%arg0, %2, %3) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32> loc("conv2")
  %8 = "onnx.Conv"(%arg0, %4, %5) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32> loc("conv3")
  %9 = "onnx.Relu"(%6)  : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32> loc("relu")
  %10 = "onnx.Sigmoid"(%7) : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32> loc("sigmoid")
  %11 = "onnx.Tanh"(%8) : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32> loc("tanh")
  %12 = "onnx.Concat"(%9, %10, %11) {axis = 1 : si64} : (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>) -> tensor<1x96x512x512xf32> loc("concat")
  return %12 : tensor<1x96x512x512xf32>

// XFAIL-CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x512x512xf32>
// XFAIL-CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<32> : tensor<3xi64>
// XFAIL-CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
// XFAIL-CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0.00999999977> : tensor<32xf32>
// XFAIL-CHECK-NOT: separator of consecutive DAGs
// XFAIL-CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Concat"([[VAR_1_]], [[VAR_1_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<32x1x3x3xf32>, tensor<32x1x3x3xf32>, tensor<32x1x3x3xf32>) -> tensor<96x1x3x3xf32> loc([[LOC_FUSED:#.+]])
// XFAIL-CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_2_]], [[VAR_2_]], [[VAR_2_]]) {axis = 0 : si64} : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<96xf32> loc([[LOC_FUSED:#.+]])
// XFAIL-CHECK:           [[VAR_5_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_3_]], [[VAR_4_]]) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>) -> tensor<1x96x512x512xf32> loc([[LOC_FUSED:#.+]])
// XFAIL-CHECK:           [[VAR_6_:%.+]]:3 = "onnx.Split"([[VAR_5_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<1x96x512x512xf32>, tensor<3xi64>) -> (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>) loc([[LOC_FUSED:#.+]])
// XFAIL-CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Relu"([[VAR_6_]]#2) : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32> loc([[LOC_RELU:#.+]])
// XFAIL-CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Sigmoid"([[VAR_6_]]#1) : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32> loc([[LOC_SIGMOID:#.+]])
// XFAIL-CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Tanh"([[VAR_6_]]#0) : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32> loc([[LOC_TANH:#.+]])
// XFAIL-CHECK:           [[VAR_10_:%.+]] = "onnx.Concat"([[VAR_7_]], [[VAR_8_]], [[VAR_9_]]) {axis = 1 : si64} : (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>) -> tensor<1x96x512x512xf32> loc([[LOC_ORIGINAL_CONCAT:#.+]])
// XFAIL-CHECK:           return [[VAR_10_]] : tensor<1x96x512x512xf32>
// XFAIL-CHECK:         }

// XFAIL-CHECK-DAG:       [[LOC_RELU:#.+]] = loc("relu")
// XFAIL-CHECK-DAG:       [[LOC_SIGMOID:#.+]] = loc("sigmoid")
// XFAIL-CHECK-DAG:       [[LOC_TANH:#.+]] = loc("tanh")
// XFAIL-CHECK-DAG:       [[LOC_ORIGINAL_CONCAT:#.+]] = loc("concat")
// XFAIL-CHECK-DAG:       [[LOC_CONV1:#.+]] = loc("conv1")
// XFAIL-CHECK-DAG:       [[LOC_CONV2:#.+]] = loc("conv2")
// XFAIL-CHECK-DAG:       [[LOC_CONV3:#.+]] = loc("conv3")
// XFAIL-CHECK-DAG:       [[LOC_FUSED]] = loc(fused[[[LOC_CONV1]], [[LOC_CONV3]], [[LOC_CONV2]]])
}

// -----

// Canonical HardSigmoid clip(mul(x, a) + b, 0, 1): the recomposed op fuses the
// locations of the Clip, Add and Mul it replaces.
// CHECK-LABEL:  func.func @hardsigmoid_canonical_loc
func.func @hardsigmoid_canonical_loc(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %a = onnx.Constant dense<0.166666672> : tensor<f32>
  %b = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %0 = "onnx.Mul"(%arg0, %a) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32> loc("mul")
  %1 = "onnx.Add"(%0, %b) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32> loc("add")
  %2 = "onnx.Clip"(%1, %zero, %one) : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32> loc("clip")
  return %2 : tensor<8xf32> loc("ret")
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 5.000000e-01 : f32} : (tensor<8xf32>) -> tensor<8xf32> loc([[LOC_FUSED:#.+]])
// CHECK:           return
// CHECK-DAG:       [[LOC_CLIP:#.+]] = loc("clip")
// CHECK-DAG:       [[LOC_ADD:#.+]] = loc("add")
// CHECK-DAG:       [[LOC_MUL:#.+]] = loc("mul")
// CHECK:           [[LOC_FUSED]] = loc(fused[[[LOC_CLIP]], [[LOC_ADD]], [[LOC_MUL]]])
}

// -----

// Div-form HardSigmoid clip(div(x + b, c), 0, 1): the recomposed op fuses the
// locations of the Clip, Div and Add it replaces. (This is the form the
// scaled-range div(clip(x + b, 0, c), c) collapses to after DivClipConstDistributive.)
// CHECK-LABEL:  func.func @hardsigmoid_div_loc
func.func @hardsigmoid_div_loc(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %three = onnx.Constant dense<3.000000e+00> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %six = onnx.Constant dense<6.000000e+00> : tensor<f32>
  %0 = "onnx.Add"(%arg0, %three) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32> loc("add")
  %1 = "onnx.Div"(%0, %six) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32> loc("div")
  %2 = "onnx.Clip"(%1, %zero, %one) : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32> loc("clip")
  return %2 : tensor<8xf32> loc("ret")
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 5.000000e-01 : f32} : (tensor<8xf32>) -> tensor<8xf32> loc([[LOC_FUSED:#.+]])
// CHECK:           return
// CHECK-DAG:       [[LOC_CLIP:#.+]] = loc("clip")
// CHECK-DAG:       [[LOC_DIV:#.+]] = loc("div")
// CHECK-DAG:       [[LOC_ADD:#.+]] = loc("add")
// CHECK:           [[LOC_FUSED]] = loc(fused[[[LOC_CLIP]], [[LOC_DIV]], [[LOC_ADD]]])
}

// -----

// Canonical HardSwish x * clip(mul(x, a) + b, 0, 1): recomposed via HardSigmoid
// then Mul(x, HardSigmoid(x)) -> HardSwish. MLIR flattens the nested fused
// locations, so the HardSwish carries the locations of all four replaced ops.
// (The scaled-range div form is normalized into this canonical form by the
// constant-propagation pass, which is exercised in onnx_recompose_constprop.mlir.)
// CHECK-LABEL:  func.func @hardswish_canonical_loc
func.func @hardswish_canonical_loc(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %a = onnx.Constant dense<0.166666672> : tensor<f32>
  %b = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %0 = "onnx.Mul"(%arg0, %a) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32> loc("mulA")
  %1 = "onnx.Add"(%0, %b) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32> loc("add")
  %2 = "onnx.Clip"(%1, %zero, %one) : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32> loc("clip")
  %3 = "onnx.Mul"(%arg0, %2) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32> loc("mulX")
  return %3 : tensor<8xf32> loc("ret")
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSwish"(%arg0) : (tensor<8xf32>) -> tensor<8xf32> loc([[LOC_FUSED:#.+]])
// CHECK:           return
// CHECK-DAG:       [[LOC_MUL_X:#.+]] = loc("mulX")
// CHECK-DAG:       [[LOC_CLIP:#.+]] = loc("clip")
// CHECK-DAG:       [[LOC_ADD:#.+]] = loc("add")
// CHECK-DAG:       [[LOC_MUL_A:#.+]] = loc("mulA")
// CHECK:           [[LOC_FUSED]] = loc(fused[[[LOC_MUL_X]], [[LOC_CLIP]], [[LOC_ADD]], [[LOC_MUL_A]]])
}
