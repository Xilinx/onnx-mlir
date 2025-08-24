// RUN: onnx-mlir-opt --dq-binary-q-opt-onnx-to-onnx %s --split-input-file | FileCheck %s

// 1) dq1-dq2(const input)-add-q-dq. remove->add,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern1a
// CHECK-NOT: onnx.Add
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return
// CHECK-NOT: onnx.DequantizeLinear
func.func @test_removebinary_pattern1a(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%36 = onnx.Constant dense<0> : tensor<ui16>
%37 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%38 = onnx.Constant dense<65535> : tensor<ui16>
%14 = onnx.Constant dense<39664> : tensor<ui16>
%15 = onnx.Constant dense<2.57987776E-5> : tensor<f32>

%960 = "onnx.DequantizeLinear"(%38, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>

%1174 = "onnx.DequantizeLinear"(%arg0, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

%1176 = "onnx.Add"(%960, %1174) : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>

%1178 = "onnx.QuantizeLinear"(%1176, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64,
output_dtype = 0 : si64,
saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>

%1180 = "onnx.DequantizeLinear"(%1178, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %1180 : tensor<1x1x1x128xf32>
}

// -----
// Test fusing a DQ -> Sub -> Q -> DQ pattern.
// The constant input to the Sub is produced by a chain: Constant -> Identity -> DequantizeLinear.
// The pass should look through the Identity op, perform the fusion, and remove the redundant Q->DQ chain.
// 2) dq1-dq2(const input)-Sub-q-dq. remove->Sub,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern1b
// CHECK-NOT: onnx.Add
// CHECK-NOT: onnx.QuantizeLinear

func.func @test_removebinary_pattern1b(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%cst_zp_zero = onnx.Constant dense<0> : tensor<ui16>
%cst_scale_small = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%cst_zp_large = onnx.Constant dense<65535> : tensor<ui16>
%cst_scale_large = onnx.Constant dense<0.152590215> : tensor<f32>

%const_input_for_chain = onnx.Constant dense<0> : tensor<ui16>

%intermediate_op = "onnx.Identity"(%const_input_for_chain) : (tensor<ui16>) -> tensor<ui16>
%const_dq = "onnx.DequantizeLinear"(%intermediate_op, %cst_scale_large, %cst_zp_large) : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>

%dq_in = "onnx.DequantizeLinear"(%arg0, %cst_scale_small, %cst_zp_zero) : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

%Sub_op = "onnx.Add"(%dq_in, %const_dq) : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<1x1x1x128xf32>

%q_out = "onnx.QuantizeLinear"(%Sub_op, %cst_scale_large, %cst_zp_large) : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%final_dq_out = "onnx.DequantizeLinear"(%q_out, %cst_scale_large, %cst_zp_large) : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

return %final_dq_out : tensor<1x1x1x128xf32>
}
//-----
// 3) dq1-const-add-q-dq. remove->add, q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern2a
// CHECK-NOT: onnx.Add
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return
// CHECK-NOT: onnx.DequantizeLinear

func.func @test_removebinary_pattern2a(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
// Original constants used in the graph.
%cst_zp_zero = onnx.Constant dense<0> : tensor<ui16>
%cst_scale_small = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%cst_zp_large = onnx.Constant dense<65535> : tensor<ui16>
%cst_scale_large = onnx.Constant dense<0.152590215> : tensor<f32>

// This DequantizeLinear was folded into a new constant:
// (0 - 65535) * 0.15259 = -10000.0
%const_neg_10k = onnx.Constant dense<-10000.0> : tensor<f32>
%dq1 = "onnx.DequantizeLinear"(%arg0, %cst_scale_small, %cst_zp_zero) : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

// The Sub op now uses the pre-calculated constant.
%Sub_op = "onnx.Add"(%dq1, %const_neg_10k) : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<1x1x1x128xf32>

%q2 = "onnx.QuantizeLinear"(%Sub_op, %cst_scale_large, %cst_zp_large) : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%dq2 = "onnx.DequantizeLinear"(%q2, %cst_scale_large, %cst_zp_large) : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

return %dq2 : tensor<1x1x1x128xf32>
}

//-----
// 4) const-dq1-add-q-dq. remove->add,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern2b
// CHECK-NOT: onnx.Add
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return
// CHECK-NOT: onnx.DequantizeLinear

func.func @test_removebinary_pattern2b(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
// Original constants used in the graph.
%cst_zp_zero = onnx.Constant dense<0> : tensor<ui16>
%cst_scale_small = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%cst_zp_large = onnx.Constant dense<65535> : tensor<ui16>
%cst_scale_large = onnx.Constant dense<0.152590215> : tensor<f32>

%const_neg_10k = onnx.Constant dense<-10000.0> : tensor<f32>
%dq1 = "onnx.DequantizeLinear"(%arg0, %cst_scale_small, %cst_zp_zero) : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

// The Sub op now uses the pre-calculated constant.
%Sub_op = "onnx.Add"(%const_neg_10k,%dq1) : (tensor<f32>,tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>

%q2 = "onnx.QuantizeLinear"(%Sub_op, %cst_scale_large, %cst_zp_large) : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%dq2 = "onnx.DequantizeLinear"(%q2, %cst_scale_large, %cst_zp_large) : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

return %dq2 : tensor<1x1x1x128xf32>
}

//-----
// 5) const-dq1-add-q-dq. kval=0. remove->add,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern3a
// CHECK-NOT: onnx.Add
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return
// CHECK-NOT: onnx.DequantizeLinear

func.func @test_removebinary_pattern3a(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
// Original constants used in the graph.
%cst_zp_zero = onnx.Constant dense<0> : tensor<ui16>
%cst_scale_small = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%cst_zp_large = onnx.Constant dense<65535> : tensor<ui16>
%cst_scale_large = onnx.Constant dense<0.152590215> : tensor<f32>

%const_neg_10k = onnx.Constant dense<0.0> : tensor<f32>
%dq1 = "onnx.DequantizeLinear"(%arg0, %cst_scale_small, %cst_zp_zero) : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

// The Sub op now uses the pre-calculated constant.
%Sub_op = "onnx.Add"(%const_neg_10k,%dq1) : (tensor<f32>,tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>

%q2 = "onnx.QuantizeLinear"(%Sub_op, %cst_scale_large, %cst_zp_large) : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%dq2 = "onnx.DequantizeLinear"(%q2, %cst_scale_large, %cst_zp_large) : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

return %dq2 : tensor<1x1x1x128xf32>
}
//-----
// 6) const-dq1-Sub-q-dq. dst_scale=0. remove->Sub,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern3b
// CHECK: onnx.Add
// CHECK: onnx.QuantizeLinear

func.func @test_removebinary_pattern3b(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
// Original constants used in the graph.
%cst_zp_zero = onnx.Constant dense<0> : tensor<ui16>
%cst_scale_small = onnx.Constant dense<0.0> : tensor<f32>
%cst_zp_large = onnx.Constant dense<65535> : tensor<ui16>
%cst_scale_large = onnx.Constant dense<0.152590215> : tensor<f32>

%const_neg_10k = onnx.Constant dense<-10000.0> : tensor<f32>
%dq1 = "onnx.DequantizeLinear"(%arg0, %cst_scale_small, %cst_zp_zero) : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

// The Sub op now uses the pre-calculated constant.
%Sub_op = "onnx.Add"(%const_neg_10k,%dq1) : (tensor<f32>,tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>

%q2 = "onnx.QuantizeLinear"(%Sub_op, %cst_scale_large, %cst_zp_large) : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%dq2 = "onnx.DequantizeLinear"(%q2, %cst_scale_large, %cst_zp_large) : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

return %dq2 : tensor<1x1x1x128xf32>
}
//-----
// 7) dq1-dq2(const input)-add-q-dq. remove->add,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern4
// CHECK-NOT: onnx.Add
// CHECK: onnx.QuantizeLinear

func.func @test_removebinary_pattern4(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%36 = onnx.Constant dense<0> : tensor<ui16>
%37 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%38 = onnx.Constant dense<65535> : tensor<ui16>
%14 = onnx.Constant dense<39664> : tensor<ui16>
%15 = onnx.Constant dense<2.57987776E-5> : tensor<f32>

%960 = "onnx.DequantizeLinear"(%38, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>

%1174 = "onnx.DequantizeLinear"(%arg0, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

%1176 = "onnx.Add"(%960, %1174) : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>

%1178 = "onnx.QuantizeLinear"(%1176, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64,
output_dtype = 0 : si64,
saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>

%1180 = "onnx.DequantizeLinear"(%1178, %15, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %1180 : tensor<1x1x1x128xf32>
}

//-----
// 8) const-dq1-Sub-tanh. remove->none
// CHECK-LABEL: func.func @test_removebinary_pattern5
// CHECK: onnx.Add
// CHECK: onnx.Tanh

func.func @test_removebinary_pattern5(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%36 = onnx.Constant dense<0> : tensor<ui16>
%37 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%38 = onnx.Constant dense<65535> : tensor<ui16>
%14 = onnx.Constant dense<39664> : tensor<ui16>
%15 = onnx.Constant dense<2.57987776E-5> : tensor<f32>

%960 = "onnx.DequantizeLinear"(%38, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>

%1174 = "onnx.DequantizeLinear"(%arg0, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

%1176 = "onnx.Add"(%960, %1174) : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
%tanh = "onnx.Tanh"(%1176) : (tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
return %tanh : tensor<1x1x1x128xf32>
}

//-----
// 9) dq1-dq2-add-q-dq1-dq2-mul-Q-DQ. multi-use of scale and zp of dq-act before binary op. remove->mul, add
// CHECK-LABEL: func.func @test_removebinary_pattern6
// CHECK-NOT: onnx.Add
// CHECK-NOT: onnx.Add

func.func @test_removebinary_pattern6(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%36 = onnx.Constant dense<0> : tensor<ui16>
%37 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%38 = onnx.Constant dense<65535> : tensor<ui16>
%39 = onnx.Constant dense<0.152590215> : tensor<f32>
%14 = onnx.Constant dense<39664> : tensor<ui16>
%15 = onnx.Constant dense<2.57987776E-5> : tensor<f32>

%960 = "onnx.DequantizeLinear"(%38, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>

%1174 = "onnx.DequantizeLinear"(%arg0, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

%1176 = "onnx.Add"(%960, %1174) {onnx_node_name = "/bert/Sub"} : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>

%1178 = "onnx.QuantizeLinear"(%1176, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64,
output_dtype = 0 : si64,
saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>

%1180 = "onnx.DequantizeLinear"(%1178, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

%961 = "onnx.DequantizeLinear"(%36, %39, %38) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>

%1182 = "onnx.Add"(%1180, %961) : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<1x1x1x128xf32>

%1184 = "onnx.QuantizeLinear"(%1182, %39, %38) {
axis = 1 : si64,
block_size = 0 : si64,
output_dtype = 0 : si64,
saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>

%1186 = "onnx.DequantizeLinear"(%1184, %39, %38) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

return %1186 : tensor<1x1x1x128xf32>
}
//-----
//
// 10) dq1-dq2(const input, per-axis length-2 on axis=0)-mul-q-dq.
// vectors wiht same values -> fusion
// CHECK-LABEL: func.func @test_removebinary_pattern7
// CHECK-NOT: onnx.Add
// CHECK-NOT: onnx.QuantizeLinear

func.func @test_removebinary_pattern7(%arg0: tensor<2x1x1x128xui16>) -> tensor<2x1x1x128xf32> {
// Per-axis (axis=0) constants (length-2 params stay the same)
%zp = onnx.Constant dense<[0, 0]> : tensor<2xui16>
%scale = onnx.Constant dense<[1.52590219E-5, 1.52590219E-5]> : tensor<2xf32>
%zpMax = onnx.Constant dense<[65535, 65535]> : tensor<2xui16>
%k = onnx.Constant dense<[0.152590215, 0.152590215]> : tensor<2xf32>

// Constant-side x: shape 2x1x1x1 (use a splat to avoid bracket hell)
%cx = onnx.Constant dense<0> : tensor<2x1x1x1xui16>

%c_dq = "onnx.DequantizeLinear"(%cx, %k, %zpMax) {
axis = 0 : si64, block_size = 0 : si64
} : (tensor<2x1x1x1xui16>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x1xf32>

// Activation-side DQ (per-axis, axis=0) → 2x1x1x128xf32
%x_dq = "onnx.DequantizeLinear"(%arg0, %scale, %zp) {
axis = 0 : si64, block_size = 0 : si64
} : (tensor<2x1x1x128xui16>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x128xf32>

// Sub now broadcasts (2x1x1x1) over (2x1x1x128) → (2x1x1x128)
%mul = "onnx.Add"(%x_dq, %c_dq)
: (tensor<2x1x1x128xf32>, tensor<2x1x1x1xf32>) -> tensor<2x1x1x128xf32>

// Re-quantize per-axis with the same (length-2) params on axis=0.
%q = "onnx.QuantizeLinear"(%mul, %k, %zpMax) {
axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64
} : (tensor<2x1x1x128xf32>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x128xui16>

// Final DQ (per-axis, axis=0).
%y = "onnx.DequantizeLinear"(%q, %k, %zpMax) {
axis = 0 : si64, block_size = 0 : si64
} : (tensor<2x1x1x128xui16>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x128xf32>

return %y : tensor<2x1x1x128xf32>
}
//-----
//
// 11) dq1-dq2(const input, per-axis length-2 on axis=0)-mul-q-dq.
// vectors wiht different values -> no fusion
// CHECK-LABEL: func.func @test_removebinary_pattern8
// CHECK: onnx.Add
// CHECK: onnx.QuantizeLinear

func.func @test_removebinary_pattern8(%arg0: tensor<2x1x1x128xui16>) -> tensor<2x1x1x128xf32> {
// Per-axis (axis=0) constants (length-2 params stay the same)
%zp = onnx.Constant dense<[0, 0]> : tensor<2xui16>
%scale = onnx.Constant dense<[1.52590219E-5, 1.52590219E-5]> : tensor<2xf32>
%zpMax = onnx.Constant dense<[65535, 1]> : tensor<2xui16>
%k = onnx.Constant dense<[0.152590215, 0.152590215]> : tensor<2xf32>

// Constant-side x: shape 2x1x1x1 (use a splat to avoid bracket hell)
%cx = onnx.Constant dense<0> : tensor<2x1x1x1xui16>

%c_dq = "onnx.DequantizeLinear"(%cx, %k, %zpMax) {
axis = 0 : si64, block_size = 0 : si64
} : (tensor<2x1x1x1xui16>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x1xf32>

// Activation-side DQ (per-axis, axis=0) → 2x1x1x128xf32
%x_dq = "onnx.DequantizeLinear"(%arg0, %scale, %zp) {
axis = 0 : si64, block_size = 0 : si64
} : (tensor<2x1x1x128xui16>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x128xf32>

// Sub now broadcasts (2x1x1x1) over (2x1x1x128) → (2x1x1x128)
%mul = "onnx.Add"(%x_dq, %c_dq)
: (tensor<2x1x1x128xf32>, tensor<2x1x1x1xf32>) -> tensor<2x1x1x128xf32>

// Re-quantize per-axis with the same (length-2) params on axis=0.
%q = "onnx.QuantizeLinear"(%mul, %k, %zpMax) {
axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64
} : (tensor<2x1x1x128xf32>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x128xui16>

// Final DQ (per-axis, axis=0).
%y = "onnx.DequantizeLinear"(%q, %k, %zpMax) {
axis = 0 : si64, block_size = 0 : si64
} : (tensor<2x1x1x128xui16>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x128xf32>

return %y : tensor<2x1x1x128xf32>
}
