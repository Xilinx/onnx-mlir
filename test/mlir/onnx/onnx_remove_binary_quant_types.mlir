// Quant-types variant of onnx_remove_binary.mlir.
// Same Q/DQ input IR — run quant-types first, then remove-binary-quant-types.
// Verify exact same scale/zp as the original pass (--qdq-canonicalize="remove-binary=true").
// RUN: onnx-mlir-opt --quant-types --remove-binary-quant-types %s -split-input-file | FileCheck %s

// ============================================================================
// CASE A: lhs = DQ, rhs = Const  (fold into Q; update Q.y_zero_point)
// zp_new = floor(0 + 10/0.1_f32) = 99, scale_new = 0.1
// ============================================================================

func.func @caseA_lhsDQ_rhsConst_foldIntoQ(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %4 = onnx.Constant dense<0> : tensor<i8>
    %5 = "onnx.DequantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %6 = onnx.Constant dense<1.000000e+01> : tensor<f32>
    %7 = "onnx.Add"(%5, %6) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %8 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %9 = onnx.Constant dense<0> : tensor<i8>
    %10 = "onnx.QuantizeLinear"(%7, %8, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %10 : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @caseA_lhsDQ_rhsConst_foldIntoQ
// CHECK: %[[S:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK: %[[ZP:.*]] = onnx.Constant dense<99> : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP]])
// CHECK-SAME: : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
// CHECK: return %[[Q]] : tensor<1x4xi8>
// CHECK-NOT: onnx.Add

// -----

// ============================================================================
// CASE A-REV: rhs = DQ, lhs = Const  (fold into Q; update Q.y_zero_point)
// ============================================================================

func.func @caseA_rev_rhsDQ_lhsConst_foldIntoQ(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %4 = onnx.Constant dense<0> : tensor<i8>
    %5 = "onnx.DequantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %6 = onnx.Constant dense<1.000000e+01> : tensor<f32>
    %7 = "onnx.Add"(%6, %5) : (tensor<f32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %8 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %9 = onnx.Constant dense<0> : tensor<i8>
    %10 = "onnx.QuantizeLinear"(%7, %8, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %10 : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @caseA_rev_rhsDQ_lhsConst_foldIntoQ
// CHECK: %[[S:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK: %[[ZP:.*]] = onnx.Constant dense<99> : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP]])
// CHECK-SAME: : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
// CHECK: return %[[Q]] : tensor<1x4xi8>
// CHECK-NOT: onnx.Add

// -----

// ============================================================================
// CASE B: both inputs are DQ; constant via dq1  (fold into Q)
// k = (10 - 0) * 5.0 = 50.0, zp_new = floor(0 + 50/0.5) = 100
// ============================================================================

func.func @caseB_bothDQ_constViaDQ1_foldIntoQ(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = "onnx.DequantizeLinear"(%2, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %4 = onnx.Constant dense<10> : tensor<i8>
    %5 = onnx.Constant dense<5.000000e+00> : tensor<f32>
    %6 = onnx.Constant dense<0> : tensor<i8>
    %7 = "onnx.DequantizeLinear"(%4, %5, %6) {axis = 1 : si64, block_size = 0 : si64} : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
    %8 = "onnx.Add"(%3, %7) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %9 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %10 = onnx.Constant dense<0> : tensor<i8>
    %11 = "onnx.QuantizeLinear"(%8, %9, %10) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %11 : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @caseB_bothDQ_constViaDQ1_foldIntoQ
// CHECK: %[[S:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK: %[[ZP:.*]] = onnx.Constant dense<100> : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP]])
// CHECK-SAME: : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
// CHECK: return %[[Q]] : tensor<1x4xi8>
// CHECK-NOT: onnx.Add

// -----

// ============================================================================
// NEGATIVE TEST: Sub with weight as first operand should NOT fold
// ============================================================================

func.func @sub_weight_first_operand_no_fold(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %4 = onnx.Constant dense<0> : tensor<i8>
    %5 = "onnx.DequantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %6 = onnx.Constant dense<1.000000e+01> : tensor<f32>
    %7 = "onnx.Sub"(%6, %5) : (tensor<f32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %8 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %9 = onnx.Constant dense<0> : tensor<i8>
    %10 = "onnx.QuantizeLinear"(%7, %8, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %10 : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @sub_weight_first_operand_no_fold
// CHECK: onnx.Sub

// -----

// ============================================================================
// NEGATIVE TEST: Div with weight as first operand should NOT fold
// ============================================================================

func.func @div_weight_first_operand_no_fold(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %4 = onnx.Constant dense<0> : tensor<i8>
    %5 = "onnx.DequantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %6 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %7 = "onnx.Div"(%6, %5) : (tensor<f32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %8 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %9 = onnx.Constant dense<0> : tensor<i8>
    %10 = "onnx.QuantizeLinear"(%7, %8, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %10 : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @div_weight_first_operand_no_fold
// CHECK: onnx.Div

// -----

// ============================================================================
// NEGATIVE TEST: Zero-point overflow for i8 (Add causing zp > 127)
// zp_new = 0 + 500/2 = 250 > 127
// ============================================================================

func.func @zp_overflow_i8_add(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %1 = onnx.Constant dense<100> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %4 = onnx.Constant dense<100> : tensor<i8>
    %5 = "onnx.DequantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %6 = onnx.Constant dense<5.000000e+02> : tensor<f32>
    %7 = "onnx.Add"(%5, %6) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %8 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %9 = onnx.Constant dense<0> : tensor<i8>
    %10 = "onnx.QuantizeLinear"(%7, %8, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %10 : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @zp_overflow_i8_add
// CHECK: onnx.Add

// -----

// ============================================================================
// POSITIVE TEST: Mul (k=5, outScale=1.0, outZp=10)
// scale_new = 1.0/5.0 = 0.2, zp_new = 10
// ============================================================================

func.func @cleanup_qdq_activation_pair_folded_into_q_mul(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %4 = onnx.Constant dense<0> : tensor<i8>
    %5 = "onnx.DequantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %6 = onnx.Constant dense<5.000000e+00> : tensor<f32>
    %7 = "onnx.Mul"(%5, %6) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %8 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %9 = onnx.Constant dense<10> : tensor<i8>
    %10 = "onnx.QuantizeLinear"(%7, %8, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %10 : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @cleanup_qdq_activation_pair_folded_into_q_mul
// CHECK: %[[S:.*]] = onnx.Constant dense<2.000000e-01> : tensor<f32>
// CHECK: %[[ZP:.*]] = onnx.Constant dense<10> : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP]])
// CHECK: return %[[Q]] : tensor<1x4xi8>
// CHECK-NOT: onnx.Mul

// -----

// ============================================================================
// POSITIVE TEST: Sub (k=15, outScale=2.0, outZp=100)
// zp_new = floor(100 - 15/2.0) = floor(92.5) = 92, scale_new = 2.0
// ============================================================================

func.func @valid_sub_no_underflow(%arg0: tensor<1x4xf32>) -> tensor<1x4xui8> {
    %0 = onnx.Constant dense<3.000000e+00> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<ui8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
    %3 = onnx.Constant dense<3.000000e+00> : tensor<f32>
    %4 = onnx.Constant dense<0> : tensor<ui8>
    %5 = "onnx.DequantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
    %6 = onnx.Constant dense<1.500000e+01> : tensor<f32>
    %7 = "onnx.Sub"(%5, %6) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %8 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %9 = onnx.Constant dense<100> : tensor<ui8>
    %10 = "onnx.QuantizeLinear"(%7, %8, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
    return %10 : tensor<1x4xui8>
}

// CHECK-LABEL: func.func @valid_sub_no_underflow
// CHECK-DAG: %[[S:.*]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<92> : tensor<ui8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP]])
// CHECK: return %[[Q]] : tensor<1x4xui8>
// CHECK-NOT: onnx.Sub

// -----

// ============================================================================
// POSITIVE TEST: Div (k=4, outScale=1.0 → scale_new = 1.0*4.0 = 4.0)
// ============================================================================

func.func @qdq_chain_div(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %4 = onnx.Constant dense<0> : tensor<i8>
    %5 = "onnx.DequantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %6 = onnx.Constant dense<4.000000e+00> : tensor<f32>
    %7 = "onnx.Div"(%5, %6) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %8 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %9 = onnx.Constant dense<0> : tensor<i8>
    %10 = "onnx.QuantizeLinear"(%7, %8, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %10 : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @qdq_chain_div
// CHECK: %[[S:.*]] = onnx.Constant dense<4.000000e+00> : tensor<f32>
// CHECK: %[[ZP:.*]] = onnx.Constant dense<0> : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP]])
// CHECK: return %[[Q]] : tensor<1x4xi8>
// CHECK-NOT: onnx.Div

// -----

// ============================================================================
// CASE B Mul with ui16 (from test_fold_mul_case_b_safe)
// k = (65535 - 0) * 0.00152590231 ≈ 99.999..
// scale_new = 0.1 / k ≈ 0.001, zp_new = 10
// ============================================================================

func.func @test_fold_mul_case_b_safe(%arg0: tensor<10x1xf32>) -> tensor<10x1xf32> {
    %0 = onnx.Constant dense<0> : tensor<ui16>
    %1 = onnx.Constant dense<5.78499521E-6> : tensor<f32>
    %2 = onnx.Constant dense<0> : tensor<ui16>
    %3 = onnx.Constant dense<0.00152590231> : tensor<f32>
    %4 = onnx.Constant dense<65535> : tensor<ui16>
    %5 = onnx.Constant dense<10> : tensor<ui16>
    %6 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %7 = "onnx.DequantizeLinear"(%4, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
    %8 = "onnx.QuantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
    %9 = "onnx.DequantizeLinear"(%8, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
    %10 = "onnx.Mul"(%9, %7) : (tensor<10x1xf32>, tensor<f32>) -> tensor<10x1xf32>
    %11 = "onnx.QuantizeLinear"(%10, %6, %5) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
    %12 = "onnx.DequantizeLinear"(%11, %6, %5) {axis = 1 : si64, block_size = 0 : si64} : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
    return %12 : tensor<10x1xf32>
}

// CHECK-LABEL: func.func @test_fold_mul_case_b_safe
// CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<10> : tensor<ui16>
// CHECK-DAG: %[[DQ_S:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK-DAG: %[[Q_S:.*]] = onnx.Constant dense<9.99999931E-4> : tensor<f32>
// CHECK: "onnx.QuantizeLinear"(%arg0, %[[Q_S]], %[[ZP]])
// CHECK: "onnx.DequantizeLinear"
// CHECK-NOT: onnx.Mul

// -----

// ============================================================================
// CASE B with value-preserving link: Reshape(const_q) → DQ
// k = (25 - 0) * 4.0 = 100.0, zp_new = 100, scale_new = 1.0
// ============================================================================

func.func @caseB_constViaReshape_foldIntoQ(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = "onnx.DequantizeLinear"(%2, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %4 = onnx.Constant dense<25> : tensor<i8>
    %5 = onnx.Constant dense<> : tensor<0xi64>
    %6 = "onnx.Reshape"(%4, %5) {allowzero = 0 : si64} : (tensor<i8>, tensor<0xi64>) -> tensor<i8>
    %7 = onnx.Constant dense<4.000000e+00> : tensor<f32>
    %8 = onnx.Constant dense<0> : tensor<i8>
    %9 = "onnx.DequantizeLinear"(%6, %7, %8) {axis = 1 : si64, block_size = 0 : si64} : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
    %10 = "onnx.Add"(%3, %9) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %11 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %12 = onnx.Constant dense<0> : tensor<i8>
    %13 = "onnx.QuantizeLinear"(%10, %11, %12) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %13 : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @caseB_constViaReshape_foldIntoQ
// CHECK-DAG: %[[S:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<100> : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP]])
// CHECK: return %[[Q]] : tensor<1x4xi8>
// CHECK-NOT: onnx.Add

// -----

// ============================================================================
// POSITIVE TEST: Add with valid zero-point (zp_new = 50)
// ============================================================================

func.func @valid_add_no_overflow(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %4 = onnx.Constant dense<0> : tensor<i8>
    %5 = "onnx.DequantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %6 = onnx.Constant dense<5.000000e+01> : tensor<f32>
    %7 = "onnx.Add"(%5, %6) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %8 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %9 = onnx.Constant dense<0> : tensor<i8>
    %10 = "onnx.QuantizeLinear"(%7, %8, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %10 : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @valid_add_no_overflow
// CHECK-DAG: %[[S:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<50> : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP]])
// CHECK: return %[[Q]] : tensor<1x4xi8>
// CHECK-NOT: onnx.Add

// -----

// ============================================================================
// Q-DQ CHAIN: Add with constant through Q→DQ chain (k=10, zp_new=10)
// ============================================================================

func.func @qdq_chain_add(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %4 = onnx.Constant dense<0> : tensor<i8>
    %5 = "onnx.DequantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %6 = onnx.Constant dense<1.000000e+01> : tensor<f32>
    %7 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %8 = onnx.Constant dense<0> : tensor<i8>
    %9 = "onnx.QuantizeLinear"(%6, %7, %8) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<f32>, tensor<f32>, tensor<i8>) -> tensor<i8>
    %10 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %11 = onnx.Constant dense<0> : tensor<i8>
    %12 = "onnx.DequantizeLinear"(%9, %10, %11) {axis = 1 : si64, block_size = 0 : si64} : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
    %13 = "onnx.Add"(%5, %12) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %14 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %15 = onnx.Constant dense<0> : tensor<i8>
    %16 = "onnx.QuantizeLinear"(%13, %14, %15) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %16 : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @qdq_chain_add
// CHECK-DAG: %[[S:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<10> : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP]])
// CHECK: return %[[Q]] : tensor<1x4xi8>
// CHECK-NOT: onnx.Add

// -----

// ============================================================================
// Q-DQ CHAIN: Mul with constant through Q→DQ chain (k=5, scale_new=0.2)
// ============================================================================

func.func @qdq_chain_mul(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %4 = onnx.Constant dense<0> : tensor<i8>
    %5 = "onnx.DequantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %6 = onnx.Constant dense<5.000000e+00> : tensor<f32>
    %7 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %8 = onnx.Constant dense<0> : tensor<i8>
    %9 = "onnx.QuantizeLinear"(%6, %7, %8) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<f32>, tensor<f32>, tensor<i8>) -> tensor<i8>
    %10 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %11 = onnx.Constant dense<0> : tensor<i8>
    %12 = "onnx.DequantizeLinear"(%9, %10, %11) {axis = 1 : si64, block_size = 0 : si64} : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
    %13 = "onnx.Mul"(%5, %12) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %14 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %15 = onnx.Constant dense<10> : tensor<i8>
    %16 = "onnx.QuantizeLinear"(%13, %14, %15) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %16 : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @qdq_chain_mul
// CHECK-DAG: %[[S:.*]] = onnx.Constant dense<2.000000e-01> : tensor<f32>
// CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<10> : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP]])
// CHECK: return %[[Q]] : tensor<1x4xi8>
// CHECK-NOT: onnx.Mul

// -----

// ============================================================================
// INTERIOR CASE: No boundary Q — pure scast → Mul → scast
// scale_new = 0.1/2.0 = 0.05, zp_new = 10
// Identity keeps the quant type visible for verification.
// ============================================================================

func.func @interior_mul(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %k = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %mul = "onnx.Mul"(%act, %k) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:10>>
  %id = "onnx.Identity"(%mul) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:10>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:10>>
  %out = "quant.scast"(%id) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:10>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @interior_mul
// CHECK-NOT: onnx.Mul
// CHECK: %[[V:.*]] = quant.scast %arg0 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-02:10>>
// CHECK: "onnx.Identity"(%[[V]]) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-02:10>>)

// -----

// ============================================================================
// INTERIOR CASE: No boundary Q — pure scast → Add → scast
// zp_new = 0 + 5/0.1 = 50, scale_new = 0.1
// ============================================================================

func.func @interior_add(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %k = onnx.Constant dense<5.000000e+00> : tensor<f32>
  %add = "onnx.Add"(%act, %k) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>
  %id = "onnx.Identity"(%add) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>
  %out = "quant.scast"(%id) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @interior_add
// CHECK-NOT: onnx.Add
// CHECK: %[[V:.*]] = quant.scast %arg0 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 1.000000e-01:50>>
// CHECK: "onnx.Identity"(%[[V]]) : (tensor<1x4x!quant.uniform<i8:f32, 1.000000e-01:50>>)
