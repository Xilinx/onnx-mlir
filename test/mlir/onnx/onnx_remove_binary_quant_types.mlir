// RUN: onnx-mlir-opt --remove-binary-quant-types %s -split-input-file | FileCheck %s

// ============================================================================
// Case A: quant activation via scast (lhs) + float constant (rhs) — Mul
// Fold into output quant type: scale_new = scale / k = 0.1 / 2.0 = 0.05
// ============================================================================

func.func @caseA_mul(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %k = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %mul = "onnx.Mul"(%act, %k) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:10>>
  %out = "quant.scast"(%mul) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:10>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @caseA_mul
// CHECK-NOT: onnx.Mul
// CHECK: return %arg0 : tensor<1x4xi8>

// -----

// ============================================================================
// Case A: Add — zp_new = zp + k/scale = 0 + 5.0/0.1 = 50
// ============================================================================

func.func @caseA_add(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %k = onnx.Constant dense<5.000000e+00> : tensor<f32>
  %add = "onnx.Add"(%act, %k) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>
  %out = "quant.scast"(%add) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @caseA_add
// CHECK-NOT: onnx.Add
// CHECK: return %arg0 : tensor<1x4xi8>

// -----

// ============================================================================
// Case A: Sub — zp_new = zp - k/scale = 0 - 5.0/0.1 = -50
// ============================================================================

func.func @caseA_sub(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %k = onnx.Constant dense<5.000000e+00> : tensor<f32>
  %sub = "onnx.Sub"(%act, %k) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>
  %out = "quant.scast"(%sub) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @caseA_sub
// CHECK-NOT: onnx.Sub
// CHECK: return %arg0 : tensor<1x4xi8>

// -----

// ============================================================================
// Case A: Div — scale_new = scale * k = 0.1 * 2.0 = 0.2
// ============================================================================

func.func @caseA_div(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %k = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %div = "onnx.Div"(%act, %k) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:5>>
  %out = "quant.scast"(%div) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:5>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @caseA_div
// CHECK-NOT: onnx.Div
// CHECK: return %arg0 : tensor<1x4xi8>

// -----

// ============================================================================
// Case A-reversed: float constant (lhs) + quant activation (rhs) — Add
// ============================================================================

func.func @caseA_rev_add(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %k = onnx.Constant dense<1.000000e+01> : tensor<f32>
  %add = "onnx.Add"(%k, %act) : (tensor<f32>, tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.2:0>>
  %out = "quant.scast"(%add) : (tensor<1x4x!quant.uniform<i8:f32, 0.2:0>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @caseA_rev_add
// CHECK-NOT: onnx.Add
// CHECK: return %arg0 : tensor<1x4xi8>

// -----

// ============================================================================
// Case A-reversed: Sub with constant as first operand — should NOT fold
// ============================================================================

func.func @caseA_rev_sub_bailout(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %k = onnx.Constant dense<5.000000e+00> : tensor<f32>
  %sub = "onnx.Sub"(%k, %act) : (tensor<f32>, tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>
  %out = "quant.scast"(%sub) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @caseA_rev_sub_bailout
// CHECK: onnx.Sub

// -----

// ============================================================================
// Case A-reversed: Div with constant as first operand — should NOT fold
// ============================================================================

func.func @caseA_rev_div_bailout(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %k = onnx.Constant dense<5.000000e+00> : tensor<f32>
  %div = "onnx.Div"(%k, %act) : (tensor<f32>, tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>
  %out = "quant.scast"(%div) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @caseA_rev_div_bailout
// CHECK: onnx.Div

// -----

// ============================================================================
// Case B: both quant-typed, rhs traces to int constant via quant.scast
// k = (10 - 0) * 5.0 = 50.0
// Fold Add into output: zp_new = 0 + 50.0/0.5 = 100
// ============================================================================

func.func @caseB_both_quant_rhs_const(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %const_q = onnx.Constant dense<10> : tensor<i8>
  %scast_const = "quant.scast"(%const_q) : (tensor<i8>) -> tensor<!quant.uniform<i8:f32, 5.0:0>>
  %add = "onnx.Add"(%act, %scast_const) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<!quant.uniform<i8:f32, 5.0:0>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %out = "quant.scast"(%add) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @caseB_both_quant_rhs_const
// CHECK-NOT: onnx.Add
// CHECK: return %arg0 : tensor<1x4xi8>

// -----

// ============================================================================
// Case B: constant through value-preserving op (Reshape) + quant.scast
// k = (25 - 0) * 4.0 = 100.0
// Fold Add into output: zp_new = 0 + 100.0/1.0 = 100
// ============================================================================

func.func @caseB_const_via_reshape(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 1.0:0>>
  %const_q = onnx.Constant dense<25> : tensor<i8>
  %shape = onnx.Constant dense<> : tensor<0xi64>
  %reshape = "onnx.Reshape"(%const_q, %shape) {allowzero = 0 : si64} : (tensor<i8>, tensor<0xi64>) -> tensor<i8>
  %scast_const = "quant.scast"(%reshape) : (tensor<i8>) -> tensor<!quant.uniform<i8:f32, 4.0:0>>
  %add = "onnx.Add"(%act, %scast_const) : (tensor<1x4x!quant.uniform<i8:f32, 1.0:0>>, tensor<!quant.uniform<i8:f32, 4.0:0>>) -> tensor<1x4x!quant.uniform<i8:f32, 1.0:0>>
  %out = "quant.scast"(%add) : (tensor<1x4x!quant.uniform<i8:f32, 1.0:0>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @caseB_const_via_reshape
// CHECK-NOT: onnx.Add
// CHECK: return %arg0 : tensor<1x4xi8>

// -----

// ============================================================================
// Branching: activation has multiple users — fold into input quant type
// ============================================================================

func.func @branching_fold_into_input(%arg0: tensor<1x4xi8>) -> (tensor<1x4xi8>, tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>) {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>
  %k = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %mul = "onnx.Mul"(%act, %k) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %out = "quant.scast"(%mul) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>) -> tensor<1x4xi8>
  %identity = "onnx.Identity"(%act) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>
  return %out, %identity : tensor<1x4xi8>, tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>
}

// CHECK-LABEL: func.func @branching_fold_into_input
// CHECK: quant.scast %arg0 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 2.000000e-01>>
// CHECK-NOT: onnx.Mul
// CHECK: onnx.Identity

// -----

// ============================================================================
// Bail-out: k=0 with Div — scale * 0 = 0 (non-positive scale)
// ============================================================================

func.func @bailout_k_zero_div(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %k = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %div = "onnx.Div"(%act, %k) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>
  %out = "quant.scast"(%div) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @bailout_k_zero_div
// CHECK: onnx.Div

// -----

// ============================================================================
// Bail-out: multi-use binary op
// ============================================================================

func.func @bailout_multi_use(%arg0: tensor<1x4xi8>) -> (tensor<1x4xi8>, tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>) {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %k = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %mul = "onnx.Mul"(%act, %k) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>
  %out = "quant.scast"(%mul) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x4xi8>
  return %out, %mul : tensor<1x4xi8>, tensor<1x4x!quant.uniform<i8:f32, 0.1:0>>
}

// CHECK-LABEL: func.func @bailout_multi_use
// CHECK: onnx.Mul

// -----

// ============================================================================
// Bail-out: non-quant output
// ============================================================================

func.func @bailout_non_quant_output(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %k = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %mul = "onnx.Mul"(%arg0, %k) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
  return %mul : tensor<1x4xf32>
}

// CHECK-LABEL: func.func @bailout_non_quant_output
// CHECK: onnx.Mul

// -----

// ============================================================================
// Bail-out: block argument activation
// ============================================================================

func.func @bailout_block_arg(%arg0: tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>) -> tensor<1x4xi8> {
  %k = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %mul = "onnx.Mul"(%arg0, %k) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.1:10>>
  %out = "quant.scast"(%mul) : (tensor<1x4x!quant.uniform<i8:f32, 0.1:10>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @bailout_block_arg
// CHECK: onnx.Mul

// -----

// ============================================================================
// Bail-out: zp overflow — new zp = 0 + 100/0.5 = 200 > 127
// ============================================================================

func.func @bailout_zp_overflow(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %k = onnx.Constant dense<1.000000e+02> : tensor<f32>
  %add = "onnx.Add"(%act, %k) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>
  %out = "quant.scast"(%add) : (tensor<1x4x!quant.uniform<i8:f32, 0.5:0>>) -> tensor<1x4xi8>
  return %out : tensor<1x4xi8>
}

// CHECK-LABEL: func.func @bailout_zp_overflow
// CHECK: onnx.Add

// -----

// ============================================================================
// Unsigned quant type: ui8 with Mul — scale_new = 0.2 / 4.0 = 0.05
// ============================================================================

func.func @unsigned_mul(%arg0: tensor<1x4xui8>) -> tensor<1x4xui8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xui8>) -> tensor<1x4x!quant.uniform<u8:f32, 0.5:128>>
  %k = onnx.Constant dense<4.000000e+00> : tensor<f32>
  %mul = "onnx.Mul"(%act, %k) : (tensor<1x4x!quant.uniform<u8:f32, 0.5:128>>, tensor<f32>) -> tensor<1x4x!quant.uniform<u8:f32, 0.2:10>>
  %out = "quant.scast"(%mul) : (tensor<1x4x!quant.uniform<u8:f32, 0.2:10>>) -> tensor<1x4xui8>
  return %out : tensor<1x4xui8>
}

// CHECK-LABEL: func.func @unsigned_mul
// CHECK-NOT: onnx.Mul
// CHECK: return %arg0 : tensor<1x4xui8>

// -----

// ============================================================================
// Unsigned quant type: zp overflow — new zp = 128 + 200/0.5 = 528 > 255
// ============================================================================

func.func @bailout_unsigned_zp_overflow(%arg0: tensor<1x4xui8>) -> tensor<1x4xui8> {
  %act = "quant.scast"(%arg0) : (tensor<1x4xui8>) -> tensor<1x4x!quant.uniform<u8:f32, 0.5:128>>
  %k = onnx.Constant dense<2.000000e+02> : tensor<f32>
  %add = "onnx.Add"(%act, %k) : (tensor<1x4x!quant.uniform<u8:f32, 0.5:128>>, tensor<f32>) -> tensor<1x4x!quant.uniform<u8:f32, 0.5:128>>
  %out = "quant.scast"(%add) : (tensor<1x4x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x4xui8>
  return %out : tensor<1x4xui8>
}

// CHECK-LABEL: func.func @bailout_unsigned_zp_overflow
// CHECK: onnx.Add
