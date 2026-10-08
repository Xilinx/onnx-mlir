// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// RUN: onnx-mlir-opt --onnx-hybrid-transform="shape-inference=false canonicalization=false constant-propagation=false recomposition=false decomposition=true enable-groupqueryattention-decompose=true hold-back-preallocated-gqa-decompose=true" %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --decompose-onnx="enable-groupqueryattention-decompose=true hold-back-preallocated-gqa-decompose=true" %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --onnx-hybrid-transform="shape-inference=false canonicalization=false constant-propagation=false recomposition=false decomposition=true enable-groupqueryattention-decompose=true hold-back-preallocated-gqa-decompose=false" %s -split-input-file | FileCheck %s --check-prefix=NO-HOLDBACK

func.func @full_rope_decode(
  %qkv: tensor<1x1x6144xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %cos: tensor<4096x48xf32>,
  %sin: tensor<4096x48xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %total = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %out, %present_k, %present_v = "onnx.Custom"(%qkv, %none, %none, %past_k, %past_v, %seqlens, %total, %cos, %sin) {
    domain_name = "com.microsoft", function_name = "GroupQueryAttention",
    do_rotary = 1 : si64, kv_num_heads = 16 : si64, num_heads = 32 : si64
  } : (tensor<1x1x6144xf32>, none, none, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>
}

// CHECK-LABEL: func.func @full_rope_decode
// CHECK-NOT: "onnx.Attention"
// CHECK: "onnx.Custom"{{.*}}function_name = "GroupQueryAttention"
// NO-HOLDBACK-LABEL: func.func @full_rope_decode
// NO-HOLDBACK-NOT: function_name = "GroupQueryAttention"
// NO-HOLDBACK: "onnx.RotaryEmbedding"
// NO-HOLDBACK: "onnx.Attention"

// -----

func.func @partial_rope_decode(
  %qkv: tensor<1x1x8192xf32>,
  %past_k: tensor<1x16x256x128xf32>,
  %past_v: tensor<1x16x256x128xf32>,
  %cos: tensor<4096x48xf32>,
  %sin: tensor<4096x48xf32>
) -> (tensor<1x1x4096xf32>, tensor<1x16x256x128xf32>, tensor<1x16x256x128xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %total = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %out, %present_k, %present_v = "onnx.Custom"(%qkv, %none, %none, %past_k, %past_v, %seqlens, %total, %cos, %sin) {
    domain_name = "com.microsoft", function_name = "GroupQueryAttention",
    do_rotary = 1 : si64, kv_num_heads = 16 : si64, num_heads = 32 : si64
  } : (tensor<1x1x8192xf32>, none, none, tensor<1x16x256x128xf32>, tensor<1x16x256x128xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> (tensor<1x1x4096xf32>, tensor<1x16x256x128xf32>, tensor<1x16x256x128xf32>)
  return %out, %present_k, %present_v : tensor<1x1x4096xf32>, tensor<1x16x256x128xf32>, tensor<1x16x256x128xf32>
}

// CHECK-LABEL: func.func @partial_rope_decode
// CHECK-NOT: "onnx.Attention"
// CHECK: "onnx.Custom"{{.*}}function_name = "GroupQueryAttention"
// NO-HOLDBACK-LABEL: func.func @partial_rope_decode
// NO-HOLDBACK-NOT: function_name = "GroupQueryAttention"
// NO-HOLDBACK: "onnx.RotaryEmbedding"
// NO-HOLDBACK: "onnx.Attention"

// -----

// Partial RoPE: held back
func.func @partial_rope_narrow_rotary_decode(
  %qkv: tensor<1x1x6144xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %cos: tensor<4096x16xf32>,
  %sin: tensor<4096x16xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %total = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %out, %present_k, %present_v = "onnx.Custom"(%qkv, %none, %none, %past_k, %past_v, %seqlens, %total, %cos, %sin) {
    domain_name = "com.microsoft", function_name = "GroupQueryAttention",
    do_rotary = 1 : si64, kv_num_heads = 16 : si64, num_heads = 32 : si64
  } : (tensor<1x1x6144xf32>, none, none, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x16xf32>, tensor<4096x16xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>
}

// CHECK-LABEL: func.func @partial_rope_narrow_rotary_decode
// CHECK-NOT: "onnx.Attention"
// CHECK: "onnx.Custom"{{.*}}function_name = "GroupQueryAttention"
// NO-HOLDBACK-LABEL: func.func @partial_rope_narrow_rotary_decode
// NO-HOLDBACK-NOT: function_name = "GroupQueryAttention"
// NO-HOLDBACK: "onnx.RotaryEmbedding"
// NO-HOLDBACK: "onnx.Attention"

// -----

// Full rotation: held back
func.func @full_rope_unalignable_half_decode(
  %qkv: tensor<1x1x7168xf32>,
  %past_k: tensor<1x16x256x112xf32>,
  %past_v: tensor<1x16x256x112xf32>,
  %cos: tensor<4096x56xf32>,
  %sin: tensor<4096x56xf32>
) -> (tensor<1x1x3584xf32>, tensor<1x16x256x112xf32>, tensor<1x16x256x112xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %total = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %out, %present_k, %present_v = "onnx.Custom"(%qkv, %none, %none, %past_k, %past_v, %seqlens, %total, %cos, %sin) {
    domain_name = "com.microsoft", function_name = "GroupQueryAttention",
    do_rotary = 1 : si64, kv_num_heads = 16 : si64, num_heads = 32 : si64
  } : (tensor<1x1x7168xf32>, none, none, tensor<1x16x256x112xf32>, tensor<1x16x256x112xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x56xf32>, tensor<4096x56xf32>) -> (tensor<1x1x3584xf32>, tensor<1x16x256x112xf32>, tensor<1x16x256x112xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3584xf32>, tensor<1x16x256x112xf32>, tensor<1x16x256x112xf32>
}

// CHECK-LABEL: func.func @full_rope_unalignable_half_decode
// CHECK-NOT: "onnx.Attention"
// CHECK: "onnx.Custom"{{.*}}function_name = "GroupQueryAttention"
// NO-HOLDBACK-LABEL: func.func @full_rope_unalignable_half_decode
// NO-HOLDBACK-NOT: function_name = "GroupQueryAttention"
// NO-HOLDBACK: "onnx.RotaryEmbedding"
// NO-HOLDBACK: "onnx.Attention"

// -----

// decompose
func.func @full_rope_head_not_multiple_of_four(
  %qkv: tensor<1x1x6272xf32>,
  %past_k: tensor<1x16x256x98xf32>,
  %past_v: tensor<1x16x256x98xf32>,
  %cos: tensor<4096x49xf32>,
  %sin: tensor<4096x49xf32>
) -> (tensor<1x1x3136xf32>, tensor<1x16x256x98xf32>, tensor<1x16x256x98xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %total = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %out, %present_k, %present_v = "onnx.Custom"(%qkv, %none, %none, %past_k, %past_v, %seqlens, %total, %cos, %sin) {
    domain_name = "com.microsoft", function_name = "GroupQueryAttention",
    do_rotary = 1 : si64, kv_num_heads = 16 : si64, num_heads = 32 : si64
  } : (tensor<1x1x6272xf32>, none, none, tensor<1x16x256x98xf32>, tensor<1x16x256x98xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x49xf32>, tensor<4096x49xf32>) -> (tensor<1x1x3136xf32>, tensor<1x16x256x98xf32>, tensor<1x16x256x98xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3136xf32>, tensor<1x16x256x98xf32>, tensor<1x16x256x98xf32>
}

// CHECK-LABEL: func.func @full_rope_head_not_multiple_of_four
// CHECK-NOT: function_name = "GroupQueryAttention"
// CHECK: "onnx.RotaryEmbedding"
// CHECK: "onnx.Attention"

// -----

// decompose
func.func @partial_rope_rotated_half_unaligned(
  %qkv: tensor<1x1x8192xf32>,
  %past_k: tensor<1x16x256x128xf32>,
  %past_v: tensor<1x16x256x128xf32>,
  %cos: tensor<4096x40xf32>,
  %sin: tensor<4096x40xf32>
) -> (tensor<1x1x4096xf32>, tensor<1x16x256x128xf32>, tensor<1x16x256x128xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %total = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %out, %present_k, %present_v = "onnx.Custom"(%qkv, %none, %none, %past_k, %past_v, %seqlens, %total, %cos, %sin) {
    domain_name = "com.microsoft", function_name = "GroupQueryAttention",
    do_rotary = 1 : si64, kv_num_heads = 16 : si64, num_heads = 32 : si64
  } : (tensor<1x1x8192xf32>, none, none, tensor<1x16x256x128xf32>, tensor<1x16x256x128xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x40xf32>, tensor<4096x40xf32>) -> (tensor<1x1x4096xf32>, tensor<1x16x256x128xf32>, tensor<1x16x256x128xf32>)
  return %out, %present_k, %present_v : tensor<1x1x4096xf32>, tensor<1x16x256x128xf32>, tensor<1x16x256x128xf32>
}

// CHECK-LABEL: func.func @partial_rope_rotated_half_unaligned
// CHECK-NOT: function_name = "GroupQueryAttention"
// CHECK: "onnx.RotaryEmbedding"
// CHECK: "onnx.Attention"

// -----

// held back
func.func @partial_rope_passthrough_half_unaligned(
  %qkv: tensor<1x1x6656xf32>,
  %past_k: tensor<1x16x256x104xf32>,
  %past_v: tensor<1x16x256x104xf32>,
  %cos: tensor<4096x32xf32>,
  %sin: tensor<4096x32xf32>
) -> (tensor<1x1x3328xf32>, tensor<1x16x256x104xf32>, tensor<1x16x256x104xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %total = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %out, %present_k, %present_v = "onnx.Custom"(%qkv, %none, %none, %past_k, %past_v, %seqlens, %total, %cos, %sin) {
    domain_name = "com.microsoft", function_name = "GroupQueryAttention",
    do_rotary = 1 : si64, kv_num_heads = 16 : si64, num_heads = 32 : si64
  } : (tensor<1x1x6656xf32>, none, none, tensor<1x16x256x104xf32>, tensor<1x16x256x104xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x32xf32>, tensor<4096x32xf32>) -> (tensor<1x1x3328xf32>, tensor<1x16x256x104xf32>, tensor<1x16x256x104xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3328xf32>, tensor<1x16x256x104xf32>, tensor<1x16x256x104xf32>
}

// CHECK-LABEL: func.func @partial_rope_passthrough_half_unaligned
// CHECK-NOT: function_name = "GroupQueryAttention"
// CHECK: "onnx.RotaryEmbedding"
// CHECK: "onnx.Attention"

// -----

func.func @full_rope_append_decode(
  %qkv: tensor<1x1x6144xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %cos: tensor<4096x48xf32>,
  %sin: tensor<4096x48xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %total = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %out, %present_k, %present_v = "onnx.Custom"(%qkv, %none, %none, %past_k, %past_v, %seqlens, %total, %cos, %sin) {
    domain_name = "com.microsoft", function_name = "GroupQueryAttention",
    do_rotary = 1 : si64, kv_num_heads = 16 : si64, num_heads = 32 : si64
  } : (tensor<1x1x6144xf32>, none, none, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @full_rope_append_decode
// CHECK-NOT: function_name = "GroupQueryAttention"
// CHECK: "onnx.RotaryEmbedding"
// CHECK: "onnx.Attention"

// -----

func.func @empty_cache_prefill(
  %qkv: tensor<1x128x6144xf32>,
  %past_k: tensor<1x16x0x96xf32>,
  %past_v: tensor<1x16x0x96xf32>,
  %cos: tensor<4096x48xf32>,
  %sin: tensor<4096x48xf32>
) -> (tensor<1x128x3072xf32>, tensor<1x16x128x96xf32>, tensor<1x16x128x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %seqlens = "onnx.Constant"() {value = dense<0> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %total = "onnx.Constant"() {value = dense<128> : tensor<i32>} : () -> tensor<i32>
  %out, %present_k, %present_v = "onnx.Custom"(%qkv, %none, %none, %past_k, %past_v, %seqlens, %total, %cos, %sin) {
    domain_name = "com.microsoft", function_name = "GroupQueryAttention",
    do_rotary = 1 : si64, kv_num_heads = 16 : si64, num_heads = 32 : si64
  } : (tensor<1x128x6144xf32>, none, none, tensor<1x16x0x96xf32>, tensor<1x16x0x96xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x128x96xf32>, tensor<1x16x128x96xf32>)
  return %out, %present_k, %present_v : tensor<1x128x3072xf32>, tensor<1x16x128x96xf32>, tensor<1x16x128x96xf32>
}

// CHECK-LABEL: func.func @empty_cache_prefill
// CHECK-NOT: function_name = "GroupQueryAttention"
// CHECK: "onnx.RotaryEmbedding"
// CHECK: "onnx.Attention"
