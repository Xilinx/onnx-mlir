// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --decompose-onnx="enable-groupqueryattention-decompose=true" %s -split-input-file | FileCheck %s

// The cos/sin caches are passed through to onnx.RotaryEmbedding without
// Slice. Missing position_ids are synthesized from runtime seqlens_k instead
// of static past_key capacity.

func.func @gqa_unsliced_cache_packed(
  %qkv: tensor<1x1x6144xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %cos_cache: tensor<4096x48xf32>,
  %sin_cache: tensor<4096x48xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%qkv, %none, %none, %past_k, %past_v, %seqlens, %total_seqlen, %cos_cache, %sin_cache) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    do_rotary = 1 : si64,
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  }: (tensor<1x1x6144xf32>, none, none, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}
// CHECK-LABEL: func.func @gqa_unsliced_cache_packed(
// CHECK-SAME: %[[QKV:[[:alnum:]_]+]]: tensor<1x1x6144xf32>
// CHECK-SAME: %[[PAST_K:[[:alnum:]_]+]]: tensor<1x16x256x96xf32>
// CHECK-SAME: %[[PAST_V:[[:alnum:]_]+]]: tensor<1x16x256x96xf32>
// CHECK-SAME: %[[COS:[[:alnum:]_]+]]: tensor<4096x48xf32>
// CHECK-SAME: %[[SIN:[[:alnum:]_]+]]: tensor<4096x48xf32>
// CHECK-NOT: "onnx.Slice"
// CHECK: %[[SEQLENS:[[:alnum:]_]+]] = onnx.Constant dense<255> : tensor<1x1xi32>
// CHECK: %[[SPLIT:[[:alnum:]_]+]]:3 = "onnx.Split"(%[[QKV]]
// CHECK: %[[SEQLENS_I64:[[:alnum:]_]+]] = "onnx.Cast"(%[[SEQLENS]])
// CHECK-SAME: -> tensor<1x1xi64>
// CHECK: %[[START:[[:alnum:]_]+]] = "onnx.Max"(
// CHECK-SAME: -> tensor<1x1xi64>
// CHECK: %[[Q_RANGE:[[:alnum:]_]+]] = "onnx.Range"(
// CHECK-SAME: -> tensor<1xi64>
// CHECK: %[[Q_RANGE_2D:[[:alnum:]_]+]] = "onnx.Reshape"(%[[Q_RANGE]]
// CHECK-SAME: -> tensor<1x1xi64>
// CHECK: %[[POS_IDS:[[:alnum:]_]+]] = "onnx.Add"(%[[START]], %[[Q_RANGE_2D]])
// CHECK-SAME: -> tensor<1x1xi64>
// CHECK: %[[COSG:[[:alnum:]_]+]] = "onnx.Gather"(%[[COS]], %[[POS_IDS]]) {axis = 0 : si64} : (tensor<4096x48xf32>, tensor<1x1xi64>) -> tensor<1x1x48xf32>
// CHECK: %[[SING:[[:alnum:]_]+]] = "onnx.Gather"(%[[SIN]], %[[POS_IDS]]) {axis = 0 : si64} : (tensor<4096x48xf32>, tensor<1x1xi64>) -> tensor<1x1x48xf32>
// CHECK: %[[ROPE_Q:[[:alnum:]_]+]] = "onnx.RotaryEmbedding"(%[[SPLIT]]#0, %[[COSG]], %[[SING]], %{{[[:alnum:]_]+}})
// CHECK-SAME: tensor<1x1x48xf32>, tensor<1x1x48xf32>, none
// CHECK: %[[ROPE_K:[[:alnum:]_]+]] = "onnx.RotaryEmbedding"(%[[SPLIT]]#1, %[[COSG]], %[[SING]], %{{[[:alnum:]_]+}})
// CHECK-SAME: tensor<1x1x48xf32>, tensor<1x1x48xf32>, none
// CHECK: %[[MASK:[[:alnum:]_]+]] = "onnx.Where"(
// CHECK-SAME: -> tensor<1x1x1x257xf32>
// CHECK: "onnx.Attention"(%[[ROPE_Q]], %[[ROPE_K]], %[[SPLIT]]#2, %[[MASK]], %[[PAST_K]], %[[PAST_V]])
// CHECK-SAME: {is_causal = 0 : si64

// -----

// Multi-step prefill (seq_len > 1) with batch > 1 and a non-zero past length:
// synthesized position_ids must be a 2D tensor derived from seqlens_k.

func.func @gqa_unsliced_cache_prefill_batched(
  %q: tensor<2x4x3072xf32>,
  %k: tensor<2x4x1536xf32>,
  %v: tensor<2x4x1536xf32>,
  %past_k: tensor<2x16x8x96xf32>,
  %past_v: tensor<2x16x8x96xf32>,
  %cos_cache: tensor<4096x48xf32>,
  %sin_cache: tensor<4096x48xf32>
) -> (tensor<2x4x3072xf32>, tensor<2x16x12x96xf32>, tensor<2x16x12x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<12> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<11> : tensor<2x1xi32>} : () -> tensor<2x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen, %cos_cache, %sin_cache) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    do_rotary = 1 : si64,
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  }: (tensor<2x4x3072xf32>, tensor<2x4x1536xf32>, tensor<2x4x1536xf32>, tensor<2x16x8x96xf32>, tensor<2x16x8x96xf32>, tensor<2x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> (tensor<2x4x3072xf32>, tensor<2x16x12x96xf32>, tensor<2x16x12x96xf32>)
  return %out, %present_k, %present_v : tensor<2x4x3072xf32>, tensor<2x16x12x96xf32>, tensor<2x16x12x96xf32>
}
// CHECK-LABEL: func.func @gqa_unsliced_cache_prefill_batched(
// CHECK-SAME: %[[Q:[[:alnum:]_]+]]: tensor<2x4x3072xf32>
// CHECK-SAME: %[[K:[[:alnum:]_]+]]: tensor<2x4x1536xf32>
// CHECK-SAME: %[[V:[[:alnum:]_]+]]: tensor<2x4x1536xf32>
// CHECK-SAME: %[[PAST_K:[[:alnum:]_]+]]: tensor<2x16x8x96xf32>
// CHECK-SAME: %[[PAST_V:[[:alnum:]_]+]]: tensor<2x16x8x96xf32>
// CHECK-SAME: %[[COS:[[:alnum:]_]+]]: tensor<4096x48xf32>
// CHECK-SAME: %[[SIN:[[:alnum:]_]+]]: tensor<4096x48xf32>
// CHECK-NOT: "onnx.Slice"
// CHECK: %[[SEQLENS:[[:alnum:]_]+]] = onnx.Constant dense<11> : tensor<2x1xi32>
// CHECK: %[[SEQLENS_I64:[[:alnum:]_]+]] = "onnx.Cast"(%[[SEQLENS]])
// CHECK-SAME: -> tensor<2x1xi64>
// CHECK: %[[START:[[:alnum:]_]+]] = "onnx.Max"(
// CHECK-SAME: -> tensor<2x1xi64>
// CHECK: %[[Q_RANGE:[[:alnum:]_]+]] = "onnx.Range"(
// CHECK-SAME: -> tensor<4xi64>
// CHECK: %[[Q_RANGE_2D:[[:alnum:]_]+]] = "onnx.Reshape"(%[[Q_RANGE]]
// CHECK-SAME: -> tensor<1x4xi64>
// CHECK: %[[POS_IDS:[[:alnum:]_]+]] = "onnx.Add"(%[[START]], %[[Q_RANGE_2D]])
// CHECK-SAME: -> tensor<2x4xi64>
// CHECK: %[[COSG:[[:alnum:]_]+]] = "onnx.Gather"(%[[COS]], %[[POS_IDS]]) {axis = 0 : si64} : (tensor<4096x48xf32>, tensor<2x4xi64>) -> tensor<2x4x48xf32>
// CHECK: %[[SING:[[:alnum:]_]+]] = "onnx.Gather"(%[[SIN]], %[[POS_IDS]]) {axis = 0 : si64} : (tensor<4096x48xf32>, tensor<2x4xi64>) -> tensor<2x4x48xf32>
// CHECK: %[[ROPE_Q:[[:alnum:]_]+]] = "onnx.RotaryEmbedding"(%[[Q]], %[[COSG]], %[[SING]], %{{[[:alnum:]_]+}})
// CHECK-SAME: tensor<2x4x48xf32>, tensor<2x4x48xf32>, none
// CHECK: %[[ROPE_K:[[:alnum:]_]+]] = "onnx.RotaryEmbedding"(%[[K]], %[[COSG]], %[[SING]], %{{[[:alnum:]_]+}})
// CHECK-SAME: tensor<2x4x48xf32>, tensor<2x4x48xf32>, none
// CHECK: %[[MASK:[[:alnum:]_]+]] = "onnx.Where"(
// CHECK-SAME: -> tensor<2x1x4x12xf32>
// CHECK: "onnx.Attention"(%[[ROPE_Q]], %[[ROPE_K]], %[[V]], %[[MASK]], %[[PAST_K]], %[[PAST_V]])
// CHECK-SAME: {is_causal = 0 : si64

// -----

// When position_ids is provided by the caller, the synthesis path is not
// taken: the existing position_ids flow through and no cache Slice is added.

func.func @gqa_unsliced_cache_with_user_position_ids(
  %q: tensor<1x4x3072xf32>,
  %k: tensor<1x4x1536xf32>,
  %v: tensor<1x4x1536xf32>,
  %past_k: tensor<1x16x8x96xf32>,
  %past_v: tensor<1x16x8x96xf32>,
  %cos_cache: tensor<4096x48xf32>,
  %sin_cache: tensor<4096x48xf32>,
  %pos_ids: tensor<1x4xi64>
) -> (tensor<1x4x3072xf32>, tensor<1x16x12x96xf32>, tensor<1x16x12x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<12> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<11> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen, %cos_cache, %sin_cache, %pos_ids) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    do_rotary = 1 : si64,
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  }: (tensor<1x4x3072xf32>, tensor<1x4x1536xf32>, tensor<1x4x1536xf32>, tensor<1x16x8x96xf32>, tensor<1x16x8x96xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<1x4xi64>) -> (tensor<1x4x3072xf32>, tensor<1x16x12x96xf32>, tensor<1x16x12x96xf32>)
  return %out, %present_k, %present_v : tensor<1x4x3072xf32>, tensor<1x16x12x96xf32>, tensor<1x16x12x96xf32>
}
// CHECK-LABEL: func.func @gqa_unsliced_cache_with_user_position_ids(
// CHECK-SAME: %[[Q:[[:alnum:]_]+]]: tensor<1x4x3072xf32>
// CHECK-SAME: %[[K:[[:alnum:]_]+]]: tensor<1x4x1536xf32>
// CHECK-SAME: %[[V:[[:alnum:]_]+]]: tensor<1x4x1536xf32>
// CHECK-SAME: %[[PAST_K:[[:alnum:]_]+]]: tensor<1x16x8x96xf32>
// CHECK-SAME: %[[PAST_V:[[:alnum:]_]+]]: tensor<1x16x8x96xf32>
// CHECK-SAME: %[[COS:[[:alnum:]_]+]]: tensor<4096x48xf32>
// CHECK-SAME: %[[SIN:[[:alnum:]_]+]]: tensor<4096x48xf32>
// CHECK-SAME: %[[POS_IDS:[[:alnum:]_]+]]: tensor<1x4xi64>
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Max"
// CHECK: %[[COSG:[[:alnum:]_]+]] = "onnx.Gather"(%[[COS]], %[[POS_IDS]]) {axis = 0 : si64} : (tensor<4096x48xf32>, tensor<1x4xi64>) -> tensor<1x4x48xf32>
// CHECK: %[[SING:[[:alnum:]_]+]] = "onnx.Gather"(%[[SIN]], %[[POS_IDS]]) {axis = 0 : si64} : (tensor<4096x48xf32>, tensor<1x4xi64>) -> tensor<1x4x48xf32>
// CHECK: %[[ROPE_Q:[[:alnum:]_]+]] = "onnx.RotaryEmbedding"(%[[Q]], %[[COSG]], %[[SING]], %{{[[:alnum:]_]+}})
// CHECK-SAME: tensor<1x4x48xf32>, tensor<1x4x48xf32>, none
// CHECK: %[[ROPE_K:[[:alnum:]_]+]] = "onnx.RotaryEmbedding"(%[[K]], %[[COSG]], %[[SING]], %{{[[:alnum:]_]+}})
// CHECK-SAME: tensor<1x4x48xf32>, tensor<1x4x48xf32>, none
// CHECK: %[[MASK:[[:alnum:]_]+]] = "onnx.Where"(
// CHECK-SAME: -> tensor<1x1x4x12xf32>
// CHECK: "onnx.Attention"(%[[ROPE_Q]], %[[ROPE_K]], %[[V]], %[[MASK]], %[[PAST_K]], %[[PAST_V]])
// CHECK-SAME: {is_causal = 0 : si64
