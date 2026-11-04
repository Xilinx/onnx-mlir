<!-- Copyright 2026 Advanced Micro Devices, Inc. or its affiliates. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# ONNX opset 27 frontend support

ONNX 1.22.0 supplies standard-domain opset 27. The Python dependency and C++
ONNX submodule must remain at the same release. The dialect generator selects
the latest implemented **schema per operator**; it does not assign version 27
to every operation. For example, Add remains schema 14.

Selections are ordered newest first. `[24, 23]` would generate unsuffixed
`onnx.Attention` for schema 24 and `onnx.AttentionV23` for the retained older
schema. A single `[23]` selection is the native schema implemented by that
operation, not a model-opset ceiling. This update deliberately keeps `[23]` for
native Attention and handles newer Attention schemas through function import.

This update covers frontend schema contracts and inference for existing
operators, adds native BitCast, CumProd and TensorScatter representations, and
uses ONNX-defined functions for Swish, RMSNormalization, CausalConvWithState and
LinearAttention. Native representation, function import and executable target
lowering are separate support levels. A target must lower or explicitly reject
native operators and the primitives introduced by function expansion.

## Existing operator contracts

The native Attention representation stays at schema 23 with six operand slots
for existing downstream and Microsoft GQA lowering. Newer standard Attention
schemas, including optional `nonpad_kv_seqlen`, expand before this boundary.
Cast and CastLike add `round_mode`; QuantizeLinear adds
`precision`; DequantizeLinear adds `output_dtype`; Range adds `stash_type`.
Existing output counts and positions do not change. Compatible C++ builders
retain the older attribute signatures and supply schema defaults. Positional
TableGen/PDLL patterns still need updating for the declared contracts.

New attributes must survive operation reconstruction. Quantization arithmetic
precision is independent of the quantized storage dtype. Decomposition makes
the compute type explicit with casts before the existing Q/DQ operations.
Algebraic Q/DQ identity checks must not assume independent precision is a no-op.

## Function expansion

Attention schemas 24 and later expand during ONNX import to preserve the
updated causal mask, explicit mask and external-cache semantics. Older models
retain native Attention. FP16/BF16 Range-27 also expands using ONNX's function,
which carries intermediate accumulation precision explicitly.

Missing native Swish, RMSNormalization, CausalConvWithState and LinearAttention
automatically use the existing schema-function fallback. Context-dependent
functions require sufficient input type/shape information. CausalConvWithState
requires a statically recoverable channel count. LinearAttention expands to
Scan; targets without Scan support require further specialization or a native
target implementation. Functions contain their own opset imports; the importer
must use those imports while processing each body.
Schema attribute defaults are bound when expanding functions, including Swish's
default `alpha = 1`, while explicit caller attributes take precedence.

## Deferred types and backend support

FP4 E2M1, FP8 E8M0 and signed/unsigned 2-bit type implementations are deferred.
They are excluded from generated dialect type constraints; the newer schemas
do not imply those formats can be stored, folded or executed by this compiler.
Existing backend restrictions on quantization, dynamic shapes, control flow and
individual operators also remain applicable.

The downstream FlexML policy of minimum opset 11, official opset 20 and beta
opset 21 is separate from this frontend's schema versions. Its Python wrapper
currently rejects models above the beta level. Updating this frontend does not
change that policy or establish opset-27 NPU qualification. In particular,
preserving the native six-operand Attention contract avoids changing existing
internal Attention/GQA patterns as part of this frontend update.

## Upstream provenance

The TensorScatter verifier/shape inference is adapted from upstream
onnx/onnx-mlir at
`bec0660f5102b0a2cd5d013687ea0898b012b118`:

- TensorScatter: https://github.com/onnx/onnx-mlir/pull/3651

The complete Attention schema function is used instead of copying upstream's
more restricted target lowering. The native Attention implementation retains
this fork's mixed-rank internal GQA support. Target-specific TensorScatter lowering,
including input alias/copy handling, must be reviewed separately from the
frontend shape/verifier port.
