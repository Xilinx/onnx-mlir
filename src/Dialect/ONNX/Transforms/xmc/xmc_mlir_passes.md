# XMC ONNX-to-MLIR Pass Pipeline

This document summarizes every pass added in `addXmcMlirPasses` (`src/Compiler/XmcMlirPasses.cpp`), in pipeline order. All passes run as nested passes on `func::FuncOp`. Summaries are derived from reading the actual pass implementations (mostly under `src/Dialect/ONNX/Transforms/xmc/`).

| # | Pass | Summary | Condition |
|---|------|---------|-----------|
| 1 | `FixNegScalePass` | Rewrites scalar `DequantizeLinear` with negative/zero scale into a positive-scale equivalent (negate scale + swap x/zp; for zero-scale emit x=0, zp=0, scale=1 so output stays 0). | Always |
| 2 | `RecomposeHardSigmoidPass` | Folds `Clip(Add(Mul(x,~1/6),~0.5),0,1)` back into a single `onnx.HardSigmoid` (alpha=0.2, beta=0.5). | Always |
| 3 | `DQBinaryQOptPass` | Folds a scalar-const `Mul` into the scale/zero-point of an adjacent `QuantizeLinear`/`DequantizeLinear`, removing the binary op. (Only Mul patterns are active; Add/Sub/Div are commented out.) | Always |
| 4 | `DedupDQsPass` | Deduplicates identical `DequantizeLinear` ops (same input X, scale, zero-point, axis, block-size): rewires all uses to a single DQ and erases the rest, preferring the DQ that feeds `func.return`. | Always |
| 5 | `ConvertQDQToRequantizePass` | Collapses `Quantize(Dequantize(x))` to `x` when params match, else inserts `XCOMPILERRequantize` carrying the (s1,zp1)→(s2,zp2) conversion. | Always |
| 6 | `QuantTypesPass` | Replaces `DequantizeLinear`/`QuantizeLinear` with `quant.StorageCast` and retypes values to `!quant.uniform`, moving to a typed-quant representation. | Always |
| 7 | `ReplaceErfToGeluPass` | Detects exact GELU `0.5*x*(1+erf(x/sqrt(2)))` subgraph and replaces it with `onnx.Gelu(approximate="none")`. | Always |
| 8 | `ReplaceTanhToGeluPass` | Detects the tanh-approximation GELU subgraph and replaces it with `onnx.Gelu(approximate="tanh")`. | Always |
| 9 | `TransferScalarConstInputDivToRequantizePass` | Replaces `Div` of a quantized activation by a scalar UINT16 quantized constant with `XCOMPILERRequantize` (kernel y_scale = s_y·real_c). (Only Div active; Mul exists but unregistered.) | Always |
| 10 | `PropagateQuantTypeThroughDataFlowPass` | Reconciles f32 <-> `!quant.uniform` type mismatches across shape/data-flow ops (Reshape, Transpose, Concat, etc.) by in-place retyping, forward or backward. | Always |
| 11 | `XmcRequantizePass` | Inserts `XCOMPILERRequantize` on data-flow ops whose input/output quant types differ (per-input for Concat), bridging quant mismatches. | Always |
| 12 | `RemoveNoOpRequantizePass` | Removes `XCOMPILERRequantize` whose input/output quant type (scale, zp, dtype) are identical, rewiring consumers to the input. | Always |
| 13 | `ConvertInstanceNormToGroupNormPass` | Fuses `Reshape->InstanceNormalization->Reshape` into `GroupNormalization`, deriving num_groups and expanding scale/bias to full channels. | Always |
| 14 | `RemoveDilationConv` | Materializes dilation into weights: expands constant Conv weights with interleaved zeros and replaces the op with an equivalent `dilation=[1,1]` Conv. | Always |
| 15 | `TransferResizeLinearToDwConv` | Replaces linear/trilinear `Resize` (integer scale) with synthesized interpolation weights via `ConvTranspose` (up) or grouped `Conv` (down); skips quantized inputs. | Always |
| 16 | `ConvWithBiasPass` | Fuses `Add(Conv(X,W), const)` into `Conv(X,W,bias)` when the constant is one value per output channel. | Always |
| 17 | `RemoveRedundantReshapePass` | Cancels redundant reshape pairs around Sigmoid and around Add/Mul/Sub, moving the op onto the original tensor shape. | Always |
| 18 | `BatchReductionToReshapeReductionPass` | For quantized 4D `ReduceSum` (batch>1, axis 3), reshapes `[N,C,H,W]->[1,N*C,H,W]`, reduces, then reshapes back. | Always |
| 19 | `TransferReduceMeanSumToConvPass` | Replaces channel-wise `ReduceMean`/`ReduceSum` with a 1x1 all-ones `Conv` (sum/mean); other axes are transposed to channel, converted, and transposed back. | Always |
| 20 | `ReplaceQDQReductionPass` | On `DQ->Reduce(Sum/Mean/Max/Min)->Q`, reshapes the reduction to rank-4 keepdims=1 form via leading/trailing reshapes, then restores output shape. | Always |
| 21 | `LowerReduceToPoolPass` | Lowers spatial `ReduceMean`->`AveragePool`/`GlobalAveragePool` and `ReduceMax`->`MaxPool`; channel `ReduceMax` uses a reshape trick. | Always |
| 22 | `TransferPoolFixToDownsampleFixPass` | Converts 1x1-kernel stride>1 zero-pad `AveragePool`/`MaxPool` into `Resize(mode=nearest)` (strided downsample). | Always |
| 23 | `RemoveRedundantReluLikeOpsPass` | Removes no-op activations: dedup `Relu(Relu)`, drops Relu/LeakyRelu on provably-nonnegative inputs, merges adjacent LeakyRelu (no requant only). | Always |
| 24 | `StandardizeSliceOpsPass` | Expands partial `Slice` into dense per-rank starts/ends/axes/steps, and converts arithmetic-index `Gather` into `Slice`. | Always |
| 25 | `MergeContinuousStridedSlicePass` | Fuses chains of back-to-back quantized `Slice` ops (matching params) into one by combining starts/ends/steps per axis. | Always |
| 26 | `ConvertMulToDepthwiseConv2dPass` | Rewrites per-channel NCHW `Mul`-by-const (optionally +Add bias/Relu) into a depthwise 1x1 `Conv` (group=channels); skips quantized. | Always |
| 27 | `TransferDepthwiseConv2dWithChannelMultiplierPass` | Splits a depthwise conv with channel multiplier >1 into N multiplier-1 depthwise convs plus a channel `Concat`. | Always |
| 28 | `RemoveUselessQLinearPoolPass` | Removes identity `AveragePool`/`MaxPool` (kernel=stride=1, shape/quant unchanged) by rewiring output to input. | Always |
| 29 | `OptimizeSliceReshapeTransposeBlockPass` | Fuses MHA-style triple `Slice->Reshape->Transpose` into `Reshape->Transpose(0,2,1,3)->Slice`, reducing layout ops before Q/K/V splits. | Always |
| 30 | `TransferSpaceToDepthToConv2dPass` | Replaces a model-specific `SpaceToDepth` (blocksize=4 on [1,3,512,512]) with a strided 4x4 `Conv` using a fixed sparse weight. | Always |
| 31 | `MergeBatchnormToConvPass` | On `Conv->Reshape->Transpose->BatchNorm`, folds BN into Conv attributes (nonlinear=BATCHNORM, c2_f/c3_f...) and drops the BN. | Always |
| 32 | `EliminateReshapeAroundSlicePass` | For `Reshape->Slice->[Add/Mul/Sub]*->Reshape` moving a single non-1 dim, removes reshapes by slicing the original tensor directly. | Always |
| 33 | `MergeSliceConcatPass` | Collapses `Slice*->Concat->InstanceNorm->Conv` (channel-partitioning slices) by feeding the unsliced input and reordering IN/conv weights. | Always |
| 34 | `TransferConvSliceToConvPass` | Fuses `Conv->Slice` into one Conv: channel slices trim weight/bias channels; spatial slices crop input and adjust padding. | Always |
| 35 | `TransferOp1dToOp2dPass` | Converts 1D conv/pool ops to 2D by reshaping `[N,C,L]<->[N,C,1,L]`, extending kernel/stride/pad attrs, reshaping output back. | Always |
| 36 | `TransferScaleToDwConv2dPass` | Replaces non-quantized `Mul(input, 1D scale)` matching last dim with a depthwise 1x1 `XFEConv` (group=channels), preserving trailing activation. | Always |
| 37 | `ConvertToChannelLastPass` | Replaces NCHW ops (Conv, Pool, Norm, Resize, etc.) with channel-last XFE variants, wrapping with NCHW<->NHWC transposes and quant-axis remap. | Always |
| 38 | `FuseMatMulAddToXFEMatMulBiasPass` | Fuses `Add(MatMul(A,B), const bias)` into `onnx.XFEMatMulBias` when bias is 1 value per output channel with compatible quant. | `opts.enableMatmulAddFusion` |
| 39 | `ConvertMatMulToXFEConvPass` | Rewrites `MatMul`/`Gemm` with a 2D constant weight into `Reshape->XFEConv->Reshape`, optionally folding a following const `Add` bias. | `opts.enableMatmulToConv` |
| 40 | `TransferSoftmaxAxisToLastPass` | Wraps Softmax with transposes to move the reduction axis to last (axis=-1); skips LogSoftmax (Log consumer). | Always |
| 41 | `ONNXTransposeOptimizationPass` | Broad transpose reduction: removes identity/consecutive transposes, pushes transposes through unary/binary/QDQ/reshape ops, folds const DQ+Transpose. | Always |
| 42 | `ConstPropONNXToONNXPass` | Compile-time constant folding of ops with all-constant inputs into `ONNXConstant` (math, reductions, shape ops, Gemm/MatMul, Split/If/Loop). | Always |
| 43 | `RemoveContinuousTransposeWithReshapePass` | Eliminates `Transpose->Reshape->Transpose` where merged permutations make the transposes inverses, leaving the input or a single Reshape. | Always |
| 44 | `TransferOp3dToOp2dPass` | Converts 5D Conv/Add(+Relu) to 4D by merging depth into channels (`[N,C,D,H,W]->[N,C*D,H,W]`), then reshaping back to 5D. | Always |
| 45 | `TransformReshapelikeOpToReshapePass` | Canonicalizes `Flatten`/`Squeeze`/`Unsqueeze`/trivial `Transpose` into explicit `Reshape` ops. | Always |
| 46 | `RemoveSemanticallyUselessOpsPass` | Removes no-ops/dead compute: identity Reshape/Concat, Mul-by-0, Sub(x,x), zero Pad, scale-1/unchanged Resize. | Always |
| 47 | `Transfer5dBlockTo4dPass` | Rewrites specific 4D->5D->(eltwise/concat/transpose)->4D blocks by merging 5D dims (usually C and D) so inner ops run in 4D. | Always |
| 48 | `Transform5DTransposeTo4DPass` | Decomposes rank>4 `Transpose` into `Reshape + <=4D Transpose + Reshape` by merging identity/consecutive permutation dims. | Always |
| 49 | `CombineTransposePairPass` | Deduplicates identical `Transpose` ops sharing the same input and perm, rewiring users to a single kept op. | Always |
| 50 | `ReplaceNDimTransposePass` | Model-specific: replaces `Transpose(perm=[2,0,3,1])->Reshape` with two transposes (`[0,2,1,3]` then `[0,1,3,2]`). | Always |
| 51 | `Transfer5dStridedSliceTo4d` | On batch=1 quantized 5D `Slice`, collapses two consecutive full-copy dims into a 4D `Slice` via reshapes and adjusted params. | Always |
| 52 | `ReplaceQDQResizePass` | Replaces rank-4 1x1-spatial `XFEResize` (matching quant params) with `Add(input, splat zero_point)` -- broadcast identity in quant domain. | Always |
| 53 | `ReplaceQuantizedTileToAddPass` | Lowers broadcastable `Tile` to identity `Add` with splat zp/0, and reorders Tile relative to TopK/GatherElements index paths. | Always |
| 54 | `ReplaceQDQClipCastPass` | Fuses `Clip(quant)->f32->Cast(->uint)` into one `XCOMPILERFusedEltwise type="CLAMP"`, lifting Clip min/max. | Always |
| 55 | `ReplaceQDQEltwisePass` | Fuses quantized eltwise ops (Add/Mul/Sub/Div, unary, Clip, Sigmoid...) into `XCOMPILERFusedEltwise`, including eltwise+ReLU chains and Expand->ADD. | Always |
| 56 | `ReplaceQDQSigmoidPass` | Intended to fuse quantized Sigmoid into `QLINEARSIGMOID`, but all patterns are commented out -- currently a no-op (Sigmoid handled by pass 55). | Always |
| 57 | `TransferBatchXCompilerFusedEltwisePass` | For quantized 4D `XCOMPILERFusedEltwise` with batch!=1, reshapes to `[1,N,C,H*W]`, runs fused eltwise, reshapes back. | Always |
| 58 | `ReplaceAdjacentOpPass` | Flattens nested same-axis `Concat` into one, and splits reused SSA operands by inserting identity Reshape copies tagged `duplicate_input`. | Always |
| 59 | `RemovePairsAndMoveDownReshapePass` | Removes canceling reshape pairs around a single-use `XCOMPILERFusedEltwise` chain when input/output shapes match and operand B stays broadcastable. | `opts.enableRemovePairsReshape` |
| 60 | `ReplaceContainedConcatPass` | When an outer concat fully contains an inner concat's inputs (same axis/quant), rewrites to `[inner_result] + remaining outer inputs`. | Always |
| 61 | `OptimizeSiblingConcatPass` | For two sibling channel-axis concats sharing one input (one feeding InstanceNorm->Conv), swaps input order preserving semantics via Concat->Slice->Concat. | Always |
| 62 | `CanonicalizeWithResultNamesPass` | Runs standard MLIR canonicalization (+optional QDQ data-movement) with a listener that preserves `ResultNames`/`TensorName` metadata across rewrites. | Always |
| 63 | `ReplaceHsigmoidAndHswishPass` | Replaces quantized `HardSigmoid` with `XCOMPILERFusedEltwise type="HSIGMOID"` and folds `HardSigmoid*const` into mul_y; HSwish not yet implemented. | Always |
| 64 | `ConvertXFEConvToDepthwiseConvPass` | Converts depthwise `XFEConv` (group==in_channels) into `XCOMPILERDepthwiseConv`, transposing weights OHWI->IHWO and remapping quant metadata. | Always |
| 65 | `FuseConvActivationPass` | Fuses a following activation (FusedEltwise or ReLU/LeakyReLU/HardSigmoid/Clip[0,6]) into `XFEConv`/`XFEConvTranspose`/`DepthwiseConv`'s `activation` attribute. | Always |
| 66 | `NormalizeConvActivationPass` | Normalizes conv fused-activation strings to hardware form (drop implicit ReLU on UINT8 zp=0, map LeakyReLU alpha to RELU/LEAKYRELU/PRELU with computed params). | Always |
| 67 | `AddRequantForOutputConvPass` | When a quantized conv/fused-eltwise has a `scast->DequantizeLinear` output path among multiple users, inserts a placeholder `XCOMPILERRequantize` (scale=1, zp=0) there. | Always |
| 68 | `ShapeInferencePass` | Greedily applies ONNX shape-inference patterns to propagate static/dynamic shapes, then updates function return types to match. | Always |

## Notes

- **Pass 56 (`ReplaceQDQSigmoidPass`) is effectively a no-op** — its rewrite patterns are all commented out; quantized Sigmoid is actually handled by pass 55 (`ReplaceQDQEltwisePass`).
- **Passes 3 and 9** have additional patterns present in source but not registered (only `Mul` / only `Div` are active, respectively).
- **Passes 30 and 50** are model-specific rewrites keyed to particular shapes/perms rather than general transforms.

## Commented-out / disabled passes

These appear in `XmcMlirPasses.cpp` but are not active:

- `OptimizeOnnxRequantizationPass` — replaced by `XmcRequantizePass`
- `SplitGroupConvPass`
- `ConvertSCastPairToRequantizePass` — covered by `XmcRequantizePass`
- `TransferOpShapeTo4dPass` — architecture-specific
- `TransferReduceHdimToReduceCdimPass`

## Source locations

Most passes live in `src/Dialect/ONNX/Transforms/xmc/`. Exceptions:

- `FixNegScalePass` → `src/Dialect/ONNX/Transforms/FixNegScale.cpp`
- `QuantTypesPass` → `src/Dialect/ONNX/Transforms/QuantTypes.cpp`
- `DedupDQsPass` → `src/Dialect/ONNX/Transforms/DedupDQs.cpp`
- `ConstPropONNXToONNXPass` → `src/Dialect/ONNX/Transforms/ConstProp.cpp`
- `ShapeInferencePass` → `src/Dialect/ONNX/Transforms/ShapeInferencePass.cpp`
- `CanonicalizeWithResultNamesPass` → `src/Dialect/ONNX/Transforms/ResultNamesUpdater.cpp`
