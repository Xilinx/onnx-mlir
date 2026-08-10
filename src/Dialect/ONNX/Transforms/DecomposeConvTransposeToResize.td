// SPDX-License-Identifier: Apache-2.0

//===- DecomposeConvTransposeToResize.td ----------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// =============================================================================
//
// Rewrite a nearest-neighbor upsampling ONNXConvTransposeOp into an
// ONNXResizeOp(mode="nearest"). the phased/single-conv ConvTranspose decomposition 
// checks if the ConvTranspose is a nearest-neighbor upsample and skips
// the decomposition for that instance.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_DECOMPOSE_CONVTRANSPOSE_TO_RESIZE
#define ONNX_DECOMPOSE_CONVTRANSPOSE_TO_RESIZE

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
#endif // OP_BASE

include "Decompose.td"

// True iff the ConvTranspose producing is a nearest-neighbor spatial
// upsample expressible as onnx.Resize.
def IsNearestUpsampleConvTranspose: Constraint<
  CPred<"onnx_mlir::isNearestUpsampleConvTranspose("
        "mlir::cast<ONNXConvTransposeOp>($_self.getDefiningOp()))">,
  "ConvTranspose is a nearest-neighbor upsample expressible as Resize">;

// Build the equivalent onnx.Resize(mode="nearest") from the ConvTranspose
// result value.
def CreateNearestResizeFromConvTranspose: NativeCodeCall<
  "onnx_mlir::createNearestResizeFromConvTranspose($_builder, $_loc, $0)">;

def ConvTransposeToResizePattern: Pat<
  (ONNXConvTransposeOp:$res $x, $w, $b, $auto_pad, $dilation, $group,
     $kernel_shape, $output_padding, $output_shape, $pads, $strides),
  (CreateNearestResizeFromConvTranspose $res),
  [(IsNearestUpsampleConvTranspose:$res)], [],
  (addBenefit 10)
>;

#endif // ONNX_DECOMPOSE_CONVTRANSPOSE_TO_RESIZE
