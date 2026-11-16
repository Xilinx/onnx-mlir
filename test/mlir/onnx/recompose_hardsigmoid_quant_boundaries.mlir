// RUN: onnx-mlir-opt --recompose-onnx="enable-relu-hardsigmoid-recomposition=false" --split-input-file %s | FileCheck %s
// RUN: onnx-mlir-opt --recompose-onnx --split-input-file %s | FileCheck %s --check-prefix=LEGACY
// RUN: onnx-mlir-opt --recompose-onnx="enable-relu-hardsigmoid-recomposition=true" --split-input-file %s | FileCheck %s --check-prefix=LEGACY
// RUN: onnx-mlir-opt --onnx-hybrid-transform="enable-relu-hardsigmoid-recomposition=false decomposition=false canonicalization=false constant-propagation=false shape-inference=false" --split-input-file %s | FileCheck %s
// RUN: onnx-mlir-opt --onnx-hybrid-transform="enable-relu-hardsigmoid-recomposition=true decomposition=false canonicalization=false constant-propagation=false shape-inference=false" --split-input-file %s | FileCheck %s --check-prefix=LEGACY
// RUN: onnx-mlir-opt --onnx-op-transform="enable-relu-hardsigmoid-recomposition=false" --split-input-file %s | FileCheck %s
// RUN: onnx-mlir-opt --onnx-op-transform="enable-relu-hardsigmoid-recomposition=true" --split-input-file %s | FileCheck %s --check-prefix=LEGACY
// RUN: onnx-mlir-opt --onnx-hybrid-transform="decomposition=false canonicalization=false constant-propagation=false shape-inference=false" --split-input-file %s | FileCheck %s --check-prefix=LEGACY
// RUN: onnx-mlir-opt --onnx-op-transform --split-input-file %s | FileCheck %s --check-prefix=LEGACY

// LEGACY-LABEL: @llnet_decoder1
// LEGACY-NOT: "onnx.QuantizeLinear"
// LEGACY: %[[HSIG:.*]] = "onnx.HardSigmoid"
// LEGACY: return %[[HSIG]]
// CHECK-LABEL: @llnet_decoder1
// CHECK: %[[RELU:.*]] = "onnx.Relu"
// CHECK: %[[QUANT:.*]] = "onnx.QuantizeLinear"(%[[RELU]],
// CHECK: %[[DEQUANT:.*]] = "onnx.DequantizeLinear"(%[[QUANT]],
// CHECK: %[[MUL:.*]] = "onnx.Mul"(%[[DEQUANT]],
// CHECK: return %[[MUL]]
func.func @llnet_decoder1(%input: tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32> {
  %alpha = onnx.Constant dense<0.19921875> : tensor<f32>
  %scale = onnx.Constant dense<0.00390625> : tensor<f32>
  %zero = onnx.Constant dense<128> : tensor<ui8>
  %relu = "onnx.Relu"(%input) : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
  %quant = "onnx.QuantizeLinear"(%relu, %scale, %zero) : (tensor<1x1x1x32xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x1x1x32xui8>
  %dequant = "onnx.DequantizeLinear"(%quant, %scale, %zero) : (tensor<1x1x1x32xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x1x1x32xf32>
  %output = "onnx.Mul"(%dequant, %alpha) : (tensor<1x1x1x32xf32>, tensor<f32>) -> tensor<1x1x1x32xf32>
  return %output : tensor<1x1x1x32xf32>
}

// -----

// LEGACY-LABEL: @llnet_decoder2
// LEGACY-NOT: "onnx.QuantizeLinear"
// LEGACY: %[[HSIG:.*]] = "onnx.HardSigmoid"
// LEGACY: return %[[HSIG]]
// CHECK-LABEL: @llnet_decoder2
// CHECK: %[[RELU:.*]] = "onnx.Relu"
// CHECK: %[[QUANT:.*]] = "onnx.QuantizeLinear"(%[[RELU]],
// CHECK: %[[DEQUANT:.*]] = "onnx.DequantizeLinear"(%[[QUANT]],
// CHECK: %[[MUL:.*]] = "onnx.Mul"(%[[DEQUANT]],
// CHECK: return %[[MUL]]
func.func @llnet_decoder2(%input: tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32> {
  %alpha = onnx.Constant dense<0.19921875> : tensor<f32>
  %scale = onnx.Constant dense<0.03125> : tensor<f32>
  %zero = onnx.Constant dense<128> : tensor<ui8>
  %relu = "onnx.Relu"(%input) : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
  %quant = "onnx.QuantizeLinear"(%relu, %scale, %zero) : (tensor<1x1x1x64xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x1x1x64xui8>
  %dequant = "onnx.DequantizeLinear"(%quant, %scale, %zero) : (tensor<1x1x1x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x1x1x64xf32>
  %output = "onnx.Mul"(%dequant, %alpha) : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<1x1x1x64xf32>
  return %output : tensor<1x1x1x64xf32>
}

// -----

// LEGACY-LABEL: @requant_after_relu_before_min
// LEGACY-NOT: "onnx.QuantizeLinear"
// LEGACY: %[[HSIG:.*]] = "onnx.HardSigmoid"
// LEGACY: return %[[HSIG]]
// CHECK-LABEL: @requant_after_relu_before_min
// CHECK-NOT: "onnx.QuantizeLinear"
// CHECK: %[[HSIG:.*]] = "onnx.HardSigmoid"
// CHECK: return %[[HSIG]]
func.func @requant_after_relu_before_min(%input: tensor<4xf32>) -> tensor<4xf32> {
  %alpha = onnx.Constant dense<0.25> : tensor<f32>
  %bound = onnx.Constant dense<4.0> : tensor<f32>
  %scale = onnx.Constant dense<0.03125> : tensor<f32>
  %zero = onnx.Constant dense<128> : tensor<ui8>
  %relu = "onnx.Relu"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  %quant = "onnx.QuantizeLinear"(%relu, %scale, %zero) : (tensor<4xf32>, tensor<f32>, tensor<ui8>) -> tensor<4xui8>
  %dequant = "onnx.DequantizeLinear"(%quant, %scale, %zero) : (tensor<4xui8>, tensor<f32>, tensor<ui8>) -> tensor<4xf32>
  %clamped = "onnx.Min"(%dequant, %bound) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %output = "onnx.Mul"(%clamped, %alpha) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  return %output : tensor<4xf32>
}

// -----

// LEGACY-LABEL: @requant_after_clip_before_mul
// LEGACY-NOT: "onnx.QuantizeLinear"
// LEGACY: %[[HSIG:.*]] = "onnx.HardSigmoid"
// LEGACY: return %[[HSIG]]
// CHECK-LABEL: @requant_after_clip_before_mul
// CHECK-NOT: "onnx.QuantizeLinear"
// CHECK: %[[HSIG:.*]] = "onnx.HardSigmoid"
// CHECK: return %[[HSIG]]
func.func @requant_after_clip_before_mul(%input: tensor<4xf32>) -> tensor<4xf32> {
  %alpha = onnx.Constant dense<0.25> : tensor<f32>
  %lower = onnx.Constant dense<0.0> : tensor<f32>
  %upper = onnx.Constant dense<4.0> : tensor<f32>
  %scale = onnx.Constant dense<0.03125> : tensor<f32>
  %zero = onnx.Constant dense<128> : tensor<ui8>
  %clamped = "onnx.Clip"(%input, %lower, %upper) : (tensor<4xf32>, tensor<f32>, tensor<f32>) -> tensor<4xf32>
  %quant = "onnx.QuantizeLinear"(%clamped, %scale, %zero) : (tensor<4xf32>, tensor<f32>, tensor<ui8>) -> tensor<4xui8>
  %dequant = "onnx.DequantizeLinear"(%quant, %scale, %zero) : (tensor<4xui8>, tensor<f32>, tensor<ui8>) -> tensor<4xf32>
  %output = "onnx.Mul"(%dequant, %alpha) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  return %output : tensor<4xf32>
}
