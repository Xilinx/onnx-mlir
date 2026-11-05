// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --unsafe-math-optimizations=false --decompose-onnx %s -split-input-file | FileCheck %s

// AveragePool * kernel_volume is not the Lp sum in reduced precision, so
// without --unsafe-math-optimizations every LpPool is left alone.

// bf16, p = 1, a 257-tap window: the sum 256 is exact, but 256/257 rounds to
// 0.99609375 and the scale then gives 255.
func.func @lppool_bf16_257_taps(%arg0: tensor<1x1x257xbf16>) -> tensor<1x1x1xbf16> {
  %0 = "onnx.LpPool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [257], p = 1 : si64, strides = [257]} : (tensor<1x1x257xbf16>) -> tensor<1x1x1xbf16>
  return %0 : tensor<1x1x1xbf16>

  // CHECK-LABEL: func.func @lppool_bf16_257_taps
  // CHECK: [[POOL:%.+]] = "onnx.LpPool"(%arg0)
  // CHECK-NOT: onnx.AveragePool
  // CHECK-NOT: onnx.Mul
  // CHECK: return [[POOL]] : tensor<1x1x1xbf16>
}

// -----

// fp16: a kernel volume or a powered value above 65504 overflows.
func.func @lppool_f16_large_window(%arg0: tensor<1x1x4096xf16>) -> tensor<1x1x1xf16> {
  %0 = "onnx.LpPool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [4096], p = 2 : si64, strides = [4096]} : (tensor<1x1x4096xf16>) -> tensor<1x1x1xf16>
  return %0 : tensor<1x1x1xf16>

  // CHECK-LABEL: func.func @lppool_f16_large_window
  // CHECK: [[POOL:%.+]] = "onnx.LpPool"(%arg0)
  // CHECK-NOT: onnx.AveragePool
  // CHECK-NOT: onnx.Sqrt
  // CHECK: return [[POOL]] : tensor<1x1x1xf16>
}

// -----

// The guard covers f32 as well, for every p.
func.func @lppool_f32_p2(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x4x4x4xf32> {
  %0 = "onnx.LpPool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], p = 2 : si64, strides = [2, 2]} : (tensor<1x4x8x8xf32>) -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>

  // CHECK-LABEL: func.func @lppool_f32_p2
  // CHECK: [[POOL:%.+]] = "onnx.LpPool"(%arg0)
  // CHECK-NOT: onnx.AveragePool
  // CHECK: return [[POOL]] : tensor<1x4x4x4xf32>
}

// -----

func.func @lppool_f32_p3(%arg0: tensor<?x1x4x4xf32>) -> tensor<?x1x2x2xf32> {
  %0 = "onnx.LpPool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], p = 3 : si64, strides = [2, 2]} : (tensor<?x1x4x4xf32>) -> tensor<?x1x2x2xf32>
  return %0 : tensor<?x1x2x2xf32>

  // CHECK-LABEL: func.func @lppool_f32_p3
  // CHECK: [[POOL:%.+]] = "onnx.LpPool"(%arg0)
  // CHECK-NOT: onnx.Pow
  // CHECK: return [[POOL]] : tensor<?x1x2x2xf32>
}
