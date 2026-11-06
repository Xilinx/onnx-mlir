// RUN: onnx-mlir-opt --unsafe-math-optimizations=true --decompose-onnx %s -split-input-file | FileCheck %s

// p = 2: Mul(x, x) -> AveragePool(count_include_pad=1) -> Mul(kernel volume) -> Sqrt.
func.func @lppool_p2(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x4x4x4xf32> {
  %0 = "onnx.LpPool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], p = 2 : si64, strides = [2, 2]} : (tensor<1x4x8x8xf32>) -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>

  // CHECK-LABEL: func.func @lppool_p2
  // CHECK-DAG: [[SCALE:%.+]] = onnx.Constant dense<4.000000e+00> : tensor<f32>
  // CHECK: [[SQ:%.+]] = "onnx.Mul"(%arg0, %arg0) : (tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
  // CHECK: [[POOL:%.+]] = "onnx.AveragePool"([[SQ]])
  // CHECK-SAME: count_include_pad = 1 : si64
  // CHECK-SAME: kernel_shape = [2, 2]
  // CHECK-SAME: strides = [2, 2]
  // CHECK-SAME: -> tensor<1x4x4x4xf32>
  // CHECK: [[SCALED:%.+]] = "onnx.Mul"([[POOL]], [[SCALE]]) : (tensor<1x4x4x4xf32>, tensor<f32>) -> tensor<1x4x4x4xf32>
  // CHECK: [[SQRT:%.+]] = "onnx.Sqrt"([[SCALED]]) : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  // CHECK: return [[SQRT]]
}

// -----

// p = 1 with padding: Abs -> AveragePool(count_include_pad=1) -> Mul(kernel volume).
func.func @lppool_p1_pads(%arg0: tensor<1x2x5x5xf32>) -> tensor<1x2x5x5xf32> {
  %0 = "onnx.LpPool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], p = 1 : si64, pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x2x5x5xf32>) -> tensor<1x2x5x5xf32>
  return %0 : tensor<1x2x5x5xf32>

  // CHECK-LABEL: func.func @lppool_p1_pads
  // CHECK-NOT: onnx.LpPool
  // CHECK-DAG: [[SCALE:%.+]] = onnx.Constant dense<9.000000e+00> : tensor<f32>
  // CHECK: [[ABS:%.+]] = "onnx.Abs"(%arg0) : (tensor<1x2x5x5xf32>) -> tensor<1x2x5x5xf32>
  // CHECK: [[POOL:%.+]] = "onnx.AveragePool"([[ABS]])
  // CHECK-SAME: count_include_pad = 1 : si64
  // CHECK-SAME: kernel_shape = [3, 3]
  // CHECK-SAME: pads = [1, 1, 1, 1]
  // CHECK-SAME: -> tensor<1x2x5x5xf32>
  // CHECK: [[SCALED:%.+]] = "onnx.Mul"([[POOL]], [[SCALE]]) : (tensor<1x2x5x5xf32>, tensor<f32>) -> tensor<1x2x5x5xf32>
  // CHECK-NOT: onnx.Sqrt
  // CHECK-NOT: onnx.Pow
  // CHECK: return [[SCALED]]
}

// -----

// Odd p keeps Abs and uses Pow for the exponent and its reciprocal.
func.func @lppool_p3(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
  %0 = "onnx.LpPool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], p = 3 : si64, strides = [2, 2]} : (tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32>
  return %0 : tensor<1x1x2x2xf32>

  // CHECK-LABEL: func.func @lppool_p3
  // CHECK-NOT: onnx.LpPool
  // CHECK: onnx.Abs
  // CHECK: onnx.Pow
  // CHECK: [[POOL:%.+]] = "onnx.AveragePool"
  // CHECK-SAME: count_include_pad = 1
  // CHECK: onnx.Pow
  // CHECK: return
}

// -----

// ceil_mode can end a window past the input and the declared pads. AveragePool
// drops those positions from its divisor, so the full kernel volume is wrong
// (this input would become [3, 3] instead of LpPool's [3, 2]). Leave LpPool.
func.func @lppool_ceil_partial_window(%arg0: tensor<1x1x4xf32>) -> tensor<1x1x2xf32> {
  %0 = "onnx.LpPool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, kernel_shape = [3], p = 1 : si64, strides = [2]} : (tensor<1x1x4xf32>) -> tensor<1x1x2xf32>
  return %0 : tensor<1x1x2xf32>

  // CHECK-LABEL: func.func @lppool_ceil_partial_window
  // CHECK: [[POOL:%.+]] = "onnx.LpPool"(%arg0)
  // CHECK-SAME: ceil_mode = 1 : si64
  // CHECK-SAME: -> tensor<1x1x2xf32>
  // CHECK-NOT: onnx.AveragePool
  // CHECK-NOT: onnx.Mul
  // CHECK: return [[POOL]] : tensor<1x1x2xf32>
}

// -----

// p > 2 with a ranked dynamic shape. Pow must keep that rank; the untyped Pow
// builder would produce tensor<*xf32> and the return would no longer match.
func.func @lppool_p3_dynamic(%arg0: tensor<?x1x4x4xf32>) -> tensor<?x1x2x2xf32> {
  %0 = "onnx.LpPool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], p = 3 : si64, strides = [2, 2]} : (tensor<?x1x4x4xf32>) -> tensor<?x1x2x2xf32>
  return %0 : tensor<?x1x2x2xf32>

  // CHECK-LABEL: func.func @lppool_p3_dynamic
  // CHECK-NOT: onnx.LpPool
  // CHECK: [[ABS:%.+]] = "onnx.Abs"(%arg0) : (tensor<?x1x4x4xf32>) -> tensor<?x1x4x4xf32>
  // CHECK: [[POW:%.+]] = "onnx.Pow"([[ABS]], {{%.+}}) : (tensor<?x1x4x4xf32>, tensor<f32>) -> tensor<?x1x4x4xf32>
  // CHECK: [[POOL:%.+]] = "onnx.AveragePool"([[POW]])
  // CHECK-SAME: count_include_pad = 1 : si64
  // CHECK-SAME: -> tensor<?x1x2x2xf32>
  // CHECK: [[SCALED:%.+]] = "onnx.Mul"([[POOL]], {{%.+}}) : (tensor<?x1x2x2xf32>, tensor<f32>) -> tensor<?x1x2x2xf32>
  // CHECK: [[ROOT:%.+]] = "onnx.Pow"([[SCALED]], {{%.+}}) : (tensor<?x1x2x2xf32>, tensor<f32>) -> tensor<?x1x2x2xf32>
  // CHECK: return [[ROOT]] : tensor<?x1x2x2xf32>
}
