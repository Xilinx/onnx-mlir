// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @invalid_direction(%x: tensor<1x2x2xf32>, %w: tensor<1x12x2xf32>,
    %r: tensor<1x12x3xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{direction attribute must be one of the strings: forward, reverse, and bidirectional}}
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none)
      {direction = "sideways", hidden_size = 3 : si64}
      : (tensor<1x2x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         none, none, none, none, none) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @unsupported_activation(%x: tensor<1x2x2xf32>, %w: tensor<1x12x2xf32>,
    %r: tensor<1x12x3xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{unsupported LSTM activation}}
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none)
      {activations = ["NotAnActivation"], hidden_size = 3 : si64}
      : (tensor<1x2x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         none, none, none, none, none) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @invalid_optional_shape(%x: tensor<1x2x2xf32>,
    %w: tensor<1x12x2xf32>, %r: tensor<1x12x3xf32>, %b: tensor<1x23xf32>)
    -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{B dimension 1 must be 24}}
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %b, %none, %none, %none, %none)
      {hidden_size = 3 : si64}
      : (tensor<1x2x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         tensor<1x23xf32>, none, none, none, none)
      -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @negative_clip(%x: tensor<1x2x2xf32>, %w: tensor<1x12x2xf32>,
    %r: tensor<1x12x3xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{clip must be non-negative}}
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none)
      {clip = -1.0 : f32, hidden_size = 3 : si64}
      : (tensor<1x2x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         none, none, none, none, none) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @invalid_sequence_lens(%x: tensor<2x2x2xf32>,
    %w: tensor<1x12x2xf32>, %r: tensor<1x12x3xf32>, %lens: tensor<1xi32>)
    -> tensor<2x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{sequence_lens length must equal the batch size}}
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %lens, %none, %none, %none)
      {hidden_size = 3 : si64}
      : (tensor<2x2x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         none, tensor<1xi32>, none, none, none)
      -> (tensor<2x1x2x3xf32>, none, none)
  return %y : tensor<2x1x2x3xf32>
}

// -----

func.func @invalid_layout(%x: tensor<1x2x2xf32>, %w: tensor<1x12x2xf32>, %r: tensor<1x12x3xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{layout must be 0 or 1}}
  %y, %h, %c = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none) {layout = 2 : si64} : (tensor<1x2x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @invalid_input_forget(%x: tensor<1x2x2xf32>, %w: tensor<1x12x2xf32>, %r: tensor<1x12x3xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{input_forget must be 0 or 1}}
  %y, %h, %c = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none) {input_forget = 2 : si64} : (tensor<1x2x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @invalid_mandatory_ranks(%x: tensor<1x2xf32>, %w: tensor<1x12x2xf32>, %r: tensor<1x12x3xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{The first input tensor must have rank 3}}
  %y, %h, %c = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none) : (tensor<1x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @invalid_w_rank(%x: tensor<1x2x2xf32>, %w: tensor<12x2xf32>, %r: tensor<1x12x3xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{The second input tensor must have rank 3}}
  %y, %h, %c = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none) : (tensor<1x2x2xf32>, tensor<12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @invalid_r_rank(%x: tensor<1x2x2xf32>, %w: tensor<1x12x2xf32>, %r: tensor<12x3xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{The third input tensor must have rank 3}}
  %y, %h, %c = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none) : (tensor<1x2x2xf32>, tensor<1x12x2xf32>, tensor<12x3xf32>, none, none, none, none, none) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @invalid_hidden_size(%x: tensor<1x2x2xf32>, %w: tensor<1x12x2xf32>, %r: tensor<1x12x3xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{hidden_size must agree with R}}
  %y, %h, %c = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none) {hidden_size = 2 : si64} : (tensor<1x2x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @invalid_optional_rank(%x: tensor<1x2x2xf32>, %w: tensor<1x12x2xf32>, %r: tensor<1x12x3xf32>, %p: tensor<9xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{P must have rank 2}}
  %y, %h, %c = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %p) : (tensor<1x2x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, tensor<9xf32>) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @invalid_initial_h(%x: tensor<1x2x2xf32>, %w: tensor<1x12x2xf32>, %r: tensor<1x12x3xf32>, %h: tensor<1x2x2xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{initial_h dimension 2 must be 3}}
  %y, %yh, %c = "onnx.LSTM"(%x, %w, %r, %none, %none, %h, %none, %none) : (tensor<1x2x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, tensor<1x2x2xf32>, none, none) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @invalid_initial_c(%x: tensor<1x2x2xf32>, %w: tensor<1x12x2xf32>, %r: tensor<1x12x3xf32>, %c: tensor<2x2x3xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{initial_c dimension 0 must be 1}}
  %y, %h, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %c, %none) : (tensor<1x2x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, tensor<2x2x3xf32>, none) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @invalid_sequence_lens_rank(%x: tensor<2x2x2xf32>, %w: tensor<1x12x2xf32>, %r: tensor<1x12x3xf32>, %lens: tensor<1x2xi32>) -> tensor<2x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{sequence_lens must be rank 1}}
  %y, %h, %c = "onnx.LSTM"(%x, %w, %r, %none, %lens, %none, %none, %none) : (tensor<2x2x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, tensor<1x2xi32>, none, none, none) -> (tensor<2x1x2x3xf32>, none, none)
  return %y : tensor<2x1x2x3xf32>
}

// -----

func.func @invalid_w_shape(%x: tensor<1x2x2xf32>, %w: tensor<2x12x2xf32>, %r: tensor<1x12x3xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{W dimension 0 must be 1}}
  %y, %h, %c = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none) : (tensor<1x2x2xf32>, tensor<2x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @invalid_r_shape(%x: tensor<1x2x2xf32>, %w: tensor<1x12x2xf32>, %r: tensor<1x11x3xf32>) -> tensor<1x1x2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{R dimension 1 must be 12}}
  %y, %h, %c = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none) : (tensor<1x2x2xf32>, tensor<1x12x2xf32>, tensor<1x11x3xf32>, none, none, none, none, none) -> (tensor<1x1x2x3xf32>, none, none)
  return %y : tensor<1x1x2x3xf32>
}

// -----

func.func @nonpositive_hidden_size(%x: tensor<1x2x2xf32>, %w: tensor<1x0x2xf32>, %r: tensor<1x0x0xf32>) -> tensor<1x1x2x0xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{hidden_size must be positive}}
  %y, %h, %c = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none) {hidden_size = 0 : si64} : (tensor<1x2x2xf32>, tensor<1x0x2xf32>, tensor<1x0x0xf32>, none, none, none, none, none) -> (tensor<1x1x2x0xf32>, none, none)
  return %y : tensor<1x1x2x0xf32>
}
