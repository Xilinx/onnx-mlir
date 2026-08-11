#loc = loc(unknown)
module attributes {
  producer.name = "onnx.quantize",
  vaimlconf.device = "stx",
  vaimlconf.device_batch_size = 1 : i32,
  vaimlconf.dynamic_modes = false,
  vaimlconf.install_dir = "/wrk/xcohdnobkup6/xiaoh/vitis_flexml/build",
  vaimlconf.library_metadata = ["${vaimlconf.install_dir}/../gather_uplift.flexml/custom_ops", "/embedded/libraryMetadata/L1", "/embedded/libraryMetadata/L1", "/embedded/libraryMetadata/L2", "${vaimlconf.install_dir}/../.venv/lib/python3.12/site-packages/vitis_mllib/L1/metadata", "${vaimlconf.install_dir}/../.venv/lib/python3.12/site-packages/vitis_mllib/L2/metadata", "/embedded/L1/metadata", "/embedded/L2/metadata", "/embedded/libraryMetadata/DMAC", "/embedded/tiling-recipe-specs"],
  vaimlconf.overlay_for_TGs = "rai_1x4x4",
  vaimlconf.single_core_compiler = "chess",
  vaimlconf.tp_size = 2 : i32,
  vaimlconf.unified_overlay = "rai_2x4x4.json"} {
  func.func @main_graph(%arg0: tensor<1x64x768xf32> {onnx.name = "/deberta/encoder/layer.11/output/Add_output_0"} loc(unknown)) -> (tensor<1x768xf32> {onnx.name = "/pooler/Gather_output_0_DequantizeLinear_Output"}) {
    %0 = onnx.Constant {ResultNames = ["/deberta/embeddings/word_embeddings/Constant_output_0"]} dense<0> : tensor<i64> loc(#loc1)
    %1 = onnx.Constant {ResultNames = ["/deberta/encoder/layer.11/output/Add_output_0_zero_point"]} dense<30800> : tensor<ui16> loc(#loc2)
    %2 = onnx.Constant {ResultNames = ["/deberta/encoder/layer.11/output/Add_output_0_scale"]} dense<6.42855535E-4> : tensor<f32> loc(#loc3)
    %3 = onnx.Constant {ResultNames = ["deberta.encoder.layer.11.output.LayerNorm.weight_scale"]} dense<0.007463207> : tensor<f32> loc(#loc4)
    %4 = onnx.Constant {ResultNames = ["deberta.encoder.layer.11.output.LayerNorm.weight_zero_point"]} dense<0> : tensor<ui8> loc(#loc5)
    %5 = onnx.Constant {ResultNames = ["deberta.encoder.layer.11.output.LayerNorm.weight_quantized"]} dense_resource<__elided__> : tensor<768xui8> loc(#loc6)
    %6 = onnx.Constant {ResultNames = ["/deberta/encoder/layer.11/output/LayerNorm/LayerNormalization_output_0_zero_point"]} dense<29511> : tensor<ui16> loc(#loc7)
    %7 = onnx.Constant {ResultNames = ["/deberta/encoder/layer.11/output/LayerNorm/LayerNormalization_output_0_scale"]} dense<4.20910917E-4> : tensor<f32> loc(#loc8)
    %8 = onnx.Constant {ResultNames = ["deberta.encoder.layer.11.output.LayerNorm.bias_quantized"]} dense_resource<__elided__> : tensor<768xi32> loc(#loc9)
    %9 = onnx.Constant {ResultNames = ["deberta.encoder.layer.11.output.LayerNorm.bias_quantized_scale"]} dense<4.79776372E-6> : tensor<1xf32> loc(#loc10)
    %10 = onnx.Constant {ResultNames = ["deberta.encoder.layer.11.output.LayerNorm.bias_quantized_zero_point"]} dense<0> : tensor<i32> loc(#loc11)
    %11 = "onnx.DequantizeLinear"(%8, %9, %10) {
      ResultNames = ["deberta.encoder.layer.11.output.LayerNorm.bias"],
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "deberta.encoder.layer.11.output.LayerNorm.bias_DequantizeLinear"} : (tensor<768xi32>, tensor<1xf32>, tensor<i32>) -> tensor<768xf32> loc(#loc12)
    %12 = "onnx.DequantizeLinear"(%5, %3, %4) {
      ResultNames = ["deberta.encoder.layer.11.output.LayerNorm.weight_DequantizeLinear_Output"],
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "deberta.encoder.layer.11.output.LayerNorm.weight_DequantizeLinear"} : (tensor<768xui8>, tensor<f32>, tensor<ui8>) -> tensor<768xf32> loc(#loc13)
    %13 = "onnx.QuantizeLinear"(%arg0, %2, %1) {
      ResultNames = ["/deberta/encoder/layer.11/output/Add_output_0_QuantizeLinear_Output"],
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "/deberta/encoder/layer.11/output/Add_output_0_QuantizeLinear",
      output_dtype = 0 : si64,
      saturate = 1 : si64} : (tensor<1x64x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x64x768xui16> loc(#loc14)
    %14 = "onnx.DequantizeLinear"(%13, %2, %1) {
      ResultNames = ["/deberta/encoder/layer.11/output/Add_output_0_DequantizeLinear_Output"],
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "/deberta/encoder/layer.11/output/Add_output_0_DequantizeLinear"} : (tensor<1x64x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x64x768xf32> loc(#loc15)
    %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%14, %12, %11) {
      ResultNames = ["/deberta/encoder/layer.11/output/LayerNorm/LayerNormalization_output_0", "", ""],
      axis = -1 : si64,
      epsilon = 1.000000e-07 : f32,
      onnx_node_name = "/deberta/encoder/layer.11/output/LayerNorm/LayerNormalization",
      stash_type = 1 : si64} : (tensor<1x64x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x64x768xf32>, none, none) loc(#loc16)
    %15 = "onnx.QuantizeLinear"(%Y, %7, %6) {
      ResultNames = ["/deberta/encoder/layer.11/output/LayerNorm/LayerNormalization_output_0_QuantizeLinear_Output"],
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "/deberta/encoder/layer.11/output/LayerNorm/LayerNormalization_output_0_QuantizeLinear",
      output_dtype = 0 : si64,
      saturate = 1 : si64} : (tensor<1x64x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x64x768xui16> loc(#loc17)
    %16 = "onnx.DequantizeLinear"(%15, %7, %6) {
      ResultNames = ["/deberta/encoder/layer.11/output/LayerNorm/LayerNormalization_output_0_DequantizeLinear_Output"],
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "/deberta/encoder/layer.11/output/LayerNorm/LayerNormalization_output_0_DequantizeLinear"} : (tensor<1x64x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x64x768xf32> loc(#loc18)
    %17 = "onnx.Gather"(%16, %0) {
      ResultNames = ["/pooler/Gather_output_0"],
      axis = 1 : si64,
      onnx_node_name = "/pooler/Gather"} : (tensor<1x64x768xf32>, tensor<i64>) -> tensor<1x768xf32> loc(#loc19)
    %18 = "onnx.QuantizeLinear"(%17, %7, %6) {
      ResultNames = ["/pooler/Gather_output_0_QuantizeLinear_Output"],
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "/pooler/Gather_output_0_QuantizeLinear",
      output_dtype = 0 : si64,
      saturate = 1 : si64} : (tensor<1x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x768xui16> loc(#loc20)
    %19 = "onnx.DequantizeLinear"(%18, %7, %6) {
      ResultNames = ["/pooler/Gather_output_0_DequantizeLinear_Output"],
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "/pooler/Gather_output_0_DequantizeLinear"} : (tensor<1x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x768xf32> loc(#loc21)
    return %19 : tensor<1x768xf32> loc(#loc)
  } loc(#loc)
  "onnx.EntryPoint"() {func = @main_graph} : () -> () loc(#loc)
} loc(#loc)
#loc1 = loc("Initializer_/deberta/embeddings/word_embeddings/Constant_output_0")
#loc2 = loc("Initializer_/deberta/encoder/layer.11/output/Add_output_0_zero_point")
#loc3 = loc("Initializer_/deberta/encoder/layer.11/output/Add_output_0_scale")
#loc4 = loc("Initializer_deberta.encoder.layer.11.output.LayerNorm.weight_scale")
#loc5 = loc("Initializer_deberta.encoder.layer.11.output.LayerNorm.weight_zero_point")
#loc6 = loc("Initializer_deberta.encoder.layer.11.output.LayerNorm.weight_quantized")
#loc7 = loc("Initializer_/deberta/encoder/layer.11/output/LayerNorm/LayerNormalization_output_0_zero_point")
#loc8 = loc("Initializer_/deberta/encoder/layer.11/output/LayerNorm/LayerNormalization_output_0_scale")
#loc9 = loc("Initializer_deberta.encoder.layer.11.output.LayerNorm.bias_quantized")
#loc10 = loc("Initializer_deberta.encoder.layer.11.output.LayerNorm.bias_quantized_scale")
#loc11 = loc("Initializer_deberta.encoder.layer.11.output.LayerNorm.bias_quantized_zero_point")
#loc12 = loc("deberta.encoder.layer.11.output.LayerNorm.bias_DequantizeLinear")
#loc13 = loc("deberta.encoder.layer.11.output.LayerNorm.weight_DequantizeLinear")
#loc14 = loc("/deberta/encoder/layer.11/output/Add_output_0_QuantizeLinear")
#loc15 = loc("/deberta/encoder/layer.11/output/Add_output_0_DequantizeLinear")
#loc16 = loc("/deberta/encoder/layer.11/output/LayerNorm/LayerNormalization")
#loc17 = loc("/deberta/encoder/layer.11/output/LayerNorm/LayerNormalization_output_0_QuantizeLinear")
#loc18 = loc("/deberta/encoder/layer.11/output/LayerNorm/LayerNormalization_output_0_DequantizeLinear")
#loc19 = loc("/pooler/Gather")
#loc20 = loc("/pooler/Gather_output_0_QuantizeLinear")
#loc21 = loc("/pooler/Gather_output_0_DequantizeLinear")
