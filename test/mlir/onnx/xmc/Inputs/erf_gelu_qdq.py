import pathlib
import subprocess
import sys


optimizer, fixture = sys.argv[1:]
source = pathlib.Path(fixture).read_text()
pipeline = [
    "--dq-binary-q-opt",
    "--convert-qdq-to-requantize",
    "--quant-types",
    "--replace-erf-to-gelu",
]


def optimize(model, passes):
    return subprocess.run(
        [optimizer, *passes], input=model, text=True, capture_output=True, check=True
    ).stdout


def check(name, model, preserve, fuse):
    folded = optimize(model, pipeline[:1])
    expected_muls = 2 if preserve else 1
    assert folded.count('"onnx.Mul"') == expected_muls, (name, folded)
    result = optimize(model, pipeline)
    assert ('"onnx.Gelu"' in result) == fuse, (name, result)
    if fuse:
        assert '"onnx.Erf"' not in result, (name, result)
        assert '"onnx.Mul"' not in result, (name, result)
        assert "0.019999999552965164:128" in result, (name, result)
    print(name + ": PASS")


check("encoder", source, True, True)
quantized_divisor = source.replace(
    "%sqrt2 = onnx.Constant dense<1.41421356> : tensor<f32>",
    """%sqrt2_code = onnx.Constant dense<91> : tensor<i8>
  %sqrt2_scale = onnx.Constant dense<0.015625> : tensor<f32>
  %sqrt2_zp = onnx.Constant dense<0> : tensor<i8>
  %sqrt2 = "onnx.DequantizeLinear"(%sqrt2_code, %sqrt2_scale, %sqrt2_zp) : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>""",
)
check("quantized_divisor", quantized_divisor, True, True)
check(
    "quantized_divisor_nonzero_zp",
    quantized_divisor.replace("dense<91>", "dense<96>").replace(
        "dense<0> : tensor<i8>", "dense<5> : tensor<i8>"
    ),
    True,
    True,
)
check(
    "wrong_quantized_divisor",
    quantized_divisor.replace("dense<91>", "dense<90>"),
    False,
    False,
)
check(
    "wrong_quantized_divisor_scale",
    quantized_divisor.replace("dense<0.015625>", "dense<0.03125>"),
    False,
    False,
)
check(
    "commuted_operands",
    source.replace("(%erf_dq, %one)", "(%one, %erf_dq)")
    .replace(
        "(tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>\n  %mul",
        "(tensor<f32>, tensor<1x4xf32>) -> tensor<1x4xf32>\n  %mul",
    )
    .replace("(%x, %add)", "(%add, %x)")
    .replace("(%mul, %half)", "(%half, %mul)")
    .replace(
        "(tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>\n  %output",
        "(tensor<f32>, tensor<1x4xf32>) -> tensor<1x4xf32>\n  %output",
    ),
    True,
    True,
)
check("wrong_divisor", source.replace("dense<1.41421356>", "dense<2.0>"), False, False)
check("wrong_offset", source.replace("dense<1.0>", "dense<2.0>"), False, False)
check("wrong_tail", source.replace("dense<0.5>", "dense<0.25>"), False, False)
check(
    "different_input",
    source.replace("(%x, %add)", "(%div_dq, %add)"),
    False,
    False,
)
check(
    "mismatched_qdq",
    source.replace("(%erf_q, %erf_scale, %zp)", "(%erf_q, %input_scale, %zp)"),
    False,
    False,
)
