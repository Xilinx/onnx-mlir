import pathlib
import subprocess
import sys


optimizer, fixture = sys.argv[1:]
source = pathlib.Path(fixture).read_text()
pipeline = [
    "--replace-erf-to-gelu",
    "--dq-binary-q-opt",
    "--convert-qdq-to-requantize",
    "--quant-types",
    "--replace-erf-to-gelu",
]


def optimize(model, passes):
    return subprocess.run(
        [optimizer, *passes], input=model, text=True, capture_output=True, check=True
    ).stdout


def check(name, model, fuse):
    recomposed = optimize(model, pipeline[:1])
    assert ('"onnx.Gelu"' in recomposed) == fuse, (name, recomposed)
    folded = optimize(model, pipeline[:2])
    expected_muls = 0 if fuse else 1
    assert folded.count('"onnx.Mul"') == expected_muls, (name, folded)
    if not fuse:
        assert "dense<4.000000e-02>" in folded, (name, folded)
    result = optimize(model, pipeline)
    assert ('"onnx.Gelu"' in result) == fuse, (name, result)
    if fuse:
        assert '"onnx.Erf"' not in result, (name, result)
        assert '"onnx.Mul"' not in result, (name, result)
        assert "0.019999999552965164:128" in result, (name, result)
    full = optimize(model, ["--xmc-passes"])
    assert ('type = "GELU"' in full) == fuse, (name, full)
    print(name + ": PASS")


check("encoder", source, True)
input_dq = next(line for line in source.splitlines() if line.startswith("  %x = "))
for name, scale, fuse in [
    ("duplicated_input_dq", "%input_scale", True),
    ("different_input_dq_scale", "%div_scale", False),
]:
    duplicate_dq = input_dq.replace("%x =", "%x_copy =").replace("%input_scale", scale)
    model = source.replace(input_dq, input_dq + "\n" + duplicate_dq).replace(
        "(%x, %add)", "(%x_copy, %add)"
    )
    result = optimize(model, ["--xmc-passes"])
    assert ('type = "GELU"' in result) == fuse, (name, result)
    assert ('"onnx.Erf"' not in result) == fuse, (name, result)
    if fuse:
        assert '"onnx.Mul"' not in result, (name, result)
        assert "0.019999999552965164:128" in result, (name, result)
    else:
        assert result.count('"onnx.Mul"') == 1, (name, result)
        assert "0.039999999105930328:128" in result, (name, result)
    print(name + ": PASS")

for name, code, scale, zero_point, dtype, fuse in [
    ("quantized_divisor", 91, 0.015625, 0, "i8", True),
    ("nonzero_zp", 96, 0.015625, 5, "i8", True),
    ("wrong_code", 90, 0.015625, 0, "i8", False),
    ("wrong_scale", 91, 0.03125, 0, "i8", False),
    ("unsigned_code", 219, 0.015625, 128, "ui8", True),
    ("negative_code", -37, 0.015625, -128, "i8", True),
]:
    model = source.replace(
        "%sqrt2 = onnx.Constant dense<1.41421356> : tensor<f32>",
        f"""%sqrt2_code = onnx.Constant dense<{code}> : tensor<{dtype}>
  %sqrt2_scale = onnx.Constant dense<{scale}> : tensor<f32>
  %sqrt2_zp = onnx.Constant dense<{zero_point}> : tensor<{dtype}>
  %sqrt2 = "onnx.DequantizeLinear"(%sqrt2_code, %sqrt2_scale, %sqrt2_zp) : (tensor<{dtype}>, tensor<f32>, tensor<{dtype}>) -> tensor<f32>""",
    )
    check(name, model, fuse)
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
)
check("wrong_divisor", source.replace("dense<1.41421356>", "dense<2.0>"), False)
check("wrong_offset", source.replace("dense<1.0>", "dense<2.0>"), False)
check(
    "different_input",
    source.replace("(%x, %add)", "(%div_dq, %add)"),
    False,
)
check(
    "mismatched_qdq",
    source.replace("(%erf_q, %erf_scale, %zp)", "(%erf_q, %input_scale, %zp)"),
    False,
)

check(
    "mismatched_qdq_zero_point",
    source.replace(
        "%zp = onnx.Constant dense<128> : tensor<ui8>",
        "%zp = onnx.Constant dense<128> : tensor<ui8>\n"
        "  %other_zp = onnx.Constant dense<127> : tensor<ui8>",
    ).replace("(%erf_q, %erf_scale, %zp)", "(%erf_q, %erf_scale, %other_zp)"),
    False,
)

for shared in ["div", "div_q", "div_dq", "erf", "erf_q", "erf_dq", "add", "mul"]:
    shared_type = "tensor<1x4xui8>" if shared.endswith("_q") else "tensor<1x4xf32>"
    shared_model = source.replace(
        "-> tensor<1x4xui8> {", f"-> (tensor<1x4xui8>, {shared_type}) {{"
    ).replace(
        "return %output : tensor<1x4xui8>",
        f"return %output, %{shared} : tensor<1x4xui8>, {shared_type}",
    )
    check("shared_" + shared, shared_model, False)

wrong_tail = source.replace("dense<0.5>", "dense<0.25>")
assert '"onnx.Gelu"' not in optimize(wrong_tail, pipeline[:1])
folded = optimize(wrong_tail, pipeline[:2])
assert folded.count('"onnx.Mul"') == 1, folded
assert "dense<8.000000e-02>" in folded, folded
assert 'type = "GELU"' not in optimize(wrong_tail, ["--xmc-passes"])

no_output_q = (
    source.replace("-> tensor<1x4xui8> {", "-> tensor<1x4xf32> {")
    .replace(
        '  %output = "onnx.QuantizeLinear"(%tail, %output_scale, %zp) : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>\n',
        "",
    )
    .replace("return %output : tensor<1x4xui8>", "return %tail : tensor<1x4xf32>")
)
assert '"onnx.Gelu"' not in optimize(no_output_q, pipeline[:1])
assert 'type = "GELU"' not in optimize(no_output_q, ["--xmc-passes"])

shared_tail = source.replace(
    "-> tensor<1x4xui8> {", "-> (tensor<1x4xui8>, tensor<1x4xf32>) {"
).replace(
    "return %output : tensor<1x4xui8>",
    "return %output, %tail : tensor<1x4xui8>, tensor<1x4xf32>",
)
result = optimize(shared_tail, pipeline[:2])
assert '"onnx.Gelu"' not in result, result
assert result.count('"onnx.Mul"') == 2, result
assert "dense<2.000000e-02>" in result, result
