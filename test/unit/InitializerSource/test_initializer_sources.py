#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
# affiliates
"""Process-level scenarios for InitializerSourceDriver; no new MLIR bindings."""
import pathlib
import struct
import subprocess
import sys
import tempfile

try:
    import onnx
    from onnx import TensorProto, helper
except ModuleNotFoundError:
    print("SKIP: InitializerSource tests require the onnx Python package")
    sys.exit(77)


def run(driver, *args, error=None):
    result = subprocess.run([driver, *map(str, args)], capture_output=True, text=True)
    if error is None:
        assert result.returncode == 0, (args, result.returncode, result.stderr)
    else:
        assert result.returncode != 0 and error in result.stderr, (args, result.stderr)
    return result.stdout


def fixture(folder, dtype=TensorProto.FLOAT, payload=None, count=4):
    if payload is None:
        payload = struct.pack("<4f", 1.25, -2.5, 3.75, 4.125)
    data = folder / "weights.bin"
    # Two ranges in one backing file, with a nonzero offset.
    data.write_bytes(b"padding!" + payload + payload)
    tensors, nodes, outputs = [], [], []
    for index in range(2):
        name = f"W{index}"
        tensor = TensorProto(
            name=name, data_type=dtype, dims=[count], data_location=TensorProto.EXTERNAL
        )
        for key, value in {
            "location": data.name,
            "offset": str(8 + index * len(payload)),
            "length": str(len(payload)),
        }.items():
            tensor.external_data.add(key=key, value=value)
        tensors.append(tensor)
        nodes.append(helper.make_node("Identity", [name], [f"Y{index}"]))
        outputs.append(helper.make_tensor_value_info(f"Y{index}", dtype, [count]))
    graph = helper.make_graph(nodes, "model", [], outputs, initializer=tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    path = folder / "model.onnx"
    path.write_bytes(model.SerializeToString())
    return path, data, payload


def nested_fixture(folder, ambiguous=False):
    data = folder / "nested.bin"
    data.write_bytes(struct.pack("<2f", 1.0, 2.0))

    def branch(name, offset, output):
        tensor_name = output if ambiguous else "W"
        tensor = TensorProto(
            name=tensor_name,
            data_type=TensorProto.FLOAT,
            dims=[1],
            data_location=TensorProto.EXTERNAL,
        )
        for key, value in {
            "location": data.name,
            "offset": str(offset),
            "length": "4",
        }.items():
            tensor.external_data.add(key=key, value=value)
        node = helper.make_node("Identity", [tensor_name], [output])
        value = helper.make_tensor_value_info(output, TensorProto.FLOAT, [1])
        return helper.make_graph([node], name, [], [value], initializer=[tensor])

    then_graph = branch("branch" if ambiguous else "then/branch", 0, "then_y")
    else_graph = branch("branch" if ambiguous else "else_branch", 4, "else_y")
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    node = helper.make_node(
        "If", ["cond"], ["Y"], then_branch=then_graph, else_branch=else_graph
    )
    graph = helper.make_graph([node], "model", [cond], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    path = folder / ("ambiguous.onnx" if ambiguous else "nested.onnx")
    path.write_bytes(model.SerializeToString())
    return path, data


def model_local_function_fixture(folder):
    data = folder / "function.bin"
    data.write_bytes(struct.pack("<2f", 1.0, 2.0))

    def branch(name, offset, output):
        tensor = TensorProto(
            name="W",
            data_type=TensorProto.FLOAT,
            dims=[1],
            data_location=TensorProto.EXTERNAL,
        )
        for key, value in {
            "location": data.name,
            "offset": str(offset),
            "length": "4",
        }.items():
            tensor.external_data.add(key=key, value=value)
        node = helper.make_node("Identity", ["W"], [output])
        value = helper.make_tensor_value_info(output, TensorProto.FLOAT, [1])
        return helper.make_graph([node], name, [], [value], initializer=[tensor])

    if_node = helper.make_node(
        "If",
        ["cond"],
        ["Y"],
        then_branch=branch("then_branch", 0, "then_y"),
        else_branch=branch("else_branch", 4, "else_y"),
    )
    function = helper.make_function(
        "test.local",
        "WithInitializer",
        ["cond"],
        ["Y"],
        [if_node],
        [helper.make_opsetid("", 18)],
    )
    call = helper.make_node("WithInitializer", ["cond"], ["Y"], domain="test.local")
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([call], "model", [cond], [output])
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("test.local", 1),
        ],
        functions=[function],
    )
    path = folder / "model-local-function.onnx"
    path.write_bytes(model.SerializeToString())
    return path


def embedded_fixture(folder, raw):
    payload = struct.pack("<4f", 1.25, -2.5, 3.75, 4.125)
    values = payload if raw else [1.25, -2.5, 3.75, 4.125]
    tensor = helper.make_tensor("W", TensorProto.FLOAT, [4], values, raw=raw)
    node = helper.make_node("Identity", ["W"], ["Y"])
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    graph = helper.make_graph([node], "model", [], [output], initializer=[tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    encoding = "raw" if raw else "typed"
    path = folder / f"embedded-{encoding}-backup.onnx"
    path.write_bytes(model.SerializeToString())
    live = folder / f"embedded-{encoding}-live.bin"
    live.write_bytes(payload)
    return path, live


def structural_fixture(folder):
    structural_data = folder / "structural.bin"
    structural_data.write_bytes(struct.pack("<4q", 2, 2, 0, 1))
    shape = helper.make_tensor("shape", TensorProto.INT64, [2], [2, 2])
    shape.ClearField("int64_data")
    shape.data_location = TensorProto.EXTERNAL
    shape.external_data.add(key="location", value=structural_data.name)
    shape.external_data.add(key="offset", value="0")
    shape.external_data.add(key="length", value="16")
    axes = TensorProto(
        name="axes",
        data_type=TensorProto.INT64,
        dims=[2],
        data_location=TensorProto.EXTERNAL,
    )
    axes.external_data.add(key="location", value=structural_data.name)
    axes.external_data.add(key="offset", value="16")
    axes.external_data.add(key="length", value="16")
    data = helper.make_tensor("data", TensorProto.FLOAT, [4], [1.0, 2.0, 3.0, 4.0])
    nodes = [
        helper.make_node("Reshape", ["data", "shape"], ["Y"]),
        helper.make_node("Identity", ["shape"], ["S"]),
        helper.make_node("ReduceSum", ["data", "axes"], ["R"], keepdims=0),
        helper.make_node("Identity", ["axes"], ["A"]),
    ]
    outputs = [
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2]),
        helper.make_tensor_value_info("S", TensorProto.INT64, [2]),
        helper.make_tensor_value_info("R", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("A", TensorProto.INT64, [2]),
    ]
    graph = helper.make_graph(
        nodes, "model", [], outputs, initializer=[data, shape, axes]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    path = folder / "structural.onnx"
    path.write_bytes(model.SerializeToString())
    return path


def non_initializer_structural_fixture(folder, constant):
    data = helper.make_tensor("data", TensorProto.FLOAT, [4], [1.0, 2.0, 3.0, 4.0])
    inputs = []
    nodes = []
    if constant:
        shape = helper.make_tensor("shape_value", TensorProto.INT64, [2], [2, 2])
        nodes.append(helper.make_node("Constant", [], ["shape"], value=shape))
    else:
        inputs.append(helper.make_tensor_value_info("shape", TensorProto.INT64, [2]))
    nodes.append(helper.make_node("Reshape", ["data", "shape"], ["Y"]))
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
    graph = helper.make_graph(nodes, "model", inputs, [output], initializer=[data])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    kind = "constant" if constant else "runtime"
    path = folder / f"structural-{kind}.onnx"
    path.write_bytes(model.SerializeToString())
    return path


def loop_structural_fixture(folder):
    structural_data = folder / "loop-structural.bin"
    structural_data.write_bytes(struct.pack("<2q", 2, 2))
    shape = TensorProto(
        name="shape",
        data_type=TensorProto.INT64,
        dims=[2],
        data_location=TensorProto.EXTERNAL,
    )
    shape.external_data.add(key="location", value=structural_data.name)
    shape.external_data.add(key="offset", value="0")
    shape.external_data.add(key="length", value="16")

    body_inputs = [
        helper.make_tensor_value_info("iteration", TensorProto.INT64, []),
        helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
        helper.make_tensor_value_info("shape_in", TensorProto.INT64, [2]),
    ]
    body_outputs = [
        helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
        helper.make_tensor_value_info("shape_out", TensorProto.INT64, [2]),
    ]
    body_data = helper.make_tensor(
        "body_data_value", TensorProto.FLOAT, [4], [1.0, 2.0, 3.0, 4.0]
    )
    body_nodes = [
        helper.make_node("Constant", [], ["body_data"], value=body_data),
        helper.make_node("Reshape", ["body_data", "shape_in"], ["reshaped"]),
        helper.make_node("Identity", ["cond_in"], ["cond_out"]),
        helper.make_node("Identity", ["shape_in"], ["shape_out"]),
    ]
    body = helper.make_graph(body_nodes, "loop_body", body_inputs, body_outputs)
    inputs = [
        helper.make_tensor_value_info("M", TensorProto.INT64, []),
        helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
    ]
    output = helper.make_tensor_value_info("final_shape", TensorProto.INT64, [2])
    loop = helper.make_node("Loop", ["M", "cond", "shape"], ["final_shape"], body=body)
    graph = helper.make_graph([loop], "model", inputs, [output], initializer=[shape])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    path = folder / "loop-structural.onnx"
    path.write_bytes(model.SerializeToString())
    return path


def constants(output):
    return [line for line in output.splitlines() if line.startswith("constant ")]


def main(driver):
    checks = 0
    with tempfile.TemporaryDirectory(prefix="initializer-source-") as temporary:
        folder = pathlib.Path(temporary)
        model, data, payload = fixture(folder)
        content = data.read_bytes()
        data.unlink()
        for entry in ("raw", "preprocessed", "array", "file"):
            out = run(driver, "--input", model, "--entry", entry, "--skip-all")
            assert out.count("source W") == 2 and "live=0" in out
            checks += 1
        run(driver, "--input", model, error="cannot read initializer source")
        checks += 1
        # Guard pages are not readable. Import and both writers must not touch
        # them. mmap guard pages are unavailable on Windows.
        if sys.platform != "win32":
            for fmt in ("assembly", "bytecode"):
                saved = folder / f"guard.{fmt}"
                out = run(
                    driver,
                    "--input",
                    model,
                    "--source=guard",
                    "--skip-all",
                    "--output",
                    saved,
                    "--format",
                    fmt,
                )
                assert out.count("live=1") == 2
                checks += 1
            for entry in ("raw", "preprocessed", "array"):
                out = run(
                    driver,
                    "--input",
                    model,
                    "--source=guard",
                    "--skip-all",
                    "--entry",
                    entry,
                )
                assert out.count("live=1") == 2
                checks += 1
        # ONNX protobuf shape inference operates on TensorProto metadata and
        # does not resolve EXTERNAL payloads. It remains safe with missing data.
        out = run(
            driver,
            "--input",
            model,
            "--entry=preprocessed",
            "--skip-all",
            "--protobuf-shape-inference",
        )
        assert out.count("source W") == 2
        checks += 1
        run(
            driver,
            "--input",
            model,
            "--entry=preprocessed",
            "--skip-all",
            "--opset-conversion",
            error="requires ONNX version conversion disabled",
        )
        checks += 1
        data.write_bytes(content)
        out = run(
            driver,
            "--input",
            model,
            "--skip-name=misspelled",
        )
        assert "source W" not in out and len(constants(out)) == 2
        checks += 1
        out = run(
            driver,
            "--input",
            model,
            "--skip-all",
            "--force-name=misspelled",
        )
        assert out.count("source W") == 2
        checks += 1
        eager = constants(run(driver, "--input", model, "--legacy"))
        default = constants(run(driver, "--input", model))
        assert eager == default and len(default) == 2
        assert all("disposable=1" in value for value in default)
        checks += 1
        selected = run(driver, "--input", model, "--skip-name=W0")
        assert "source W0 live=0" in selected
        assert "source W1" not in selected and len(constants(selected)) == 1
        checks += 1
        # An exact outermost name wins before '@' is interpreted as the start
        # of a nested MLIR symbol reference.
        proto = onnx.load_model(model, load_external_data=False)
        proto.graph.initializer[0].name = "@W0"
        proto.graph.node[0].input[0] = "@W0"
        at_root_model = folder / "at-root.onnx"
        at_root_model.write_bytes(proto.SerializeToString())
        selected = run(driver, "--input", at_root_model, "--skip-name=@W0")
        assert "source @W0 live=0" in selected
        assert "source W1" not in selected and len(constants(selected)) == 1
        checks += 1
        # skip-all is authoritative and does not validate an otherwise unknown
        # skip-name entry.
        out = run(driver, "--input", model, "--skip-all", "--skip-name=does-not-exist")
        assert out.count("source W") == 2
        checks += 1
        # Caller-owned live contents with no model at the advertised backup path.
        # The caller keeps the contents alive across cloning and the second pass.
        missing = folder / "missing.onnx"
        out = run(
            driver,
            "--input",
            model,
            "--source=live",
            "--live-data",
            data,
            "--backup-path",
            missing,
            "--skip-all",
            "--materialize-after",
            "--clone-before-load",
        )
        assert constants(out) == default
        checks += 1
        # No skip => default materialization must use live storage despite absent disk.
        out = run(
            driver,
            "--input",
            model,
            "--source=live",
            "--live-data",
            data,
            "--backup-path",
            missing,
        )
        assert constants(out) == default
        checks += 1
        missing_saved = folder / "missing-model.mlir"
        run(
            driver,
            "--input",
            model,
            "--source=live",
            "--live-data",
            data,
            "--backup-path",
            missing,
            "--skip-all",
            "--output",
            missing_saved,
        )
        run(
            driver,
            "--input",
            missing_saved,
            "--read-ir",
            error="cannot read initializer backup model",
        )
        checks += 1
        if sys.platform != "win32":
            run(
                driver,
                "--input",
                model,
                "--source=guard",
                "--skip-all",
                "--omit-backup",
                error="requires caller-owned data and an explicit backup model",
            )
            checks += 1
        relative = folder / "relative-backup.mlir"
        run(
            driver,
            "--input",
            model,
            "--source=live",
            "--live-data",
            data,
            "--backup-path",
            model.name,
            "--skip-all",
            "--output",
            relative,
        )
        assert constants(run(driver, "--input", relative, "--read-ir")) == default
        checks += 1
        # A memory source may resolve to an initializer embedded in the backup
        # model rather than to external data.
        for raw in (True, False):
            embedded_model, embedded_live = embedded_fixture(folder, raw)
            embedded_expected = constants(
                run(driver, "--input", embedded_model, "--legacy")
            )
            encoding = "raw" if raw else "typed"
            embedded_saved = folder / f"embedded-{encoding}.mlir"
            run(
                driver,
                "--input",
                embedded_model,
                "--source=live",
                "--live-data",
                embedded_live,
                "--skip-all",
                "--output",
                embedded_saved,
            )
            assert (
                constants(run(driver, "--input", embedded_saved, "--read-ir"))
                == embedded_expected
            )
            checks += 1
        # Each run() starts a distinct process. The saved live source cannot rely
        # on a registry, pointer, original protobuf, or original owner on reload.
        cases = [
            (TensorProto.FLOAT, payload, 4),
            (TensorProto.INT64, struct.pack("<4q", -5, 0, 10, 2**40), 4),
            (TensorProto.BOOL, bytes([1, 0, 1, 1]), 4),
            (TensorProto.FLOAT16, struct.pack("<4e", 1, -2, 3, 4), 4),
            (TensorProto.INT4, bytes([0x78, 0x0F, 0x03]), 5),
            (TensorProto.FLOAT, b"", 0),
        ]
        for dtype, raw, count in cases:
            model, data, _ = fixture(folder, dtype, raw, count)
            expected = constants(run(driver, "--input", model, "--legacy"))
            for source in ("file", "live"):
                for fmt in ("assembly", "bytecode"):
                    saved = folder / f"{dtype}-{source}.{fmt}"
                    args = [
                        "--input",
                        model,
                        "--source",
                        source,
                        "--skip-all",
                        "--output",
                        saved,
                        "--format",
                        fmt,
                    ]
                    if source == "live":
                        args += ["--live-data", data]
                    run(driver, *args)
                    encoded = saved.read_bytes()
                    assert b"_ORT_MEM_ADDR_" not in encoded
                    if fmt == "assembly":
                        assert b"dense<" not in encoded
                    else:
                        assert encoded[:4] == b"ML\xefR"
                    # Inspect deserialized source before resolving in another run.
                    disk_only = run(driver, "--input", saved, "--read-ir", "--skip-all")
                    assert disk_only.count("live=0") == 2
                    assert (
                        constants(run(driver, "--input", saved, "--read-ir"))
                        == expected
                    )
                    checks += 1
        model, data, _ = fixture(folder)
        # Embedded initializers follow the same source path as external ones.
        proto = onnx.load_model(model, load_external_data=False)
        proto.graph.initializer.append(
            helper.make_tensor("embedded", TensorProto.FLOAT, [1], [7.0])
        )
        model.write_bytes(proto.SerializeToString())
        data.unlink()
        out = run(driver, "--input", model, "--skip-all")
        assert out.count("source W") == 2
        assert "source embedded live=1" in out and not constants(out)
        checks += 1
        for raw in (True, False):
            embedded_model, _ = embedded_fixture(folder, raw)
            expected = constants(run(driver, "--input", embedded_model, "--legacy"))
            out = run(
                driver,
                "--input",
                embedded_model,
                "--skip-all",
            )
            assert "source W live=1" in out and not constants(out)
            checks += 1
            out = run(
                driver,
                "--input",
                embedded_model,
                "--entry=file",
                "--skip-all",
            )
            assert "source W live=0" in out and not constants(out)
            checks += 1
            for fmt in ("assembly", "bytecode"):
                saved = folder / f"protobuf-{raw}.{fmt}"
                run(
                    driver,
                    "--input",
                    embedded_model,
                    "--skip-all",
                    "--format",
                    fmt,
                    "--output",
                    saved,
                )
                disk_only = run(driver, "--input", saved, "--read-ir", "--skip-all")
                assert "source W live=0" in disk_only
                assert constants(run(driver, "--input", saved, "--read-ir")) == expected
                checks += 1

        # A structural use consumes the source globally, including its ordinary
        # data use, while an unrelated embedded data initializer stays deferred.
        structural_model = structural_fixture(folder)
        out = run(
            driver,
            "--input",
            structural_model,
            "--skip-all",
        )
        assert "source data live=1" in out
        assert "source shape" not in out and "source axes" not in out
        assert len(constants(out)) == 2
        checks += 1
        for constant in (False, True):
            non_initializer_model = non_initializer_structural_fixture(folder, constant)
            out = run(
                driver,
                "--input",
                non_initializer_model,
                "--skip-all",
            )
            assert "source data live=1" in out
            assert len(constants(out)) == int(constant)
            checks += 1
        out = run(
            driver,
            "--input",
            loop_structural_fixture(folder),
            "--skip-all",
        )
        assert "source shape" not in out
        checks += 1
        out = run(
            driver,
            "--input",
            embedded_model,
            "--skip-all",
            "--force-name=W",
        )
        assert "source W" not in out and len(constants(out)) == 1
        checks += 1
        model, data, _ = fixture(folder)
        proto = onnx.load_model(model, load_external_data=False)
        for field in proto.graph.initializer[0].external_data:
            if field.key == "offset":
                field.value = "08"
        model.write_bytes(proto.SerializeToString())
        run(driver, "--input", model, "--skip-all")
        checks += 1
        for field in proto.graph.initializer[0].external_data:
            if field.key == "offset":
                field.value = "0x08"
        model.write_bytes(proto.SerializeToString())
        run(driver, "--input", model, "--skip-all", error="invalid external offset")
        checks += 1
        model, data, _ = fixture(folder)
        proto = onnx.load_model(model, load_external_data=False)
        for field in proto.graph.initializer[0].external_data:
            if field.key == "length":
                field.value = "0"
        model.write_bytes(proto.SerializeToString())
        run(driver, "--input", model, "--skip-all", error="payload length")
        checks += 1
        model, data, _ = fixture(folder)
        proto = onnx.load_model(model, load_external_data=False)
        for field in proto.graph.initializer[0].external_data:
            if field.key == "location":
                field.value = str(data.resolve())
        model.write_bytes(proto.SerializeToString())
        run(
            driver,
            "--input",
            model,
            "--skip-all",
            error="must be a relative path without '..'",
        )
        checks += 1
        model, data, _ = fixture(folder)
        proto = onnx.load_model(model, load_external_data=False)
        for field in proto.graph.initializer[0].external_data:
            if field.key == "location":
                field.value = "../weights.bin"
        model.write_bytes(proto.SerializeToString())
        run(
            driver,
            "--input",
            model,
            "--skip-all",
            error="must be a relative path without '..'",
        )
        checks += 1
        model, data = nested_fixture(folder)
        out = run(driver, "--input", model, "--skip-all")
        assert 'source @"then/branch"::@W' in out
        assert "source @else_branch::@W" in out
        checks += 1
        selected = run(driver, "--input", model, '--skip-name=@"then/branch"::@W')
        assert 'source @"then/branch"::@W' in selected
        assert "source @else_branch::@W" not in selected
        assert len(constants(selected)) == 1
        checks += 1
        nested_expected = constants(run(driver, "--input", model, "--legacy"))
        nested_saved = folder / "nested-live.mlir"
        run(
            driver,
            "--input",
            model,
            "--source=live",
            "--live-data",
            data,
            "--skip-all",
            "--output",
            nested_saved,
        )
        assert (
            constants(run(driver, "--input", nested_saved, "--read-ir"))
            == nested_expected
        )
        checks += 1
        model, data = nested_fixture(folder, ambiguous=True)
        run(
            driver,
            "--input",
            model,
            "--skip-all",
            error="graph-name scope is ambiguous",
        )
        checks += 1
        model = model_local_function_fixture(folder)
        run(driver, "--input", model, "--legacy")
        run(
            driver,
            "--input",
            model,
            "--skip-all",
            error="inside a model-local function",
        )
        checks += 1
        model, data, _ = fixture(folder)
        saved = folder / "renamed-backup.mlir"
        run(
            driver,
            "--input",
            model,
            "--source=live",
            "--live-data",
            data,
            "--skip-all",
            "--output",
            saved,
        )
        proto = onnx.load_model(model, load_external_data=False)
        proto.graph.initializer[0].name = "renamed"
        model.write_bytes(proto.SerializeToString())
        run(
            driver,
            "--input",
            saved,
            "--read-ir",
            error="cannot find initializer in backup model",
        )
        checks += 1
        model, data, _ = fixture(folder)
        saved = folder / "wrong-backup-type.mlir"
        run(
            driver,
            "--input",
            model,
            "--source=live",
            "--live-data",
            data,
            "--skip-all",
            "--output",
            saved,
        )
        proto = onnx.load_model(model, load_external_data=False)
        proto.graph.initializer[0].data_type = TensorProto.INT32
        model.write_bytes(proto.SerializeToString())
        run(
            driver,
            "--input",
            saved,
            "--read-ir",
            error="backup initializer type or shape disagrees with source",
        )
        checks += 1
        model, data, _ = fixture(folder)
        # A valid range spelling with the wrong payload size fails during import.
        proto = onnx.load_model(model, load_external_data=False)
        for field in proto.graph.initializer[0].external_data:
            if field.key == "length":
                field.value = "1"
        model.write_bytes(proto.SerializeToString())
        run(driver, "--input", model, "--skip-all", error="payload length")
        checks += 1
        model, data, _ = fixture(folder)
        content = data.read_bytes()
        data.write_bytes(content[:9])
        run(driver, "--input", model, error="source range exceeds available bytes")
        checks += 1
        model, data, _ = fixture(folder)
        saved = folder / "wrong-type.mlir"
        run(driver, "--input", model, "--skip-all", "--output", saved)
        saved.write_text(
            saved.read_text().replace("tensor<4xf32>", "tensor<4xcomplex<f32>>")
        )
        run(
            driver,
            "--input",
            saved,
            "--read-ir",
            error="unsupported source element type",
        )
        checks += 1
        model, data, _ = fixture(folder)
        saved = folder / "wrong-length.mlir"
        run(driver, "--input", model, "--skip-all", "--output", saved)
        saved.write_text(
            saved.read_text().replace("length = 16 : i64", "length = 8 : i64")
        )
        run(
            driver,
            "--input",
            saved,
            "--read-ir",
            error="source length disagrees with tensor payload size",
        )
        checks += 1
        model, data, _ = fixture(folder, TensorProto.COMPLEX64)
        run(
            driver,
            "--input",
            model,
            "--skip-all",
            error="unsupported external initializer element type",
        )
        checks += 1
        model, data, _ = fixture(folder)
        # Bad metadata fails during import without dereferencing the live source.
        proto = onnx.load_model(model, load_external_data=False)
        for field in proto.graph.initializer[0].external_data:
            if field.key == "offset":
                field.value = "not-an-offset"
        model.write_bytes(proto.SerializeToString())
        run(driver, "--input", model, "--skip-all", error="invalid external offset")
        checks += 1
    print(f"PASS: {checks} initializer-source scenarios")


if __name__ == "__main__":
    main(str(pathlib.Path(sys.argv[1]).resolve()))
