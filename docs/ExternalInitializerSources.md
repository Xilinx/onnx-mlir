<!--- SPDX-License-Identifier: Apache-2.0 -->
<!--- Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates -->

# External Initializer Sources

This document describes `ONNXInitializerSourceOp`, which represents an external
ONNX initializer without reading its data during model import.

## Motivation

ONNX initializers are normally decoded into `ElementsAttr` values during model
import. After this conversion, the compiler cannot distinguish an initializer
from a constant produced by a compiler transformation. Import must also read the
external data before the compiler knows whether its contents are needed.

`ONNXInitializerSourceOp` preserves the initializer name and external file range
until an explicit materialization pass. This allows a compiler to keep selected
initializers available for later processing while using the existing eager
behavior for the other initializers.

This feature is experimental and disabled by default. It currently applies only
to initializers with `TensorProto::EXTERNAL`. Initializers stored in the model
protobuf continue to use the existing import path.

## ONNXInitializerSourceOp

The operation produces a ranked tensor value. For example:

```mlir
%0 = "onnx.InitializerSource"() {
  source = #onnx.initializer_source<{
    data_type = 1 : i64,
    length = 64 : i64,
    name = "weight",
    offset = 0 : i64,
    path = "/models/weights.bin",
    scope = "main_graph"
  }>
} : () -> tensor<4x4xf32>
```

The source attribute contains:

- an absolute normalized path to the external data file;
- the ONNX graph scope and initializer name;
- the byte offset and length; and
- the ONNX `TensorProto` data type.

The verifier checks the source fields, result element type, tensor shape, and
payload length. A new ONNX data type must have explicit decoder support before
it can be used by this operation.

`ONNXInitializerSourceOp` is `Pure` because the operation does not read the
external data and the source represents an immutable model snapshot. The
operation does not have the `ConstantLike` trait or a fold hook. Folding the
operation would read the data implicitly and would replace the named initializer
with an ordinary compiler constant.

## Import and Materialization

`ImportOptions::useInitializerSources` enables the new import path. The frontend
creates `ONNXInitializerSourceOp` before calling the existing tensor decoder. It
constructs the tensor type and source metadata without opening or mapping the
external data file.

By default, import then runs the initializer-source materialization pass before
returning the module. This preserves the existing behavior for callers that opt
in only to test the new import path. The pass reads each source with the
existing external-data cache and numerical decoders and replaces it with an
`ONNXConstantOp` containing a `DisposableElementsAttr`.

The following options can leave source operations in the module for testing:

- `skipAllInitializerMaterialization` leaves all sources unchanged.
- `skipInitializerMaterialization` leaves the named sources unchanged.

Initializer names have the form `graph[/nested_graph]/initializer`. Names must
match exactly. An unknown name is an error, and the skip-all option takes
precedence over the name list.

Keeping an `ONNXInitializerSourceOp` in the module does not mean that all later
passes support it. Those passes must be updated before the operation can remain
unmaterialized through the complete compiler pipeline.

ONNX protobuf shape inference may run before import. Shape inference does not
resolve external files, and data propagation is disabled by default. ONNX
version conversion is rejected because preservation of initializer sources
through that step has not been verified.

## In-Memory External Data

ONNX Runtime may use the process-local `_ORT_MEM_ADDR_` marker for an external
initializer. In this case, the caller adds an entry to
`ImportOptions::externalInitializerBackups` using the initializer's scoped name.
The entry contains:

- a non-owning, read-only `llvm::ArrayRef<char>` over the live data;
- a path to the disk backup; and
- the backup offset and length.

This disk-range form documents the current prototype, not the intended durable
contract for `_ORT_MEM_ADDR_`. The marker does not define how a raw memory range
maps to an on-disk range. The durable form should instead record the backup ONNX
model and structured initializer identity. Reload can then find the saved
`TensorProto` and either decode its embedded data or follow its ordinary
`TensorProto::EXTERNAL` metadata. Until that lookup is implemented, serialized
ORT-memory sources are an experimental limitation rather than a stable format.

Import checks the address and size metadata without reading the live data. The
caller must ensure that the live data and the disk backup contain the same
immutable tensor snapshot.

The source attribute copies the view, not its storage. The caller must keep that
storage alive and bytewise immutable until the `MLIRContext` passed to the
import entry point is destroyed. Cloning the IR copies the same non-owning view
and does not extend its lifetime. This requirement can be longer than the
lifetime of buffers managed by `DisposableGarbageCollector`.

## Caller Lifetime Requirement

All `ImportFrontendModel*` entry points borrow `ImportOptions` and their ordinary
input buffer or protobuf only for the duration of the call. Process-local source
references have a longer lifetime. A caller that supplies live initializer data
must keep the referenced model snapshot alive and bytewise immutable until the
`MLIRContext` passed to the import call is destroyed. This includes backing data
reached through a future model resolver or retained `const TensorProto *`.

The current external-initializer implementation retains only the live views in
`ImportOptions::externalInitializerBackups`; it does not retain the input
`ModelProto`. Future protobuf-backed sources may retain a caller-owned model,
tensor, or resolver under the same lifetime contract. The file and array import
entry points parse temporary protobuf objects, so they cannot expose pointers
into those objects as deferred sources. They must use a durable disk locator or
a separate long-lived source supplied by the caller.

## Assembly and Bytecode

Assembly and bytecode contain only the external file metadata. They do not
contain the process-local address, live owner, or tensor data. Parsing the saved
IR creates a disk-only source. A different process can therefore load the tensor
from the disk backup after the original live buffer is gone.

The external file must not change while it is referenced by a compilation. This
is also required by the existing whole-file external-data cache.

## Future Protobuf Sources

The source attribute can later distinguish external-data ranges from named
tensors located through a model protobuf. A protobuf source would retain the
backup model path, structured graph path, initializer name, data type, and
result type. The same source kind is the durable fallback for an
`_ORT_MEM_ADDR_` tensor. It would not retain a byte offset into the protobuf.
Tensor data may be encoded as `raw_data` or typed repeated fields, and protobuf
wire offsets are not a stable interface.

Materialization would load the backup model, find the named initializer,
validate its metadata, and use the existing tensor decoder. An optional
process-local view or resolver would remain non-owning and would be omitted from
assembly and bytecode, as it is for external-data sources. A live resolver may
return the existing `const TensorProto *` without reparsing. The caller must
keep the model alive and immutable; import must not retain pointers into
temporary models created by file or array entry points. The resolver and all
returned tensors obey the caller lifetime requirement above.

## Current Limitations

The new import path handles dense tensors in the graph initializer list.
Tensor-valued node attributes, including attributes of ONNX `Constant` nodes,
continue to use eager import. Sparse initializers are not handled by this path.
ONNX model-local functions do not have initializer lists, and their
tensor-valued node attributes also remain eager.

References to initializers stored in the model protobuf, evaluation of
constant-foldable expressions containing initializer sources, weights packing,
and standalone weights repacking are future work.

See the generated [ONNX dialect reference](Dialects/onnx.md) for the operation
definition. See
[Constant Propagation for ONNX operations](ConstPropagationPass.md) for the
`DisposableElementsAttr` representation used after materialization.
