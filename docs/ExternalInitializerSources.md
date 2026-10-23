<!--- SPDX-License-Identifier: Apache-2.0 -->
<!--- Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates -->

# Initializer Sources

This document describes `ONNXInitializerSourceOp`, which represents an ONNX
initializer without reading its data during model import. The initializer may
be stored in the model protobuf or in an external data file.

## Motivation

ONNX initializers are normally decoded into `ElementsAttr` values during model
import. After this conversion, the compiler cannot distinguish an initializer
from a constant produced by a compiler transformation. Import must also read the
external data before the compiler knows whether its contents are needed.

`ONNXInitializerSourceOp` preserves the initializer name and durable storage
description until an explicit materialization request. This allows a compiler
to keep initializers available for later processing without constructing their
MLIR element values during import.

This feature is experimental and disabled by default. When enabled, it applies
to supported dense tensors in the graph initializer list, including both
`TensorProto::EXTERNAL` tensors and tensors stored in the model protobuf.

## ONNXInitializerSourceOp

The operation produces a ranked tensor value. For example:

```mlir
%0 = "onnx.InitializerSource"() {
  kind = "external_data",
  length = 64 : i64,
  name = "weight",
  offset = 0 : i64,
  path = "/models/weights.bin",
  scope = []
} : () -> tensor<4x4xf32>
```

The operation contains an explicit source kind, the ONNX graph scope and
initializer name, and an absolute path. Its result type records the tensor's
element type and shape. The scope is an array of nested graph names; it is empty
for the outermost graph. Keeping the components separate avoids assigning
separator semantics to ONNX names. An `external_data` source also contains its
byte offset and length.
For an ONNX Runtime `_ORT_MEM_ADDR_` initializer, an `onnx_model` source records
the path of a backup copy of the original ONNX model. This backup model has
ordinary durable tensor storage instead of process-local addresses. After the
IR is serialized and the live memory is unavailable, materialization finds the
named initializer in that model.

The verifier checks the source fields, result element type, tensor shape, and
the payload length of direct external ranges. A new ONNX data type must have
explicit decoder support before it can be used by this operation.

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
in only to test the new import path. The pass reads direct external sources with
the existing cache and numerical decoders. For a model source, it first uses a
live borrowed `TensorProto` when one is available. Otherwise, it caches the
parsed backup model, finds the named `TensorProto`, and decodes its embedded data
or follows its external-data metadata. It replaces each source with an
`ONNXConstantOp` containing a `DisposableElementsAttr`. Other transformations
can use `InitializerSourceMaterializer` to perform the same conversion for one
source. The materializer keeps the parsed backup-model cache across calls.

Some ONNX operands contain values needed for compilation, such as the `shape`
operand of `Reshape`. Generated ONNX operations identify these operands through
`StructuralOperandOpInterface`. The materialization pass consumes a directly
connected initializer source in such a position even when the source would
otherwise be skipped. This classification describes the operand role; the
operand is still allowed to be a runtime value or an ordinary constant.

The following options can leave source operations in the module for testing:

- `skipAllInitializerMaterialization` leaves all sources unchanged.
- `skipInitializerMaterialization` leaves the named sources unchanged.
- `forceInitializerMaterialization` materializes named sources even when one of
  the skip options applies. It is an escape hatch for downstream code that has
  not yet adopted initializer sources.

An outermost initializer is selected by its unmodified ONNX name, for example
`weight`. A nested initializer uses MLIR nested symbol-reference syntax, for
example `@graph::@nested_graph::@weight`. MLIR quoting makes each component
unambiguous, such as `@"graph/with/slash"::@weight`. An exact outermost name is
checked before parsing a nested symbol reference, so an outermost name beginning
with `@` remains selectable without modification. An unmatched name produces a
warning. Skip-all makes the skip-name list redundant; force-name still
materializes a selected source.

The symbol-reference spelling is a selector syntax, not a complete structural
path through an ONNX protobuf. The current scope records only each nested
`GraphProto::name()`. It has these limitations:

- Every graph name in the path must identify one graph-valued node attribute.
  If sibling graph attributes have the same graph name, the importer reports an
  ambiguity before creating sources in either graph. A future selector syntax
  may add node and attribute identity for this case.
- An initializer in a graph nested inside a model-local `FunctionProto` cannot
  be resolved from a backup model using this path. The importer may expand the
  same function definition at several call sites, producing several MLIR
  occurrences with the same selector. Source-enabled import rejects such an
  initializer until function and graph-attribute provenance is represented.
- If an outermost initializer name is also a valid spelling of a nested symbol
  reference, exact outermost-name matching wins. The nested initializer cannot
  be selected individually with that spelling.

MLIR quoting prevents separators inside an ONNX name from being misparsed, but
it cannot supply the missing node, attribute, graph-list, or function identity.

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
`ImportOptions::externalInitializerBackups` using an `InitializerIdentity`
containing the graph-scope components and initializer name.
The entry contains:

- a non-owning, read-only `llvm::ArrayRef<char>` over caller-owned data;
- the path to the backup ONNX model.

The `_ORT_MEM_ADDR_` marker does not define an on-disk byte range. Import
therefore creates an `onnx_model` source containing the absolute backup-model
path and the initializer's structured identity. After assembly or bytecode
reload, materialization finds the saved `TensorProto` and either decodes its
embedded data or follows its ordinary `TensorProto::EXTERNAL` metadata.

Import checks the address and size metadata without reading the caller-owned
data. The caller must ensure that the data and the named tensor in the backup
model contain the same immutable tensor snapshot.

The optional `#onnx.caller_owned_data` attribute stores a non-owning reference
to raw bytes or an embedded `TensorProto`, not the storage or a recipe for
reconstructing it. The caller must keep that storage alive and bytewise
immutable until the `MLIRContext` passed to the import entry point is destroyed.
Cloning the IR copies the same reference and does not extend its lifetime. This
requirement can be longer than the lifetime of buffers managed by
`DisposableGarbageCollector`.

## Embedded Initializers and Caller Lifetime

All `ImportFrontendModel*` entry points borrow `ImportOptions` and their ordinary
input buffer only for the duration of the call. Process-local initializer-source
references have a longer lifetime. The entry points that take a caller-owned
`ModelProto` retain non-owning pointers to its embedded `TensorProto`
initializers. The caller must keep that model alive and bytewise immutable until
the `MLIRContext` passed to the import call is destroyed.

File and array entry points parse temporary protobuf objects and therefore
create disk-only sources. A file entry point uses its input path as the backup
model path. An array caller must provide `initializerSourceModelPath` when
embedded initializers are selected. ModelProto callers must also provide this
path so assembly and bytecode have a durable fallback after the live pointers
are discarded.

## Assembly and Bytecode

Assembly and bytecode serialize `#onnx.caller_owned_data` as an empty marker.
They do not contain the process-local address, caller-owned data, or tensor
data. Parsing the saved IR creates an empty marker, which behaves like an absent
attribute and makes the operation use its durable metadata. A different process
can therefore load a direct external range or find a named tensor in the backup
model after the original caller-owned data is gone.

External files and backup models must not change while referenced by a
compilation. This is also required by the existing whole-file external-data
cache and parsed-model cache.

## Protobuf Tensor Lookup

An `onnx_model` source retains the backup model path, structured graph path,
initializer name, and result type. It does not retain a byte offset into the
protobuf. Tensor data may be encoded as `raw_data` or typed repeated fields, and
protobuf wire offsets are not a stable interface.

Materialization caches the backup model, finds the named initializer, validates
its data type and shape, and uses the existing tensor decoder. If the located
tensor is external, its location is resolved relative to the backup model. A
second `_ORT_MEM_ADDR_` marker is rejected because it is not a durable fallback.
The same lookup supports initializers that were originally stored in the model
protobuf and `_ORT_MEM_ADDR_` tensors whose durable representation is the named
tensor in the backup model.

## Current Limitations

The new import path handles dense tensors in the model graph's initializer
lists, subject to the nested-scope restrictions above. Tensor-valued node
attributes, including attributes of ONNX `Constant` nodes, continue to use
eager import. Sparse initializers are not handled by this path. Initializers in
graphs nested inside model-local functions and tensor-valued attributes in
those functions cannot be deferred; source-enabled import rejects the former
and imports the latter eagerly.

Evaluation of constant-foldable expressions containing initializer sources,
weights packing, and standalone weights repacking are future work. Structural
materialization currently recognizes only a source directly connected to the
structural operand; it does not yet walk through a foldable expression.

See the generated [ONNX dialect reference](Dialects/onnx.md) for the operation
definition. See
[Constant Propagation for ONNX operations](ConstPropagationPass.md) for the
`DisposableElementsAttr` representation used after materialization.
