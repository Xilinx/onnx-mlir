<!--- SPDX-License-Identifier: Apache-2.0 -->

# Define and Use Command-line Options for ONNX-MLIR

Command-line options can be used to alter the default behavior of onnx-mlir, or onnx-mlir-opt, and help user experimenting, debugging or performance tuning. We implemented command-line in ONNX-MLIR based on the command-line utility provided by LLVM. We did not define `Option` or `ListOption` with MLIR pass classes(see discussion). 
 
## Organize Options
Refer [llvm document](https://llvm.org/docs/CommandLine.html) for basic idea of how to define an option. In ONNX-MLIR, options are put into groups (`llvm::cl::OptionCategory`). All command-line options for onnx-mlir are in the `OnnxMlirOptions` group.

## Code structure
Command-line options should be placed in `src/Compiler/CompilerOptions.cpp` and declared in `src/Compiler/CompilerOptions.hpp`.

## Define an option
- Add a declaration of the option in `src/Compiler/CompilerOptions.hpp`
- In `src/Compiler/CompilerOptions.cpp`, define the option
- Do **not** include `src/Compiler/CompilerOptions.hpp` in new source files; it should only be used in the onnx-mlir and onnn-mlir-opt command-line tools.
- Do create 'Pass Options' to pass information to specific passses and transformations

## Define an option local to a transformation
Use MLIR's Pass Options to configure passes.

## XMC early DQ deduplication

Set `ONNX_MLIR_ENABLE_EARLY_DEDUP_DQS=1` for a compilation process to run the
existing `DedupDQs` pass before early Erf-to-GELU recomposition in the XMC
pipeline. This allows equivalent input DQs cloned by ORT to share a value before
GELU matching and scalar folding. The setting applies to both the command-line
tools and embedded callers of the shared XMC pipeline.

This extra invocation is disabled by default; only the exact value `1` enables
it. Unset or other values preserve the previous pipeline. The existing later
`DedupDQs` invocation remains unchanged. Set the variable only for the affected
model's compilation process, rather than globally, to leave other models on the
default behavior.
