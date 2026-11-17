// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#ifndef DEFERRED_CONSTANT_OP_INTERFACE
#define DEFERRED_CONSTANT_OP_INTERFACE

include "mlir/IR/OpBase.td"

def DeferredConstantOpInterface : OpInterface<"DeferredConstantOpInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    Identifies operation results that have constant semantics but whose
    contents are deliberately unavailable to ordinary folding. Reading the
    contents requires an explicit materialization step.
  }];

  let methods = [
    InterfaceMethod<
        /*description=*/"Return whether the specified result is a deferred "
                        "constant.",
        /*retTy=*/"bool",
        /*methodName=*/"isDeferredConstantResult",
        /*args=*/(ins "unsigned":$index),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{ return true; }]>
  ];
}

#endif // DEFERRED_CONSTANT_OP_INTERFACE
