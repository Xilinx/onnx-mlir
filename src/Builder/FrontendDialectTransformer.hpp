/*
 * SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
 */

//===--------- FrontendDialectTransformer.hpp - MLIR Operations -----------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_FRONTEND_TRANSFORMER_H
#define ONNX_MLIR_FRONTEND_TRANSFORMER_H

#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "onnx/onnx_pb.h"

#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"

#include "src/Builder/FrontendDialectHelper.hpp"
#include "src/Support/ErrorHandling.hpp"

namespace mlir {
class MLIRContext;
template <typename OpTy>
class OwningOpRef;
class ModuleOp;
} // namespace mlir

//===----------------------------------------------------------------------===//
// Import a model into the ONNX-MLIR dialect.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

// A caller guarantees that the non-owning live view remains valid and bytewise
// immutable until the MLIRContext passed to ImportFrontendModel* is destroyed.
// The live view and named tensor in the backup model describe the same
// immutable tensor snapshot.
// Live storage starts at the ORT address (not at a whole-model allocation).
struct ExternalInitializerBackup {
  std::string modelPath;
  llvm::ArrayRef<char> live;
};

/*!
 * Options to control the translation of an ONNX model to ONNX-MLIR.
 */
struct ImportOptions {
  bool verboseOutput = false;
  // Use types/shapes in the input-model for translation (for intermediate
  // variables)
  bool useOnnxModelTypes = true;
  bool runOnnxShapeInference = true;
  bool useOnnxModelTypesForCustomOps = true;
  bool invokeOnnxVersionConverter = false;
  bool allowSorting = true;
  bool useOutputNameAsLocation = false;
  bool addResultNamesAttr = false;

  // If true, type inference will be used to
  // infer missing output types. This is done by copying the, potential
  // inferred, output type of the node connected to the output. According to
  // ONNX, all outputs MUST have types. Therefore this option has to be
  // considered as a stretch best effort.
  bool allowMissingOutputTypes = false;

  // Custom shape information for the graph inputs.
  // Its format is 'input_id:dim,dim,dim|input_id:dim,dim,dim'
  // E.g. An ONNX model has two dynamic inputs
  //   - (arg0: tensor<?x?x?xf32>, arg1: tensor<?x5xf32>)
  // If we want to compile the model for static dimensions, we can use:
  //   - shapeInformation = '0:3,4,5|1:10,5'
  // to obtain a model with two staic inputs:
  //   - (arg0: tensor<3x4x5xf32>, arg1: tensor<10x5xf32>)
  //
  std::string shapeInformation = "";
  // Custom onnx.dim_params attributes for the graph inputs for specifying
  // relationship among their dynamic dimensions.
  // Its format is 'input_id:dim_id=sym,dim_id=sym,...|input_id:
  // dim_id=sym,dim_id=sym,...|input_id...'
  // E.g. An ONNX model has two dynamic inputs
  //   - (arg0: tensor<?x5xf32>, arg1: tensor<?x5xf32>)
  // If we want to specify that the first unknown dimension of arg0 and the
  // first unknown dimension of arg1 are the same, we can assign the two
  // dimensions to the same symbol "batch" as follows.
  //   - dimParams = '0:0=batch|1:0=batch'
  //
  std::string dimParams = "";
  // Directory to look for external data if any tensor has external
  // data location. If empty then external data is disabled.
  std::string externalDataDir = "";

  std::vector<std::string> functionsToDecompose = {};

  // Experimental external-initializer source support. Keep these fields at
  // the end so existing aggregate initialization, if any, is not reordered.
  bool useInitializerSources = false;
  bool skipAllInitializerMaterialization = false;
  // Outermost names are unchanged. Nested names use MLIR symbol references,
  // for example @graph::@nested::@name.
  std::vector<std::string> skipInitializerMaterialization;

  // Required for every TensorProto::EXTERNAL initializer whose location is
  // "*/_ORT_MEM_ADDR_/*" when useInitializerSources is enabled. The key is
  // the initializer identity: its unmodified ONNX name in the outermost graph,
  // or an MLIR nested symbol reference such as @graph::@nested::@name. The
  // value supplies a non-owning view beginning at the address encoded in the
  // TensorProto and a durable backup-model path containing the same scoped
  // tensor. Relative backup paths are resolved against externalDataDir. Import
  // fails if an in-memory initializer has no matching complete entry.
  std::map<std::string, ExternalInitializerBackup> externalInitializerBackups;
};

// Lifetime contract for ImportFrontendModel* source references:
//
// ImportOptions and ordinary input buffers are borrowed only for the duration
// of an import call. Storage referenced by a process-local initializer source
// is different: the caller must keep it alive and bytewise immutable until the
// MLIRContext passed to the import call is destroyed. Cloning the returned IR
// copies the non-owning reference and does not extend the storage lifetime.
//
// The current external-initializer mode retains only live views supplied in
// ExternalInitializerBackup; it does not retain the input ModelProto. A source
// mode that retains a ModelProto, TensorProto, or resolver reference has the
// same context-lifetime and immutability requirement. File and array entry
// points parse temporary ModelProto objects and therefore cannot provide such
// references unless the caller supplies separate long-lived source storage.

/*!
 *  Load an ONNX model proto from a file into a protobuf object.
 *  @param model_fname file name pointing to the onnx model protobuf.
 *  @param errorMessage error message.
 *  @param model Out parameter: the onnx model protobuf.
 */
[[nodiscard]] std::error_code LoadModelProtoFromFile(
    llvm::StringRef model_fname, std::string &errorMessage,
    onnx::ModelProto &model);

/*!
 *  Import an ONNX model array into the ONNX Dialect.
 *  @param onnxBuffer buffer containing onnx model protobuf.
 *  @param bufferSize size of buffer containing onnx model protobuf.
 *  @param MLIR::module generated for the ONNX model.
 *  @pre Process-local source storage in options obeys the lifetime contract
 *  above.
 */
[[nodiscard]] std::error_code ImportFrontendModelArray(const void *onnxBuffer,
    int bufferSize, mlir::MLIRContext &context,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string &errorMessage,
    const ImportOptions &options = ImportOptions());

/*!
 *  Import an ONNX model file into the ONNX Dialect.
 *  @param model_fname file name pointing to the onnx model protobuf.
 *  @param MLIR::module generated for the ONNX model.
 *  @pre Process-local source storage in options obeys the lifetime contract
 *  above.
 */
[[nodiscard]] std::error_code ImportFrontendModelFile(
    llvm::StringRef model_fname, mlir::MLIRContext &context,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string &errorMessage,
    const ImportOptions &options = ImportOptions());

/*!
 *  Import an ONNX model proto with pre-processing into the ONNX Dialect.
 *  The pre-processing depends on @param options and can involve topologically
 *  sorting, shape inference and opset conversion.
 *  @param model the onnx model protobuf.
 *  @param MLIR::module generated for the ONNX model.
 *  @pre Process-local source storage in model or options obeys the lifetime
 *  contract above if the selected source mode retains references to it.
 */
[[nodiscard]] std::error_code ImportFrontendModelWithPreProcessing(
    onnx::ModelProto &model, mlir::MLIRContext &context,
    mlir::OwningOpRef<mlir::ModuleOp> &module, const ImportOptions &options,
    std::string &errorMessage);

/*!
 *  Import an ONNX model proto into the ONNX Dialect.
 *  @param model the onnx model protobuf.
 *  @param MLIR::module generated for the ONNX model.
 *  @pre Process-local source storage in model or options obeys the lifetime
 *  contract above if the selected source mode retains references to it.
 */
[[nodiscard]] std::error_code ImportFrontendModel(const onnx::ModelProto &model,
    mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string &errorMessage, const ImportOptions &options = ImportOptions());

/*!
 *  TODO: Import models into other extension dialects that cover the
 *  operations specific to other frameworks such as Tensorflow or Pytorch.
 */
} // namespace onnx_mlir
#endif
