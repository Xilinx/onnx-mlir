// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
// Test-only scenario driver. Python owns fixtures and separate-process stages;
// this executable owns address-bearing ONNX tensors and real MLIR contexts.
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "src/Builder/InitializerSource.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#ifndef _WIN32
#include <sys/mman.h>
#endif

using namespace mlir;
using namespace onnx_mlir;
namespace cl = llvm::cl;
static cl::OptionCategory initializerSourceCategory(
    "initializer-source test options");
static cl::opt<std::string> input("input", cl::Required,
    cl::desc("Input ONNX model, or MLIR with --read-ir"),
    cl::value_desc("path"), cl::cat(initializerSourceCategory));
static cl::opt<std::string> output("output",
    cl::desc("Write the resulting MLIR"), cl::value_desc("path"),
    cl::cat(initializerSourceCategory));
static cl::opt<std::string> sourceMode("source", cl::init("file"),
    cl::desc("Initializer source: file, live, or guard"),
    cl::value_desc("mode"), cl::cat(initializerSourceCategory));
static cl::opt<std::string> entry("entry", cl::init("raw"),
    cl::desc("Import API: raw, preprocessed, array, or file"),
    cl::value_desc("api"), cl::cat(initializerSourceCategory));
static cl::opt<std::string> liveData("live-data",
    cl::desc("Bytes backing live initializers"), cl::value_desc("path"),
    cl::cat(initializerSourceCategory));
static cl::opt<std::string> backupPath("backup-path",
    cl::desc("Durable backup ONNX model for live initializers"),
    cl::value_desc("path"), cl::cat(initializerSourceCategory));
static cl::opt<std::string> format("format", cl::init("assembly"),
    cl::desc("Output format: assembly or bytecode"), cl::value_desc("format"),
    cl::cat(initializerSourceCategory));
static cl::opt<bool> readIR("read-ir",
    cl::desc("Parse --input as MLIR and materialize its sources"),
    cl::init(false), cl::cat(initializerSourceCategory));
static cl::opt<bool> skipAll("skip-all",
    cl::desc("Leave every initializer source unmaterialized"), cl::init(false),
    cl::cat(initializerSourceCategory));
static cl::list<std::string> skipNames("skip-name",
    cl::desc("Leave one fully scoped initializer path unmaterialized"),
    cl::value_desc("scope/name"), cl::ZeroOrMore,
    cl::cat(initializerSourceCategory));
static cl::opt<bool> materializeAfter("materialize-after",
    cl::desc("Materialize after importer options and protobuf are destroyed"),
    cl::init(false), cl::cat(initializerSourceCategory));
static cl::opt<bool> omitBackup("omit-backup",
    cl::desc("Do not register a durable backup for live initializers"),
    cl::init(false), cl::cat(initializerSourceCategory));
static cl::opt<bool> legacy("legacy",
    cl::desc("Use the original eager initializer import"), cl::init(false),
    cl::cat(initializerSourceCategory));
static cl::opt<bool> protobufShapeInference("protobuf-shape-inference",
    cl::desc("Run ONNX protobuf shape inference before import"),
    cl::init(false), cl::cat(initializerSourceCategory));
static cl::opt<bool> opsetConversion("opset-conversion",
    cl::desc("Request ONNX opset conversion before import"), cl::init(false),
    cl::cat(initializerSourceCategory));
static cl::opt<bool> cloneBeforeLoad("clone-before-load",
    cl::desc("Clone the MLIR module before delayed materialization"),
    cl::init(false), cl::cat(initializerSourceCategory));

#ifndef _WIN32
class GuardedBuffer final : public llvm::MemoryBuffer {
public:
  explicit GuardedBuffer(size_t length) : length(length ? length : 1) {
    data = mmap(
        nullptr, this->length, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED)
      llvm::report_fatal_error("cannot allocate guarded test buffer");
    init(static_cast<char *>(data), static_cast<char *>(data) + length, false);
  }
  ~GuardedBuffer() override { munmap(data, length); }
  BufferKind getBufferKind() const override { return MemoryBuffer_Malloc; }

private:
  void *data;
  size_t length;
};
#endif

static bool configureLive(onnx::GraphProto &graph, std::string scope,
    ImportOptions &options,
    std::vector<std::unique_ptr<llvm::MemoryBuffer>> &liveOwners,
    bool isMainGraph = false) {
  if (!isMainGraph) {
    if (!scope.empty())
      scope += "/";
    scope += graph.name();
  }
  for (auto &tensor : *graph.mutable_initializer()) {
    if (tensor.data_location() != onnx::TensorProto::EXTERNAL)
      continue;
    uint64_t offset = 0, length = 0;
    std::string location;
    for (const auto &field : tensor.external_data()) {
      if (field.key() == "location")
        location = field.value();
      if (field.key() == "offset" &&
          StringRef(field.value()).getAsInteger(10, offset))
        return false;
      if (field.key() == "length" &&
          StringRef(field.value()).getAsInteger(10, length))
        return false;
    }
    ExternalInitializerBackup source;
    source.offset = offset;
    source.length = length;
    SmallString<256> original(options.externalDataDir);
    llvm::sys::path::append(original, location);
    source.path =
        backupPath.empty() ? original.str().str() : backupPath.getValue();
    std::unique_ptr<llvm::MemoryBuffer> liveOwner;
    if (sourceMode == "guard") {
#ifndef _WIN32
      liveOwner = std::make_unique<GuardedBuffer>(length);
#else
      llvm::errs() << "guarded-buffer scenario requires mmap\n";
      return false;
#endif
    } else {
      auto bytes = llvm::MemoryBuffer::getFile(liveData, false, false);
      if (!bytes || offset > (*bytes)->getBufferSize() ||
          length > (*bytes)->getBufferSize() - offset) {
        llvm::errs() << "cannot read live fixture/range\n";
        return false;
      }
      liveOwner = llvm::MemoryBuffer::getMemBufferCopy(
          (*bytes)->getBuffer().substr(offset, length));
    }
    source.live =
        ArrayRef<char>(liveOwner->getBufferStart(), liveOwner->getBufferSize());
    liveOwners.push_back(std::move(liveOwner));
    tensor.clear_external_data();
    auto add = [&](StringRef key, StringRef value) {
      auto *field = tensor.add_external_data();
      field->set_key(key.str());
      field->set_value(value.str());
    };
    add("location", "*/_ORT_MEM_ADDR_/*");
    add("offset",
        std::to_string(reinterpret_cast<uintptr_t>(source.live.data())));
    add("length", std::to_string(length));
    if (!omitBackup) {
      auto identity =
          scope.empty() ? tensor.name() : scope + "/" + tensor.name();
      options.externalInitializerBackups[identity] = source;
    }
  }
  for (auto &node : *graph.mutable_node())
    for (auto &attr : *node.mutable_attribute())
      if (attr.has_g() &&
          !configureLive(*attr.mutable_g(), scope, options, liveOwners, false))
        return false;
  return true;
}

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions(initializerSourceCategory);
  cl::ParseCommandLineOptions(argc, argv,
      "Test-only driver for ONNX initializer-source scenarios. It can import "
      "file-backed, caller-owned, or guarded tensors; defer materialization; "
      "and round-trip MLIR assembly or bytecode. The Python unit test creates "
      "fixtures and invokes this command in separate processes.\n");
  if (sourceMode != "file" && sourceMode != "live" && sourceMode != "guard")
    return 2;
  auto ownedContext = std::make_unique<MLIRContext>();
  MLIRContext &context = *ownedContext;
  context.loadDialect<ONNXDialect, func::FuncDialect>();
  OwningOpRef<ModuleOp> module;
  // The test driver acts as the VAIML/FlexML caller and keeps these buffers
  // alive while in-memory initializer sources may be read.
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> liveOwners;
  if (readIR) {
    module = parseSourceFile<ModuleOp>(input, &context);
  } else {
    onnx::ModelProto model;
    std::string error;
    if (LoadModelProtoFromFile(input.getValue(), error, model)) {
      llvm::errs() << error << "\n";
      return 1;
    }
    ImportOptions options;
    options.useInitializerSources = !legacy;
    options.runOnnxShapeInference = protobufShapeInference;
    options.invokeOnnxVersionConverter = opsetConversion;
    options.addResultNamesAttr = true;
    options.skipAllInitializerMaterialization = skipAll;
    options.skipInitializerMaterialization.assign(
        skipNames.begin(), skipNames.end());
    options.externalDataDir =
        llvm::sys::path::parent_path(input.getValue()).str();
    if (sourceMode != "file" &&
        !configureLive(*model.mutable_graph(), "", options, liveOwners, true))
      return 1;
    std::error_code ec;
    if (entry == "raw")
      ec = ImportFrontendModel(model, context, module, error, options);
    else if (entry == "preprocessed")
      ec = ImportFrontendModelWithPreProcessing(
          model, context, module, options, error);
    else if (entry == "array") {
      std::string encoded = model.SerializeAsString();
      ec = ImportFrontendModelArray(
          encoded.data(), encoded.size(), context, module, error, options);
    } else if (entry == "file" && sourceMode == "file")
      ec = ImportFrontendModelFile(
          input.getValue(), context, module, error, options);
    else {
      llvm::errs() << "unsupported entry/source combination\n";
      return 2;
    }
    if (ec) {
      llvm::errs() << error << "\n";
      return 1;
    }
  } // Caller options and the original protobuf die here; liveOwners remains.
  if (!module || failed(verify(*module)))
    return 1;
  if (cloneBeforeLoad)
    module = cast<ModuleOp>(module->clone());
  if (materializeAfter || readIR) {
    PassManager pm(&context);
    pm.addPass(createMaterializeInitializerSourcesPass(
        readIR ? bool(skipAll) : false,
        readIR ? ArrayRef<std::string>(skipNames) : ArrayRef<std::string>()));
    if (failed(pm.run(*module)))
      return 1;
  }
  module->walk([&](ONNXInitializerSourceOp op) {
    auto source = cast<InitializerSourceAttr>(op.getSource());
    auto disk = source.getDisk();
    auto scope = disk.getAs<StringAttr>("scope").getValue();
    llvm::outs() << "source ";
    if (!scope.empty())
      llvm::outs() << scope << "/";
    llvm::outs() << disk.getAs<StringAttr>("name").getValue()
                 << " live=" << (source.getLive().data() != nullptr) << "\n";
  });
  module->walk([&](ONNXConstantOp op) {
    auto value = op.getValueAttr();
    if (!value)
      return;
    llvm::outs() << "constant disposable=" << isa<DisposableElementsAttr>(value)
                 << " ";
    if (auto disposable = dyn_cast<DisposableElementsAttr>(value))
      llvm::outs() << disposable.toDenseElementsAttr();
    else
      llvm::outs() << value;
    llvm::outs() << "\n";
  });
  if (!output.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream out(output, ec, llvm::sys::fs::OF_None);
    if (ec)
      return 1;
    if (format == "bytecode") {
      if (failed(writeBytecodeToFile(*module, out)))
        return 1;
    } else if (format == "assembly") {
      module->print(out);
    } else
      return 2;
  }
  module = {};
  ownedContext.reset();
  return 0;
}
