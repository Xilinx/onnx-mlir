// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
#include "src/Builder/InitializerSource.hpp"
#include "mlir/IR/Builders.h"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <limits>
#include <map>

using namespace mlir;
namespace onnx_mlir {
namespace {
constexpr StringLiteral ortMemoryLocation = "*/_ORT_MEM_ADDR_/*";
constexpr StringLiteral externalDataSourceKind = "external_data";
constexpr StringLiteral modelSourceKind = "onnx_model";

using ModelCache =
    std::map<std::string, std::shared_ptr<const onnx::ModelProto>>;

bool supportedDataType(int type) {
  return isSupportedInitializerSourceDataType(type);
}

bool containsParentTraversal(StringRef path) {
  for (auto component = llvm::sys::path::begin(path),
            end = llvm::sys::path::end(path);
      component != end; ++component)
    if (*component == "..")
      return true;
  return false;
}

std::string initializerIdentity(DictionaryAttr disk) {
  StringRef scope = disk.getAs<StringAttr>("scope").getValue();
  StringRef name = disk.getAs<StringAttr>("name").getValue();
  return scope.empty() ? name.str() : scope.str() + "/" + name.str();
}

FailureOr<ElementsAttr> resolveFileRange(ONNXInitializerSourceOp op,
    RankedTensorType type, int dataType, StringRef path, uint64_t offset,
    uint64_t length) {
  auto loaded = getExternalDataFile(path);
  if (!loaded) {
    op.emitError("cannot read initializer source ")
        << path << ": " << loaded.getError().message();
    return failure();
  }
  std::shared_ptr<const llvm::MemoryBuffer> buffer = *loaded;
  if (offset > buffer->getBufferSize() ||
      length > buffer->getBufferSize() - offset) {
    op.emitError("initializer source range exceeds available bytes");
    return failure();
  }
  return externalTensorBytesToElmAttr(type, dataType, buffer, offset, length);
}

FailureOr<const onnx::TensorProto *> findInitializerInModel(
    ONNXInitializerSourceOp op, const onnx::ModelProto &model, StringRef scope,
    StringRef name) {
  const onnx::GraphProto *graph = &model.graph();
  SmallVector<StringRef> components;
  scope.split(components, '/', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef component : components) {
    const onnx::GraphProto *match = nullptr;
    for (const auto &node : graph->node()) {
      for (const auto &attr : node.attribute()) {
        auto consider = [&](const onnx::GraphProto &candidate) {
          if (candidate.name() != component)
            return success();
          if (match)
            return failure();
          match = &candidate;
          return success();
        };
        if (attr.has_g() && failed(consider(attr.g()))) {
          op.emitError("ambiguous graph component in initializer source: ")
              << component;
          return failure();
        }
        for (const auto &candidate : attr.graphs())
          if (failed(consider(candidate))) {
            op.emitError("ambiguous graph component in initializer source: ")
                << component;
            return failure();
          }
      }
    }
    if (!match) {
      op.emitError("cannot find graph component in initializer source: ")
          << component;
      return failure();
    }
    graph = match;
  }
  const onnx::TensorProto *match = nullptr;
  for (const auto &tensor : graph->initializer()) {
    if (tensor.name() != name)
      continue;
    if (match) {
      op.emitError("ambiguous initializer in backup model: ") << name;
      return failure();
    }
    match = &tensor;
  }
  if (!match) {
    op.emitError("cannot find initializer in backup model: ") << name;
    return failure();
  }
  return match;
}

FailureOr<ElementsAttr> resolveModelTensor(ONNXInitializerSourceOp op,
    DictionaryAttr disk, RankedTensorType type, int dataType,
    uint64_t expectedSize, ModelCache &models) {
  StringRef modelPath = disk.getAs<StringAttr>("path").getValue();
  auto found = models.find(modelPath.str());
  if (found == models.end()) {
    auto model = std::make_shared<onnx::ModelProto>();
    std::string errorMessage;
    if (LoadModelProtoFromFile(modelPath, errorMessage, *model)) {
      op.emitError("cannot read initializer backup model ")
          << modelPath << ": " << errorMessage;
      return failure();
    }
    found = models.emplace(modelPath.str(), std::move(model)).first;
  }
  auto tensor = findInitializerInModel(op, *found->second,
      disk.getAs<StringAttr>("scope").getValue(),
      disk.getAs<StringAttr>("name").getValue());
  if (failed(tensor))
    return failure();
  if ((*tensor)->data_type() != dataType ||
      !std::equal((*tensor)->dims().begin(), (*tensor)->dims().end(),
          type.getShape().begin(), type.getShape().end())) {
    op.emitError("backup initializer type or shape disagrees with source");
    return failure();
  }
  if ((*tensor)->data_location() != onnx::TensorProto::EXTERNAL)
    return onnxTensorProtoToElmAttr(op.getContext(), "", **tensor);

  std::string location;
  uint64_t offset = 0, length = expectedSize;
  bool hasLength = false;
  for (const auto &entry : (*tensor)->external_data()) {
    if (entry.key() == "location")
      location = entry.value();
    else if (entry.key() == "offset" || entry.key() == "length") {
      uint64_t value;
      if (StringRef(entry.value()).getAsInteger(10, value)) {
        op.emitError("invalid external metadata in backup initializer: ")
            << entry.key();
        return failure();
      }
      if (entry.key() == "offset")
        offset = value;
      else {
        length = value;
        hasLength = true;
      }
    }
  }
  if (location.empty() || location == ortMemoryLocation) {
    op.emitError("backup initializer has no durable external location");
    return failure();
  }
  if (llvm::sys::path::is_absolute(location) ||
      containsParentTraversal(location)) {
    op.emitError("backup initializer location must be relative without '..'");
    return failure();
  }
  if (hasLength && length != expectedSize) {
    op.emitError("backup initializer length disagrees with source");
    return failure();
  }
  SmallString<256> path(llvm::sys::path::parent_path(modelPath));
  llvm::sys::path::append(path, location);
  llvm::sys::path::remove_dots(path, true);
  return resolveFileRange(op, type, dataType, path, offset, length);
}

FailureOr<ElementsAttr> resolve(
    ONNXInitializerSourceOp op, ModelCache &models) {
  auto source = cast<InitializerSourceAttr>(op.getSource());
  auto disk = source.getDisk();
  int dataType = disk.getAs<IntegerAttr>("data_type").getInt();
  auto type = cast<RankedTensorType>(op.getOutput().getType());
  if (!supportedDataType(dataType) ||
      mlirTypeOfBType(btypeOfOnnxDataType(dataType), op.getContext()) !=
          type.getElementType()) {
    op.emitError("source data_type disagrees with result element type");
    return failure();
  }
  auto size = getInitializerSourcePayloadSize(type);
  if (failed(size)) {
    op.emitError("has invalid or overflowing result shape");
    return failure();
  }
  ArrayRef<char> live = source.getLive();
  if (live.data()) {
    if (live.size() != *size) {
      op.emitError("initializer source range exceeds available live bytes");
      return failure();
    }
    return externalTensorBytesToElmAttr(type, dataType, live);
  }
  StringRef kind = disk.getAs<StringAttr>("kind").getValue();
  if (kind == modelSourceKind)
    return resolveModelTensor(op, disk, type, dataType, *size, models);
  uint64_t offset = disk.getAs<IntegerAttr>("offset").getInt();
  uint64_t length = disk.getAs<IntegerAttr>("length").getInt();
  if (length != *size) {
    op.emitError("source length disagrees with tensor payload size");
    return failure();
  }
  auto path = disk.getAs<StringAttr>("path").getValue();
  return resolveFileRange(op, type, dataType, path, offset, length);
}

struct MaterializeInitializerSourcesPass
    : PassWrapper<MaterializeInitializerSourcesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializeInitializerSourcesPass)
  MaterializeInitializerSourcesPass() = default;
  MaterializeInitializerSourcesPass(const MaterializeInitializerSourcesPass &p)
      : PassWrapper(p) {}
  Option<bool> skipAll{*this, "skip-all", llvm::cl::init(false),
      llvm::cl::desc("Leave all initializer sources unresolved")};
  ListOption<std::string> skipNames{*this, "skip-names",
      llvm::cl::desc("Scoped initializer names to leave unresolved"),
      llvm::cl::ZeroOrMore};
  StringRef getArgument() const final {
    return "onnx-materialize-initializer-sources";
  }
  StringRef getDescription() const final {
    return "Resolve ONNX initializer sources into disposable "
           "constants";
  }
  void runOnOperation() final {
    if (skipAll)
      return;
    SmallVector<ONNXInitializerSourceOp> sources;
    ModelCache models;
    getOperation().walk(
        [&](ONNXInitializerSourceOp op) { sources.push_back(op); });
    for (const auto &name : skipNames) {
      bool found = llvm::any_of(sources, [&](ONNXInitializerSourceOp op) {
        auto disk = cast<InitializerSourceAttr>(op.getSource()).getDisk();
        return name == initializerIdentity(disk);
      });
      if (!found) {
        getOperation().emitError("unknown initializer source in skip-names: ")
            << name;
        signalPassFailure();
        return;
      }
    }
    for (auto op : sources) {
      auto disk = cast<InitializerSourceAttr>(op.getSource()).getDisk();
      auto identity = initializerIdentity(disk);
      if (llvm::is_contained(skipNames, identity))
        continue;
      auto value = resolve(op, models);
      if (failed(value)) {
        signalPassFailure();
        return;
      }
      OpBuilder builder(op);
      auto constant = OnnxBuilder(builder, op.getLoc()).constant(*value);
      for (auto attr : op->getAttrs())
        if (attr.getName() != "source")
          constant.getDefiningOp()->setAttr(attr.getName(), attr.getValue());
      op.getOutput().replaceAllUsesWith(constant);
      op.erase();
    }
  }
};
} // namespace

std::unique_ptr<Pass> createMaterializeInitializerSourcesPass(
    bool skipAll, ArrayRef<std::string> skipNames) {
  auto pass = std::make_unique<MaterializeInitializerSourcesPass>();
  pass->skipAll = skipAll;
  pass->skipNames = skipNames;
  return pass;
}

FailureOr<InitializerSourceAttr> createInitializerSource(
    const onnx::TensorProto &tensor, RankedTensorType type, StringRef scope,
    const ImportOptions &options, Location loc) {
  if (!supportedDataType(tensor.data_type())) {
    emitError(loc, "unsupported external initializer element type");
    return failure();
  }
  if (llvm::any_of(type.getShape(), [](int64_t d) { return d < 0; })) {
    emitError(loc, "negative initializer dimension");
    return failure();
  }
  auto size = getInitializerSourcePayloadSize(type);
  if (failed(size)) {
    emitError(loc, "invalid or overflowing initializer payload size");
    return failure();
  }
  std::string location;
  bool hasLength = false;
  uint64_t offset = 0, length = 0;
  for (const auto &entry : tensor.external_data()) {
    if (entry.key() == "location")
      location = entry.value();
    else if (entry.key() == "offset" || entry.key() == "length") {
      uint64_t value;
      if (StringRef(entry.value()).getAsInteger(10, value)) {
        emitError(loc) << "invalid external " << entry.key();
        return failure();
      }
      if (entry.key() == "offset")
        offset = value;
      else {
        length = value;
        hasLength = true;
      }
    }
  }
  ArrayRef<char> live;
  std::string path;
  StringRef kind = externalDataSourceKind;
  if (location == ortMemoryLocation) {
    auto identity =
        scope.empty() ? tensor.name() : scope.str() + "/" + tensor.name();
    auto backup = options.externalInitializerBackups.find(identity);
    if (backup == options.externalInitializerBackups.end() ||
        backup->second.modelPath.empty() || !backup->second.live.data()) {
      emitError(loc)
          << "memory initializer " << identity
          << " requires a borrowed live view and explicit backup model";
      return failure();
    }
    const auto &source = backup->second;
    live = source.live;
    // Only compare address metadata; never read bytes during import.
    if (offset != reinterpret_cast<uintptr_t>(live.data()) || length != *size ||
        live.size() != *size) {
      emitError(
          loc, "memory initializer address/length disagrees with live view");
      return failure();
    }
    if (llvm::sys::path::is_relative(source.modelPath)) {
      if (options.externalDataDir.empty()) {
        emitError(
            loc, "relative backup model requires an explicit source base");
        return failure();
      }
      SmallString<256> joined(options.externalDataDir);
      llvm::sys::path::append(joined, source.modelPath);
      path = joined.str().str();
    } else {
      path = source.modelPath;
    }
    kind = modelSourceKind;
  } else {
    if (location.empty()) {
      emitError(loc, "external initializer requires a location");
      return failure();
    }
    if (llvm::sys::path::is_absolute(location) ||
        containsParentTraversal(location)) {
      emitError(loc,
          "external initializer location must be a relative path without '..'");
      return failure();
    }
    if (options.externalDataDir.empty()) {
      emitError(
          loc, "relative external location requires an explicit source base");
      return failure();
    }
    SmallString<256> joined(options.externalDataDir);
    llvm::sys::path::append(joined, location);
    path = joined.str().str();
    if (!hasLength)
      length = *size;
  }
  if (length != *size || (kind == externalDataSourceKind &&
                             (offset > uint64_t(INT64_MAX) ||
                                 length > uint64_t(INT64_MAX) - offset))) {
    emitError(loc, "invalid initializer source range or payload length");
    return failure();
  }
  SmallString<256> absolute(path);
  if (auto ec = llvm::sys::fs::make_absolute(absolute)) {
    emitError(loc) << "cannot resolve initializer source path: "
                   << ec.message();
    return failure();
  }
  llvm::sys::path::remove_dots(absolute, true);
  Builder builder(loc.getContext());
  SmallVector<NamedAttribute> fields{
      builder.getNamedAttr("kind", builder.getStringAttr(kind)),
      builder.getNamedAttr("path", builder.getStringAttr(absolute)),
      builder.getNamedAttr("scope", builder.getStringAttr(scope)),
      builder.getNamedAttr("name", builder.getStringAttr(tensor.name())),
      builder.getNamedAttr(
          "data_type", builder.getI64IntegerAttr(tensor.data_type()))};
  if (kind == externalDataSourceKind) {
    fields.push_back(
        builder.getNamedAttr("offset", builder.getI64IntegerAttr(offset)));
    fields.push_back(
        builder.getNamedAttr("length", builder.getI64IntegerAttr(length)));
  }
  auto disk = builder.getDictionaryAttr(fields);
  return InitializerSourceAttr::get(disk, live);
}
} // namespace onnx_mlir
