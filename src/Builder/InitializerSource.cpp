// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
#include "src/Builder/InitializerSource.hpp"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "llvm/ADT/StringSet.h"
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

SmallVector<StringRef> initializerScope(DictionaryAttr disk) {
  SmallVector<StringRef> result;
  for (Attribute component : disk.getAs<ArrayAttr>("scope"))
    result.push_back(cast<StringAttr>(component).getValue());
  return result;
}

std::string initializerIdentity(DictionaryAttr disk) {
  auto scope = initializerScope(disk);
  SmallVector<std::string> ownedScope;
  llvm::transform(scope, std::back_inserter(ownedScope),
      [](StringRef component) { return component.str(); });
  return getInitializerIdentity(
      disk.getContext(), ownedScope, disk.getAs<StringAttr>("name").getValue());
}

bool matchesNestedIdentity(SymbolRefAttr identity, DictionaryAttr disk) {
  auto scope = initializerScope(disk);
  auto nested = identity.getNestedReferences();
  if (scope.empty() || nested.size() != scope.size())
    return false;
  if (identity.getRootReference() != scope.front())
    return false;
  for (auto [reference, component] :
      llvm::zip(nested.drop_back(), ArrayRef<StringRef>(scope).drop_front()))
    if (reference.getValue() != component)
      return false;
  return nested.back().getValue() == disk.getAs<StringAttr>("name").getValue();
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
    ONNXInitializerSourceOp op, const onnx::ModelProto &model, ArrayAttr scope,
    StringRef name) {
  const onnx::GraphProto *graph = &model.graph();
  for (Attribute scopeAttr : scope) {
    StringRef component = cast<StringAttr>(scopeAttr).getValue();
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
  auto tensor =
      findInitializerInModel(op, *found->second, disk.getAs<ArrayAttr>("scope"),
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
    getOperation().walk(
        [&](ONNXInitializerSourceOp op) { sources.push_back(op); });
    llvm::StringSet<> skippedIdentities;
    for (const auto &name : skipNames) {
      SmallVector<ONNXInitializerSourceOp> matches;
      llvm::copy_if(sources, std::back_inserter(matches),
          [&](ONNXInitializerSourceOp op) {
            auto disk = cast<InitializerSourceAttr>(op.getSource()).getDisk();
            return initializerScope(disk).empty() &&
                   name == disk.getAs<StringAttr>("name").getValue();
          });
      // An exact outer-scope tensor name takes precedence, including names
      // beginning with '@' that could otherwise parse as a symbol reference.
      if (matches.empty() && StringRef(name).starts_with("@")) {
        auto parsed = parseAttribute(name, getOperation().getContext());
        auto symbol = dyn_cast_or_null<SymbolRefAttr>(parsed);
        if (!symbol || symbol.getNestedReferences().empty()) {
          getOperation().emitError(
              "invalid nested initializer source in skip-names: ")
              << name;
          signalPassFailure();
          return;
        }
        llvm::copy_if(sources, std::back_inserter(matches),
            [&](ONNXInitializerSourceOp op) {
              auto disk = cast<InitializerSourceAttr>(op.getSource()).getDisk();
              return matchesNestedIdentity(symbol, disk);
            });
      }
      if (matches.empty()) {
        getOperation().emitError("unknown initializer source in skip-names: ")
            << name;
        signalPassFailure();
        return;
      }
      if (matches.size() != 1) {
        getOperation().emitError("ambiguous initializer source in skip-names: ")
            << name;
        signalPassFailure();
        return;
      }
      auto disk =
          cast<InitializerSourceAttr>(matches.front().getSource()).getDisk();
      skippedIdentities.insert(initializerIdentity(disk));
    }
    InitializerSourceMaterializer materializer;
    for (auto op : sources) {
      auto disk = cast<InitializerSourceAttr>(op.getSource()).getDisk();
      auto identity = initializerIdentity(disk);
      if (skippedIdentities.contains(identity))
        continue;
      if (failed(materializer.materialize(op))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

struct InitializerSourceMaterializer::Impl {
  ModelCache models;
};

InitializerSourceMaterializer::InitializerSourceMaterializer()
    : impl(std::make_unique<Impl>()) {}

InitializerSourceMaterializer::~InitializerSourceMaterializer() = default;

FailureOr<ONNXConstantOp> InitializerSourceMaterializer::materialize(
    ONNXInitializerSourceOp op) {
  auto value = resolve(op, impl->models);
  if (failed(value))
    return failure();
  OpBuilder builder(op);
  Value constantValue = OnnxBuilder(builder, op.getLoc()).constant(*value);
  auto constant = cast<ONNXConstantOp>(constantValue.getDefiningOp());
  for (auto attr : op->getAttrs())
    if (attr.getName() != "source")
      constant->setAttr(attr.getName(), attr.getValue());
  op.getOutput().replaceAllUsesWith(constantValue);
  op.erase();
  return constant;
}

std::unique_ptr<Pass> createMaterializeInitializerSourcesPass(
    bool skipAll, ArrayRef<std::string> skipNames) {
  auto pass = std::make_unique<MaterializeInitializerSourcesPass>();
  pass->skipAll = skipAll;
  pass->skipNames = skipNames;
  return pass;
}

FailureOr<InitializerSourceAttr> createInitializerSource(
    const onnx::TensorProto &tensor, RankedTensorType type,
    ArrayRef<std::string> scope, const ImportOptions &options, Location loc) {
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
        getInitializerIdentity(loc.getContext(), scope, tensor.name());
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
  SmallVector<StringRef> scopeRefs;
  llvm::transform(scope, std::back_inserter(scopeRefs),
      [](const std::string &component) { return StringRef(component); });
  SmallVector<NamedAttribute> fields{
      builder.getNamedAttr("kind", builder.getStringAttr(kind)),
      builder.getNamedAttr("path", builder.getStringAttr(absolute)),
      builder.getNamedAttr("scope", builder.getStrArrayAttr(scopeRefs)),
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

std::string getInitializerIdentity(
    MLIRContext *context, ArrayRef<std::string> scope, StringRef name) {
  if (scope.empty())
    return name.str();
  SmallVector<FlatSymbolRefAttr> nested;
  for (StringRef component : scope.drop_front())
    nested.push_back(FlatSymbolRefAttr::get(context, component));
  nested.push_back(FlatSymbolRefAttr::get(context, name));
  auto identity =
      SymbolRefAttr::get(StringAttr::get(context, scope.front()), nested);
  std::string result;
  llvm::raw_string_ostream stream(result);
  identity.print(stream);
  return result;
}
} // namespace onnx_mlir
