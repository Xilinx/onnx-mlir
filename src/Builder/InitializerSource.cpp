// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
#include "src/Builder/InitializerSource.hpp"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <limits>
#include <map>

using namespace mlir;
namespace onnx_mlir {
#define GEN_PASS_DEF_MATERIALIZEINITIALIZERSOURCESPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"
namespace {
constexpr StringLiteral ortMemoryLocation = "*/_ORT_MEM_ADDR_/*";
constexpr StringLiteral externalDataSourceKind = "external_data";
constexpr StringLiteral modelSourceKind = "onnx_model";

using ModelCache =
    std::map<std::string, std::shared_ptr<const onnx::ModelProto>>;

bool containsParentTraversal(StringRef path) {
  for (auto component = llvm::sys::path::begin(path),
            end = llvm::sys::path::end(path);
      component != end; ++component)
    if (*component == "..")
      return true;
  return false;
}

SmallVector<StringRef> initializerScope(ONNXInitializerSourceOp op) {
  SmallVector<StringRef> result;
  for (Attribute component : op.getScope())
    result.push_back(cast<StringAttr>(component).getValue());
  return result;
}

InitializerIdentity initializerIdentity(ONNXInitializerSourceOp op) {
  InitializerIdentity identity;
  llvm::transform(initializerScope(op), std::back_inserter(identity.scope),
      [](StringRef component) { return component.str(); });
  identity.name = op.getName().str();
  return identity;
}

bool matchesNestedIdentity(SymbolRefAttr identity, ONNXInitializerSourceOp op) {
  auto scope = initializerScope(op);
  auto nested = identity.getNestedReferences();
  if (scope.empty() || nested.size() != scope.size())
    return false;
  if (identity.getRootReference() != scope.front())
    return false;
  for (auto [reference, component] :
      llvm::zip(nested.drop_back(), ArrayRef<StringRef>(scope).drop_front()))
    if (reference.getValue() != component)
      return false;
  return nested.back().getValue() == op.getName();
}

FailureOr<ElementsAttr> resolveFileRange(ONNXInitializerSourceOp op,
    RankedTensorType type, StringRef path, uint64_t offset, uint64_t length) {
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
  return externalTensorBytesToElmAttr(type, buffer, offset, length);
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
    RankedTensorType type, BType btype, uint64_t expectedSize,
    ModelCache &models) {
  StringRef modelPath = op.getPath();
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
      findInitializerInModel(op, *found->second, op.getScope(), op.getName());
  if (failed(tensor))
    return failure();
  if ((*tensor)->data_type() != onnxDataTypeOfBType(btype) ||
      !llvm::equal((*tensor)->dims(), type.getShape())) {
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
  return resolveFileRange(op, type, path, offset, length);
}

FailureOr<ElementsAttr> resolve(
    ONNXInitializerSourceOp op, ModelCache &models) {
  auto type = cast<RankedTensorType>(op.getOutput().getType());
  std::optional<BType> btype =
      btypeOfMlirTypeIfSupported(type.getElementType());
  if (!btype) {
    op.emitError("has unsupported source element type ")
        << type.getElementType();
    return failure();
  }
  auto size = getInitializerSourcePayloadSize(type);
  if (failed(size)) {
    op.emitError("has invalid or overflowing result shape");
    return failure();
  }
  ArrayRef<char> callerOwnedData;
  if (auto attr = op.getCallerOwnedDataAttr())
    callerOwnedData = attr.getData();
  if (callerOwnedData.data()) {
    if (callerOwnedData.size() != *size) {
      op.emitError(
          "initializer source range exceeds available caller-owned data");
      return failure();
    }
    return externalTensorBytesToElmAttr(type, callerOwnedData);
  }
  StringRef kind = op.getKind();
  if (kind == modelSourceKind)
    return resolveModelTensor(op, type, *btype, *size, models);
  uint64_t offset = *op.getOffset();
  uint64_t length = *op.getLength();
  if (length != *size) {
    op.emitError("source length disagrees with tensor payload size");
    return failure();
  }
  return resolveFileRange(op, type, op.getPath(), offset, length);
}

struct MaterializeInitializerSourcesPass
    : impl::MaterializeInitializerSourcesPassBase<
          MaterializeInitializerSourcesPass> {
  using Base::Base;
  void runOnOperation() final {
    if (skipAll)
      return;
    SmallVector<ONNXInitializerSourceOp> sources;
    getOperation().walk(
        [&](ONNXInitializerSourceOp op) { sources.push_back(op); });
    std::set<InitializerIdentity> skippedIdentities;
    for (const auto &name : skipNames) {
      SmallVector<ONNXInitializerSourceOp> matches;
      llvm::copy_if(sources, std::back_inserter(matches),
          [&](ONNXInitializerSourceOp op) {
            return initializerScope(op).empty() && name == op.getName();
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
              return matchesNestedIdentity(symbol, op);
            });
      }
      if (matches.empty()) {
        getOperation().emitWarning("unknown initializer source in skip-names: ")
            << name;
        continue;
      }
      for (auto match : matches)
        skippedIdentities.insert(initializerIdentity(match));
    }
    InitializerSourceMaterializer materializer;
    IRRewriter rewriter(getOperation().getContext());
    for (auto op : sources) {
      auto identity = initializerIdentity(op);
      if (skippedIdentities.contains(identity))
        continue;
      if (failed(materializer.materialize(op, rewriter))) {
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
    ONNXInitializerSourceOp op, RewriterBase &rewriter) {
  auto value = resolve(op, impl->models);
  if (failed(value))
    return failure();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  Value constantValue = OnnxBuilder(rewriter, op.getLoc()).constant(*value);
  auto constant = cast<ONNXConstantOp>(constantValue.getDefiningOp());
  for (auto attr : op->getAttrs())
    if (!llvm::is_contained(
            ONNXInitializerSourceOp::getAttributeNames(), attr.getName()))
      constant->setAttr(attr.getName(), attr.getValue());
  rewriter.replaceOp(op, constantValue);
  return constant;
}

std::unique_ptr<Pass> createMaterializeInitializerSourcesPass(
    bool skipAll, ArrayRef<std::string> skipNames) {
  MaterializeInitializerSourcesPassOptions options;
  options.skipAll = skipAll;
  llvm::append_range(options.skipNames, skipNames);
  return createMaterializeInitializerSourcesPass(std::move(options));
}

FailureOr<ONNXInitializerSourceOp> createInitializerSource(OpBuilder &opBuilder,
    const onnx::TensorProto &tensor, RankedTensorType type,
    ArrayRef<std::string> scope, const ImportOptions &options, Location loc) {
  if (!mlirTypeOfBType(
          btypeOfOnnxDataType(tensor.data_type()), loc.getContext())) {
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
  ArrayRef<char> callerOwnedData;
  std::string path;
  StringRef kind = externalDataSourceKind;
  if (location == ortMemoryLocation) {
    InitializerIdentity identity{scope.vec(), tensor.name()};
    auto backup = options.externalInitializerBackups.find(identity);
    if (backup == options.externalInitializerBackups.end() ||
        backup->second.backupModelPath.empty() ||
        !backup->second.callerOwnedData.data()) {
      emitError(loc)
          << "memory initializer "
          << getInitializerIdentity(loc.getContext(), identity)
          << " requires caller-owned data and an explicit backup model";
      return failure();
    }
    const auto &source = backup->second;
    callerOwnedData = source.callerOwnedData;
    // Only compare address metadata; never read bytes during import.
    if (offset != reinterpret_cast<uintptr_t>(callerOwnedData.data()) ||
        length != *size || callerOwnedData.size() != *size) {
      emitError(loc,
          "memory initializer address/length disagrees with caller-owned data");
      return failure();
    }
    if (llvm::sys::path::is_relative(source.backupModelPath)) {
      if (options.externalDataDir.empty()) {
        emitError(
            loc, "relative backup model requires an explicit source base");
        return failure();
      }
      SmallString<256> joined(options.externalDataDir);
      llvm::sys::path::append(joined, source.backupModelPath);
      path = joined.str().str();
    } else {
      path = source.backupModelPath;
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
  Builder builder(loc.getContext());
  SmallVector<StringRef> scopeRefs;
  llvm::transform(scope, std::back_inserter(scopeRefs),
      [](const std::string &component) { return StringRef(component); });
  SmallVector<NamedAttribute> fields{
      builder.getNamedAttr("kind", builder.getStringAttr(kind)),
      builder.getNamedAttr("path", builder.getStringAttr(absolute)),
      builder.getNamedAttr("scope", builder.getStrArrayAttr(scopeRefs)),
      builder.getNamedAttr("name", builder.getStringAttr(tensor.name()))};
  if (kind == externalDataSourceKind) {
    fields.push_back(
        builder.getNamedAttr("offset", builder.getI64IntegerAttr(offset)));
    fields.push_back(
        builder.getNamedAttr("length", builder.getI64IntegerAttr(length)));
  }
  OperationState state(loc, ONNXInitializerSourceOp::getOperationName());
  state.addTypes(type);
  state.addAttributes(fields);
  if (callerOwnedData.data())
    state.addAttribute("caller_owned_data",
        CallerOwnedDataAttr::get(loc.getContext(), callerOwnedData));
  return cast<ONNXInitializerSourceOp>(opBuilder.create(state));
}

std::string getInitializerIdentity(
    MLIRContext *context, const InitializerIdentity &identity) {
  if (identity.scope.empty())
    return identity.name;
  SmallVector<FlatSymbolRefAttr> nested;
  for (StringRef component : ArrayRef<std::string>(identity.scope).drop_front())
    nested.push_back(FlatSymbolRefAttr::get(context, component));
  nested.push_back(FlatSymbolRefAttr::get(context, identity.name));
  auto symbol = SymbolRefAttr::get(
      StringAttr::get(context, identity.scope.front()), nested);
  std::string result;
  llvm::raw_string_ostream stream(result);
  symbol.print(stream);
  return result;
}
} // namespace onnx_mlir
