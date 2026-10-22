// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
#include "src/Builder/InitializerSource.hpp"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "src/Interface/StructuralOperandOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <limits>
#include <map>
#include <optional>

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
  if (auto attr = op.getCallerOwnedDataAttr()) {
    if (const onnx::TensorProto *tensor = attr.getTensor()) {
      if (tensor->data_type() != onnxDataTypeOfBType(*btype) ||
          tensor->data_location() == onnx::TensorProto::EXTERNAL ||
          !llvm::equal(tensor->dims(), type.getShape())) {
        op.emitError("caller-owned initializer type, shape, or storage "
                     "disagrees with source");
        return failure();
      }
      return onnxTensorProtoToElmAttr(op.getContext(), "", *tensor);
    }
    callerOwnedData = attr.getData();
  }
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
    SmallVector<ONNXInitializerSourceOp> initialSources;
    getOperation().walk(
        [&](ONNXInitializerSourceOp op) { initialSources.push_back(op); });
    auto resolveNames = [&](const auto &names, StringRef option,
                            std::set<InitializerIdentity> &identities) {
      for (const auto &name : names) {
        SmallVector<ONNXInitializerSourceOp> matches;
        llvm::copy_if(initialSources, std::back_inserter(matches),
            [&](ONNXInitializerSourceOp op) {
              return initializerScope(op).empty() && name == op.getName();
            });
        // An exact outer-scope tensor name takes precedence, including names
        // beginning with '@' that could otherwise parse as a symbol reference.
        if (matches.empty() && StringRef(name).starts_with("@")) {
          auto parsed = parseAttribute(name, getOperation().getContext());
          auto symbol = dyn_cast_or_null<SymbolRefAttr>(parsed);
          if (!symbol || symbol.getNestedReferences().empty()) {
            getOperation().emitError("invalid nested initializer source in ")
                << option << ": " << name;
            return failure();
          }
          llvm::copy_if(initialSources, std::back_inserter(matches),
              [&](ONNXInitializerSourceOp op) {
                return matchesNestedIdentity(symbol, op);
              });
        }
        if (matches.empty()) {
          getOperation().emitWarning("unknown initializer source in ")
              << option << ": " << name;
          continue;
        }
        for (ONNXInitializerSourceOp match : matches)
          identities.insert(initializerIdentity(match));
      }
      return success();
    };
    std::set<InitializerIdentity> forcedIdentities;
    std::set<InitializerIdentity> skippedIdentities;
    if (failed(resolveNames(forceNames, "force-names", forcedIdentities)) ||
        (!skipAll &&
            failed(resolveNames(skipNames, "skip-names", skippedIdentities)))) {
      signalPassFailure();
      return;
    }

    InitializerSourceMaterializer materializer;
    IRRewriter rewriter(getOperation().getContext());

    llvm::SetVector<ONNXInitializerSourceOp> structuralSources;
    getOperation().walk([&](StructuralOperandOpInterface op) {
      for (auto [index, operand] : llvm::enumerate(op->getOperands()))
        if (op.isStructuralOperand(index))
          if (auto source = operand.getDefiningOp<ONNXInitializerSourceOp>())
            structuralSources.insert(source);
    });
    for (ONNXInitializerSourceOp source : structuralSources)
      if (failed(materializer.materialize(source, rewriter))) {
        signalPassFailure();
        return;
      }

    SmallVector<ONNXInitializerSourceOp> sources;
    getOperation().walk(
        [&](ONNXInitializerSourceOp op) { sources.push_back(op); });
    for (ONNXInitializerSourceOp source : sources)
      if (forcedIdentities.contains(initializerIdentity(source)) &&
          failed(materializer.materialize(source, rewriter))) {
        signalPassFailure();
        return;
      }
    if (skipAll)
      return;

    sources.clear();
    getOperation().walk(
        [&](ONNXInitializerSourceOp op) { sources.push_back(op); });
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

std::unique_ptr<Pass> createMaterializeInitializerSourcesPass(bool skipAll,
    ArrayRef<std::string> skipNames, ArrayRef<std::string> forceNames) {
  MaterializeInitializerSourcesPassOptions options;
  options.skipAll = skipAll;
  llvm::append_range(options.skipNames, skipNames);
  llvm::append_range(options.forceNames, forceNames);
  return createMaterializeInitializerSourcesPass(std::move(options));
}

FailureOr<ONNXInitializerSourceOp> createInitializerSource(OpBuilder &opBuilder,
    const onnx::TensorProto &tensor, RankedTensorType type,
    ArrayRef<std::string> scope, const ImportOptions &options, Location loc,
    bool retainTensorReference) {
  if (!mlirTypeOfBType(
          btypeOfOnnxDataType(tensor.data_type()), loc.getContext())) {
    emitError(loc, "unsupported initializer source element type");
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
  Builder builder(loc.getContext());
  SmallVector<StringRef> scopeRefs;
  llvm::transform(scope, std::back_inserter(scopeRefs),
      [](const std::string &component) { return StringRef(component); });
  auto createSource = [&](StringRef kind, StringRef path,
                          std::optional<uint64_t> offset,
                          std::optional<uint64_t> length,
                          CallerOwnedDataAttr callerOwnedData = {}) {
    SmallVector<NamedAttribute> fields{
        builder.getNamedAttr("kind", builder.getStringAttr(kind)),
        builder.getNamedAttr("path", builder.getStringAttr(path)),
        builder.getNamedAttr("scope", builder.getStrArrayAttr(scopeRefs)),
        builder.getNamedAttr("name", builder.getStringAttr(tensor.name()))};
    if (offset)
      fields.push_back(
          builder.getNamedAttr("offset", builder.getI64IntegerAttr(*offset)));
    if (length)
      fields.push_back(
          builder.getNamedAttr("length", builder.getI64IntegerAttr(*length)));
    OperationState state(loc, ONNXInitializerSourceOp::getOperationName());
    state.addTypes(type);
    state.addAttributes(fields);
    if (callerOwnedData)
      state.addAttribute("caller_owned_data", callerOwnedData);
    return cast<ONNXInitializerSourceOp>(opBuilder.create(state));
  };
  auto resolveModelPath = [&](StringRef modelPath) -> FailureOr<std::string> {
    if (modelPath.empty()) {
      emitError(loc, "embedded initializer source requires a backup model");
      return failure();
    }
    SmallString<256> path;
    if (llvm::sys::path::is_relative(modelPath)) {
      if (options.externalDataDir.empty()) {
        emitError(
            loc, "relative backup model requires an explicit source base");
        return failure();
      }
      path = options.externalDataDir;
      llvm::sys::path::append(path, modelPath);
    } else {
      path = modelPath;
    }
    if (auto ec = llvm::sys::fs::make_absolute(path)) {
      emitError(loc) << "cannot resolve initializer source path: "
                     << ec.message();
      return failure();
    }
    return path.str().str();
  };

  if (tensor.data_location() != onnx::TensorProto::EXTERNAL) {
    auto path = resolveModelPath(options.initializerSourceModelPath);
    if (failed(path))
      return failure();
    CallerOwnedDataAttr callerOwnedTensor;
    if (retainTensorReference)
      callerOwnedTensor = CallerOwnedDataAttr::get(loc.getContext(), &tensor);
    return createSource(
        modelSourceKind, *path, std::nullopt, std::nullopt, callerOwnedTensor);
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
  CallerOwnedDataAttr callerOwnedDataAttr;
  if (callerOwnedData.data())
    callerOwnedDataAttr =
        CallerOwnedDataAttr::get(loc.getContext(), callerOwnedData);
  if (kind == externalDataSourceKind)
    return createSource(kind, absolute, offset, length, callerOwnedDataAttr);
  return createSource(
      kind, absolute, std::nullopt, std::nullopt, callerOwnedDataAttr);
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
