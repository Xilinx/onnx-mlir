// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
#include "src/Builder/InitializerSource.hpp"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "src/Dialect/ONNX/CallerOwnedDataAttr.hpp"
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

// ONNX external_data metadata only; parsing never reads tensor contents.
// An absent length is distinct from an explicitly specified zero.
struct ExternalDataEntries {
  std::string location;
  uint64_t offset = 0;
  std::optional<uint64_t> length;
};

FailureOr<ExternalDataEntries> parseExternalDataEntries(
    const onnx::TensorProto &tensor, Location loc,
    StringRef invalidEntryMessage) {
  ExternalDataEntries result;
  for (const auto &entry : tensor.external_data()) {
    if (entry.key() == "location")
      result.location = entry.value();
    else if (entry.key() == "offset" || entry.key() == "length") {
      uint64_t value;
      if (StringRef(entry.value()).getAsInteger(10, value)) {
        emitError(loc) << invalidEntryMessage << entry.key();
        return failure();
      }
      if (entry.key() == "offset")
        result.offset = value;
      else
        result.length = value;
    }
  }
  return result;
}

bool containsParentTraversal(StringRef path) {
  for (auto component = llvm::sys::path::begin(path),
            end = llvm::sys::path::end(path);
      component != end; ++component)
    if (*component == "..")
      return true;
  return false;
}

SmallVector<StringRef> getInitializerScope(ONNXInitializerSourceOp op) {
  SmallVector<StringRef> result;
  for (Attribute component : op.getScope())
    result.push_back(cast<StringAttr>(component).getValue());
  return result;
}

InitializerIdentity getSourceIdentity(ONNXInitializerSourceOp op) {
  InitializerIdentity identity;
  llvm::transform(getInitializerScope(op), std::back_inserter(identity.scope),
      [](StringRef component) { return component.str(); });
  identity.name = op.getName().str();
  return identity;
}

ONNXInitializerSourceOp findStructuralInitializerSource(Value value) {
  while (auto argument = dyn_cast<BlockArgument>(value)) {
    Block *block = argument.getOwner();
    auto loop = dyn_cast<ONNXLoopOp>(block->getParentOp());
    unsigned index = argument.getArgNumber();
    if (!loop || loop.getRegion().empty() ||
        block != &loop.getRegion().front() || index == 0 ||
        index >= loop->getNumOperands())
      return {};
    // Body argument 0 is the iteration counter, not the trip-count operand.
    // The condition and loop-carried arguments align with the initial operands;
    // materializing those operands does not make later iterations constant.
    value = loop->getOperand(index);
  }
  return value.getDefiningOp<ONNXInitializerSourceOp>();
}

bool matchesNestedIdentity(SymbolRefAttr identity, ONNXInitializerSourceOp op) {
  auto scope = getInitializerScope(op);
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

// Reads and decodes the specified byte range using the external-file cache.
FailureOr<ElementsAttr> readExternalInitializerRange(ONNXInitializerSourceOp op,
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
      op.emitError("ambiguous initializer in source model: ") << name;
      return failure();
    }
    match = &tensor;
  }
  if (!match) {
    op.emitError("cannot find initializer in source model: ") << name;
    return failure();
  }
  return match;
}

// Cache parsed source models across tensor lookups.
FailureOr<ElementsAttr> readInitializerFromSourceModel(
    ONNXInitializerSourceOp op, RankedTensorType type, BType btype,
    uint64_t expectedSize, ModelCache &models) {
  StringRef modelPath = op.getPath();
  auto found = models.find(modelPath.str());
  if (found == models.end()) {
    auto model = std::make_shared<onnx::ModelProto>();
    std::string errorMessage;
    if (LoadModelProtoFromFile(modelPath, errorMessage, *model)) {
      op.emitError("cannot read initializer source model ")
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
    op.emitError("initializer type or shape disagrees with source");
    return failure();
  }
  if ((*tensor)->data_location() != onnx::TensorProto::EXTERNAL)
    return onnxTensorProtoToElmAttr(op.getContext(), "", **tensor);

  auto metadata = parseExternalDataEntries(**tensor, op.getLoc(),
      "invalid external metadata in source initializer: ");
  if (failed(metadata))
    return failure();
  StringRef location = metadata->location;
  if (location.empty() || location == ortMemoryLocation) {
    op.emitError("source initializer has no durable external location");
    return failure();
  }
  if (llvm::sys::path::is_absolute(location) ||
      containsParentTraversal(location)) {
    op.emitError("source initializer location must be relative without '..'");
    return failure();
  }
  if (metadata->length && *metadata->length != expectedSize) {
    op.emitError("initializer length disagrees with source");
    return failure();
  }
  SmallString<256> path(llvm::sys::path::parent_path(modelPath));
  llvm::sys::path::append(path, location);
  return readExternalInitializerRange(op, type, path, metadata->offset,
      metadata->length.value_or(expectedSize));
}

// Resolves caller-owned data first, falling back to the durable source locator.
// This reads payloads but does not modify IR.
FailureOr<ElementsAttr> resolveInitializerContents(
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
    return readInitializerFromSourceModel(op, type, *btype, *size, models);
  uint64_t offset = *op.getOffset();
  uint64_t length = *op.getLength();
  if (length != *size) {
    op.emitError("source length disagrees with tensor payload size");
    return failure();
  }
  return readExternalInitializerRange(op, type, op.getPath(), offset, length);
}

// Durable locator and optional non-owning reference to caller-owned data.
// Preparation validates metadata and paths without opening files or reading
// tensor contents.
struct InitializerSourceDescriptor {
  StringRef kind;
  std::string path;
  std::optional<uint64_t> offset;
  std::optional<uint64_t> length;
  CallerOwnedDataAttr callerOwnedData;
};

FailureOr<std::string> makeAbsoluteSourcePath(StringRef path, Location loc) {
  SmallString<256> absolute(path);
  if (auto ec = llvm::sys::fs::make_absolute(absolute)) {
    emitError(loc) << "cannot resolve initializer source path: "
                   << ec.message();
    return failure();
  }
  return absolute.str().str();
}

// Embedded tensors use named source-model lookup. ModelProto entry points can
// also supply a non-owning reference to the caller-owned tensor.
FailureOr<InitializerSourceDescriptor> prepareEmbeddedSource(
    const onnx::TensorProto &tensor, bool referenceCallerOwnedTensor,
    const ImportOptions &options, Location loc) {
  StringRef modelPath = options.initializerSourceModelPath;
  if (modelPath.empty()) {
    emitError(loc, "embedded initializer source requires a source model");
    return failure();
  }
  SmallString<256> path(modelPath);
  if (llvm::sys::path::is_relative(path)) {
    if (options.externalDataDir.empty()) {
      emitError(loc, "relative source model requires an explicit source base");
      return failure();
    }
    path = options.externalDataDir;
    llvm::sys::path::append(path, modelPath);
  }
  auto absolute = makeAbsoluteSourcePath(path, loc);
  if (failed(absolute))
    return failure();
  CallerOwnedDataAttr callerOwnedTensor;
  if (referenceCallerOwnedTensor)
    callerOwnedTensor = CallerOwnedDataAttr::get(loc.getContext(), &tensor);
  return InitializerSourceDescriptor{modelSourceKind, *absolute, std::nullopt,
      std::nullopt, callerOwnedTensor};
}

FailureOr<InitializerSourceDescriptor> prepareCallerOwnedSource(
    const onnx::TensorProto &tensor, ArrayRef<std::string> scope,
    const ExternalDataEntries &metadata, uint64_t expectedSize,
    const ImportOptions &options, Location loc) {
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
  ArrayRef<char> data = source.callerOwnedData;
  // Compare address metadata only; never dereference caller-owned bytes here.
  if (metadata.offset != reinterpret_cast<uintptr_t>(data.data()) ||
      metadata.length.value_or(0) != expectedSize ||
      data.size() != expectedSize) {
    emitError(loc,
        "memory initializer address/length disagrees with caller-owned data");
    return failure();
  }
  SmallString<256> path(source.backupModelPath);
  if (llvm::sys::path::is_relative(path)) {
    if (options.externalDataDir.empty()) {
      emitError(loc, "relative backup model requires an explicit source base");
      return failure();
    }
    path = options.externalDataDir;
    llvm::sys::path::append(path, source.backupModelPath);
  }
  auto absolute = makeAbsoluteSourcePath(path, loc);
  if (failed(absolute))
    return failure();
  return InitializerSourceDescriptor{modelSourceKind, *absolute, std::nullopt,
      std::nullopt, CallerOwnedDataAttr::get(loc.getContext(), data)};
}

FailureOr<InitializerSourceDescriptor> prepareExternalFileSource(
    const ExternalDataEntries &metadata, uint64_t expectedSize,
    const ImportOptions &options, Location loc) {
  if (metadata.location.empty()) {
    emitError(loc, "external initializer requires a location");
    return failure();
  }
  if (llvm::sys::path::is_absolute(metadata.location) ||
      containsParentTraversal(metadata.location)) {
    emitError(loc,
        "external initializer location must be a relative path without '..'");
    return failure();
  }
  if (options.externalDataDir.empty()) {
    emitError(
        loc, "relative external location requires an explicit source base");
    return failure();
  }
  uint64_t length = metadata.length.value_or(expectedSize);
  if (length != expectedSize || metadata.offset > uint64_t(INT64_MAX) ||
      length > uint64_t(INT64_MAX) - metadata.offset) {
    emitError(loc, "invalid initializer source range or payload length");
    return failure();
  }
  SmallString<256> path(options.externalDataDir);
  llvm::sys::path::append(path, metadata.location);
  auto absolute = makeAbsoluteSourcePath(path, loc);
  if (failed(absolute))
    return failure();
  return InitializerSourceDescriptor{
      externalDataSourceKind, *absolute, metadata.offset, length, {}};
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
              return getInitializerScope(op).empty() && name == op.getName();
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
          identities.insert(getSourceIdentity(match));
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

    auto cache = createInitializerSourceCache();
    IRRewriter rewriter(getOperation().getContext());

    llvm::SetVector<ONNXInitializerSourceOp> structuralSources;
    getOperation().walk([&](StructuralOperandOpInterface op) {
      for (auto [index, operand] : llvm::enumerate(op->getOperands()))
        if (op.isStructuralOperand(index))
          if (auto source = findStructuralInitializerSource(operand))
            structuralSources.insert(source);
    });
    for (ONNXInitializerSourceOp source : structuralSources)
      if (failed(materializeInitializerSource(source, rewriter, *cache))) {
        signalPassFailure();
        return;
      }

    SmallVector<ONNXInitializerSourceOp> sources;
    getOperation().walk(
        [&](ONNXInitializerSourceOp op) { sources.push_back(op); });
    for (ONNXInitializerSourceOp source : sources) {
      auto identity = getSourceIdentity(source);
      if (!forcedIdentities.contains(identity) &&
          (skipAll || skippedIdentities.contains(identity)))
        continue;
      if (failed(materializeInitializerSource(source, rewriter, *cache))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

struct InitializerSourceCache {
  ModelCache models;
};

std::shared_ptr<InitializerSourceCache> createInitializerSourceCache() {
  return std::make_shared<InitializerSourceCache>();
}

FailureOr<ONNXConstantOp> materializeInitializerSource(
    ONNXInitializerSourceOp op, RewriterBase &rewriter,
    InitializerSourceCache &cache) {
  auto value = resolveInitializerContents(op, cache.models);
  if (failed(value))
    return failure();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  Value constantValue = OnnxBuilder(rewriter, op.getLoc()).constant(*value);
  auto constant = cast<ONNXConstantOp>(constantValue.getDefiningOp());
  for (auto attr : op->getDiscardableAttrs())
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

FailureOr<ONNXInitializerSourceOp> createInitializerSource(OpBuilder &builder,
    const onnx::TensorProto &tensor, RankedTensorType type,
    ArrayRef<std::string> scope, const ImportOptions &options, Location loc,
    bool isInput, bool referenceCallerOwnedTensor) {
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
  auto source = [&]() -> FailureOr<InitializerSourceDescriptor> {
    if (tensor.data_location() != onnx::TensorProto::EXTERNAL)
      return prepareEmbeddedSource(
          tensor, referenceCallerOwnedTensor, options, loc);
    auto metadata = parseExternalDataEntries(tensor, loc, "invalid external ");
    if (failed(metadata))
      return failure();
    if (metadata->location == ortMemoryLocation)
      return prepareCallerOwnedSource(
          tensor, scope, *metadata, *size, options, loc);
    return prepareExternalFileSource(*metadata, *size, options, loc);
  }();
  if (failed(source))
    return failure();
  SmallVector<StringRef> scopeRefs;
  llvm::transform(scope, std::back_inserter(scopeRefs),
      [](const std::string &component) { return StringRef(component); });
  return builder.create<ONNXInitializerSourceOp>(loc, type, source->kind,
      source->path, tensor.name(), builder.getStrArrayAttr(scopeRefs),
      source->offset ? builder.getI64IntegerAttr(*source->offset)
                     : IntegerAttr{},
      source->length ? builder.getI64IntegerAttr(*source->length)
                     : IntegerAttr{},
      source->callerOwnedData, isInput);
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
