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

using namespace mlir;
namespace onnx_mlir {
namespace {
constexpr StringLiteral ortMemoryLocation = "*/_ORT_MEM_ADDR_/*";

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

FailureOr<ElementsAttr> resolve(ONNXInitializerSourceOp op) {
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
  uint64_t offset = disk.getAs<IntegerAttr>("offset").getInt();
  uint64_t length = disk.getAs<IntegerAttr>("length").getInt();
  if (length != *size) {
    op.emitError("source length disagrees with tensor payload size");
    return failure();
  }
  ArrayRef<char> live = source.getLive();
  if (live.data()) {
    if (live.size() != length) {
      op.emitError("initializer source range exceeds available live bytes");
      return failure();
    }
    return externalTensorBytesToElmAttr(type, dataType, live);
  }
  auto path = disk.getAs<StringAttr>("path").getValue();
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
    return "Resolve external ONNX initializer sources into disposable "
           "constants";
  }
  void runOnOperation() final {
    if (skipAll)
      return;
    SmallVector<ONNXInitializerSourceOp> sources;
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
      auto value = resolve(op);
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
  if (location == ortMemoryLocation) {
    auto identity =
        scope.empty() ? tensor.name() : scope.str() + "/" + tensor.name();
    auto backup = options.externalInitializerBackups.find(identity);
    if (backup == options.externalInitializerBackups.end() ||
        backup->second.path.empty() || !backup->second.live.data()) {
      emitError(loc)
          << "memory initializer " << identity
          << " requires a borrowed live view and explicit disk backup";
      return failure();
    }
    const auto &source = backup->second;
    live = source.live;
    // Only compare address metadata; never read bytes during import.
    if (offset != reinterpret_cast<uintptr_t>(live.data()) || length != *size ||
        live.size() != *size || source.length != *size) {
      emitError(loc, "memory initializer address/length disagrees with live "
                     "view or backup");
      return failure();
    }
    if (llvm::sys::path::is_relative(source.path)) {
      if (options.externalDataDir.empty()) {
        emitError(loc, "relative disk backup requires an explicit source base");
        return failure();
      }
      SmallString<256> joined(options.externalDataDir);
      llvm::sys::path::append(joined, source.path);
      path = joined.str().str();
    } else {
      path = source.path;
    }
    offset = source.offset;
    length = source.length;
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
  if (length != *size || offset > uint64_t(INT64_MAX) ||
      length > uint64_t(INT64_MAX) - offset) {
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
  auto disk = builder.getDictionaryAttr(
      {builder.getNamedAttr("path", builder.getStringAttr(absolute)),
          builder.getNamedAttr("scope", builder.getStringAttr(scope)),
          builder.getNamedAttr("name", builder.getStringAttr(tensor.name())),
          builder.getNamedAttr("offset", builder.getI64IntegerAttr(offset)),
          builder.getNamedAttr("length", builder.getI64IntegerAttr(length)),
          builder.getNamedAttr(
              "data_type", builder.getI64IntegerAttr(tensor.data_type()))});
  return InitializerSourceAttr::get(disk, live);
}
} // namespace onnx_mlir
