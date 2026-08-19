#ifndef ONNX_MLIR_DECOMPOSE_LSTM_H
#define ONNX_MLIR_DECOMPOSE_LSTM_H

#include <functional>

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

/// Optional caller policy that decides whether a particular LSTM may be
/// decomposed. An empty predicate permits every otherwise supported LSTM.
using LSTMDecompositionPredicate = std::function<bool(mlir::Operation *)>;

/// Populate LSTM-to-basic-ONNX-operations decomposition.
/// Static sequences of length greater than one are represented by onnx.Loop.
void populateDecomposeLSTMPatterns(mlir::RewritePatternSet &patterns,
    mlir::PatternBenefit benefit = mlir::PatternBenefit(1),
    LSTMDecompositionPredicate predicate = {});

} // namespace onnx_mlir

#endif
