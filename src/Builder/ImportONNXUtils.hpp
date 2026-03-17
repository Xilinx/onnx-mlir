/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ImportONNXUtils.hpp ----------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// Helper methods for importing and cleaning of onnx models.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_IMPORT_UTILS_H
#define ONNX_MLIR_IMPORT_UTILS_H

#include <set>
#include <string>

#include "onnx/onnx_pb.h"

// Check whether the graph nodes are in topological order.
// |outerScopeNames| contains names defined in enclosing graphs that are
// implicitly available (e.g. for Loop body / If branch subgraphs).
bool IsTopologicallySorted(const onnx::GraphProto &graph,
    const std::set<std::string> &outerScopeNames = {});

// Sort graph into lexicographically smallest topological ordering.
// |outerScopeNames| contains names defined in enclosing graphs that are
// implicitly available and don't create intra-graph dependencies.
// Returns true if sorted successfully, false if the graph has a cycle.
bool SortGraph(onnx::GraphProto *graph,
    const std::set<std::string> &outerScopeNames = {});

// Recursively sort all subgraphs (Loop bodies, If branches, etc.) that are
// not topologically sorted. Propagates the set of available names from
// enclosing scopes so that outer-scope references are correctly resolved.
// Returns true if all subgraphs were sorted successfully.
bool SortAllSubgraphs(onnx::GraphProto *graph,
    const std::set<std::string> &outerScopeNames = {});
#endif
