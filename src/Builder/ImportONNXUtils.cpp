/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ImportONNXUtils.hpp ----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Helper methods for importing and cleaning of onnx models.
//
//===----------------------------------------------------------------------===//

#include <map>
#include <set>
#include <vector>

#include "src/Builder/ImportONNXUtils.hpp"

// Collect all names that are "pre-defined" for a graph: its inputs,
// initializers, and any names inherited from enclosing scopes.
static std::set<std::string> getAvailableNames(
    const onnx::GraphProto &graph,
    const std::set<std::string> &outerScopeNames) {
  std::set<std::string> available(outerScopeNames);
  for (const auto &initializer : graph.initializer())
    available.insert(initializer.name());
  for (const auto &input : graph.input())
    available.insert(input.name());
  // Empty input names are placeholders and should always be ignored.
  available.insert("");
  return available;
}

bool IsTopologicallySorted(const onnx::GraphProto &graph,
    const std::set<std::string> &outerScopeNames) {
  std::set<std::string> visited = getAvailableNames(graph, outerScopeNames);
  for (const auto &node : graph.node()) {
    for (const auto &input : node.input()) {
      if (!visited.count(input))
        return false;
    }
    for (const auto &output : node.output()) {
      visited.insert(output);
    }
  }
  return true;
}

// Sort graph into lexicographically smallest topological ordering.
// Returns true if sorted succesfully and false otherwise.
bool SortGraph(onnx::GraphProto *graph,
    const std::set<std::string> &outerScopeNames) {
  int nNodes = graph->node().size();
  // Map of edges / node-outputs to their parent ops
  std::map<std::string, int> origIndex;
  int index = 0;
  for (const auto &node : graph->node()) {
    for (const auto &output : node.output()) {
      origIndex[output] = index;
    }
    index++;
  }
  assert(index == nNodes);

  // Names that don't create intra-graph dependencies: graph inputs,
  // initializers, outer-scope names, and empty placeholders.
  std::set<std::string> predefined = getAvailableNames(*graph, outerScopeNames);

  // Users tracks idx of the ops which consumes a given ops outputs.
  std::vector<std::vector<int>> users(nNodes);
  index = 0;
  for (const auto &node : graph->node()) {
    for (const auto &input : node.input()) {
      if (predefined.count(input))
        continue;
      // Input not predefined and not produced by any node in this graph:
      // the graph references an undefined name and cannot be sorted.
      if (!origIndex.count(input))
        return false;
      // Add current node as a user of the op that produces input.
      users[origIndex[input]].push_back(index);
    }
    index++;
  }

  // inDegrees stores the number of inputs to a given node not counting
  // predefined names.
  std::vector<int> inDegrees(nNodes, 0);
  index = 0;
  for (const auto &node : graph->node()) {
    for (const auto &input : node.input()) {
      if (!predefined.count(input)) {
        inDegrees[index]++;
      }
    }
    index++;
  }
  assert(index == nNodes);

  // Create a set and inserting all nodes with indegree 0.
  std::multiset<int> nodeList;
  for (int i = 0; i < nNodes; i++) {
    if (inDegrees[i] == 0) {
      nodeList.insert(i);
    }
  }

  // The number of visited nodes.
  int nVisited = 0;
  // The final topological order.
  std::vector<int> topOrder;

  // Now we follow Kahn's algorithm for topological sorting
  while (!nodeList.empty()) {
    // Extract node with minimum number from multiset
    // and add it to topological order.
    int u = *nodeList.begin();
    nodeList.erase(nodeList.begin());
    topOrder.push_back(u);

    // Iterate through all its users
    // and decreament inDegrees by 1.
    for (auto v : users[u]) {
      // If inDegree becomes zero, add it to queue.
      if (--inDegrees[v] == 0) {
        nodeList.insert(v);
      }
    }
    nVisited++;
  }
  // No possible topological order.
  if (nVisited != nNodes) {
    return false;
  }

  // Generate SwapElements to reach desired order.
  std::vector<int> curOrder(nNodes);
  for (int i = 0; i < nNodes; i++)
    curOrder[i] = i;
  for (int resIndex = 0; resIndex < nNodes; resIndex++) {
    if (topOrder[resIndex] == curOrder[resIndex])
      continue;
    for (int search = resIndex + 1; search < nNodes; search++) {
      if (topOrder[resIndex] == curOrder[search]) {
        graph->mutable_node()->SwapElements(resIndex, search);
        std::swap(curOrder[search], curOrder[resIndex]);
        break;
      }
    }
  }
  return true; // Succesfully sorted graph.
}

bool SortAllSubgraphs(onnx::GraphProto *graph,
    const std::set<std::string> &outerScopeNames) {
  // Build the full set of names visible in this graph: outer scope + this
  // graph's own inputs, initializers, and all node outputs.
  std::set<std::string> scopeNames = getAvailableNames(*graph, outerScopeNames);
  for (const auto &node : graph->node()) {
    for (const auto &output : node.output())
      scopeNames.insert(output);
  }

  // Recurse into subgraphs. They can see everything defined in this graph.
  for (auto &node : *graph->mutable_node()) {
    for (auto &attr : *node.mutable_attribute()) {
      if (attr.has_g()) {
        if (!SortAllSubgraphs(attr.mutable_g(), scopeNames))
          return false;
      }
      for (auto &g : *attr.mutable_graphs()) {
        if (!SortAllSubgraphs(&g, scopeNames))
          return false;
      }
    }
  }

  // Sort this graph itself if needed.
  if (!IsTopologicallySorted(*graph, outerScopeNames)) {
    if (!SortGraph(graph, outerScopeNames))
      return false;
  }
  return true;
}
