#include "modules/common/tracking/associator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"

namespace apollo {
namespace common {
namespace tracking {
namespace {

absl::Status Validate(const AssociationProblem& p) {
  if (!std::isfinite(p.unmatched_cost) || p.unmatched_cost <= 0.0 ||
      !std::isfinite(2.0 * p.unmatched_cost)) {
    return absl::InvalidArgumentError(
        "unmatched_cost must be finite and positive");
  }
  if (p.observation_count != 0 &&
      p.track_count >
          std::numeric_limits<size_t>::max() / p.observation_count) {
    return absl::InvalidArgumentError("association dimensions overflow");
  }
  if (p.edges.size() != p.track_count * p.observation_count) {
    return absl::InvalidArgumentError("association matrix shape mismatch");
  }
  if (p.track_count >=
          std::numeric_limits<size_t>::max() - p.observation_count ||
      !std::isfinite(
          p.unmatched_cost *
          static_cast<double>(p.track_count + p.observation_count))) {
    return absl::InvalidArgumentError(
        "assignment objective or dimensions overflow");
  }
  for (const auto& edge : p.edges) {
    if (edge.allowed && (!std::isfinite(edge.value) || edge.value < 0.0)) {
      return absl::InvalidArgumentError(
          "allowed cost must be finite and nonnegative");
    }
  }
  return absl::OkStatus();
}

AssignmentResult BuildResult(const AssociationProblem& p,
                             const std::vector<size_t>& columns) {
  AssignmentResult result;
  std::vector<bool> used(p.observation_count, false);
  for (size_t row = 0; row < p.track_count; ++row) {
    const size_t col = columns[row];
    if (col < p.observation_count) {
      const double cost = p.edges[row * p.observation_count + col].value;
      result.matches.push_back({row, col, cost});
      result.total_cost += cost;
      used[col] = true;
    } else {
      result.unmatched_tracks.push_back(row);
      result.total_cost += p.unmatched_cost;
    }
  }
  for (size_t col = 0; col < p.observation_count; ++col) {
    if (!used[col]) {
      result.unmatched_observations.push_back(col);
      result.total_cost += p.unmatched_cost;
    }
  }
  return result;
}

bool HasMatchableEdges(const AssociationProblem& p) {
  return std::any_of(
      p.edges.begin(), p.edges.end(), [&](const CostEvaluation& edge) {
        return edge.allowed && edge.value < 2.0 * p.unmatched_cost;
      });
}

struct FlowEdge {
  size_t to;
  size_t reverse;
  int capacity;
  double cost;
};

void AddEdge(size_t from, size_t to, double cost,
             std::vector<std::vector<FlowEdge>>* graph) {
  const size_t reverse_from = (*graph)[to].size();
  const size_t reverse_to = (*graph)[from].size();
  (*graph)[from].push_back({to, reverse_from, 1, cost});
  (*graph)[to].push_back({from, reverse_to, 0, -cost});
}

}  // namespace

absl::StatusOr<AssignmentResult> HungarianAssociator::Associate(
    const AssociationProblem& p) const {
  const auto status = Validate(p);
  if (!status.ok()) {
    return status;
  }
  if (!HasMatchableEdges(p)) {
    return BuildResult(p,
                       std::vector<size_t>(p.track_count, p.observation_count));
  }
  const size_t n = p.track_count;
  const size_t m = p.observation_count + n;
  if (m < n || m == std::numeric_limits<size_t>::max()) {
    return absl::InvalidArgumentError("assignment dimensions overflow");
  }
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
  std::vector<size_t> owner(m + 1, 0), previous(m + 1, 0);
  for (size_t row = 1; row <= n; ++row) {
    owner[0] = row;
    size_t col = 0;
    std::vector<double> best(m + 1, infinity);
    std::vector<bool> visited(m + 1, false);
    do {
      visited[col] = true;
      const size_t current_row = owner[col];
      double delta = infinity;
      size_t next = 0;
      for (size_t candidate = 1; candidate <= m; ++candidate) {
        if (visited[candidate]) {
          continue;
        }
        double cost = 2.0 * p.unmatched_cost;
        if (candidate <= p.observation_count) {
          const auto& edge =
              p.edges[(current_row - 1) * p.observation_count + candidate - 1];
          cost = edge.allowed && edge.value < 2.0 * p.unmatched_cost
                     ? edge.value
                     : infinity;
        }
        const double reduced = cost - u[current_row] - v[candidate];
        if (reduced < best[candidate]) {
          best[candidate] = reduced;
          previous[candidate] = col;
        }
        if (best[candidate] < delta) {
          delta = best[candidate];
          next = candidate;
        }
      }
      if (!std::isfinite(delta)) {
        return absl::InternalError("assignment has no augmenting path");
      }
      for (size_t candidate = 0; candidate <= m; ++candidate) {
        if (visited[candidate]) {
          u[owner[candidate]] += delta;
          v[candidate] -= delta;
        } else {
          best[candidate] -= delta;
        }
      }
      col = next;
    } while (owner[col] != 0);
    do {
      const size_t next = previous[col];
      owner[col] = owner[next];
      col = next;
    } while (col != 0);
  }
  std::vector<size_t> columns(n, p.observation_count);
  for (size_t col = 1; col <= m; ++col) {
    if (owner[col] != 0) {
      columns[owner[col] - 1] = col - 1;
    }
  }
  return BuildResult(p, columns);
}

absl::StatusOr<AssignmentResult> MinCostFlowAssociator::Associate(
    const AssociationProblem& p) const {
  const auto status = Validate(p);
  if (!status.ok()) {
    return status;
  }
  if (!HasMatchableEdges(p)) {
    return BuildResult(p,
                       std::vector<size_t>(p.track_count, p.observation_count));
  }
  const size_t source = 0;
  const size_t observation_offset = 1 + p.track_count;
  const size_t sink = observation_offset + p.observation_count;
  if (sink < observation_offset || observation_offset == 0 ||
      sink == std::numeric_limits<size_t>::max()) {
    return absl::InvalidArgumentError("flow dimensions overflow");
  }
  std::vector<std::vector<FlowEdge>> graph(sink + 1);
  for (size_t row = 0; row < p.track_count; ++row) {
    AddEdge(source, row + 1, 0.0, &graph);
    AddEdge(row + 1, sink, 2.0 * p.unmatched_cost, &graph);
    for (size_t col = 0; col < p.observation_count; ++col) {
      const auto& edge = p.edges[row * p.observation_count + col];
      if (edge.allowed && edge.value < 2.0 * p.unmatched_cost) {
        AddEdge(row + 1, observation_offset + col, edge.value, &graph);
      }
    }
  }
  for (size_t col = 0; col < p.observation_count; ++col) {
    AddEdge(observation_offset + col, sink, 0.0, &graph);
  }
  for (size_t flow = 0; flow < p.track_count; ++flow) {
    std::vector<double> distance(graph.size(),
                                 std::numeric_limits<double>::infinity());
    std::vector<size_t> parent(graph.size(), graph.size());
    std::vector<size_t> parent_edge(graph.size(), 0);
    distance[source] = 0.0;
    // Bellman-Ford handles negative residual edges without heuristic retries.
    for (size_t pass = 1; pass < graph.size(); ++pass) {
      bool changed = false;
      for (size_t from = 0; from < graph.size(); ++from) {
        if (!std::isfinite(distance[from])) {
          continue;
        }
        for (size_t index = 0; index < graph[from].size(); ++index) {
          const auto& edge = graph[from][index];
          if (edge.capacity > 0 &&
              distance[from] + edge.cost < distance[edge.to]) {
            distance[edge.to] = distance[from] + edge.cost;
            parent[edge.to] = from;
            parent_edge[edge.to] = index;
            changed = true;
          }
        }
      }
      if (!changed) {
        break;
      }
    }
    if (parent[sink] == graph.size()) {
      return absl::InternalError("flow has no augmenting path");
    }
    size_t path_length = 0;
    for (size_t to = sink; to != source; to = parent[to]) {
      if (parent[to] == graph.size() || ++path_length >= graph.size()) {
        return absl::InternalError("invalid residual augmenting path");
      }
      auto& edge = graph[parent[to]][parent_edge[to]];
      --edge.capacity;
      ++graph[to][edge.reverse].capacity;
    }
  }
  std::vector<size_t> columns(p.track_count, p.observation_count);
  for (size_t row = 0; row < p.track_count; ++row) {
    for (const auto& edge : graph[row + 1]) {
      if (edge.to >= observation_offset && edge.to < sink &&
          edge.capacity == 0) {
        columns[row] = edge.to - observation_offset;
      }
    }
  }
  return BuildResult(p, columns);
}

}  // namespace tracking
}  // namespace common
}  // namespace apollo
