#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace apollo {
namespace common {
namespace tracking {

enum class TrackStatus { kTentative, kConfirmed, kLost, kDeleted };

template <typename Payload>
struct Observation {
  uint64_t id = 0;
  std::shared_ptr<const Payload> payload;
};

template <typename Payload>
struct TrackingFrame {
  double timestamp = 0.0;
  std::string coordinate_frame;
  std::vector<Observation<Payload>> observations;
};

template <typename State>
struct StateSample {
  double timestamp = 0.0;
  State state;
  bool observed = false;
  std::optional<double> association_cost;
};

template <typename State, typename Payload>
struct TrackSnapshot {
  uint64_t id = 0;
  TrackStatus status = TrackStatus::kTentative;
  State state;
  double first_seen_time = 0.0;
  double last_observed_time = 0.0;
  double state_time = 0.0;
  size_t hits = 0;
  size_t consecutive_hits = 0;
  size_t consecutive_misses = 0;
  bool observed = false;
  std::shared_ptr<const Payload> last_observation;
  std::vector<StateSample<State>> history;
  // Absent for births; retained through missed frames with last_observed_time.
  std::optional<double> last_association_cost;
};

struct CostComponent {
  std::string name;
  double value = 0.0;
  double weight = 0.0;
};

struct CostEvaluation {
  bool allowed = false;
  double value = 0.0;
  std::string reason;
  std::vector<CostComponent> components;

  static CostEvaluation Accept(double value) { return {true, value, {}, {}}; }
  static CostEvaluation Reject(std::string reason) {
    return {false, 0.0, std::move(reason), {}};
  }
};

struct Match {
  size_t track_index = 0;
  size_t observation_index = 0;
  double cost = 0.0;
};

struct AssignmentResult {
  std::vector<Match> matches;
  std::vector<size_t> unmatched_tracks;
  std::vector<size_t> unmatched_observations;
  double total_cost = 0.0;
};

struct AssociationProblem {
  size_t track_count = 0;
  size_t observation_count = 0;
  // Row-major, with forbidden edges represented explicitly.
  std::vector<CostEvaluation> edges;
  double unmatched_cost = 1.0;
};

struct LifecycleEvent {
  uint64_t track_id = 0;
  TrackStatus from = TrackStatus::kDeleted;
  TrackStatus to = TrackStatus::kTentative;
  std::string reason;
};

struct BirthDecision {
  bool allowed = true;
  std::string reason;
};

struct BirthRejection {
  uint64_t observation_id = 0;
  std::string reason;
};

struct AssociationTrace {
  uint64_t track_id = 0;
  uint64_t observation_id = 0;
  CostEvaluation evaluation;
};

struct TrackingDiagnostics {
  size_t candidate_count = 0;
  size_t allowed_count = 0;
  size_t rejected_count = 0;
  size_t omitted_trace_count = 0;
  double elapsed_seconds = 0.0;
  std::vector<AssociationTrace> association_traces;
  std::vector<LifecycleEvent> lifecycle_events;
  std::vector<BirthRejection> rejected_births;
};

struct LifecycleConfig {
  size_t confirmation_hits = 3;
  size_t tentative_max_misses = 0;
  double lost_timeout_seconds = 1.0;
  size_t history_limit = 0;
  bool publish_tentative = false;
  bool publish_lost = false;
};

struct TrackerConfig {
  LifecycleConfig lifecycle;
  double unmatched_cost = 1.0;
  size_t max_candidate_pairs = 1000000;
  size_t trace_limit = 0;
  size_t max_tracks = 1000;
  size_t max_observations_per_frame = 10000;
  size_t max_cost_components = 32;
};

struct IdentifiedMatch {
  uint64_t track_id = 0;
  uint64_t observation_id = 0;
  double cost = 0.0;
};

template <typename State, typename Payload>
struct TrackingResult {
  double timestamp = 0.0;
  std::vector<TrackSnapshot<State, Payload>> tracks;
  std::vector<IdentifiedMatch> matches;
  std::vector<uint64_t> unmatched_track_ids;
  std::vector<uint64_t> unmatched_observation_ids;
  TrackingDiagnostics diagnostics;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
