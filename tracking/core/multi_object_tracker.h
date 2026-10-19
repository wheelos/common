// Copyright 2026 WheelOS. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tracking/association/assignment_solver.h"
#include "tracking/association/association_strategy.h"
#include "tracking/core/track_manager.h"
#include "tracking/costs/association_cost.h"

namespace apollo {
namespace common {
namespace tracking {

template <typename State, typename Payload>
class MultiObjectTracker {
 public:
  using Track = TrackSnapshot<State, Payload>;
  using Result = TrackingResult<State, Payload>;

  MultiObjectTracker(
      TrackerConfig config,
      std::shared_ptr<const StateEstimator<State, Payload>> estimator,
      std::shared_ptr<const AssociationCost<State, Payload>> cost,
      std::shared_ptr<const AssignmentSolver> assignment_solver,
      std::shared_ptr<const TrackLifecyclePolicy<State, Payload>> lifecycle =
          std::make_shared<TrackLifecyclePolicy<State, Payload>>(),
      std::shared_ptr<const AssociationStrategy> association_strategy = nullptr)
      : config_(std::move(config)),
        estimator_(std::move(estimator)),
        cost_(std::move(cost)),
        assignment_solver_(std::move(assignment_solver)),
        association_strategy_(std::move(association_strategy)),
        manager_(config_.lifecycle, std::move(lifecycle)) {}

  // Plugins must be side-effect-free; state is committed only on success.
  absl::StatusOr<Result> Update(const TrackingFrame<Payload>& frame) {
    const auto begin = std::chrono::steady_clock::now();
    const auto valid = Validate(frame);
    if (!valid.ok()) {
      return valid;
    }
    Result result;
    result.timestamp = frame.timestamp;
    std::string coordinate_frame = frame.coordinate_frame;
    auto working = tracks_;
    uint64_t next_id = next_id_;
    for (auto& track : working) {
      track.observed = false;
      const auto expiration =
          manager_.Expire(frame.timestamp, &track, &result.diagnostics);
      if (!expiration.ok()) {
        return expiration;
      }
      if (track.status == TrackStatus::kDeleted) {
        result.tracking_events.push_back(
            {TrackingEventType::kExpired, track.id});
        continue;
      }
      const auto predicted =
          estimator_->Predict(frame.timestamp - track.state_time, &track.state);
      if (!predicted.ok()) {
        return predicted;
      }
      track.state_time = frame.timestamp;
    }
    manager_.Prune(&working);
    const size_t count = frame.observations.size();
    if (count != 0 && working.size() > config_.max_candidate_pairs / count) {
      return absl::ResourceExhaustedError(
          "association candidate limit exceeded");
    }
    AssociationProblem problem;
    problem.track_count = working.size();
    problem.observation_count = count;
    problem.unmatched_cost = config_.unmatched_cost;
    problem.max_solver_work = config_.max_solver_work;
    problem.edges.reserve(working.size() * count);
    for (const auto& track : working) {
      for (const auto& observation : frame.observations) {
        auto evaluation = cost_->Evaluate(track, observation);
        if (!evaluation.ok()) {
          return evaluation.status();
        }
        const auto cost_valid =
            ValidateCostEvaluation(*evaluation, config_.max_cost_components);
        if (!cost_valid.ok()) {
          return cost_valid;
        }
        ++result.diagnostics.candidate_count;
        if (evaluation->allowed) {
          ++result.diagnostics.allowed_count;
        } else {
          ++result.diagnostics.rejected_count;
        }
        if (result.diagnostics.association_traces.size() <
            config_.trace_limit) {
          result.diagnostics.association_traces.push_back(
              {track.id, observation.id, *evaluation});
        } else {
          ++result.diagnostics.omitted_trace_count;
        }
        problem.edges.push_back(std::move(*evaluation));
      }
    }
    absl::StatusOr<AssignmentResult> assignment =
        association_strategy_
            ? association_strategy_->Associate(problem, GetConfidences(frame),
                                               GetTrackStatuses(working),
                                               *assignment_solver_)
            : assignment_solver_->Solve(problem);
    if (!assignment.ok()) {
      return assignment.status();
    }
    const auto assignment_status = ValidateAssignment(problem, *assignment);
    if (!assignment_status.ok()) {
      return assignment_status;
    }
    for (const auto& match : assignment->matches) {
      auto& track = working[match.track_index];
      const auto& observation = frame.observations[match.observation_index];
      const auto updated =
          estimator_->Update(*observation.payload, &track.state);
      if (!updated.ok()) {
        return updated;
      }
      const auto hit = manager_.Hit(frame.timestamp, observation, &track,
                                    &result.diagnostics, match.cost);
      if (!hit.ok()) {
        return hit;
      }
      result.matches.push_back({track.id, observation.id, match.cost});
      result.tracking_events.push_back(
          {TrackingEventType::kMatched, track.id, observation.id, match.cost});
      if (track.status == TrackStatus::kDeleted) {
        result.tracking_events.push_back(
            {TrackingEventType::kExpired, track.id});
      }
    }
    for (size_t index : assignment->unmatched_tracks) {
      auto& track = working[index];
      result.unmatched_track_ids.push_back(track.id);
      const auto miss =
          manager_.Miss(frame.timestamp, &track, &result.diagnostics);
      if (!miss.ok()) {
        return miss;
      }
      if (track.status == TrackStatus::kDeleted) {
        result.tracking_events.push_back(
            {TrackingEventType::kExpired, track.id});
      }
    }
    manager_.Prune(&working);
    for (size_t index : assignment->unmatched_observations) {
      const auto& observation = frame.observations[index];
      result.unmatched_observation_ids.push_back(observation.id);
      if (association_strategy_ &&
          !association_strategy_->CanStartTrack(observation.confidence)) {
        result.diagnostics.rejected_births.push_back(
            {observation.id,
             "association strategy rejected track birth confidence"});
        continue;
      }
      auto birth = manager_.Admit(observation);
      if (!birth.ok()) {
        return birth.status();
      }
      if (!birth->allowed) {
        result.diagnostics.rejected_births.push_back(
            {observation.id, std::move(birth->reason)});
        continue;
      }
      if (working.size() >= config_.max_tracks) {
        return absl::ResourceExhaustedError("active track limit exceeded");
      }
      if (next_id == std::numeric_limits<uint64_t>::max()) {
        return absl::ResourceExhaustedError("track IDs exhausted");
      }
      auto state = estimator_->Initialize(*observation.payload);
      if (!state.ok()) {
        return state.status();
      }
      auto track = manager_.Birth(next_id++, frame.timestamp, std::move(*state),
                                  observation, &result.diagnostics);
      if (!track.ok()) {
        return track.status();
      }
      if (track->status != TrackStatus::kDeleted) {
        const uint64_t track_id = track->id;
        working.push_back(std::move(*track));
        result.tracking_events.push_back(
            {TrackingEventType::kAdded, track_id, observation.id});
      } else {
        result.tracking_events.push_back(
            {TrackingEventType::kExpired, track->id});
      }
    }
    for (auto& track : working) {
      manager_.Record(&track);
      if (manager_.Publish(track)) {
        result.tracks.push_back(track);
      }
    }
    result.diagnostics.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
            .count();
    tracks_ = std::move(working);
    next_id_ = next_id;
    last_timestamp_ = frame.timestamp;
    coordinate_frame_.swap(coordinate_frame);
    initialized_ = true;
    return result;
  }

  const std::vector<Track>& tracks() const { return tracks_; }

  // Clearing never recycles IDs. A new instance defines a new ID namespace.
  void Reset() {
    tracks_.clear();
    initialized_ = false;
    coordinate_frame_.clear();
  }

 private:
  absl::Status Validate(const TrackingFrame<Payload>& frame) const {
    if (!estimator_ || !cost_ || !assignment_solver_) {
      return absl::InvalidArgumentError("tracking plugins must not be null");
    }
    const auto lifecycle_valid = manager_.Validate();
    if (!lifecycle_valid.ok()) {
      return lifecycle_valid;
    }
    if (!std::isfinite(config_.unmatched_cost) ||
        config_.unmatched_cost <= 0.0 ||
        !std::isfinite(2.0 * config_.unmatched_cost) ||
        config_.max_solver_work == 0) {
      return absl::InvalidArgumentError("invalid tracking configuration");
    }
    if (!std::isfinite(frame.timestamp) || frame.coordinate_frame.empty()) {
      return absl::InvalidArgumentError(
          "finite time and coordinate frame required");
    }
    if (initialized_ && (frame.timestamp <= last_timestamp_ ||
                         frame.coordinate_frame != coordinate_frame_ ||
                         !std::isfinite(frame.timestamp - last_timestamp_))) {
      return absl::InvalidArgumentError(
          "frames must have strictly increasing times and identical "
          "coordinates");
    }
    if (frame.observations.size() > config_.max_observations_per_frame) {
      return absl::ResourceExhaustedError("observation limit exceeded");
    }
    std::unordered_set<uint64_t> ids;
    for (const auto& observation : frame.observations) {
      if (!observation.payload || !ids.insert(observation.id).second ||
          !std::isfinite(observation.confidence) ||
          observation.confidence < 0.0 || observation.confidence > 1.0) {
        return absl::InvalidArgumentError(
            "invalid payload, confidence, or duplicate observation ID");
      }
      const auto valid = estimator_->Validate(*observation.payload);
      if (!valid.ok()) {
        return valid;
      }
    }
    return absl::OkStatus();
  }

  static std::vector<double> GetConfidences(
      const TrackingFrame<Payload>& frame) {
    std::vector<double> confidences;
    confidences.reserve(frame.observations.size());
    for (const auto& observation : frame.observations) {
      confidences.push_back(observation.confidence);
    }
    return confidences;
  }

  static std::vector<TrackStatus> GetTrackStatuses(
      const std::vector<Track>& tracks) {
    std::vector<TrackStatus> statuses;
    statuses.reserve(tracks.size());
    for (const auto& track : tracks) {
      statuses.push_back(track.status);
    }
    return statuses;
  }

  static absl::Status ValidateAssignment(const AssociationProblem& problem,
                                         const AssignmentResult& assignment) {
    double expected_cost = 0.0;
    std::vector<bool> rows(problem.track_count, false);
    std::vector<bool> columns(problem.observation_count, false);
    for (const auto& match : assignment.matches) {
      if (match.track_index >= rows.size() ||
          match.observation_index >= columns.size() ||
          rows[match.track_index] || columns[match.observation_index]) {
        return absl::InternalError(
            "assignment solver returned invalid or duplicate match");
      }
      const auto& edge = problem.edges[match.track_index * columns.size() +
                                       match.observation_index];
      if (!edge.allowed || !std::isfinite(match.cost) ||
          match.cost != edge.value) {
        return absl::InternalError(
            "assignment solver returned forbidden or incorrect cost");
      }
      rows[match.track_index] = true;
      columns[match.observation_index] = true;
      expected_cost += match.cost;
    }
    for (size_t index : assignment.unmatched_tracks) {
      if (index >= rows.size() || rows[index]) {
        return absl::InternalError("invalid unmatched track");
      }
      rows[index] = true;
      expected_cost += problem.unmatched_cost;
    }
    for (size_t index : assignment.unmatched_observations) {
      if (index >= columns.size() || columns[index]) {
        return absl::InternalError("invalid unmatched observation");
      }
      columns[index] = true;
      expected_cost += problem.unmatched_cost;
    }
    if (std::find(rows.begin(), rows.end(), false) != rows.end() ||
        std::find(columns.begin(), columns.end(), false) != columns.end()) {
      return absl::InternalError(
          "assignment solver returned incomplete partition");
    }
    if (!std::isfinite(expected_cost) ||
        !std::isfinite(assignment.total_cost) || assignment.total_cost < 0.0 ||
        std::abs(expected_cost - assignment.total_cost) >
            1e-10 * std::max(1.0, expected_cost)) {
      return absl::InternalError(
          "assignment solver returned an invalid objective");
    }
    return absl::OkStatus();
  }

  TrackerConfig config_;
  std::shared_ptr<const StateEstimator<State, Payload>> estimator_;
  std::shared_ptr<const AssociationCost<State, Payload>> cost_;
  std::shared_ptr<const AssignmentSolver> assignment_solver_;
  std::shared_ptr<const AssociationStrategy> association_strategy_;
  TrackManager<State, Payload> manager_;
  std::vector<Track> tracks_;
  uint64_t next_id_ = 1;
  double last_timestamp_ = 0.0;
  std::string coordinate_frame_;
  bool initialized_ = false;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
