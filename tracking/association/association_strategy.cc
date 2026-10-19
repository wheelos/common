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

#include "tracking/association/association_strategy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "absl/status/status.h"

namespace apollo {
namespace common {
namespace tracking {
namespace {

absl::Status ValidateAssignment(const AssociationProblem& problem,
                                const AssignmentResult& assignment) {
  std::vector<bool> tracks(problem.track_count, false);
  std::vector<bool> observations(problem.observation_count, false);
  double expected_cost = 0.0;
  for (const auto& match : assignment.matches) {
    if (match.track_index >= tracks.size() ||
        match.observation_index >= observations.size() ||
        tracks[match.track_index] || observations[match.observation_index]) {
      return absl::InternalError("assignment solver returned invalid match");
    }
    const auto& edge =
        problem.edges[match.track_index * problem.observation_count +
                      match.observation_index];
    if (!edge.allowed || !std::isfinite(match.cost) ||
        match.cost != edge.value) {
      return absl::InternalError("assignment solver returned invalid cost");
    }
    tracks[match.track_index] = true;
    observations[match.observation_index] = true;
    expected_cost += match.cost;
  }
  for (size_t index : assignment.unmatched_tracks) {
    if (index >= tracks.size() || tracks[index]) {
      return absl::InternalError("assignment solver returned invalid track");
    }
    tracks[index] = true;
    expected_cost += problem.unmatched_cost;
  }
  for (size_t index : assignment.unmatched_observations) {
    if (index >= observations.size() || observations[index]) {
      return absl::InternalError(
          "assignment solver returned invalid observation");
    }
    observations[index] = true;
    expected_cost += problem.unmatched_cost;
  }
  if (std::find(tracks.begin(), tracks.end(), false) != tracks.end() ||
      std::find(observations.begin(), observations.end(), false) !=
          observations.end() ||
      !std::isfinite(expected_cost) || !std::isfinite(assignment.total_cost) ||
      std::abs(expected_cost - assignment.total_cost) >
          1e-10 * std::max(1.0, expected_cost)) {
    return absl::InternalError(
        "assignment solver returned an incomplete partition or objective");
  }
  return absl::OkStatus();
}

AssociationProblem MakeSubproblem(
    const AssociationProblem& problem, const std::vector<size_t>& track_indices,
    const std::vector<size_t>& observation_indices) {
  AssociationProblem subproblem;
  subproblem.track_count = track_indices.size();
  subproblem.observation_count = observation_indices.size();
  subproblem.unmatched_cost = problem.unmatched_cost;
  subproblem.max_solver_work = problem.max_solver_work;
  subproblem.edges.reserve(track_indices.size() * observation_indices.size());
  for (size_t track : track_indices) {
    for (size_t observation : observation_indices) {
      subproblem.edges.push_back(
          problem.edges[track * problem.observation_count + observation]);
    }
  }
  return subproblem;
}

}  // namespace

absl::Status ByteTrackAssociationStrategy::ValidateConfig() const {
  if (!std::isfinite(config_.low_score_threshold) ||
      !std::isfinite(config_.high_score_threshold) ||
      !std::isfinite(config_.new_track_threshold) ||
      config_.low_score_threshold < 0.0 ||
      config_.low_score_threshold >= config_.high_score_threshold ||
      config_.high_score_threshold > 1.0 ||
      config_.new_track_threshold < config_.high_score_threshold ||
      config_.new_track_threshold > 1.0) {
    return absl::InvalidArgumentError(
        "invalid ByteTrack confidence thresholds");
  }
  return absl::OkStatus();
}

bool ByteTrackAssociationStrategy::CanStartTrack(double confidence) const {
  return ValidateConfig().ok() && std::isfinite(confidence) &&
         confidence >= config_.new_track_threshold && confidence <= 1.0;
}

absl::StatusOr<AssignmentResult> ByteTrackAssociationStrategy::Associate(
    const AssociationProblem& problem,
    const std::vector<double>& observation_confidences,
    const std::vector<TrackStatus>& track_statuses,
    const AssignmentSolver& solver) const {
  const auto config_status = ValidateConfig();
  if (!config_status.ok()) {
    return config_status;
  }
  if (problem.observation_count != observation_confidences.size() ||
      problem.track_count != track_statuses.size() ||
      (problem.observation_count != 0 &&
       problem.track_count >
           std::numeric_limits<size_t>::max() / problem.observation_count) ||
      problem.edges.size() != problem.track_count * problem.observation_count) {
    return absl::InvalidArgumentError(
        "ByteTrack association input dimensions mismatch");
  }

  std::vector<size_t> high_confidence_observations;
  std::vector<size_t> low_confidence_observations;
  for (size_t index = 0; index < observation_confidences.size(); ++index) {
    const double confidence = observation_confidences[index];
    if (!std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0) {
      return absl::InvalidArgumentError(
          "ByteTrack observation confidence must be in [0, 1]");
    }
    if (confidence >= config_.high_score_threshold) {
      high_confidence_observations.push_back(index);
    } else if (confidence >= config_.low_score_threshold) {
      low_confidence_observations.push_back(index);
    }
  }

  std::vector<size_t> established_tracks;
  std::vector<size_t> tentative_tracks;
  for (size_t index = 0; index < track_statuses.size(); ++index) {
    if (track_statuses[index] == TrackStatus::kConfirmed ||
        track_statuses[index] == TrackStatus::kLost) {
      established_tracks.push_back(index);
    } else if (track_statuses[index] == TrackStatus::kTentative) {
      tentative_tracks.push_back(index);
    }
  }
  const auto high_problem =
      MakeSubproblem(problem, established_tracks, high_confidence_observations);
  auto high_assignment = solver.Solve(high_problem);
  if (!high_assignment.ok()) {
    return high_assignment.status();
  }
  const auto high_status = ValidateAssignment(high_problem, *high_assignment);
  if (!high_status.ok()) {
    return high_status;
  }

  AssignmentResult result;
  std::vector<bool> matched_tracks(problem.track_count, false);
  std::vector<bool> matched_observations(problem.observation_count, false);
  for (const auto& match : high_assignment->matches) {
    const size_t track = established_tracks[match.track_index];
    const size_t observation =
        high_confidence_observations[match.observation_index];
    result.matches.push_back({track, observation, match.cost});
    matched_tracks[track] = true;
    matched_observations[observation] = true;
  }

  std::vector<size_t> active_unmatched_tracks;
  for (size_t local_track : high_assignment->unmatched_tracks) {
    const size_t track = established_tracks[local_track];
    if (track_statuses[track] == TrackStatus::kConfirmed) {
      active_unmatched_tracks.push_back(track);
    }
  }
  const auto low_problem = MakeSubproblem(problem, active_unmatched_tracks,
                                          low_confidence_observations);
  auto low_assignment = solver.Solve(low_problem);
  if (!low_assignment.ok()) {
    return low_assignment.status();
  }
  const auto low_status = ValidateAssignment(low_problem, *low_assignment);
  if (!low_status.ok()) {
    return low_status;
  }
  for (const auto& match : low_assignment->matches) {
    const size_t track = active_unmatched_tracks[match.track_index];
    const size_t observation =
        low_confidence_observations[match.observation_index];
    result.matches.push_back({track, observation, match.cost});
    matched_tracks[track] = true;
    matched_observations[observation] = true;
  }

  std::vector<size_t> unmatched_high_confidence_observations;
  for (size_t observation : high_confidence_observations) {
    if (!matched_observations[observation]) {
      unmatched_high_confidence_observations.push_back(observation);
    }
  }
  const auto tentative_problem = MakeSubproblem(
      problem, tentative_tracks, unmatched_high_confidence_observations);
  auto tentative_assignment = solver.Solve(tentative_problem);
  if (!tentative_assignment.ok()) {
    return tentative_assignment.status();
  }
  const auto tentative_status =
      ValidateAssignment(tentative_problem, *tentative_assignment);
  if (!tentative_status.ok()) {
    return tentative_status;
  }
  for (const auto& match : tentative_assignment->matches) {
    const size_t track = tentative_tracks[match.track_index];
    const size_t observation =
        unmatched_high_confidence_observations[match.observation_index];
    result.matches.push_back({track, observation, match.cost});
    matched_tracks[track] = true;
    matched_observations[observation] = true;
  }

  for (size_t index = 0; index < matched_tracks.size(); ++index) {
    if (!matched_tracks[index]) {
      result.unmatched_tracks.push_back(index);
    }
  }
  for (size_t index = 0; index < matched_observations.size(); ++index) {
    if (!matched_observations[index]) {
      result.unmatched_observations.push_back(index);
    }
  }
  result.total_cost = 0.0;
  for (const auto& match : result.matches) {
    result.total_cost += match.cost;
  }
  result.total_cost +=
      problem.unmatched_cost *
      static_cast<double>(result.unmatched_tracks.size() +
                          result.unmatched_observations.size());
  if (!std::isfinite(result.total_cost)) {
    return absl::InvalidArgumentError(
        "ByteTrack association objective overflow");
  }
  return result;
}

}  // namespace tracking
}  // namespace common
}  // namespace apollo
