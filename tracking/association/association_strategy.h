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

#include <vector>

#include "absl/status/statusor.h"
#include "tracking/association/assignment_solver.h"
#include "tracking/core/types.h"

namespace apollo {
namespace common {
namespace tracking {

class AssociationStrategy {
 public:
  virtual ~AssociationStrategy() = default;

  virtual absl::StatusOr<AssignmentResult> Associate(
      const AssociationProblem& problem,
      const std::vector<double>& observation_confidences,
      const std::vector<TrackStatus>& track_statuses,
      const AssignmentSolver& solver) const = 0;

  virtual bool CanStartTrack(double confidence) const = 0;
};

struct ByteTrackConfig {
  double high_score_threshold = 0.5;
  double low_score_threshold = 0.1;
  double new_track_threshold = 0.6;
};

class ByteTrackAssociationStrategy final : public AssociationStrategy {
 public:
  explicit ByteTrackAssociationStrategy(ByteTrackConfig config = {})
      : config_(config) {}

  absl::StatusOr<AssignmentResult> Associate(
      const AssociationProblem& problem,
      const std::vector<double>& observation_confidences,
      const std::vector<TrackStatus>& track_statuses,
      const AssignmentSolver& solver) const override;

  bool CanStartTrack(double confidence) const override;

 private:
  absl::Status ValidateConfig() const;

  ByteTrackConfig config_;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
