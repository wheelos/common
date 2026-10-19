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

#include "absl/status/statusor.h"
#include "tracking/core/types.h"

namespace apollo {
namespace common {
namespace tracking {

class AssignmentSolver {
 public:
  virtual ~AssignmentSolver() = default;
  virtual absl::StatusOr<AssignmentResult> Solve(
      const AssociationProblem& problem) const = 0;
};

class HungarianSolver final : public AssignmentSolver {
 public:
  absl::StatusOr<AssignmentResult> Solve(
      const AssociationProblem& problem) const override;
};

// Single-frame min-cost flow. No delayed output or historical reassignment.
class MinCostFlowSolver final : public AssignmentSolver {
 public:
  absl::StatusOr<AssignmentResult> Solve(
      const AssociationProblem& problem) const override;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
