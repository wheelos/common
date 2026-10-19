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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tracking/core/lifecycle.h"
#include "tracking/core/types.h"

namespace apollo {
namespace common {
namespace tracking {

template <typename State, typename Payload>
class StateEstimator {
 public:
  virtual ~StateEstimator() = default;
  virtual absl::Status Validate(const Payload& observation) const = 0;
  virtual absl::StatusOr<State> Initialize(
      const Payload& observation) const = 0;
  virtual absl::Status Predict(double dt, State* state) const = 0;
  virtual absl::Status Update(const Payload& observation,
                              State* state) const = 0;
};

template <typename State, typename Payload>
class AssociationCost {
 public:
  virtual ~AssociationCost() = default;
  virtual absl::StatusOr<CostEvaluation> Evaluate(
      const TrackSnapshot<State, Payload>& track,
      const Observation<Payload>& observation) const = 0;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
