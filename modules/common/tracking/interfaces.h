#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "modules/common/tracking/lifecycle.h"
#include "modules/common/tracking/types.h"

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
