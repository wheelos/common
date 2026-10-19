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

#include <cmath>
#include <memory>
#include <utility>

#include "Eigen/Cholesky"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tracking/estimation/predictor.h"

namespace apollo {
namespace common {
namespace tracking {

template <typename Payload>
class MahalanobisCost final : public AssociationCost<GaussianState, Payload> {
 public:
  MahalanobisCost(std::shared_ptr<const MeasurementModel<Payload>> measurement,
                  double squared_gate)
      : measurement_(std::move(measurement)), squared_gate_(squared_gate) {}

  absl::StatusOr<CostEvaluation> Evaluate(
      const TrackSnapshot<GaussianState, Payload>& track,
      const Observation<Payload>& observation) const override {
    if (!measurement_ || !observation.payload ||
        !std::isfinite(squared_gate_) || squared_gate_ <= 0.0) {
      return absl::InvalidArgumentError("invalid Mahalanobis configuration");
    }
    const auto state_valid = ValidateGaussian(track.state);
    if (!state_valid.ok()) {
      return state_valid;
    }
    const auto observation_valid = measurement_->Validate(*observation.payload);
    if (!observation_valid.ok()) {
      return observation_valid;
    }
    auto projection = measurement_->Project(track.state, *observation.payload);
    if (!projection.ok()) {
      return projection.status();
    }
    const auto valid = ValidateProjection(track.state, *projection);
    if (!valid.ok()) {
      return valid;
    }
    const Eigen::MatrixXd innovation = projection->jacobian *
                                           track.state.covariance *
                                           projection->jacobian.transpose() +
                                       projection->covariance;
    const auto covariance_valid =
        ValidateCovariance(innovation, projection->residual.size(), true);
    if (!covariance_valid.ok()) {
      return covariance_valid;
    }
    const Eigen::LDLT<Eigen::MatrixXd> factor(innovation);
    const double squared =
        projection->residual.dot(factor.solve(projection->residual));
    if (!std::isfinite(squared) || squared < 0.0) {
      return absl::InvalidArgumentError("invalid Mahalanobis distance");
    }
    return squared <= squared_gate_
               ? CostEvaluation::Accept(squared / squared_gate_)
               : CostEvaluation::Reject("Mahalanobis gate");
  }

 private:
  std::shared_ptr<const MeasurementModel<Payload>> measurement_;
  double squared_gate_;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
