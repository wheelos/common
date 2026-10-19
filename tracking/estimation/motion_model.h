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

#include <cstddef>

#include "Eigen/Core"
#include "absl/status/status.h"

namespace apollo {
namespace common {
namespace tracking {

struct GaussianState {
  Eigen::VectorXd mean;
  Eigen::MatrixXd covariance;
};

absl::Status ValidateGaussian(const GaussianState& state);
absl::Status ValidateCovariance(const Eigen::MatrixXd& covariance,
                                Eigen::Index dimension, bool positive_definite);

class MotionModel {
 public:
  virtual ~MotionModel() = default;
  virtual Eigen::Index state_dimension() const = 0;
  virtual absl::Status Validate() const;
  virtual absl::Status ValidateState(const GaussianState& state) const;
  virtual absl::Status Predict(double dt, GaussianState* state) const = 0;
};

// State layout: [position(D), velocity(D)].
class ConstantVelocityModel final : public MotionModel {
 public:
  explicit ConstantVelocityModel(size_t dimensions = 3,
                                 double acceleration_noise = 1.0);
  Eigen::Index state_dimension() const override;
  absl::Status Validate() const override;
  absl::Status Predict(double dt, GaussianState* state) const override;

 private:
  size_t dimensions_;
  double noise_;
};

// State layout: [position(D), velocity(D), acceleration(D)].
class ConstantAccelerationModel final : public MotionModel {
 public:
  explicit ConstantAccelerationModel(size_t dimensions = 3,
                                     double jerk_noise = 1.0);
  Eigen::Index state_dimension() const override;
  absl::Status Validate() const override;
  absl::Status Predict(double dt, GaussianState* state) const override;

 private:
  size_t dimensions_;
  double noise_;
};

// Upright kinematic bicycle, rear-axle reference:
// [x, y, z, yaw, signed_speed, steering_angle]. Wheelbase is not box length.
class BicycleModel final : public MotionModel {
 public:
  explicit BicycleModel(double wheelbase, double process_noise = 0.1);
  Eigen::Index state_dimension() const override { return 6; }
  absl::Status Validate() const override;
  absl::Status ValidateState(const GaussianState& state) const override;
  absl::Status Predict(double dt, GaussianState* state) const override;

 private:
  double wheelbase_;
  double noise_;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
