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

#include <Eigen/Core>
#include <Eigen/LU>

namespace wheel {
namespace vision {
namespace lidar {

struct RigidTransform3f {
  Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
  Eigen::Vector3f translation = Eigen::Vector3f::Zero();

  inline bool IsFinite() const {
    return rotation.allFinite() && translation.allFinite();
  }

  // IsFinite alone also accepts scale, shear, and reflection.
  inline bool IsValid(float tolerance = 1e-4f) const {
    return std::isfinite(tolerance) && tolerance > 0.0f && IsFinite() &&
           (rotation.transpose() * rotation - Eigen::Matrix3f::Identity())
                   .cwiseAbs()
                   .maxCoeff() <= tolerance &&
           std::abs(rotation.determinant() - 1.0f) <= tolerance;
  }
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
