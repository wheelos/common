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

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tracking/core/types.h"

namespace apollo {
namespace common {
namespace tracking {

struct Box3D {
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d size = Eigen::Vector3d::Ones();
  double yaw = 0.0;
};

absl::Status ValidateBox(const Box3D& box);

class CenterDistanceCost {
 public:
  explicit CenterDistanceCost(double gate, bool use_bev = false)
      : gate_(gate), use_bev_(use_bev) {}
  absl::StatusOr<CostEvaluation> Evaluate(const Box3D& predicted,
                                          const Box3D& observed) const;

 private:
  double gate_;
  bool use_bev_;
};

class BevIouCost {
 public:
  absl::StatusOr<CostEvaluation> Evaluate(const Box3D& predicted,
                                          const Box3D& observed) const;
};

// Rotated XY footprint with upright Z extent, not arbitrary roll/pitch boxes.
class Iou3dCost {
 public:
  absl::StatusOr<CostEvaluation> Evaluate(const Box3D& predicted,
                                          const Box3D& observed) const;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
