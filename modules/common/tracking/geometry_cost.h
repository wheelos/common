#pragma once

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "modules/common/tracking/types.h"

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
