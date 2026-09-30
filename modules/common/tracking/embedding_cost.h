#pragma once

#include "Eigen/Core"
#include "absl/status/statusor.h"

#include "modules/common/tracking/types.h"

namespace apollo {
namespace common {
namespace tracking {

class CosineEmbeddingCost {
 public:
  absl::StatusOr<CostEvaluation> Evaluate(
      const Eigen::VectorXd& predicted, const Eigen::VectorXd& observed) const;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
