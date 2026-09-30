#include "modules/common/tracking/embedding_cost.h"

#include <algorithm>
#include <cmath>

#include "absl/status/status.h"

namespace apollo {
namespace common {
namespace tracking {

absl::StatusOr<CostEvaluation> CosineEmbeddingCost::Evaluate(
    const Eigen::VectorXd& predicted, const Eigen::VectorXd& observed) const {
  if (predicted.size() == 0 || predicted.size() != observed.size() ||
      !predicted.allFinite() || !observed.allFinite()) {
    return absl::InvalidArgumentError("embedding shape or values invalid");
  }
  const double first_norm = predicted.stableNorm();
  const double second_norm = observed.stableNorm();
  if (!std::isfinite(first_norm) || !std::isfinite(second_norm) ||
      first_norm <= 0.0 || second_norm <= 0.0) {
    return absl::InvalidArgumentError(
        "embedding norm must be finite and positive");
  }
  const double cosine = (predicted / first_norm).dot(observed / second_norm);
  return CostEvaluation::Accept(0.5 * (1.0 - std::clamp(cosine, -1.0, 1.0)));
}

}  // namespace tracking
}  // namespace common
}  // namespace apollo
