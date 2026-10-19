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

#include "tracking/costs/embedding_cost.h"

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
