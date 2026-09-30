#pragma once

#include "absl/status/statusor.h"

#include "modules/common/tracking/types.h"

namespace apollo {
namespace common {
namespace tracking {

class Associator {
 public:
  virtual ~Associator() = default;
  virtual absl::StatusOr<AssignmentResult> Associate(
      const AssociationProblem& problem) const = 0;
};

class HungarianAssociator final : public Associator {
 public:
  absl::StatusOr<AssignmentResult> Associate(
      const AssociationProblem& problem) const override;
};

// Single-frame min-cost flow. No delayed output or historical reassignment.
class MinCostFlowAssociator final : public Associator {
 public:
  absl::StatusOr<AssignmentResult> Associate(
      const AssociationProblem& problem) const override;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
