#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "modules/common/tracking/interfaces.h"

namespace apollo {
namespace common {
namespace tracking {

inline absl::Status ValidateCostEvaluation(
    const CostEvaluation& evaluation,
    size_t component_limit = std::numeric_limits<size_t>::max()) {
  if (!std::isfinite(evaluation.value) || evaluation.value < 0.0 ||
      (!evaluation.allowed && evaluation.reason.empty())) {
    return absl::InvalidArgumentError(
        "invalid cost value or missing gate reason");
  }
  if (evaluation.components.size() > component_limit) {
    return absl::ResourceExhaustedError("cost component limit exceeded");
  }
  std::unordered_set<std::string> names;
  for (const auto& component : evaluation.components) {
    if (component.name.empty() || !names.insert(component.name).second ||
        !std::isfinite(component.value) || component.value < 0.0 ||
        !std::isfinite(component.weight) || component.weight < 0.0) {
      return absl::InvalidArgumentError("invalid cost component");
    }
  }
  return absl::OkStatus();
}

template <typename State, typename Payload>
class FunctionalAssociationCost final : public AssociationCost<State, Payload> {
 public:
  using Function = std::function<absl::StatusOr<CostEvaluation>(
      const TrackSnapshot<State, Payload>&, const Observation<Payload>&)>;
  explicit FunctionalAssociationCost(Function function)
      : function_(std::move(function)) {}
  absl::StatusOr<CostEvaluation> Evaluate(
      const TrackSnapshot<State, Payload>& track,
      const Observation<Payload>& observation) const override {
    if (!function_ || !observation.payload) {
      return absl::InvalidArgumentError("missing cost function or observation");
    }
    auto evaluation = function_(track, observation);
    if (!evaluation.ok()) {
      return evaluation.status();
    }
    const auto valid = ValidateCostEvaluation(*evaluation);
    if (!valid.ok()) {
      return valid;
    }
    return evaluation;
  }

 private:
  Function function_;
};

template <typename State, typename Payload>
class CompositeAssociationCost final : public AssociationCost<State, Payload> {
 public:
  struct Term {
    std::string name;
    double weight = 1.0;
    std::shared_ptr<const AssociationCost<State, Payload>> cost;
  };

  explicit CompositeAssociationCost(std::vector<Term> terms)
      : terms_(std::move(terms)) {}

  absl::StatusOr<CostEvaluation> Evaluate(
      const TrackSnapshot<State, Payload>& track,
      const Observation<Payload>& observation) const override {
    double total_weight = 0.0;
    std::unordered_set<std::string> names;
    for (const auto& term : terms_) {
      if (term.name.empty() || !names.insert(term.name).second || !term.cost ||
          !std::isfinite(term.weight) || term.weight <= 0.0) {
        return absl::InvalidArgumentError("invalid composite cost term");
      }
      total_weight += term.weight;
    }
    if (!std::isfinite(total_weight) || total_weight <= 0.0) {
      return absl::InvalidArgumentError("invalid composite cost weights");
    }
    CostEvaluation combined = CostEvaluation::Accept(0.0);
    for (const auto& term : terms_) {
      auto evaluation = term.cost->Evaluate(track, observation);
      if (!evaluation.ok()) {
        return evaluation.status();
      }
      const auto valid = ValidateCostEvaluation(*evaluation);
      if (!valid.ok()) {
        return valid;
      }
      if (!evaluation->allowed) {
        combined.allowed = false;
        combined.reason = term.name + ": " + evaluation->reason;
        return combined;
      }
      if (evaluation->value > 1.0) {
        return absl::InvalidArgumentError(
            "composite costs must be normalized to [0, 1]");
      }
      combined.value += evaluation->value * (term.weight / total_weight);
      combined.components.push_back(
          {term.name, evaluation->value, term.weight});
    }
    return combined;
  }

 private:
  std::vector<Term> terms_;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
