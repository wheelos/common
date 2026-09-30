#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "modules/common/tracking/interfaces.h"

namespace apollo {
namespace common {
namespace tracking {

template <typename State>
struct MultiModelState {
  std::string model_name;
  State estimate;
};

template <typename State, typename Payload>
class MultiModelEstimator final
    : public StateEstimator<MultiModelState<State>, Payload> {
 public:
  using Estimator = StateEstimator<State, Payload>;
  using Registry =
      std::unordered_map<std::string, std::shared_ptr<const Estimator>>;
  using Selector =
      std::function<absl::StatusOr<std::string>(const Payload& observation)>;

  MultiModelEstimator(Registry models, Selector selector)
      : models_(std::move(models)), selector_(std::move(selector)) {}

  absl::Status Validate(const Payload& observation) const override {
    auto name = Select(observation);
    if (!name.ok()) {
      return name.status();
    }
    auto estimator = Resolve(*name);
    if (!estimator.ok()) {
      return estimator.status();
    }
    return (*estimator)->Validate(observation);
  }

  absl::StatusOr<MultiModelState<State>> Initialize(
      const Payload& observation) const override {
    auto name = Select(observation);
    if (!name.ok()) {
      return name.status();
    }
    auto estimator = Resolve(*name);
    if (!estimator.ok()) {
      return estimator.status();
    }
    const auto valid = (*estimator)->Validate(observation);
    if (!valid.ok()) {
      return valid;
    }
    auto estimate = (*estimator)->Initialize(observation);
    if (!estimate.ok()) {
      return estimate.status();
    }
    return MultiModelState<State>{std::move(*name), std::move(*estimate)};
  }

  absl::Status Predict(double dt,
                       MultiModelState<State>* state) const override {
    if (state == nullptr || !std::isfinite(dt) || dt < 0.0) {
      return absl::InvalidArgumentError("invalid multi-model prediction input");
    }
    auto estimator = Resolve(state->model_name);
    if (!estimator.ok()) {
      return estimator.status();
    }
    State predicted = state->estimate;
    const auto status = (*estimator)->Predict(dt, &predicted);
    if (!status.ok()) {
      return status;
    }
    state->estimate = std::move(predicted);
    return absl::OkStatus();
  }

  absl::Status Update(const Payload& observation,
                      MultiModelState<State>* state) const override {
    if (state == nullptr) {
      return absl::InvalidArgumentError("null multi-model state");
    }
    // Birth selects the model; subsequent updates cannot reinterpret its
    // layout.
    auto estimator = Resolve(state->model_name);
    if (!estimator.ok()) {
      return estimator.status();
    }
    const auto valid = (*estimator)->Validate(observation);
    if (!valid.ok()) {
      return valid;
    }
    State updated = state->estimate;
    const auto status = (*estimator)->Update(observation, &updated);
    if (!status.ok()) {
      return status;
    }
    state->estimate = std::move(updated);
    return absl::OkStatus();
  }

 private:
  absl::Status ValidateRegistry() const {
    if (models_.empty()) {
      return absl::InvalidArgumentError("empty tracking model registry");
    }
    for (const auto& entry : models_) {
      if (entry.first.empty() || !entry.second) {
        return absl::InvalidArgumentError(
            "invalid tracking model registration");
      }
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> Select(const Payload& observation) const {
    const auto valid = ValidateRegistry();
    if (!valid.ok()) {
      return valid;
    }
    if (!selector_) {
      return absl::InvalidArgumentError("missing tracking model selector");
    }
    auto name = selector_(observation);
    if (!name.ok()) {
      return name.status();
    }
    if (name->empty()) {
      return absl::InvalidArgumentError(
          "model selector returned an empty name");
    }
    return name;
  }

  absl::StatusOr<std::shared_ptr<const Estimator>> Resolve(
      const std::string& name) const {
    const auto valid = ValidateRegistry();
    if (!valid.ok()) {
      return valid;
    }
    const auto entry = models_.find(name);
    if (entry == models_.end()) {
      return absl::NotFoundError("unregistered tracking model: " + name);
    }
    return entry->second;
  }

  Registry models_;
  Selector selector_;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
