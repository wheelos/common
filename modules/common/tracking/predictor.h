#pragma once

#include <cmath>
#include <memory>
#include <utility>

#include "Eigen/Cholesky"
#include "absl/status/statusor.h"

#include "modules/common/tracking/interfaces.h"
#include "modules/common/tracking/motion_model.h"
#include "modules/common/tracking/multi_model_estimator.h"

namespace apollo {
namespace common {
namespace tracking {

struct MeasurementProjection {
  // User-defined residual allows angles, nonlinear geometry and tensor
  // adapters.
  Eigen::VectorXd residual;
  Eigen::MatrixXd jacobian;
  Eigen::MatrixXd covariance;
};

template <typename Payload>
class MeasurementModel {
 public:
  virtual ~MeasurementModel() = default;
  virtual absl::Status Validate(const Payload& observation) const = 0;
  virtual absl::StatusOr<GaussianState> Initialize(
      const Payload& observation) const = 0;
  virtual absl::StatusOr<MeasurementProjection> Project(
      const GaussianState& state, const Payload& observation) const = 0;
};

absl::Status ValidateProjection(const GaussianState& state,
                                const MeasurementProjection& projection);

template <typename Payload>
class Predictor final : public StateEstimator<GaussianState, Payload> {
 public:
  Predictor(std::shared_ptr<const MotionModel> motion,
            std::shared_ptr<const MeasurementModel<Payload>> measurement)
      : motion_(std::move(motion)), measurement_(std::move(measurement)) {}

  absl::Status Validate(const Payload& observation) const override {
    if (!motion_ || !measurement_) {
      return absl::InvalidArgumentError("invalid predictor plugins");
    }
    const auto valid = motion_->Validate();
    if (!valid.ok()) {
      return valid;
    }
    return measurement_->Validate(observation);
  }

  absl::StatusOr<GaussianState> Initialize(
      const Payload& observation) const override {
    const auto valid = Validate(observation);
    if (!valid.ok()) {
      return valid;
    }
    auto state = measurement_->Initialize(observation);
    if (!state.ok()) {
      return state.status();
    }
    const auto state_valid = motion_->ValidateState(*state);
    if (!state_valid.ok()) {
      return state_valid;
    }
    return state;
  }

  absl::Status Predict(double dt, GaussianState* state) const override {
    if (!motion_ || state == nullptr || !std::isfinite(dt) || dt < 0.0) {
      return absl::InvalidArgumentError(
          "invalid predictor prediction parameters");
    }
    const auto valid = motion_->ValidateState(*state);
    if (!valid.ok()) {
      return valid;
    }
    GaussianState predicted = *state;
    const auto status = motion_->Predict(dt, &predicted);
    if (!status.ok()) {
      return status;
    }
    const auto predicted_valid = motion_->ValidateState(predicted);
    if (!predicted_valid.ok()) {
      return predicted_valid;
    }
    *state = std::move(predicted);
    return absl::OkStatus();
  }

  absl::Status Update(const Payload& observation,
                      GaussianState* state) const override {
    if (state == nullptr) {
      return absl::InvalidArgumentError("null state");
    }
    const auto valid = Validate(observation);
    if (!valid.ok()) {
      return valid;
    }
    const auto state_valid = motion_->ValidateState(*state);
    if (!state_valid.ok()) {
      return state_valid;
    }
    auto projected = measurement_->Project(*state, observation);
    if (!projected.ok()) {
      return projected.status();
    }
    const auto projection_valid = ValidateProjection(*state, *projected);
    if (!projection_valid.ok()) {
      return projection_valid;
    }
    const auto& h = projected->jacobian;
    Eigen::MatrixXd innovation =
        h * state->covariance * h.transpose() + projected->covariance;
    const auto innovation_valid =
        ValidateCovariance(innovation, projected->residual.size(), true);
    if (!innovation_valid.ok()) {
      return innovation_valid;
    }
    const Eigen::LDLT<Eigen::MatrixXd> factor(innovation);
    Eigen::MatrixXd gain = factor.solve(h * state->covariance).transpose();
    GaussianState updated;
    updated.mean = state->mean + gain * projected->residual;
    const Eigen::MatrixXd correction =
        Eigen::MatrixXd::Identity(state->mean.size(), state->mean.size()) -
        gain * h;
    // Joseph form preserves covariance symmetry and positive semidefiniteness.
    updated.covariance =
        correction * state->covariance * correction.transpose() +
        gain * projected->covariance * gain.transpose();
    updated.covariance =
        (0.5 * (updated.covariance + updated.covariance.transpose())).eval();
    const auto updated_valid = motion_->ValidateState(updated);
    if (!updated_valid.ok()) {
      return updated_valid;
    }
    *state = std::move(updated);
    return absl::OkStatus();
  }

 private:
  std::shared_ptr<const MotionModel> motion_;
  std::shared_ptr<const MeasurementModel<Payload>> measurement_;
};

template <typename Payload>
using MultiModelPredictor = MultiModelEstimator<GaussianState, Payload>;

struct VectorMeasurement {
  Eigen::VectorXd value;
  Eigen::MatrixXd covariance;
};

// Observes the first D state elements; other elements initialize to zero.
class PositionMeasurementModel final
    : public MeasurementModel<VectorMeasurement> {
 public:
  PositionMeasurementModel(Eigen::Index state_dimension,
                           Eigen::Index measurement_dimension,
                           double initial_variance = 1.0);
  absl::Status Validate(const VectorMeasurement& observation) const override;
  absl::StatusOr<GaussianState> Initialize(
      const VectorMeasurement& observation) const override;
  absl::StatusOr<MeasurementProjection> Project(
      const GaussianState& state,
      const VectorMeasurement& observation) const override;

 private:
  Eigen::Index state_dimension_;
  Eigen::Index measurement_dimension_;
  double initial_variance_;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
