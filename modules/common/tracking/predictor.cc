#include "modules/common/tracking/predictor.h"

#include <cmath>

namespace apollo {
namespace common {
namespace tracking {

absl::Status ValidateProjection(const GaussianState& state,
                                const MeasurementProjection& projection) {
  const auto valid = ValidateGaussian(state);
  if (!valid.ok()) {
    return valid;
  }
  if (projection.residual.size() <= 0 || !projection.residual.allFinite() ||
      projection.jacobian.rows() != projection.residual.size() ||
      projection.jacobian.cols() != state.mean.size() ||
      !projection.jacobian.allFinite()) {
    return absl::InvalidArgumentError(
        "invalid measurement residual or Jacobian");
  }
  return ValidateCovariance(projection.covariance, projection.residual.size(),
                            false);
}

PositionMeasurementModel::PositionMeasurementModel(
    Eigen::Index state_dimension, Eigen::Index measurement_dimension,
    double initial_variance)
    : state_dimension_(state_dimension),
      measurement_dimension_(measurement_dimension),
      initial_variance_(initial_variance) {}

absl::Status PositionMeasurementModel::Validate(
    const VectorMeasurement& observation) const {
  if (state_dimension_ <= 0 || measurement_dimension_ <= 0 ||
      measurement_dimension_ > state_dimension_ ||
      !std::isfinite(initial_variance_) || initial_variance_ <= 0.0 ||
      observation.value.size() != measurement_dimension_ ||
      !observation.value.allFinite()) {
    return absl::InvalidArgumentError(
        "invalid position measurement or dimensions");
  }
  return ValidateCovariance(observation.covariance, measurement_dimension_,
                            true);
}

absl::StatusOr<GaussianState> PositionMeasurementModel::Initialize(
    const VectorMeasurement& observation) const {
  const auto valid = Validate(observation);
  if (!valid.ok()) {
    return valid;
  }
  GaussianState state;
  state.mean = Eigen::VectorXd::Zero(state_dimension_);
  state.mean.head(measurement_dimension_) = observation.value;
  state.covariance =
      initial_variance_ *
      Eigen::MatrixXd::Identity(state_dimension_, state_dimension_);
  state.covariance.topLeftCorner(
      measurement_dimension_, measurement_dimension_) = observation.covariance;
  return state;
}

absl::StatusOr<MeasurementProjection> PositionMeasurementModel::Project(
    const GaussianState& state, const VectorMeasurement& observation) const {
  const auto valid = Validate(observation);
  if (!valid.ok()) {
    return valid;
  }
  const auto state_valid = ValidateGaussian(state);
  if (!state_valid.ok()) {
    return state_valid;
  }
  if (state.mean.size() != state_dimension_) {
    return absl::InvalidArgumentError(
        "position projection state dimension mismatch");
  }
  MeasurementProjection projection;
  projection.residual =
      observation.value - state.mean.head(measurement_dimension_);
  projection.jacobian =
      Eigen::MatrixXd::Zero(measurement_dimension_, state_dimension_);
  projection.jacobian.leftCols(measurement_dimension_).setIdentity();
  projection.covariance = observation.covariance;
  return projection;
}

}  // namespace tracking
}  // namespace common
}  // namespace apollo
