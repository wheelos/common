#include "modules/common/tracking/motion_model.h"

#include <cmath>
#include <limits>
#include <utility>

#include "Eigen/Cholesky"

namespace apollo {
namespace common {
namespace tracking {
namespace {

absl::Status ValidatePrediction(double dt, const MotionModel& model,
                                const GaussianState* state) {
  if (state == nullptr || !std::isfinite(dt) || dt < 0.0) {
    return absl::InvalidArgumentError("invalid prediction parameters");
  }
  return model.ValidateState(*state);
}

absl::Status ValidateNoise(double noise) {
  if (!std::isfinite(noise) || noise < 0.0) {
    return absl::InvalidArgumentError(
        "process noise must be finite and nonnegative");
  }
  return absl::OkStatus();
}

absl::Status CommitPrediction(const Eigen::MatrixXd& transition,
                              const Eigen::MatrixXd& noise,
                              const Eigen::VectorXd& mean,
                              GaussianState* state) {
  GaussianState predicted;
  predicted.mean = mean;
  predicted.covariance =
      transition * state->covariance * transition.transpose() + noise;
  predicted.covariance =
      (0.5 * (predicted.covariance + predicted.covariance.transpose())).eval();
  const auto valid = ValidateGaussian(predicted);
  if (!valid.ok()) {
    return valid;
  }
  *state = std::move(predicted);
  return absl::OkStatus();
}

Eigen::Index Dimension(size_t dimensions, size_t blocks) {
  if (dimensions == 0 ||
      dimensions >
          static_cast<size_t>(std::numeric_limits<Eigen::Index>::max()) /
              blocks) {
    return 0;
  }
  return static_cast<Eigen::Index>(dimensions * blocks);
}

}  // namespace

absl::Status ValidateCovariance(const Eigen::MatrixXd& covariance,
                                Eigen::Index dimension,
                                bool positive_definite) {
  if (dimension <= 0 || covariance.rows() != dimension ||
      covariance.cols() != dimension || !covariance.allFinite() ||
      !covariance.isApprox(covariance.transpose(), 1e-10)) {
    return absl::InvalidArgumentError("invalid covariance shape or symmetry");
  }
  const Eigen::LDLT<Eigen::MatrixXd> factor(covariance);
  if (factor.info() != Eigen::Success || !factor.vectorD().allFinite() ||
      (positive_definite ? factor.vectorD().minCoeff() <= 0.0
                         : factor.vectorD().minCoeff() < 0.0)) {
    return absl::InvalidArgumentError(
        "covariance must be positive (semi)definite");
  }
  return absl::OkStatus();
}

absl::Status ValidateGaussian(const GaussianState& state) {
  if (state.mean.size() == 0 || !state.mean.allFinite()) {
    return absl::InvalidArgumentError("state must be nonempty and finite");
  }
  return ValidateCovariance(state.covariance, state.mean.size(), false);
}

absl::Status MotionModel::Validate() const {
  return state_dimension() > 0
             ? absl::OkStatus()
             : absl::InvalidArgumentError("invalid motion state dimension");
}

absl::Status MotionModel::ValidateState(const GaussianState& state) const {
  const auto valid = Validate();
  if (!valid.ok()) {
    return valid;
  }
  const auto gaussian_valid = ValidateGaussian(state);
  if (!gaussian_valid.ok()) {
    return gaussian_valid;
  }
  if (state.mean.size() != state_dimension()) {
    return absl::InvalidArgumentError("motion state dimension mismatch");
  }
  return absl::OkStatus();
}

ConstantVelocityModel::ConstantVelocityModel(size_t dimensions,
                                             double acceleration_noise)
    : dimensions_(dimensions), noise_(acceleration_noise) {}

Eigen::Index ConstantVelocityModel::state_dimension() const {
  return Dimension(dimensions_, 2);
}

absl::Status ConstantVelocityModel::Validate() const {
  const auto valid = MotionModel::Validate();
  return valid.ok() ? ValidateNoise(noise_) : valid;
}

absl::Status ConstantVelocityModel::Predict(double dt,
                                            GaussianState* state) const {
  const auto valid = ValidatePrediction(dt, *this, state);
  if (!valid.ok()) {
    return valid;
  }
  const Eigen::Index d = static_cast<Eigen::Index>(dimensions_);
  Eigen::MatrixXd transition =
      Eigen::MatrixXd::Identity(state_dimension(), state_dimension());
  transition.topRightCorner(d, d) = dt * Eigen::MatrixXd::Identity(d, d);
  Eigen::MatrixXd injection = Eigen::MatrixXd::Zero(state_dimension(), d);
  injection.topRows(d) = 0.5 * dt * dt * Eigen::MatrixXd::Identity(d, d);
  injection.bottomRows(d) = dt * Eigen::MatrixXd::Identity(d, d);
  return CommitPrediction(transition,
                          noise_ * injection * injection.transpose(),
                          transition * state->mean, state);
}

ConstantAccelerationModel::ConstantAccelerationModel(size_t dimensions,
                                                     double jerk_noise)
    : dimensions_(dimensions), noise_(jerk_noise) {}

Eigen::Index ConstantAccelerationModel::state_dimension() const {
  return Dimension(dimensions_, 3);
}

absl::Status ConstantAccelerationModel::Validate() const {
  const auto valid = MotionModel::Validate();
  return valid.ok() ? ValidateNoise(noise_) : valid;
}

absl::Status ConstantAccelerationModel::Predict(double dt,
                                                GaussianState* state) const {
  const auto valid = ValidatePrediction(dt, *this, state);
  if (!valid.ok()) {
    return valid;
  }
  const Eigen::Index d = static_cast<Eigen::Index>(dimensions_);
  Eigen::MatrixXd transition =
      Eigen::MatrixXd::Identity(state_dimension(), state_dimension());
  transition.block(0, d, d, d) = dt * Eigen::MatrixXd::Identity(d, d);
  transition.block(0, 2 * d, d, d) =
      0.5 * dt * dt * Eigen::MatrixXd::Identity(d, d);
  transition.block(d, 2 * d, d, d) = dt * Eigen::MatrixXd::Identity(d, d);
  Eigen::MatrixXd injection = Eigen::MatrixXd::Zero(state_dimension(), d);
  injection.topRows(d) = dt * dt * dt / 6.0 * Eigen::MatrixXd::Identity(d, d);
  injection.middleRows(d, d) = 0.5 * dt * dt * Eigen::MatrixXd::Identity(d, d);
  injection.bottomRows(d) = dt * Eigen::MatrixXd::Identity(d, d);
  return CommitPrediction(transition,
                          noise_ * injection * injection.transpose(),
                          transition * state->mean, state);
}

BicycleModel::BicycleModel(double wheelbase, double process_noise)
    : wheelbase_(wheelbase), noise_(process_noise) {}

absl::Status BicycleModel::Validate() const {
  if (!std::isfinite(wheelbase_) || wheelbase_ <= 0.0) {
    return absl::InvalidArgumentError("wheelbase must be finite and positive");
  }
  return ValidateNoise(noise_);
}

absl::Status BicycleModel::ValidateState(const GaussianState& state) const {
  const auto valid = MotionModel::ValidateState(state);
  if (!valid.ok()) {
    return valid;
  }
  if (std::abs(state.mean[5]) >= 1.5) {
    return absl::InvalidArgumentError("invalid bicycle steering angle");
  }
  return absl::OkStatus();
}

absl::Status BicycleModel::Predict(double dt, GaussianState* state) const {
  const auto valid = ValidatePrediction(dt, *this, state);
  if (!valid.ok()) {
    return valid;
  }
  const double yaw = state->mean[3];
  const double speed = state->mean[4];
  const double steering = state->mean[5];
  const double tangent = std::tan(steering);
  const double turn = speed * tangent / wheelbase_ * dt;
  const double half_turn = 0.5 * turn;
  const double sinc = std::abs(half_turn) < 1e-6
                          ? 1.0 - half_turn * half_turn / 6.0
                          : std::sin(half_turn) / half_turn;
  const double sinc_derivative =
      std::abs(half_turn) < 1e-6
          ? -half_turn / 3.0
          : (half_turn * std::cos(half_turn) - std::sin(half_turn)) /
                (half_turn * half_turn);
  const double angle = yaw + half_turn;
  const double travel = speed * dt * sinc;
  const double turn_speed = tangent / wheelbase_ * dt;
  const double turn_steering =
      speed * dt / (wheelbase_ * std::cos(steering) * std::cos(steering));
  Eigen::VectorXd mean = state->mean;
  mean[0] += travel * std::cos(angle);
  mean[1] += travel * std::sin(angle);
  mean[3] = std::atan2(std::sin(yaw + turn), std::cos(yaw + turn));
  Eigen::MatrixXd transition = Eigen::MatrixXd::Identity(6, 6);
  transition(0, 3) = -travel * std::sin(angle);
  transition(1, 3) = travel * std::cos(angle);
  for (Eigen::Index column = 4; column <= 5; ++column) {
    const double turn_derivative = column == 4 ? turn_speed : turn_steering;
    const double travel_derivative =
        (column == 4 ? dt * sinc : 0.0) +
        speed * dt * sinc_derivative * 0.5 * turn_derivative;
    transition(0, column) = travel_derivative * std::cos(angle) -
                            travel * std::sin(angle) * 0.5 * turn_derivative;
    transition(1, column) = travel_derivative * std::sin(angle) +
                            travel * std::cos(angle) * 0.5 * turn_derivative;
    transition(3, column) = turn_derivative;
  }
  return CommitPrediction(
      transition, noise_ * dt * Eigen::MatrixXd::Identity(6, 6), mean, state);
}

}  // namespace tracking
}  // namespace common
}  // namespace apollo
