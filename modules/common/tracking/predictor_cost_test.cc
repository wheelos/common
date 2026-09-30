#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "modules/common/tracking/cost.h"

namespace apollo {
namespace common {
namespace tracking {
namespace {

GaussianState State(Eigen::Index dimension) {
  return {Eigen::VectorXd::Zero(dimension),
          Eigen::MatrixXd::Identity(dimension, dimension)};
}

TEST(MotionModelTest, ArbitraryDimensionsAndRealDt) {
  auto cv = State(24);
  cv.mean.tail(12).setConstant(2.0);
  ASSERT_TRUE(ConstantVelocityModel(12, 0.0).Predict(0.25, &cv).ok());
  EXPECT_TRUE(cv.mean.head(12).isApprox(Eigen::VectorXd::Constant(12, 0.5)));
  EXPECT_DOUBLE_EQ(cv.covariance(0, 0), 1.0625);
  auto ca = State(9);
  ca.mean.segment(3, 3).setConstant(2.0);
  ca.mean.tail(3).setConstant(4.0);
  ASSERT_TRUE(ConstantAccelerationModel(3, 0.0).Predict(0.5, &ca).ok());
  EXPECT_DOUBLE_EQ(ca.mean[0], 1.5);
  EXPECT_DOUBLE_EQ(ca.mean[3], 4.0);
  EXPECT_DOUBLE_EQ(ca.mean[6], 4.0);
  auto zero = cv;
  ASSERT_TRUE(ConstantVelocityModel(12).Predict(0.0, &zero).ok());
  EXPECT_TRUE(zero.mean.isApprox(cv.mean));
  EXPECT_TRUE(zero.covariance.isApprox(cv.covariance));
  EXPECT_FALSE(ConstantVelocityModel(0).Predict(1.0, &cv).ok());
  EXPECT_FALSE(ConstantVelocityModel(3).Predict(1.0, &cv).ok());
  EXPECT_FALSE(ConstantVelocityModel(12).Predict(-1.0, &cv).ok());
  EXPECT_FALSE(ConstantVelocityModel(12, -1.0).Predict(1.0, &cv).ok());
}

TEST(MotionModelTest, BicycleStraightTurningAndNearZeroSteering) {
  auto straight = State(6);
  straight.mean[4] = 10.0;
  ASSERT_TRUE(BicycleModel(2.5, 0.0).Predict(0.5, &straight).ok());
  EXPECT_NEAR(straight.mean[0], 5.0, 1e-12);
  EXPECT_DOUBLE_EQ(straight.mean[1], 0.0);
  auto near_zero = State(6);
  near_zero.mean[4] = 10.0;
  near_zero.mean[5] = 1e-10;
  ASSERT_TRUE(BicycleModel(2.5).Predict(0.5, &near_zero).ok());
  EXPECT_NEAR(near_zero.mean[0], 5.0, 1e-9);
  EXPECT_NEAR(near_zero.mean[1], 0.0, 1e-9);
  auto turning = State(6);
  turning.mean[4] = 5.0;
  turning.mean[5] = std::atan(0.5);
  ASSERT_TRUE(BicycleModel(2.5, 0.0).Predict(1.0, &turning).ok());
  EXPECT_NEAR(turning.mean[3], 1.0, 1e-12);
  EXPECT_NEAR(turning.mean[0], 5.0 * std::sin(1.0), 1e-12);
  EXPECT_NEAR(turning.mean[1], 5.0 * (1.0 - std::cos(1.0)), 1e-12);
  EXPECT_TRUE(ValidateGaussian(turning).ok());
  EXPECT_FALSE(BicycleModel(0.0).Predict(1.0, &turning).ok());
  turning.mean[5] = 1.55;
  EXPECT_FALSE(BicycleModel(2.5).Predict(1.0, &turning).ok());
}

TEST(PredictorTest, MeasurementUpdateAndMahalanobisUseSameProjection) {
  auto measurement = std::make_shared<PositionMeasurementModel>(6, 3);
  Predictor<VectorMeasurement> predictor(
      std::make_shared<ConstantVelocityModel>(3, 0.0), measurement);
  VectorMeasurement observed{Eigen::VectorXd::Ones(3),
                             Eigen::MatrixXd::Identity(3, 3)};
  auto state = State(6);
  TrackSnapshot<GaussianState, VectorMeasurement> track;
  track.state = state;
  MahalanobisCost<VectorMeasurement> cost(measurement, 3.0);
  auto distance =
      cost.Evaluate(track, {1, std::make_shared<VectorMeasurement>(observed)});
  ASSERT_TRUE(distance.ok()) << distance.status();
  EXPECT_DOUBLE_EQ(distance->value, 0.5);
  ASSERT_TRUE(predictor.Update(observed, &state).ok());
  EXPECT_DOUBLE_EQ(state.mean[0], 0.5);
  EXPECT_DOUBLE_EQ(state.covariance(0, 0), 0.5);
  EXPECT_TRUE(ValidateGaussian(state).ok());
  observed.value.setConstant(100.0);
  auto gated =
      cost.Evaluate(track, {2, std::make_shared<VectorMeasurement>(observed)});
  ASSERT_TRUE(gated.ok());
  EXPECT_FALSE(gated->allowed);
  EXPECT_FALSE(gated->reason.empty());
  observed.covariance.setZero();
  EXPECT_FALSE(predictor.Initialize(observed).ok());
}

TEST(PredictorTest, InvalidMotionConfigurationFailsBeforeInitialization) {
  const std::vector<std::shared_ptr<const MotionModel>> invalid_models{
      std::make_shared<ConstantVelocityModel>(0),
      std::make_shared<ConstantVelocityModel>(
          std::numeric_limits<size_t>::max()),
      std::make_shared<ConstantVelocityModel>(3, -1.0),
      std::make_shared<ConstantVelocityModel>(
          3, std::numeric_limits<double>::quiet_NaN()),
      std::make_shared<ConstantAccelerationModel>(2, -1.0),
      std::make_shared<BicycleModel>(0.0),
      std::make_shared<BicycleModel>(std::numeric_limits<double>::infinity()),
      std::make_shared<BicycleModel>(2.5, -1.0)};
  VectorMeasurement observed{Eigen::VectorXd::Zero(3),
                             Eigen::MatrixXd::Identity(3, 3)};
  auto measurement = std::make_shared<PositionMeasurementModel>(6, 3);
  for (const auto& model : invalid_models) {
    Predictor<VectorMeasurement> predictor(model, measurement);
    EXPECT_FALSE(predictor.Validate(observed).ok());
    EXPECT_FALSE(predictor.Initialize(observed).ok());
    auto state = State(6);
    EXPECT_FALSE(predictor.Predict(0.1, &state).ok());
    EXPECT_FALSE(predictor.Update(observed, &state).ok());
    EXPECT_TRUE(state.mean.isZero());
    EXPECT_TRUE(state.covariance.isIdentity());
  }
}

TEST(PredictorTest, BicycleConstraintsApplyAtInitializationAndCorrection) {
  Predictor<VectorMeasurement> predictor(
      std::make_shared<BicycleModel>(2.5),
      std::make_shared<PositionMeasurementModel>(6, 6));
  VectorMeasurement observed{Eigen::VectorXd::Zero(6),
                             0.01 * Eigen::MatrixXd::Identity(6, 6)};
  ASSERT_TRUE(predictor.Initialize(observed).ok());
  observed.value[5] = 2.0;
  EXPECT_FALSE(predictor.Initialize(observed).ok());
  auto state = State(6);
  EXPECT_FALSE(predictor.Update(observed, &state).ok());
  EXPECT_TRUE(state.mean.isZero());
  EXPECT_TRUE(state.covariance.isIdentity());
  state.mean[5] = 1.5;
  EXPECT_FALSE(predictor.Update(observed, &state).ok());
  EXPECT_FALSE(predictor.Predict(0.1, &state).ok());
}

class AngleMeasurement final : public MeasurementModel<double> {
 public:
  absl::Status Validate(const double& yaw) const override {
    return std::isfinite(yaw) ? absl::OkStatus()
                              : absl::InvalidArgumentError("nonfinite yaw");
  }
  absl::StatusOr<GaussianState> Initialize(const double& yaw) const override {
    auto state = State(1);
    state.mean[0] = yaw;
    return state;
  }
  absl::StatusOr<MeasurementProjection> Project(
      const GaussianState& state, const double& yaw) const override {
    const double difference = yaw - state.mean[0];
    return MeasurementProjection{
        Eigen::VectorXd::Constant(
            1, std::atan2(std::sin(difference), std::cos(difference))),
        Eigen::MatrixXd::Identity(1, 1), Eigen::MatrixXd::Constant(1, 1, 0.01)};
  }
};

class IdentityMotion final : public MotionModel {
 public:
  Eigen::Index state_dimension() const override { return 1; }
  absl::Status Predict(double, GaussianState*) const override {
    return absl::OkStatus();
  }
};

TEST(PredictorTest, CustomAngularMeasurementWrapsResidual) {
  Predictor<double> predictor(std::make_shared<IdentityMotion>(),
                              std::make_shared<AngleMeasurement>());
  auto state = predictor.Initialize(3.13);
  ASSERT_TRUE(state.ok());
  ASSERT_TRUE(predictor.Update(-3.13, &*state).ok());
  EXPECT_LT(std::abs(state->mean[0] - 3.13), 0.03);
}

class InvalidStateMotion final : public MotionModel {
 public:
  Eigen::Index state_dimension() const override { return 1; }
  absl::Status Predict(double, GaussianState* state) const override {
    state->mean[0] = std::numeric_limits<double>::quiet_NaN();
    return absl::OkStatus();
  }
};

TEST(PredictorTest, InvalidPredictionOutputDoesNotMutateInputState) {
  Predictor<double> predictor(std::make_shared<InvalidStateMotion>(),
                              std::make_shared<AngleMeasurement>());
  auto state = State(1);
  state.mean[0] = 3.13;
  EXPECT_FALSE(predictor.Predict(0.1, &state).ok());
  EXPECT_DOUBLE_EQ(state.mean[0], 3.13);
  EXPECT_TRUE(state.covariance.isIdentity());
  EXPECT_FALSE(predictor.Predict(-0.1, &state).ok());
  EXPECT_FALSE(
      predictor.Predict(std::numeric_limits<double>::quiet_NaN(), &state).ok());
  EXPECT_FALSE(predictor.Predict(0.1, nullptr).ok());
}

TEST(PredictorTest, RejectsNonfiniteAsymmetricAndIndefiniteCovariance) {
  auto state = State(2);
  state.covariance(0, 1) = 1.0;
  EXPECT_FALSE(ValidateGaussian(state).ok());
  state.covariance << 1.0, 2.0, 2.0, 1.0;
  EXPECT_FALSE(ValidateGaussian(state).ok());
  state.covariance << 0.0, 1.0, 1.0, 0.0;
  EXPECT_FALSE(ValidateGaussian(state).ok());
  state.covariance.setIdentity();
  state.mean[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(ValidateGaussian(state).ok());
}

TEST(CostTest, RotatedBevAndUpright3dIou) {
  Box3D a;
  a.size = Eigen::Vector3d(4.0, 2.0, 2.0);
  auto same = BevIouCost().Evaluate(a, a);
  ASSERT_TRUE(same.ok()) << same.status();
  EXPECT_NEAR(same->value, 0.0, 1e-10);
  Box3D b = a;
  b.yaw = std::acos(-1.0) * 0.5;
  auto rotated = BevIouCost().Evaluate(a, b);
  ASSERT_TRUE(rotated.ok()) << rotated.status();
  EXPECT_NEAR(rotated->value, 2.0 / 3.0, 1e-10);
  b = a;
  b.center.z() = 1.0;
  EXPECT_NEAR(BevIouCost().Evaluate(a, b)->value, 0.0, 1e-10);
  EXPECT_NEAR(Iou3dCost().Evaluate(a, b)->value, 2.0 / 3.0, 1e-10);
  b.center.z() = 2.0;
  EXPECT_DOUBLE_EQ(Iou3dCost().Evaluate(a, b)->value, 1.0);
  b.center.x() = 10.0;
  EXPECT_DOUBLE_EQ(BevIouCost().Evaluate(a, b)->value, 1.0);
  b.size.x() = 0.0;
  EXPECT_FALSE(BevIouCost().Evaluate(a, b).ok());
}

TEST(PredictorTest, InvalidStateFailsBeforeUserProjection) {
  Predictor<double> predictor(std::make_shared<IdentityMotion>(),
                              std::make_shared<AngleMeasurement>());
  auto empty = GaussianState{};
  EXPECT_FALSE(predictor.Update(0.0, &empty).ok());
  auto wrong_dimensions = State(2);
  EXPECT_FALSE(predictor.Update(0.0, &wrong_dimensions).ok());
  TrackSnapshot<GaussianState, double> track;
  track.state = empty;
  MahalanobisCost<double> cost(std::make_shared<AngleMeasurement>(), 9.0);
  EXPECT_FALSE(cost.Evaluate(track, {1, std::make_shared<double>(0.0)}).ok());
}

TEST(CostTest, DegenerateGeometryReturnsErrorsRatherThanTerminating) {
  Box3D tiny;
  tiny.size = Eigen::Vector3d(1e-6, 1e-6, 1.0);
  EXPECT_FALSE(BevIouCost().Evaluate(tiny, tiny).ok());
  EXPECT_FALSE(Iou3dCost().Evaluate(tiny, tiny).ok());
  Box3D thin;
  thin.size = Eigen::Vector3d(1e12, 1e-7, 1.0);
  thin.yaw = std::acos(-1.0) * 0.25;
  EXPECT_FALSE(BevIouCost().Evaluate(thin, thin).ok());
  Box3D large_position;
  large_position.center = Eigen::Vector3d(1e12, -1e12, 1e12);
  auto same = Iou3dCost().Evaluate(large_position, large_position);
  ASSERT_TRUE(same.ok()) << same.status();
  EXPECT_NEAR(same->value, 0.0, 1e-12);
}

TEST(CostTest,
     NearTouchingRotatedBoxesDoNotConstructDegenerateOverlapPolygons) {
  Box3D first;
  first.size = Eigen::Vector3d(2.0, 2.0, 1.0);
  Box3D second = first;
  second.yaw = std::acos(-1.0) * 0.25;
  const double delta = std::sqrt(1.5e-10);
  second.center.x() = 1.0 + std::sqrt(2.0) * 0.5 - delta * 0.5;
  second.center.y() = second.center.x();
  const double intersection = delta * delta * 0.5;
  auto result = BevIouCost().Evaluate(first, second);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NEAR(result->value, 1.0 - intersection / (8.0 - intersection), 1e-12);
  auto reversed = BevIouCost().Evaluate(second, first);
  ASSERT_TRUE(reversed.ok()) << reversed.status();
  EXPECT_NEAR(reversed->value, result->value, 1e-12);
}

TEST(CostTest, CenterGatesAndEmbeddingDimensions) {
  Box3D a, b;
  b.center.z() = 5.0;
  EXPECT_FALSE(CenterDistanceCost(1.0).Evaluate(a, b)->allowed);
  EXPECT_TRUE(CenterDistanceCost(1.0, true).Evaluate(a, b)->allowed);
  EXPECT_FALSE(CenterDistanceCost(0.0).Evaluate(a, b).ok());
  Eigen::VectorXd feature = Eigen::VectorXd::Ones(12);
  auto same = CosineEmbeddingCost().Evaluate(feature, feature);
  ASSERT_TRUE(same.ok());
  EXPECT_NEAR(same->value, 0.0, 1e-12);
  EXPECT_NEAR(CosineEmbeddingCost().Evaluate(feature, -feature)->value, 1.0,
              1e-12);
  EXPECT_FALSE(
      CosineEmbeddingCost().Evaluate(feature, Eigen::VectorXd::Ones(11)).ok());
  EXPECT_FALSE(
      CosineEmbeddingCost().Evaluate(feature, Eigen::VectorXd::Zero(12)).ok());
}

}  // namespace
}  // namespace tracking
}  // namespace common
}  // namespace apollo
