#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "modules/common/tracking/association_cost.h"
#include "modules/common/tracking/embedding_cost.h"
#include "modules/common/tracking/multi_model_estimator.h"
#include "modules/common/tracking/multi_object_tracker.h"
#include "modules/common/tracking/predictor.h"

namespace apollo {
namespace common {
namespace tracking {
namespace {

struct Detection {
  std::string model;
  std::string category;
  Eigen::VectorXd position;
  Eigen::VectorXd embedding;
  Eigen::VectorXd initial_state;
};

using MixedState = MultiModelState<GaussianState>;
using MixedTrack = TrackSnapshot<MixedState, Detection>;
using MixedPredictor = MultiModelPredictor<Detection>;
using MixedTracker = MultiObjectTracker<MixedState, Detection>;
using FunctionalCost = FunctionalAssociationCost<MixedState, Detection>;

class Measurement final : public MeasurementModel<Detection> {
 public:
  Measurement(Eigen::Index state_dimension, Eigen::Index dimensions)
      : state_dimension_(state_dimension),
        position_(state_dimension, dimensions) {}

  absl::Status Validate(const Detection& observation) const override {
    if (observation.initial_state.size() != state_dimension_ ||
        !observation.initial_state.allFinite() ||
        observation.embedding.size() == 0 ||
        !observation.embedding.allFinite()) {
      return absl::InvalidArgumentError("invalid custom observation schema");
    }
    return position_.Validate(Position(observation));
  }

  absl::StatusOr<GaussianState> Initialize(
      const Detection& observation) const override {
    const auto valid = Validate(observation);
    if (!valid.ok()) {
      return valid;
    }
    auto state = position_.Initialize(Position(observation));
    if (!state.ok()) {
      return state.status();
    }
    state->mean = observation.initial_state;
    return state;
  }

  absl::StatusOr<MeasurementProjection> Project(
      const GaussianState& state, const Detection& observation) const override {
    const auto valid = Validate(observation);
    if (!valid.ok()) {
      return valid;
    }
    return position_.Project(state, Position(observation));
  }

 private:
  static VectorMeasurement Position(const Detection& observation) {
    return {observation.position,
            0.01 * Eigen::MatrixXd::Identity(observation.position.size(),
                                             observation.position.size())};
  }

  Eigen::Index state_dimension_;
  PositionMeasurementModel position_;
};

class CustomMotion final : public MotionModel {
 public:
  Eigen::Index state_dimension() const override { return 5; }
  absl::Status Predict(double dt, GaussianState* state) const override {
    if (state == nullptr || state->mean.size() != 5) {
      return absl::InvalidArgumentError("custom state dimensions");
    }
    state->mean[0] += dt;
    return ValidateGaussian(*state);
  }
};

std::shared_ptr<MixedPredictor> MakePredictor() {
  MixedPredictor::Registry registry;
  registry.emplace("cv_2d", std::make_shared<Predictor<Detection>>(
                                std::make_shared<ConstantVelocityModel>(2, 0.0),
                                std::make_shared<Measurement>(4, 2)));
  registry.emplace("ca_3d",
                   std::make_shared<Predictor<Detection>>(
                       std::make_shared<ConstantAccelerationModel>(3, 0.0),
                       std::make_shared<Measurement>(9, 3)));
  registry.emplace("bicycle", std::make_shared<Predictor<Detection>>(
                                  std::make_shared<BicycleModel>(2.5, 0.0),
                                  std::make_shared<Measurement>(6, 3)));
  registry.emplace("custom", std::make_shared<Predictor<Detection>>(
                                 std::make_shared<CustomMotion>(),
                                 std::make_shared<Measurement>(5, 5)));
  return std::make_shared<MixedPredictor>(
      std::move(registry),
      [](const Detection& observation) -> absl::StatusOr<std::string> {
        return observation.model;
      });
}

std::shared_ptr<CompositeAssociationCost<MixedState, Detection>> MakeCost() {
  auto compatibility = std::make_shared<FunctionalCost>(
      [](const MixedTrack& track, const Observation<Detection>& observed) {
        const auto& detection = *observed.payload;
        return detection.model == track.state.model_name &&
                       detection.category == track.last_observation->category
                   ? CostEvaluation::Accept(0.0)
                   : CostEvaluation::Reject("model/category incompatibility");
      });
  auto position = std::make_shared<FunctionalCost>(
      [](const MixedTrack& track, const Observation<Detection>& observed)
          -> absl::StatusOr<CostEvaluation> {
        const auto& measurement = observed.payload->position;
        if (measurement.size() > track.state.estimate.mean.size()) {
          return absl::InvalidArgumentError("position schema mismatch");
        }
        const double distance =
            (track.state.estimate.mean.head(measurement.size()) - measurement)
                .stableNorm();
        return distance <= 2.0 ? CostEvaluation::Accept(distance / 2.0)
                               : CostEvaluation::Reject("position gate");
      });
  auto embedding = std::make_shared<FunctionalCost>(
      [](const MixedTrack& track, const Observation<Detection>& observed) {
        return CosineEmbeddingCost().Evaluate(track.last_observation->embedding,
                                              observed.payload->embedding);
      });
  return std::make_shared<CompositeAssociationCost<MixedState, Detection>>(
      std::vector<CompositeAssociationCost<MixedState, Detection>::Term>{
          {"compatibility", 1.0, compatibility},
          {"position", 2.0, position},
          {"embedding", 3.0, embedding}});
}

TrackerConfig Config() {
  TrackerConfig config;
  config.lifecycle.confirmation_hits = 1;
  config.lifecycle.lost_timeout_seconds = 5.0;
  config.lifecycle.publish_lost = true;
  config.lifecycle.history_limit = 3;
  config.trace_limit = 100;
  return config;
}

Observation<Detection> MakeDetection(uint64_t id, std::string model,
                                     Eigen::Index state_dimension,
                                     Eigen::Index position_dimension,
                                     Eigen::Index embedding_dimension,
                                     double x) {
  Detection observation;
  observation.model = std::move(model);
  observation.category = "user-type";
  observation.initial_state = Eigen::VectorXd::Zero(state_dimension);
  observation.initial_state[0] = x;
  observation.position =
      observation.initial_state.head(position_dimension).eval();
  observation.embedding = Eigen::VectorXd::Ones(embedding_dimension);
  if (observation.model == "cv_2d") {
    observation.initial_state[2] = 2.0;
  } else if (observation.model == "ca_3d") {
    observation.initial_state[3] = 1.0;
    observation.initial_state[6] = 2.0;
  } else if (observation.model == "bicycle") {
    observation.initial_state[4] = 4.0;
  }
  return {id, std::make_shared<Detection>(std::move(observation))};
}

TEST(MultiModelTest, DifferentModelsAndDimensionsCoexistInOneTracker) {
  MixedTracker tracker(Config(), MakePredictor(), MakeCost(),
                       std::make_shared<HungarianAssociator>());
  auto birth = tracker.Update({0.0,
                               "local",
                               {MakeDetection(10, "cv_2d", 4, 2, 8, 0.0),
                                MakeDetection(20, "ca_3d", 9, 3, 12, 10.0),
                                MakeDetection(30, "bicycle", 6, 3, 16, 20.0),
                                MakeDetection(40, "custom", 5, 5, 7, 30.0)}});
  ASSERT_TRUE(birth.ok()) << birth.status();
  ASSERT_EQ(birth->tracks.size(), 4);
  auto prediction = tracker.Update({0.5, "local", {}});
  ASSERT_TRUE(prediction.ok()) << prediction.status();
  ASSERT_EQ(prediction->tracks.size(), 4);
  const std::vector<Eigen::Index> dimensions{4, 9, 6, 5};
  const std::vector<double> positions{1.0, 10.75, 22.0, 30.5};
  for (size_t index = 0; index < dimensions.size(); ++index) {
    const auto& track = prediction->tracks[index];
    EXPECT_EQ(track.id, birth->tracks[index].id);
    EXPECT_EQ(track.state.estimate.mean.size(), dimensions[index]);
    EXPECT_NEAR(track.state.estimate.mean[0], positions[index], 1e-12);
    EXPECT_FALSE(track.observed);
    EXPECT_EQ(track.status, TrackStatus::kLost);
  }
  auto restored = tracker.Update({1.0,
                                  "local",
                                  {MakeDetection(40, "custom", 5, 5, 7, 31.0),
                                   MakeDetection(30, "bicycle", 6, 3, 16, 24.0),
                                   MakeDetection(20, "ca_3d", 9, 3, 12, 12.0),
                                   MakeDetection(10, "cv_2d", 4, 2, 8, 2.0)}});
  ASSERT_TRUE(restored.ok()) << restored.status();
  ASSERT_EQ(restored->tracks.size(), 4);
  ASSERT_EQ(restored->matches.size(), 4);
  const std::vector<double> expected{2.0, 12.0, 24.0, 31.0};
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(restored->tracks[index].id, birth->tracks[index].id);
    EXPECT_NEAR(restored->tracks[index].state.estimate.mean[0], expected[index],
                1e-12);
    EXPECT_EQ(restored->tracks[index].status, TrackStatus::kConfirmed);
    EXPECT_EQ(restored->tracks[index].history.size(), 3);
    EXPECT_EQ(restored->matches[index].observation_id, (index + 1) * 10);
  }
}

TEST(MultiModelTest, MultiFeatureScoresResolveSpatialAmbiguityAndAreTraceable) {
  MixedTracker tracker(Config(), MakePredictor(), MakeCost(),
                       std::make_shared<HungarianAssociator>());
  auto first = std::make_shared<Detection>(
      *MakeDetection(1, "cv_2d", 4, 2, 2, 0.0).payload);
  first->embedding << 1.0, 0.0;
  auto second = std::make_shared<Detection>(*first);
  second->embedding << 0.0, 1.0;
  auto birth = tracker.Update({0.0, "local", {{1, first}, {2, second}}});
  ASSERT_TRUE(birth.ok());
  first = std::make_shared<Detection>(*first);
  second = std::make_shared<Detection>(*second);
  first->position[0] = 0.2;
  second->position[0] = 0.2;
  auto update = tracker.Update({0.1, "local", {{22, second}, {11, first}}});
  ASSERT_TRUE(update.ok()) << update.status();
  ASSERT_EQ(update->matches.size(), 2);
  EXPECT_EQ(update->matches[0].track_id, birth->tracks[0].id);
  EXPECT_EQ(update->matches[0].observation_id, 11);
  EXPECT_EQ(update->matches[1].track_id, birth->tracks[1].id);
  EXPECT_EQ(update->matches[1].observation_id, 22);
  ASSERT_EQ(update->diagnostics.association_traces.size(), 4);
  const auto& wrong = update->diagnostics.association_traces[0].evaluation;
  const auto& correct = update->diagnostics.association_traces[1].evaluation;
  ASSERT_EQ(correct.components.size(), 3);
  EXPECT_EQ(correct.components[1].name, "position");
  EXPECT_NEAR(correct.components[1].value, 0.0, 1e-12);
  EXPECT_EQ(correct.components[2].name, "embedding");
  EXPECT_NEAR(correct.components[2].value, 0.0, 1e-12);
  EXPECT_NEAR(wrong.components[2].value, 0.5, 1e-12);
  EXPECT_NEAR(wrong.value, 0.25, 1e-12);
  EXPECT_NEAR(correct.value, 0.0, 1e-12);
}

class ModelLifecycle final
    : public TrackLifecyclePolicy<MixedState, Detection> {
 public:
  absl::StatusOr<BirthDecision> OnBirth(
      const Observation<Detection>& observation) const override {
    return observation.payload->category == "ignore"
               ? BirthDecision{false, "user birth rule"}
               : BirthDecision{};
  }
  TrackStatus OnMiss(const Track& track, double timestamp,
                     const LifecycleConfig& config) const override {
    return track.state.model_name == "bicycle"
               ? TrackStatus::kDeleted
               : TrackLifecyclePolicy::OnMiss(track, timestamp, config);
  }
};

TEST(MultiModelTest, LifecyclePluginIsIndependentOfPredictionAndScoring) {
  MixedTracker tracker(Config(), MakePredictor(), MakeCost(),
                       std::make_shared<HungarianAssociator>(),
                       std::make_shared<ModelLifecycle>());
  auto ignored = std::make_shared<Detection>(
      *MakeDetection(30, "custom", 5, 5, 7, 30.0).payload);
  ignored->category = "ignore";
  auto birth = tracker.Update({0.0,
                               "local",
                               {MakeDetection(10, "cv_2d", 4, 2, 8, 0.0),
                                MakeDetection(20, "bicycle", 6, 3, 16, 20.0),
                                {30, ignored}}});
  ASSERT_TRUE(birth.ok()) << birth.status();
  ASSERT_EQ(birth->tracks.size(), 2);
  ASSERT_EQ(birth->diagnostics.rejected_births.size(), 1);
  auto miss = tracker.Update({0.5, "local", {}});
  ASSERT_TRUE(miss.ok());
  ASSERT_EQ(miss->tracks.size(), 1);
  EXPECT_EQ(miss->tracks[0].state.model_name, "cv_2d");
  EXPECT_EQ(miss->tracks[0].status, TrackStatus::kLost);
  EXPECT_NEAR(miss->tracks[0].state.estimate.mean[0], 1.0, 1e-12);
  ASSERT_EQ(miss->diagnostics.lifecycle_events.size(), 2);
  EXPECT_EQ(miss->diagnostics.lifecycle_events[1].to, TrackStatus::kDeleted);
}

TEST(MultiModelTest, UnknownModelsAndSchemaErrorsFailWithoutChangingTracks) {
  auto predictor = MakePredictor();
  const auto valid = MakeDetection(1, "cv_2d", 4, 2, 8, 0.0);
  auto initialized = predictor->Initialize(*valid.payload);
  ASSERT_TRUE(initialized.ok());
  auto changed_label = *valid.payload;
  changed_label.model = "unregistered";
  EXPECT_TRUE(predictor->Update(changed_label, &*initialized).ok());
  EXPECT_EQ(initialized->model_name, "cv_2d");
  auto missing = *initialized;
  missing.model_name = "unregistered";
  EXPECT_FALSE(predictor->Predict(0.5, &missing).ok());
  EXPECT_FALSE(predictor->Predict(-0.5, &*initialized).ok());
  EXPECT_FALSE(
      predictor->Predict(std::numeric_limits<double>::infinity(), &*initialized)
          .ok());
  MixedTracker tracker(Config(), predictor, MakeCost(),
                       std::make_shared<HungarianAssociator>());
  ASSERT_TRUE(tracker.Update({0.0, "local", {valid}}).ok());
  EXPECT_FALSE(
      tracker
          .Update({0.1, "local", {MakeDetection(2, "unknown", 4, 2, 8, 0.2)}})
          .ok());
  auto invalid_schema = std::make_shared<Detection>(
      *MakeDetection(2, "ca_3d", 9, 3, 8, 0.2).payload);
  invalid_schema->initial_state.conservativeResize(4);
  EXPECT_FALSE(tracker.Update({0.1, "local", {{2, invalid_schema}}}).ok());
  EXPECT_DOUBLE_EQ(tracker.tracks()[0].state_time, 0.0);
  EXPECT_DOUBLE_EQ(tracker.tracks()[0].state.estimate.mean[0], 0.0);
  auto recovered =
      tracker.Update({0.1, "local", {MakeDetection(2, "cv_2d", 4, 2, 8, 0.2)}});
  ASSERT_TRUE(recovered.ok()) << recovered.status();
  EXPECT_EQ(recovered->tracks[0].id, tracker.tracks()[0].id);
  EXPECT_NEAR(recovered->tracks[0].state.estimate.mean[0], 0.2, 1e-12);
}

class MutatingFailureEstimator final
    : public StateEstimator<GaussianState, Detection> {
 public:
  absl::Status Validate(const Detection&) const override {
    return absl::OkStatus();
  }
  absl::StatusOr<GaussianState> Initialize(
      const Detection& observation) const override {
    return GaussianState{
        observation.initial_state,
        Eigen::MatrixXd::Identity(observation.initial_state.size(),
                                  observation.initial_state.size())};
  }
  absl::Status Predict(double, GaussianState* state) const override {
    state->mean[0] = 999.0;
    return absl::InvalidArgumentError("prediction failed after mutation");
  }
  absl::Status Update(const Detection&, GaussianState* state) const override {
    state->mean[0] = 999.0;
    return absl::InvalidArgumentError("update failed after mutation");
  }
};

TEST(MultiModelTest, FailedPluginMutationsDoNotEscapeTheRoutingLayer) {
  MixedPredictor predictor(
      {{"failing", std::make_shared<MutatingFailureEstimator>()}},
      [](const Detection&) -> absl::StatusOr<std::string> {
        return "failing";
      });
  const auto observation = MakeDetection(1, "cv_2d", 4, 2, 8, 0.0);
  auto state = predictor.Initialize(*observation.payload);
  ASSERT_TRUE(state.ok());
  EXPECT_FALSE(predictor.Predict(0.1, &*state).ok());
  EXPECT_DOUBLE_EQ(state->estimate.mean[0], 0.0);
  EXPECT_FALSE(predictor.Update(*observation.payload, &*state).ok());
  EXPECT_DOUBLE_EQ(state->estimate.mean[0], 0.0);
  EXPECT_EQ(state->model_name, "failing");
}

TEST(MultiModelTest, RegistryAndSelectorConfigurationErrorsAreExplicit) {
  const auto observation = MakeDetection(1, "cv_2d", 4, 2, 8, 0.0);
  MixedPredictor empty({}, [](const Detection&) {
    return absl::StatusOr<std::string>("cv_2d");
  });
  EXPECT_FALSE(empty.Validate(*observation.payload).ok());
  MixedPredictor invalid({{"cv_2d", nullptr}}, {});
  EXPECT_FALSE(invalid.Validate(*observation.payload).ok());
  auto estimator = std::make_shared<Predictor<Detection>>(
      std::make_shared<ConstantVelocityModel>(2),
      std::make_shared<Measurement>(4, 2));
  MixedPredictor no_selector({{"cv_2d", estimator}}, {});
  EXPECT_FALSE(no_selector.Validate(*observation.payload).ok());
  MixedPredictor selector_error(
      {{"cv_2d", estimator}},
      [](const Detection&) -> absl::StatusOr<std::string> {
        return absl::InvalidArgumentError("selector cannot determine model");
      });
  EXPECT_FALSE(selector_error.Initialize(*observation.payload).ok());
}

}  // namespace
}  // namespace tracking
}  // namespace common
}  // namespace apollo
