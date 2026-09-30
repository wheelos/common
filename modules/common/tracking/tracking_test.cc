#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "modules/common/tracking/association_cost.h"
#include "modules/common/tracking/gaussian_cost.h"
#include "modules/common/tracking/multi_object_tracker.h"
#include "modules/common/tracking/predictor.h"

namespace apollo {
namespace common {
namespace tracking {
namespace {

// Deliberately not an Apollo object, Eigen state or fixed tracking schema.
struct TensorDetection {
  std::vector<size_t> shape;
  std::vector<float> tensor;
  std::string category;
};

struct TensorState {
  std::vector<float> tensor;
  std::string category;
  size_t predictions = 0;
};

class TensorEstimator final
    : public StateEstimator<TensorState, TensorDetection> {
 public:
  absl::Status Validate(const TensorDetection& observation) const override {
    size_t size = 1;
    for (size_t extent : observation.shape) {
      if (extent == 0 || size > std::numeric_limits<size_t>::max() / extent) {
        return absl::InvalidArgumentError("invalid tensor shape");
      }
      size *= extent;
    }
    if (observation.shape.empty() || size != observation.tensor.size()) {
      return absl::InvalidArgumentError("tensor dimensions mismatch");
    }
    for (float value : observation.tensor) {
      if (!std::isfinite(value)) {
        return absl::InvalidArgumentError("nonfinite tensor");
      }
    }
    return absl::OkStatus();
  }
  absl::StatusOr<TensorState> Initialize(
      const TensorDetection& observation) const override {
    return TensorState{observation.tensor, observation.category, 0};
  }
  absl::Status Predict(double dt, TensorState* state) const override {
    if (dt < 0.0) {
      return absl::InvalidArgumentError("negative dt");
    }
    ++state->predictions;
    return absl::OkStatus();
  }
  absl::Status Update(const TensorDetection& observation,
                      TensorState* state) const override {
    if (observation.category == "fail-update") {
      return absl::InvalidArgumentError("injected update failure");
    }
    state->tensor = observation.tensor;
    return absl::OkStatus();
  }
};

using TensorTracker = MultiObjectTracker<TensorState, TensorDetection>;

TrackerConfig Config() {
  TrackerConfig config;
  config.lifecycle.confirmation_hits = 2;
  config.lifecycle.lost_timeout_seconds = 2.0;
  config.lifecycle.publish_tentative = true;
  config.lifecycle.publish_lost = true;
  config.lifecycle.history_limit = 3;
  config.trace_limit = 1;
  return config;
}

TensorTracker MakeTracker(TrackerConfig config = Config()) {
  auto cost =
      std::make_shared<FunctionalAssociationCost<TensorState, TensorDetection>>(
          [](const TrackSnapshot<TensorState, TensorDetection>& track,
             const Observation<TensorDetection>& observation)
              -> absl::StatusOr<CostEvaluation> {
            const auto& payload = *observation.payload;
            if (payload.category != track.state.category &&
                payload.category != "fail-update") {
              return CostEvaluation::Reject("category");
            }
            if (payload.tensor.size() != track.state.tensor.size()) {
              return CostEvaluation::Reject("feature dimension");
            }
            double squared = 0.0;
            for (size_t index = 0; index < payload.tensor.size(); ++index) {
              const double delta =
                  payload.tensor[index] - track.state.tensor[index];
              squared += delta * delta;
            }
            return squared < 1.0 ? CostEvaluation::Accept(squared)
                                 : CostEvaluation::Reject("tensor distance");
          });
  return TensorTracker(config, std::make_shared<TensorEstimator>(), cost,
                       std::make_shared<HungarianAssociator>());
}

Observation<TensorDetection> Detection(uint64_t id, float value,
                                       std::string category = "custom-class") {
  return {id, std::make_shared<TensorDetection>(TensorDetection{
                  {2, 6}, std::vector<float>(12, value), std::move(category)})};
}

TrackingFrame<TensorDetection> Frame(
    double time, std::vector<Observation<TensorDetection>> observations = {}) {
  return {time, "user-coordinate-frame", std::move(observations)};
}

TEST(TrackingTest, CustomTensorStateAndLifecycle) {
  auto tracker = MakeTracker();
  auto birth = tracker.Update(Frame(0.0, {Detection(10, 0.0f)}));
  ASSERT_TRUE(birth.ok()) << birth.status();
  ASSERT_EQ(birth->tracks.size(), 1);
  EXPECT_EQ(birth->tracks[0].status, TrackStatus::kTentative);
  const uint64_t id = birth->tracks[0].id;
  auto confirmed = tracker.Update(Frame(0.2, {Detection(20, 0.01f)}));
  ASSERT_TRUE(confirmed.ok()) << confirmed.status();
  EXPECT_EQ(confirmed->tracks[0].status, TrackStatus::kConfirmed);
  EXPECT_EQ(confirmed->tracks[0].state.tensor.size(), 12);
  EXPECT_EQ(confirmed->matches[0].observation_id, 20);
  auto lost = tracker.Update(Frame(0.5));
  ASSERT_TRUE(lost.ok()) << lost.status();
  EXPECT_EQ(lost->tracks[0].status, TrackStatus::kLost);
  EXPECT_FALSE(lost->tracks[0].observed);
  EXPECT_DOUBLE_EQ(lost->tracks[0].last_observed_time, 0.2);
  EXPECT_DOUBLE_EQ(lost->tracks[0].state_time, 0.5);
  auto restored = tracker.Update(Frame(0.8, {Detection(30, 0.02f)}));
  ASSERT_TRUE(restored.ok()) << restored.status();
  EXPECT_EQ(restored->tracks[0].id, id);
  EXPECT_EQ(restored->tracks[0].status, TrackStatus::kConfirmed);
  EXPECT_EQ(restored->tracks[0].state.predictions, 3);
  EXPECT_EQ(restored->tracks[0].history.size(), 3);
  EXPECT_FALSE(restored->tracks[0].history[1].observed);
  auto deleted = tracker.Update(Frame(3.0, {Detection(40, 0.02f)}));
  ASSERT_TRUE(deleted.ok()) << deleted.status();
  ASSERT_EQ(deleted->tracks.size(), 1);
  EXPECT_NE(deleted->tracks[0].id, id);
  EXPECT_EQ(deleted->diagnostics.lifecycle_events[0].to, TrackStatus::kDeleted);
  EXPECT_EQ(deleted->tracks[0].status, TrackStatus::kTentative);
}

TEST(TrackingTest, TentativeMissAndPublicationPolicy) {
  TrackerConfig config = Config();
  config.lifecycle.publish_tentative = false;
  config.lifecycle.publish_lost = false;
  auto tracker = MakeTracker(config);
  auto birth = tracker.Update(Frame(0.0, {Detection(1, 0.0f)}));
  ASSERT_TRUE(birth.ok());
  EXPECT_TRUE(birth->tracks.empty());
  EXPECT_EQ(tracker.tracks().size(), 1);
  auto missed = tracker.Update(Frame(0.1));
  ASSERT_TRUE(missed.ok());
  EXPECT_TRUE(tracker.tracks().empty());
  EXPECT_EQ(missed->diagnostics.lifecycle_events[0].to, TrackStatus::kDeleted);
  ASSERT_TRUE(tracker.Update(Frame(0.2, {Detection(1, 0.0f)})).ok());
  auto confirmed = tracker.Update(Frame(0.3, {Detection(2, 0.01f)}));
  ASSERT_TRUE(confirmed.ok());
  EXPECT_EQ(confirmed->tracks.size(), 1);
  auto lost = tracker.Update(Frame(0.4));
  ASSERT_TRUE(lost.ok());
  EXPECT_TRUE(lost->tracks.empty());
  EXPECT_EQ(tracker.tracks()[0].status, TrackStatus::kLost);
}

TEST(TrackingTest, StrictTimeCoordinatesAndAtomicFailure) {
  auto tracker = MakeTracker();
  ASSERT_TRUE(tracker.Update(Frame(0.0, {Detection(1, 0.0f)})).ok());
  EXPECT_FALSE(tracker.Update(Frame(0.0)).ok());
  EXPECT_FALSE(tracker.Update(Frame(-1.0)).ok());
  auto invalid_frame = Frame(0.1);
  invalid_frame.coordinate_frame = "different";
  EXPECT_FALSE(tracker.Update(invalid_frame).ok());
  EXPECT_FALSE(
      tracker.Update(Frame(std::numeric_limits<double>::quiet_NaN())).ok());
  EXPECT_FALSE(
      tracker.Update(Frame(0.1, {Detection(1, 0.0f), Detection(1, 0.0f)}))
          .ok());
  EXPECT_FALSE(tracker.Update(Frame(0.1, {{3, nullptr}})).ok());
  auto bad_tensor = std::make_shared<TensorDetection>(
      TensorDetection{{2, 6}, std::vector<float>(11), "custom-class"});
  EXPECT_FALSE(tracker.Update(Frame(0.1, {{1, bad_tensor}})).ok());
  EXPECT_FALSE(
      tracker.Update(Frame(0.1, {Detection(1, 0.0f, "fail-update")})).ok());
  EXPECT_EQ(tracker.tracks()[0].state.predictions, 0);
  EXPECT_EQ(tracker.tracks()[0].hits, 1);
  auto valid = tracker.Update(Frame(0.1, {Detection(1, 0.0f)}));
  ASSERT_TRUE(valid.ok()) << valid.status();
  EXPECT_EQ(valid->tracks[0].state.predictions, 1);
  EXPECT_EQ(valid->tracks[0].hits, 2);
}

TEST(TrackingTest, BoundedDiagnosticsAndCandidateLimit) {
  auto tracker = MakeTracker();
  ASSERT_TRUE(
      tracker
          .Update(Frame(0.0, {Detection(1, 0.0f), Detection(2, 1.0f, "other")}))
          .ok());
  auto result = tracker.Update(
      Frame(0.1, {Detection(3, 0.0f), Detection(4, 1.0f, "other")}));
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->diagnostics.candidate_count, 4);
  EXPECT_EQ(result->diagnostics.allowed_count, 2);
  EXPECT_EQ(result->diagnostics.rejected_count, 2);
  EXPECT_EQ(result->diagnostics.association_traces.size(), 1);
  EXPECT_EQ(result->diagnostics.omitted_trace_count, 3);
  EXPECT_GE(result->diagnostics.elapsed_seconds, 0.0);
  auto config = Config();
  config.max_candidate_pairs = 1;
  config.trace_limit = 0;
  auto limited = MakeTracker(config);
  ASSERT_TRUE(limited.Update(Frame(0.0, {Detection(1, 0.0f)})).ok());
  EXPECT_FALSE(
      limited.Update(Frame(0.1, {Detection(1, 0.0f), Detection(2, 1.0f)}))
          .ok());
  EXPECT_EQ(limited.tracks()[0].state.predictions, 0);
  auto accepted = limited.Update(Frame(0.1, {Detection(1, 0.0f)}));
  ASSERT_TRUE(accepted.ok());
  EXPECT_TRUE(accepted->diagnostics.association_traces.empty());
}

TEST(TrackingTest, ResetNeverReusesIdsAndReplayIsDeterministic) {
  auto first = MakeTracker();
  auto second = MakeTracker();
  for (double time : {0.0, 0.2, 0.4, 0.6}) {
    const auto a = first.Update(Frame(time, {Detection(1, 0.01f)}));
    const auto b = second.Update(Frame(time, {Detection(1, 0.01f)}));
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    EXPECT_EQ(a->tracks[0].id, b->tracks[0].id);
    EXPECT_EQ(a->tracks[0].status, b->tracks[0].status);
    EXPECT_EQ(a->tracks[0].state.tensor, b->tracks[0].state.tensor);
  }
  const auto original_id = first.tracks()[0].id;
  first.Reset();
  auto born = first.Update(Frame(0.0, {Detection(1, 0.0f)}));
  ASSERT_TRUE(born.ok());
  EXPECT_GT(born->tracks[0].id, original_id);
  EXPECT_EQ(second.tracks()[0].hits, 4);
}

class BadAssociator final : public Associator {
 public:
  absl::StatusOr<AssignmentResult> Associate(
      const AssociationProblem&) const override {
    return AssignmentResult{};
  }
};

TEST(TrackingTest, InvalidPluginOutputIsNotSilentlyAccepted) {
  auto cost =
      std::make_shared<FunctionalAssociationCost<TensorState, TensorDetection>>(
          [](const TrackSnapshot<TensorState, TensorDetection>&,
             const Observation<TensorDetection>&) {
            return CostEvaluation::Accept(0.0);
          });
  TensorTracker tracker(Config(), std::make_shared<TensorEstimator>(), cost,
                        std::make_shared<BadAssociator>());
  EXPECT_FALSE(tracker.Update(Frame(0.0, {Detection(1, 0.0f)})).ok());
  EXPECT_TRUE(tracker.tracks().empty());
  auto bad_cost =
      std::make_shared<FunctionalAssociationCost<TensorState, TensorDetection>>(
          [](const TrackSnapshot<TensorState, TensorDetection>&,
             const Observation<TensorDetection>&) {
            return CostEvaluation::Accept(
                std::numeric_limits<double>::quiet_NaN());
          });
  TensorTracker invalid(Config(), std::make_shared<TensorEstimator>(), bad_cost,
                        std::make_shared<HungarianAssociator>());
  ASSERT_TRUE(invalid.Update(Frame(0.0, {Detection(1, 0.0f)})).ok());
  EXPECT_FALSE(invalid.Update(Frame(0.1, {Detection(1, 0.0f)})).ok());
  EXPECT_EQ(invalid.tracks()[0].state.predictions, 0);
}

TEST(TrackingTest, PayloadRemainsAliveAndImmutableAfterCallerRelease) {
  auto tracker = MakeTracker();
  auto observation = Detection(1, 0.0f);
  std::weak_ptr<const TensorDetection> weak = observation.payload;
  ASSERT_TRUE(tracker.Update(Frame(0.0, {observation})).ok());
  observation.payload.reset();
  EXPECT_FALSE(weak.expired());
  EXPECT_EQ(tracker.tracks()[0].last_observation->tensor.size(), 12);
  tracker.Reset();
  EXPECT_TRUE(weak.expired());
}

TEST(TrackingTest, CompositeCostsPreserveTermsAndHardGates) {
  using Cost = FunctionalAssociationCost<TensorState, TensorDetection>;
  auto first = std::make_shared<Cost>(
      [](const TrackSnapshot<TensorState, TensorDetection>&,
         const Observation<TensorDetection>&) {
        return CostEvaluation::Accept(0.2);
      });
  auto second = std::make_shared<Cost>(
      [](const TrackSnapshot<TensorState, TensorDetection>&,
         const Observation<TensorDetection>&) {
        return CostEvaluation::Accept(0.8);
      });
  CompositeAssociationCost<TensorState, TensorDetection> composite(
      {{"geometry", 3.0, first}, {"embedding", 1.0, second}});
  TrackSnapshot<TensorState, TensorDetection> track;
  auto observation = Detection(1, 0.0f);
  auto result = composite.Evaluate(track, observation);
  ASSERT_TRUE(result.ok());
  EXPECT_NEAR(result->value, 0.35, 1e-12);
  ASSERT_EQ(result->components.size(), 2);
  EXPECT_EQ(result->components[1].name, "embedding");
  auto gate = std::make_shared<Cost>(
      [](const TrackSnapshot<TensorState, TensorDetection>&,
         const Observation<TensorDetection>&) {
        return CostEvaluation::Reject("incompatible category");
      });
  CompositeAssociationCost<TensorState, TensorDetection> gated(
      {{"category", 1.0, gate}, {"geometry", 1.0, first}});
  auto rejected = gated.Evaluate(track, observation);
  ASSERT_TRUE(rejected.ok());
  EXPECT_FALSE(rejected->allowed);
  EXPECT_EQ(rejected->reason, "category: incompatible category");
  CompositeAssociationCost<TensorState, TensorDetection> invalid(
      {{"term", -1.0, first}});
  EXPECT_FALSE(invalid.Evaluate(track, observation).ok());
}

struct NonDefaultState {
  explicit NonDefaultState(int input) : value(input) {}
  int value;
};

class NonDefaultEstimator final : public StateEstimator<NonDefaultState, int> {
 public:
  absl::Status Validate(const int&) const override { return absl::OkStatus(); }
  absl::StatusOr<NonDefaultState> Initialize(const int& value) const override {
    return NonDefaultState(value);
  }
  absl::Status Predict(double, NonDefaultState*) const override {
    return absl::OkStatus();
  }
  absl::Status Update(const int& value, NonDefaultState* state) const override {
    state->value = value;
    return absl::OkStatus();
  }
};

TEST(TrackingTest, UserStateNeedNotBeDefaultConstructible) {
  auto cost = std::make_shared<FunctionalAssociationCost<NonDefaultState, int>>(
      [](const TrackSnapshot<NonDefaultState, int>&, const Observation<int>&) {
        return CostEvaluation::Accept(0.0);
      });
  MultiObjectTracker<NonDefaultState, int> tracker(
      Config(), std::make_shared<NonDefaultEstimator>(), cost,
      std::make_shared<HungarianAssociator>());
  auto birth =
      tracker.Update({0.0, "scalar", {{1, std::make_shared<int>(42)}}});
  ASSERT_TRUE(birth.ok()) << birth.status();
  EXPECT_EQ(birth->tracks[0].state.value, 42);
  auto update =
      tracker.Update({0.1, "scalar", {{2, std::make_shared<int>(43)}}});
  ASSERT_TRUE(update.ok());
  EXPECT_EQ(update->tracks[0].state.value, 43);
}

class TensorLifecycle final
    : public TrackLifecyclePolicy<TensorState, TensorDetection> {
 public:
  absl::StatusOr<BirthDecision> OnBirth(
      const Observation<TensorDetection>& observation) const override {
    if (observation.payload->category == "reject-birth") {
      return BirthDecision{false, "user category not eligible for birth"};
    }
    return BirthDecision{};
  }
  TrackStatus OnHit(const Track& track,
                    const LifecycleConfig& config) const override {
    if (track.state.tensor[0] > 0.1f) {
      return TrackStatus::kDeleted;
    }
    return TrackLifecyclePolicy::OnHit(track, config);
  }
};

TEST(TrackingTest, LifecyclePoliciesInspectUserPayloadAndState) {
  auto cost =
      std::make_shared<FunctionalAssociationCost<TensorState, TensorDetection>>(
          [](const TrackSnapshot<TensorState, TensorDetection>&,
             const Observation<TensorDetection>&) {
            return CostEvaluation::Accept(0.0);
          });
  TensorTracker tracker(Config(), std::make_shared<TensorEstimator>(), cost,
                        std::make_shared<HungarianAssociator>(),
                        std::make_shared<TensorLifecycle>());
  auto rejected =
      tracker.Update(Frame(0.0, {Detection(1, 0.0f, "reject-birth")}));
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_TRUE(tracker.tracks().empty());
  ASSERT_EQ(rejected->diagnostics.rejected_births.size(), 1);
  EXPECT_EQ(rejected->diagnostics.rejected_births[0].observation_id, 1);
  ASSERT_TRUE(tracker.Update(Frame(0.1, {Detection(2, 0.0f)})).ok());
  auto deleted = tracker.Update(Frame(0.2, {Detection(3, 0.2f)}));
  ASSERT_TRUE(deleted.ok());
  EXPECT_TRUE(tracker.tracks().empty());
  ASSERT_EQ(deleted->diagnostics.lifecycle_events.size(), 1);
  EXPECT_EQ(deleted->diagnostics.lifecycle_events[0].to, TrackStatus::kDeleted);
}

TEST(TrackingTest, GaussianPredictorTracksCrossingObjects) {
  auto measurement = std::make_shared<PositionMeasurementModel>(2, 1, 100.0);
  auto predictor = std::make_shared<Predictor<VectorMeasurement>>(
      std::make_shared<ConstantVelocityModel>(1, 0.01), measurement);
  auto cost =
      std::make_shared<MahalanobisCost<VectorMeasurement>>(measurement, 25.0);
  auto config = Config();
  config.lifecycle.confirmation_hits = 1;
  MultiObjectTracker<GaussianState, VectorMeasurement> tracker(
      config, predictor, cost, std::make_shared<HungarianAssociator>());
  auto detection = [](uint64_t id, double position) {
    VectorMeasurement value;
    value.value = Eigen::VectorXd::Constant(1, position);
    value.covariance = Eigen::MatrixXd::Constant(1, 1, 0.001);
    return Observation<VectorMeasurement>{
        id, std::make_shared<VectorMeasurement>(std::move(value))};
  };
  auto born =
      tracker.Update({0.0, "local", {detection(1, -3), detection(2, 3)}});
  ASSERT_TRUE(born.ok()) << born.status();
  const uint64_t rightward = born->tracks[0].id;
  const uint64_t leftward = born->tracks[1].id;
  for (int step = 1; step <= 6; ++step) {
    auto result =
        tracker.Update({static_cast<double>(step),
                        "local",
                        {detection(2, 3.0 - step), detection(1, -3.0 + step)}});
    ASSERT_TRUE(result.ok()) << result.status();
    ASSERT_EQ(result->matches.size(), 2);
    if (step != 3) {
      for (const auto& match : result->matches) {
        EXPECT_EQ(match.track_id,
                  match.observation_id == 1 ? rightward : leftward);
      }
    }
    ASSERT_EQ(result->tracks.size(), 2);
  }
  EXPECT_NEAR(tracker.tracks()[0].state.mean[1], 1.0, 0.05);
  EXPECT_NEAR(tracker.tracks()[1].state.mean[1], -1.0, 0.05);
}

}  // namespace
}  // namespace tracking
}  // namespace common
}  // namespace apollo
