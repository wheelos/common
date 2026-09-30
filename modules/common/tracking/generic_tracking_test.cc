#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "modules/common/tracking/association_cost.h"
#include "modules/common/tracking/multi_object_tracker.h"

namespace apollo {
namespace common {
namespace tracking {
namespace {

struct Payload {
  double position = 0.0;
};

struct State {
  double position = 0.0;
  size_t predictions = 0;
};

using Tracker = MultiObjectTracker<State, Payload>;
using Policy = TrackLifecyclePolicy<State, Payload>;
using Cost = FunctionalAssociationCost<State, Payload>;

class Estimator final : public StateEstimator<State, Payload> {
 public:
  absl::Status Validate(const Payload& observation) const override {
    return std::isfinite(observation.position)
               ? absl::OkStatus()
               : absl::InvalidArgumentError("nonfinite position");
  }
  absl::StatusOr<State> Initialize(const Payload& observation) const override {
    return State{observation.position, 0};
  }
  absl::Status Predict(double dt, State* state) const override {
    state->position += dt;
    ++state->predictions;
    return absl::OkStatus();
  }
  absl::Status Update(const Payload& observation, State* state) const override {
    state->position = observation.position;
    return absl::OkStatus();
  }
};

std::shared_ptr<Cost> MakeCost() {
  return std::make_shared<Cost>(
      [](const Tracker::Track& track, const Observation<Payload>& observation) {
        const double distance =
            std::abs(track.state.position - observation.payload->position);
        return distance <= 2.0 ? CostEvaluation::Accept(distance / 2.0)
                               : CostEvaluation::Reject("position gate");
      });
}

TrackerConfig Config() {
  TrackerConfig config;
  config.lifecycle.confirmation_hits = 1;
  config.lifecycle.lost_timeout_seconds = 1.0;
  config.lifecycle.publish_lost = true;
  config.lifecycle.history_limit = 5;
  config.trace_limit = 1;
  return config;
}

Observation<Payload> Observe(uint64_t id, double position) {
  return {id, std::make_shared<Payload>(Payload{position})};
}

Tracker MakeTracker(
    TrackerConfig config = Config(),
    std::shared_ptr<const Policy> policy = std::make_shared<Policy>()) {
  return Tracker(config, std::make_shared<Estimator>(), MakeCost(),
                 std::make_shared<HungarianAssociator>(), std::move(policy));
}

class RetentionPolicy final : public Policy {
 public:
  absl::StatusOr<bool> ShouldExpire(const Track& track, double timestamp,
                                    const LifecycleConfig&) const override {
    return timestamp - track.last_observed_time > 3.0;
  }
  bool ShouldPublish(const Track& track,
                     const LifecycleConfig&) const override {
    return track.observed;
  }
};

TEST(GenericTrackingTest, PolicyOwnsExpirationAndPublication) {
  auto tracker = MakeTracker(Config(), std::make_shared<RetentionPolicy>());
  auto born = tracker.Update({0.0, "local", {Observe(1, 0.0)}});
  ASSERT_TRUE(born.ok()) << born.status();
  const uint64_t id = born->tracks[0].id;
  auto lost = tracker.Update({2.0, "local", {}});
  ASSERT_TRUE(lost.ok()) << lost.status();
  EXPECT_TRUE(lost->tracks.empty());
  ASSERT_EQ(tracker.tracks().size(), 1);
  EXPECT_EQ(tracker.tracks()[0].status, TrackStatus::kLost);
  EXPECT_DOUBLE_EQ(tracker.tracks()[0].state.position, 2.0);
  auto restored = tracker.Update({2.5, "local", {Observe(2, 2.5)}});
  ASSERT_TRUE(restored.ok()) << restored.status();
  ASSERT_EQ(restored->tracks.size(), 1);
  EXPECT_EQ(restored->tracks[0].id, id);
  auto expired = tracker.Update({6.0, "local", {}});
  ASSERT_TRUE(expired.ok());
  EXPECT_TRUE(tracker.tracks().empty());
  ASSERT_EQ(expired->diagnostics.lifecycle_events.size(), 1);
  EXPECT_EQ(expired->diagnostics.lifecycle_events[0].to, TrackStatus::kDeleted);
}

TEST(GenericTrackingTest, DefaultExpirationBoundaryIsPreserved) {
  auto tracker = MakeTracker();
  ASSERT_TRUE(tracker.Update({0.0, "local", {Observe(1, 0.0)}}).ok());
  auto boundary = tracker.Update({1.0, "local", {}});
  ASSERT_TRUE(boundary.ok());
  ASSERT_EQ(boundary->tracks.size(), 1);
  EXPECT_EQ(boundary->tracks[0].status, TrackStatus::kLost);
  auto expired = tracker.Update({1.0001, "local", {}});
  ASSERT_TRUE(expired.ok());
  EXPECT_TRUE(tracker.tracks().empty());
}

class InvalidTransitionPolicy final : public Policy {
 public:
  TrackStatus OnHit(const Track& track,
                    const LifecycleConfig& config) const override {
    return track.hits > 1 ? TrackStatus::kTentative
                          : Policy::OnHit(track, config);
  }
};

class ExpirationErrorPolicy final : public Policy {
 public:
  absl::StatusOr<bool> ShouldExpire(const Track&, double,
                                    const LifecycleConfig&) const override {
    return absl::InvalidArgumentError("expiration policy failure");
  }
};

TEST(GenericTrackingTest, PolicyErrorsAndInvalidTransitionsRollBackTheFrame) {
  for (const auto& policy : std::vector<std::shared_ptr<const Policy>>{
           std::make_shared<InvalidTransitionPolicy>(),
           std::make_shared<ExpirationErrorPolicy>()}) {
    auto tracker = MakeTracker(Config(), policy);
    ASSERT_TRUE(tracker.Update({0.0, "local", {Observe(1, 0.0)}}).ok());
    auto failure = tracker.Update({0.1, "local", {Observe(2, 0.1)}});
    EXPECT_FALSE(failure.ok());
    ASSERT_EQ(tracker.tracks().size(), 1);
    EXPECT_DOUBLE_EQ(tracker.tracks()[0].state_time, 0.0);
    EXPECT_EQ(tracker.tracks()[0].hits, 1);
    EXPECT_EQ(tracker.tracks()[0].state.predictions, 0);
    EXPECT_EQ(tracker.tracks()[0].status, TrackStatus::kConfirmed);
  }
}

class RejectionPolicy final : public Policy {
 public:
  absl::StatusOr<BirthDecision> OnBirth(
      const Observation<Payload>& observation) const override {
    return observation.payload->position < 0.0 ? BirthDecision{false, {}}
                                               : BirthDecision{};
  }
};

TEST(GenericTrackingTest, BirthRejectionsRequireReasonsAndDoNotConsumeIds) {
  auto tracker = MakeTracker(Config(), std::make_shared<RejectionPolicy>());
  EXPECT_FALSE(tracker.Update({0.0, "local", {Observe(1, -1.0)}}).ok());
  EXPECT_TRUE(tracker.tracks().empty());
  auto valid = tracker.Update({0.0, "local", {Observe(2, 0.0)}});
  ASSERT_TRUE(valid.ok());
  EXPECT_EQ(valid->tracks[0].id, 1);
}

TEST(GenericTrackingTest, ResourceLimitsFailExplicitlyWithoutPartialCommit) {
  auto config = Config();
  config.max_tracks = 1;
  config.max_observations_per_frame = 2;
  auto tracker = MakeTracker(config);
  ASSERT_TRUE(tracker.Update({0.0, "local", {Observe(1, 0.0)}}).ok());
  EXPECT_FALSE(
      tracker
          .Update({0.1,
                   "local",
                   {Observe(1, 0.1), Observe(2, 10.0), Observe(3, 20.0)}})
          .ok());
  EXPECT_FALSE(
      tracker.Update({0.1, "local", {Observe(1, 0.1), Observe(2, 10.0)}}).ok());
  EXPECT_EQ(tracker.tracks()[0].hits, 1);
  EXPECT_EQ(tracker.tracks()[0].state.predictions, 0);
  auto valid = tracker.Update({0.1, "local", {Observe(1, 0.1)}});
  ASSERT_TRUE(valid.ok());
  EXPECT_EQ(valid->tracks[0].id, 1);
  auto component_config = Config();
  component_config.max_cost_components = 0;
  auto composite = std::make_shared<CompositeAssociationCost<State, Payload>>(
      std::vector<CompositeAssociationCost<State, Payload>::Term>{
          {"position", 1.0, MakeCost()}});
  Tracker limited(component_config, std::make_shared<Estimator>(), composite,
                  std::make_shared<HungarianAssociator>());
  ASSERT_TRUE(limited.Update({0.0, "local", {Observe(1, 0.0)}}).ok());
  EXPECT_FALSE(limited.Update({0.1, "local", {Observe(1, 0.1)}}).ok());
  EXPECT_EQ(limited.tracks()[0].state.predictions, 0);
}

TEST(GenericTrackingTest, RepeatedUpdatesKeepHistoryAndTracesBounded) {
  auto tracker = MakeTracker();
  ASSERT_TRUE(tracker.Update({0.0, "local", {Observe(1, 0.0)}}).ok());
  for (size_t step = 1; step < 2000; ++step) {
    const double time = static_cast<double>(step) * 0.01;
    const std::vector<Observation<Payload>> observations =
        step % 5 == 0 ? std::vector<Observation<Payload>>{}
                      : std::vector<Observation<Payload>>{Observe(step, time)};
    auto result = tracker.Update({time, "local", observations});
    ASSERT_TRUE(result.ok()) << result.status();
    ASSERT_EQ(result->tracks.size(), 1);
    EXPECT_EQ(result->tracks[0].id, 1);
    EXPECT_EQ(result->tracks[0].state.predictions, step);
    EXPECT_LE(result->tracks[0].history.size(), 5);
    EXPECT_LE(result->diagnostics.association_traces.size(), 1);
    EXPECT_NEAR(result->tracks[0].state.position, time, 1e-12);
  }
}

TEST(GenericTrackingTest, ExtremeFiniteTimesCannotOverflowPredictionIntervals) {
  auto tracker = MakeTracker();
  ASSERT_TRUE(tracker.Update({-1e308, "local", {Observe(1, 0.0)}}).ok());
  EXPECT_FALSE(tracker.Update({1e308, "local", {}}).ok());
  EXPECT_EQ(tracker.tracks()[0].state.predictions, 0);
}

TEST(GenericTrackingTest, CounterExhaustionIsNotSilentlyWrapped) {
  auto config = Config();
  TrackManager<State, Payload> manager(config.lifecycle,
                                       std::make_shared<Policy>());
  Tracker::Track track;
  track.status = TrackStatus::kConfirmed;
  track.hits = std::numeric_limits<size_t>::max();
  TrackingDiagnostics diagnostics;
  EXPECT_FALSE(manager.Hit(0.1, Observe(1, 0.1), &track, &diagnostics).ok());
  EXPECT_EQ(track.hits, std::numeric_limits<size_t>::max());
  track.consecutive_misses = std::numeric_limits<size_t>::max();
  EXPECT_FALSE(manager.Miss(0.1, &track, &diagnostics).ok());
}

TEST(GenericTrackingTest, CompositeCostDoesNotHideMissingRejectionReasons) {
  auto rejected = std::make_shared<Cost>(
      [](const Tracker::Track&, const Observation<Payload>&) {
        return CostEvaluation::Reject({});
      });
  CompositeAssociationCost<State, Payload> composite(
      {{"broken-term", 1.0, rejected}});
  Tracker::Track track;
  EXPECT_FALSE(composite.Evaluate(track, Observe(1, 0.0)).ok());
}

class AssociationQualityPolicy final : public Policy {
 public:
  TrackStatus OnHit(const Track& track,
                    const LifecycleConfig& config) const override {
    return track.last_association_cost && *track.last_association_cost > 0.25
               ? TrackStatus::kDeleted
               : Policy::OnHit(track, config);
  }
};

TEST(GenericTrackingTest, LifecycleUsesPreUpdateScoresWithoutRecomputingCosts) {
  auto tracker =
      MakeTracker(Config(), std::make_shared<AssociationQualityPolicy>());
  auto born = tracker.Update({0.0, "local", {Observe(1, 0.0)}});
  ASSERT_TRUE(born.ok());
  EXPECT_FALSE(born->tracks[0].last_association_cost.has_value());
  auto matched = tracker.Update({0.1, "local", {Observe(2, 0.3)}});
  ASSERT_TRUE(matched.ok());
  ASSERT_TRUE(matched->tracks[0].last_association_cost.has_value());
  EXPECT_NEAR(*matched->tracks[0].last_association_cost, 0.1, 1e-12);
  ASSERT_TRUE(matched->tracks[0].history.back().association_cost.has_value());
  EXPECT_NEAR(*matched->tracks[0].history.back().association_cost, 0.1, 1e-12);
  auto missed = tracker.Update({0.2, "local", {}});
  ASSERT_TRUE(missed.ok());
  EXPECT_NEAR(*missed->tracks[0].last_association_cost, 0.1, 1e-12);
  EXPECT_FALSE(missed->tracks[0].history.back().association_cost.has_value());
  // Filtering would make the residual zero; the actual pre-update cost is 0.5.
  auto rejected = tracker.Update({0.3, "local", {Observe(3, 1.5)}});
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_TRUE(tracker.tracks().empty());
  ASSERT_EQ(rejected->matches.size(), 1);
  EXPECT_NEAR(rejected->matches[0].cost, 0.5, 1e-12);
  EXPECT_EQ(rejected->diagnostics.lifecycle_events[0].to,
            TrackStatus::kDeleted);
}

class InvalidObjectiveAssociator final : public Associator {
 public:
  absl::StatusOr<AssignmentResult> Associate(
      const AssociationProblem& problem) const override {
    auto result = HungarianAssociator().Associate(problem);
    if (!result.ok()) {
      return result.status();
    }
    result->total_cost += 1.0;
    return result;
  }
};

TEST(GenericTrackingTest,
     InvalidSolverObjectivesAndTraceComponentsAreRejected) {
  Tracker invalid(Config(), std::make_shared<Estimator>(), MakeCost(),
                  std::make_shared<InvalidObjectiveAssociator>());
  EXPECT_FALSE(invalid.Update({0.0, "local", {Observe(1, 0.0)}}).ok());
  EXPECT_TRUE(invalid.tracks().empty());
  auto bad_cost = std::make_shared<Cost>(
      [](const Tracker::Track&, const Observation<Payload>&) {
        auto evaluation = CostEvaluation::Accept(0.0);
        evaluation.components.push_back(
            {"bad-score", std::numeric_limits<double>::quiet_NaN(), 1.0});
        return evaluation;
      });
  Tracker bad_trace(Config(), std::make_shared<Estimator>(), bad_cost,
                    std::make_shared<HungarianAssociator>());
  ASSERT_TRUE(bad_trace.Update({0.0, "local", {Observe(1, 0.0)}}).ok());
  EXPECT_FALSE(bad_trace.Update({0.1, "local", {Observe(1, 0.1)}}).ok());
  EXPECT_EQ(bad_trace.tracks()[0].state.predictions, 0);
}

}  // namespace
}  // namespace tracking
}  // namespace common
}  // namespace apollo
