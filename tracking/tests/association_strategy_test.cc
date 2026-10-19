// Copyright 2026 WheelOS. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tracking/association/association_strategy.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"

namespace apollo {
namespace common {
namespace tracking {
namespace {

TEST(ByteTrackAssociationStrategyTest,
     MatchesHighScoresFirstAndLowScoresOnlyToUnmatchedActiveTracks) {
  AssociationProblem problem;
  problem.track_count = 3;
  problem.observation_count = 3;
  problem.unmatched_cost = 1.0;
  problem.edges = {
      CostEvaluation::Accept(0.1),         CostEvaluation::Accept(0.8),
      CostEvaluation::Accept(0.9),         CostEvaluation::Reject("high gate"),
      CostEvaluation::Accept(0.2),         CostEvaluation::Accept(0.7),
      CostEvaluation::Reject("high gate"), CostEvaluation::Accept(0.8),
      CostEvaluation::Accept(0.01),
  };
  const std::vector<double> confidences = {0.9, 0.3, 0.2};
  const std::vector<TrackStatus> statuses = {
      TrackStatus::kConfirmed, TrackStatus::kConfirmed, TrackStatus::kLost};

  ByteTrackAssociationStrategy strategy;
  const auto result =
      strategy.Associate(problem, confidences, statuses, HungarianSolver());
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->matches.size(), 2);
  EXPECT_EQ(result->matches[0].track_index, 0);
  EXPECT_EQ(result->matches[0].observation_index, 0);
  EXPECT_EQ(result->matches[1].track_index, 1);
  EXPECT_EQ(result->matches[1].observation_index, 1);
  ASSERT_EQ(result->unmatched_tracks.size(), 1);
  EXPECT_EQ(result->unmatched_tracks[0], 2);
  ASSERT_EQ(result->unmatched_observations.size(), 1);
  EXPECT_EQ(result->unmatched_observations[0], 2);
  EXPECT_DOUBLE_EQ(result->total_cost, 2.3);
}

TEST(ByteTrackAssociationStrategyTest,
     EstablishedTracksGetPriorityOverTentativeTracks) {
  AssociationProblem problem;
  problem.track_count = 2;
  problem.observation_count = 1;
  problem.unmatched_cost = 1.0;
  problem.edges = {CostEvaluation::Accept(0.2), CostEvaluation::Accept(0.1)};
  const std::vector<double> confidences = {0.9};
  const std::vector<TrackStatus> statuses = {TrackStatus::kConfirmed,
                                             TrackStatus::kTentative};

  ByteTrackAssociationStrategy strategy;
  const auto result =
      strategy.Associate(problem, confidences, statuses, HungarianSolver());
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->matches.size(), 1);
  EXPECT_EQ(result->matches[0].track_index, 0);
  EXPECT_EQ(result->matches[0].observation_index, 0);
  ASSERT_EQ(result->unmatched_tracks.size(), 1);
  EXPECT_EQ(result->unmatched_tracks[0], 1);
  EXPECT_TRUE(result->unmatched_observations.empty());
}

TEST(ByteTrackAssociationStrategyTest,
     TentativeTracksUseRemainingHighConfidenceObservations) {
  AssociationProblem problem;
  problem.track_count = 2;
  problem.observation_count = 2;
  problem.unmatched_cost = 1.0;
  problem.edges = {CostEvaluation::Accept(0.1),
                   CostEvaluation::Reject("established track gate"),
                   CostEvaluation::Accept(0.0), CostEvaluation::Accept(0.2)};
  const std::vector<double> confidences = {0.9, 0.9};
  const std::vector<TrackStatus> statuses = {TrackStatus::kConfirmed,
                                             TrackStatus::kTentative};

  ByteTrackAssociationStrategy strategy;
  const auto result =
      strategy.Associate(problem, confidences, statuses, HungarianSolver());
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->matches.size(), 2);
  EXPECT_EQ(result->matches[0].track_index, 0);
  EXPECT_EQ(result->matches[0].observation_index, 0);
  EXPECT_EQ(result->matches[1].track_index, 1);
  EXPECT_EQ(result->matches[1].observation_index, 1);
  EXPECT_TRUE(result->unmatched_tracks.empty());
  EXPECT_TRUE(result->unmatched_observations.empty());
}

TEST(ByteTrackAssociationStrategyTest, ValidatesThresholdOrdering) {
  ByteTrackAssociationStrategy strategy(ByteTrackConfig{0.2, 0.5, 0.7});
  AssociationProblem problem;
  const auto result = strategy.Associate(problem, {}, {}, HungarianSolver());
  EXPECT_FALSE(result.ok());
}

}  // namespace
}  // namespace tracking
}  // namespace common
}  // namespace apollo
