#include "modules/common/tracking/associator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace apollo {
namespace common {
namespace tracking {
namespace {

double ExhaustiveCost(const AssociationProblem& problem, size_t row,
                      std::vector<bool>* used) {
  if (row == problem.track_count) {
    return static_cast<double>(std::count(used->begin(), used->end(), false)) *
           problem.unmatched_cost;
  }
  double best = problem.unmatched_cost + ExhaustiveCost(problem, row + 1, used);
  for (size_t col = 0; col < problem.observation_count; ++col) {
    const auto& edge = problem.edges[row * problem.observation_count + col];
    if ((*used)[col] || !edge.allowed) {
      continue;
    }
    (*used)[col] = true;
    best = std::min(best, edge.value + ExhaustiveCost(problem, row + 1, used));
    (*used)[col] = false;
  }
  return best;
}

void CheckPartition(const AssociationProblem& problem,
                    const AssignmentResult& result) {
  std::vector<int> rows(problem.track_count, 0);
  std::vector<int> cols(problem.observation_count, 0);
  for (const auto& match : result.matches) {
    ASSERT_LT(match.track_index, rows.size());
    ASSERT_LT(match.observation_index, cols.size());
    const auto& edge =
        problem
            .edges[match.track_index * cols.size() + match.observation_index];
    EXPECT_TRUE(edge.allowed);
    EXPECT_EQ(edge.value, match.cost);
    ++rows[match.track_index];
    ++cols[match.observation_index];
  }
  for (size_t row : result.unmatched_tracks) {
    ASSERT_LT(row, rows.size());
    ++rows[row];
  }
  for (size_t col : result.unmatched_observations) {
    ASSERT_LT(col, cols.size());
    ++cols[col];
  }
  for (int count : rows) {
    EXPECT_EQ(count, 1);
  }
  for (int count : cols) {
    EXPECT_EQ(count, 1);
  }
}

TEST(AssociatorTest, GlobalOptimumNotGreedy) {
  AssociationProblem p{
      2,
      2,
      {CostEvaluation::Accept(0.1), CostEvaluation::Accept(0.2),
       CostEvaluation::Accept(0.2), CostEvaluation::Reject("gate")},
      1.0};
  for (const auto& solver : std::vector<std::shared_ptr<Associator>>{
           std::make_shared<HungarianAssociator>(),
           std::make_shared<MinCostFlowAssociator>()}) {
    const auto result = solver->Associate(p);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_DOUBLE_EQ(result->total_cost, 0.4);
    ASSERT_EQ(result->matches.size(), 2);
    EXPECT_EQ(result->matches[0].observation_index, 1);
    EXPECT_EQ(result->matches[1].observation_index, 0);
    CheckPartition(p, *result);
  }
}

TEST(AssociatorTest, EmptyForbiddenAndUnmatchedPenalty) {
  for (size_t rows = 0; rows <= 3; ++rows) {
    for (size_t cols = 0; cols <= 3; ++cols) {
      AssociationProblem p;
      p.track_count = rows;
      p.observation_count = cols;
      p.edges.assign(rows * cols, CostEvaluation::Reject("gate"));
      for (const auto& solver : std::vector<std::shared_ptr<Associator>>{
               std::make_shared<HungarianAssociator>(),
               std::make_shared<MinCostFlowAssociator>()}) {
        auto result = solver->Associate(p);
        ASSERT_TRUE(result.ok()) << result.status();
        EXPECT_TRUE(result->matches.empty());
        EXPECT_DOUBLE_EQ(result->total_cost, static_cast<double>(rows + cols));
        CheckPartition(p, *result);
      }
    }
  }
  AssociationProblem p{1, 1, {CostEvaluation::Accept(2.0)}, 1.0};
  EXPECT_TRUE(HungarianAssociator().Associate(p)->matches.empty());
  EXPECT_TRUE(MinCostFlowAssociator().Associate(p)->matches.empty());
}

TEST(AssociatorTest, BothSolversMatchExhaustiveOracle) {
  std::mt19937 random(42);
  for (size_t rows = 0; rows <= 4; ++rows) {
    for (size_t cols = 0; cols <= 4; ++cols) {
      for (int sample = 0; sample < 30; ++sample) {
        AssociationProblem p;
        p.track_count = rows;
        p.observation_count = cols;
        for (size_t index = 0; index < rows * cols; ++index) {
          p.edges.push_back(
              random() % 4 == 0
                  ? CostEvaluation::Reject("random gate")
                  : CostEvaluation::Accept(static_cast<double>(random() % 31) /
                                           10.0));
        }
        std::vector<bool> used(cols, false);
        const double expected = ExhaustiveCost(p, 0, &used);
        const auto hungarian = HungarianAssociator().Associate(p);
        const auto flow = MinCostFlowAssociator().Associate(p);
        ASSERT_TRUE(hungarian.ok()) << hungarian.status();
        ASSERT_TRUE(flow.ok()) << flow.status();
        EXPECT_NEAR(hungarian->total_cost, expected, 1e-10);
        EXPECT_NEAR(flow->total_cost, expected, 1e-10);
        CheckPartition(p, *hungarian);
        CheckPartition(p, *flow);
        EXPECT_EQ(HungarianAssociator().Associate(p)->total_cost,
                  hungarian->total_cost);
      }
    }
  }
}

TEST(AssociatorTest, EmptyLargeInputsUseTheUnmatchedFastPath) {
  AssociationProblem p;
  p.track_count = 10000;
  const auto hungarian = HungarianAssociator().Associate(p);
  const auto flow = MinCostFlowAssociator().Associate(p);
  ASSERT_TRUE(hungarian.ok());
  ASSERT_TRUE(flow.ok());
  EXPECT_EQ(hungarian->unmatched_tracks.size(), 10000);
  EXPECT_EQ(flow->unmatched_tracks.size(), 10000);
  EXPECT_DOUBLE_EQ(hungarian->total_cost, 10000.0);
  EXPECT_DOUBLE_EQ(flow->total_cost, 10000.0);
}

TEST(AssociatorTest, RejectsOverflowingObjectives) {
  AssociationProblem p;
  p.track_count = 10;
  p.unmatched_cost = 1e308 / 10.0;
  p.observation_count = 10;
  p.edges.assign(100, CostEvaluation::Reject("gate"));
  EXPECT_FALSE(HungarianAssociator().Associate(p).ok());
  EXPECT_FALSE(MinCostFlowAssociator().Associate(p).ok());
}

TEST(AssociatorTest, RejectsBadShapeAndInvalidCosts) {
  AssociationProblem p{1, 1, {}, 1.0};
  EXPECT_FALSE(HungarianAssociator().Associate(p).ok());
  EXPECT_FALSE(MinCostFlowAssociator().Associate(p).ok());
  for (double value : {-1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    p.edges = {CostEvaluation::Accept(value)};
    EXPECT_FALSE(HungarianAssociator().Associate(p).ok());
    EXPECT_FALSE(MinCostFlowAssociator().Associate(p).ok());
  }
  p.edges = {CostEvaluation::Accept(0.0)};
  p.unmatched_cost = 0.0;
  EXPECT_FALSE(HungarianAssociator().Associate(p).ok());
  EXPECT_FALSE(MinCostFlowAssociator().Associate(p).ok());
}

}  // namespace
}  // namespace tracking
}  // namespace common
}  // namespace apollo
