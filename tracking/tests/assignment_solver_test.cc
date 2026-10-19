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

#include "tracking/association/assignment_solver.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "absl/status/status.h"

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

TEST(AssignmentSolverTest, GlobalOptimumNotGreedy) {
  AssociationProblem p{
      2,
      2,
      {CostEvaluation::Accept(0.1), CostEvaluation::Accept(0.2),
       CostEvaluation::Accept(0.2), CostEvaluation::Reject("gate")},
      1.0};
  for (const auto& solver : std::vector<std::shared_ptr<AssignmentSolver>>{
           std::make_shared<HungarianSolver>(),
           std::make_shared<MinCostFlowSolver>()}) {
    const auto result = solver->Solve(p);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_DOUBLE_EQ(result->total_cost, 0.4);
    ASSERT_EQ(result->matches.size(), 2);
    EXPECT_EQ(result->matches[0].observation_index, 1);
    EXPECT_EQ(result->matches[1].observation_index, 0);
    CheckPartition(p, *result);
  }
}

TEST(AssignmentSolverTest, EmptyForbiddenAndUnmatchedPenalty) {
  for (size_t rows = 0; rows <= 3; ++rows) {
    for (size_t cols = 0; cols <= 3; ++cols) {
      AssociationProblem p;
      p.track_count = rows;
      p.observation_count = cols;
      p.edges.assign(rows * cols, CostEvaluation::Reject("gate"));
      for (const auto& solver : std::vector<std::shared_ptr<AssignmentSolver>>{
               std::make_shared<HungarianSolver>(),
               std::make_shared<MinCostFlowSolver>()}) {
        auto result = solver->Solve(p);
        ASSERT_TRUE(result.ok()) << result.status();
        EXPECT_TRUE(result->matches.empty());
        EXPECT_DOUBLE_EQ(result->total_cost, static_cast<double>(rows + cols));
        CheckPartition(p, *result);
      }
    }
  }
  AssociationProblem p{1, 1, {CostEvaluation::Accept(2.0)}, 1.0};
  EXPECT_TRUE(HungarianSolver().Solve(p)->matches.empty());
  EXPECT_TRUE(MinCostFlowSolver().Solve(p)->matches.empty());
}

TEST(AssignmentSolverTest, BothSolversMatchExhaustiveOracle) {
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
        const auto hungarian = HungarianSolver().Solve(p);
        const auto flow = MinCostFlowSolver().Solve(p);
        ASSERT_TRUE(hungarian.ok()) << hungarian.status();
        ASSERT_TRUE(flow.ok()) << flow.status();
        EXPECT_NEAR(hungarian->total_cost, expected, 1e-10);
        EXPECT_NEAR(flow->total_cost, expected, 1e-10);
        CheckPartition(p, *hungarian);
        CheckPartition(p, *flow);
        EXPECT_EQ(HungarianSolver().Solve(p)->total_cost,
                  hungarian->total_cost);
      }
    }
  }
}

TEST(AssignmentSolverTest, EmptyLargeInputsUseTheUnmatchedFastPath) {
  AssociationProblem p;
  p.track_count = 10000;
  const auto hungarian = HungarianSolver().Solve(p);
  const auto flow = MinCostFlowSolver().Solve(p);
  ASSERT_TRUE(hungarian.ok());
  ASSERT_TRUE(flow.ok());
  EXPECT_EQ(hungarian->unmatched_tracks.size(), 10000);
  EXPECT_EQ(flow->unmatched_tracks.size(), 10000);
  EXPECT_DOUBLE_EQ(hungarian->total_cost, 10000.0);
  EXPECT_DOUBLE_EQ(flow->total_cost, 10000.0);
}

TEST(AssignmentSolverTest, RejectsOverflowingObjectives) {
  AssociationProblem p;
  p.track_count = 10;
  p.unmatched_cost = 1e308 / 10.0;
  p.observation_count = 10;
  p.edges.assign(100, CostEvaluation::Reject("gate"));
  EXPECT_FALSE(HungarianSolver().Solve(p).ok());
  EXPECT_FALSE(MinCostFlowSolver().Solve(p).ok());
}

TEST(AssignmentSolverTest, MinCostFlowHonorsWorkBudget) {
  AssociationProblem p;
  p.track_count = 2;
  p.observation_count = 2;
  p.edges.assign(4, CostEvaluation::Accept(0.1));
  p.max_solver_work = 1;
  const auto result = MinCostFlowSolver().Solve(p);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kResourceExhausted);
}

TEST(AssignmentSolverTest, RejectsBadShapeAndInvalidCosts) {
  AssociationProblem p{1, 1, {}, 1.0};
  EXPECT_FALSE(HungarianSolver().Solve(p).ok());
  EXPECT_FALSE(MinCostFlowSolver().Solve(p).ok());
  for (double value : {-1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    p.edges = {CostEvaluation::Accept(value)};
    EXPECT_FALSE(HungarianSolver().Solve(p).ok());
    EXPECT_FALSE(MinCostFlowSolver().Solve(p).ok());
  }
  p.edges = {CostEvaluation::Accept(0.0)};
  p.unmatched_cost = 0.0;
  EXPECT_FALSE(HungarianSolver().Solve(p).ok());
  EXPECT_FALSE(MinCostFlowSolver().Solve(p).ok());
}

}  // namespace
}  // namespace tracking
}  // namespace common
}  // namespace apollo
