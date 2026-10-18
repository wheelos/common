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

#include "pointcloud/filter/crop_box_filter.h"

#include <cmath>
#include <limits>
#include <set>
#include <utility>

#include <gtest/gtest.h>

namespace wheel {
namespace vision {
namespace lidar {
namespace {

TEST(CropBoxFilterTest, RemovesPointsInsideBoxAndRejectsNonFinite) {
  PointCloudBuffer source;
  ASSERT_TRUE(
      source.Allocate(5U, PointLayout::kInterleaved, MemoryType::kUnified,
                      CoordinateFrame::kLidar,
                      {{PointField::kX, ScalarType::kFloat32},
                       {PointField::kY, ScalarType::kFloat32},
                       {PointField::kZ, ScalarType::kFloat32},
                       {PointField::kIntensity, ScalarType::kFloat32}}));

  const float xs[5] = {0.0F, 2.0F, 3.0F, NAN, -2.0F};
  const float ys[5] = {0.0F, 0.0F, 0.0F, 0.0F, 1.5F};
  const float zs[5] = {0.0F, 0.5F, 3.0F, 0.0F, 0.2F};
  for (uint32_t index = 0U; index < source.point_count(); ++index) {
    *source.mutable_field_element<float>(PointField::kX, index) = xs[index];
    *source.mutable_field_element<float>(PointField::kY, index) = ys[index];
    *source.mutable_field_element<float>(PointField::kZ, index) = zs[index];
    *source.mutable_field_element<float>(PointField::kIntensity, index) =
        10.0F + static_cast<float>(index);
  }

  CropBoxFilterWorkspace workspace;
  ASSERT_TRUE(workspace.Allocate(source.point_count()));

  CropBoxFilter filter({.min_x = -1.0F,
                        .max_x = 1.0F,
                        .min_y = -1.0F,
                        .max_y = 1.0F,
                        .min_z = -1.0F,
                        .max_z = 1.0F,
                        .reject_non_finite = true,
                        .mode = CropBoxMode::kRemoveInside});
  PointCloudBuffer output;
  ASSERT_TRUE(filter.Filter(source.view(), &output, &workspace));

  ASSERT_EQ(output.point_count(), 3U);
  std::set<float> selected;
  for (uint32_t i = 0U; i < output.point_count(); ++i) {
    const float x = *output.field_element<float>(PointField::kX, i);
    selected.insert(x);
    const float intensity =
        *output.field_element<float>(PointField::kIntensity, i);
    EXPECT_FLOAT_EQ(intensity, x == 2.0f ? 11.0f : (x == 3.0f ? 12.0f : 14.0f));
  }
  EXPECT_EQ(selected, (std::set<float>{-2.0f, 2.0f, 3.0f}));
}

TEST(CropBoxFilterTest, KeepsPointsInsideConfiguredHeightWindow) {
  PointCloudBuffer source;
  ASSERT_TRUE(source.Allocate(4U, PointLayout::kPlanar, MemoryType::kUnified,
                              CoordinateFrame::kLidar,
                              {{PointField::kX, ScalarType::kFloat32},
                               {PointField::kY, ScalarType::kFloat32},
                               {PointField::kZ, ScalarType::kFloat32},
                               {PointField::kRing, ScalarType::kUint16}}));

  const float zs[4] = {-0.5F, 0.4F, 1.2F, -1.5F};
  for (uint32_t index = 0U; index < source.point_count(); ++index) {
    *source.mutable_field_element<float>(PointField::kX, index) =
        static_cast<float>(index);
    *source.mutable_field_element<float>(PointField::kY, index) = 0.0F;
    *source.mutable_field_element<float>(PointField::kZ, index) = zs[index];
    *source.mutable_field_element<uint16_t>(PointField::kRing, index) =
        static_cast<uint16_t>(index + 1U);
  }

  CropBoxFilterWorkspace workspace;
  ASSERT_TRUE(workspace.Allocate(source.point_count(), MemoryType::kDevice));

  CropBoxFilter filter(
      {.min_z = -1.0F, .max_z = 1.0F, .mode = CropBoxMode::kKeepInside});
  PointCloudBuffer output;
  ASSERT_TRUE(filter.Filter(source.view(), &output, &workspace));

  ASSERT_EQ(output.point_count(), 2U);
  std::set<uint16_t> rings;
  for (uint32_t i = 0U; i < output.point_count(); ++i) {
    const uint16_t ring = *output.field_element<uint16_t>(PointField::kRing, i);
    ASSERT_GE(ring, 1U);
    ASSERT_LE(ring, 2U);
    rings.insert(ring);
    EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kZ, i),
                    zs[ring - 1U]);
  }
  EXPECT_EQ(rings, (std::set<uint16_t>{1U, 2U}));
}

TEST(CropBoxFilterTest, HandlesInclusiveBoundsEmptySelectionAndInvalidCalls) {
  PointCloudBuffer source;
  ASSERT_TRUE(source.Allocate(3U, PointLayout::kPlanar, MemoryType::kUnified,
                              CoordinateFrame::kLidar,
                              {{PointField::kX, ScalarType::kFloat32},
                               {PointField::kY, ScalarType::kFloat32},
                               {PointField::kZ, ScalarType::kFloat32}}));
  for (uint32_t i = 0U; i < source.point_count(); ++i) {
    *source.mutable_field_element<float>(PointField::kX, i) =
        static_cast<float>(i) - 1.0f;
    *source.mutable_field_element<float>(PointField::kY, i) = 0.0f;
    *source.mutable_field_element<float>(PointField::kZ, i) = 0.0f;
  }
  CropBoxFilter filter({.min_x = -1.0f, .max_x = 1.0f});
  CropBoxFilterWorkspace workspace;
  PointCloudBuffer output;
  EXPECT_EQ(filter.FilterWithStatus(source.view(), &output, &workspace).code,
            StatusCode::kInsufficientCapacity);
  ASSERT_TRUE(workspace.Allocate(3U));
  PointCloudView copy = source.view();
  EXPECT_EQ(filter.FilterWithStatus(copy, &source, &workspace).code,
            StatusCode::kInvalidArgument);
  EXPECT_EQ(source.point_count(), 3U);
  EXPECT_EQ(filter.FilterWithStatus(source.view(), nullptr, &workspace).code,
            StatusCode::kInvalidArgument);
  ASSERT_TRUE(filter.FilterWithStatus(source.view(), &output, &workspace).ok());
  EXPECT_EQ(output.point_count(), 3U);
  CropBoxFilter reject_all({.min_x = 5.0f, .max_x = 6.0f});
  ASSERT_TRUE(reject_all.Filter(source.view(), &output, &workspace));
  EXPECT_EQ(output.point_count(), 0U);
  EXPECT_EQ(output.field_count(), 3U);
  CropBoxFilter invalid({.mode = static_cast<CropBoxMode>(255)});
  EXPECT_EQ(invalid.FilterWithStatus(source.view(), &output, &workspace).code,
            StatusCode::kInvalidArgument);
}

TEST(CropBoxFilterTest, MatchesCpuSelectionAcrossBlocksOnNonDefaultStream) {
  constexpr uint32_t kCount = 1031U;
  PointCloudBuffer source;
  ASSERT_TRUE(source.Allocate(kCount, PointLayout::kInterleaved,
                              MemoryType::kUnified, CoordinateFrame::kLidar,
                              {{PointField::kX, ScalarType::kFloat32},
                               {PointField::kY, ScalarType::kFloat32},
                               {PointField::kZ, ScalarType::kFloat32},
                               {PointField::kRing, ScalarType::kUint32}}));
  for (uint32_t i = 0U; i < kCount; ++i) {
    *source.mutable_field_element<float>(PointField::kX, i) =
        i % 29U == 0U ? std::numeric_limits<float>::quiet_NaN()
                      : static_cast<float>(i % 21U) - 10.0f;
    *source.mutable_field_element<float>(PointField::kY, i) =
        static_cast<float>(i % 13U) - 6.0f;
    *source.mutable_field_element<float>(PointField::kZ, i) =
        static_cast<float>(i % 7U) - 3.0f;
    *source.mutable_field_element<uint32_t>(PointField::kRing, i) = i;
  }
  CropBoxFilterWorkspace workspace;
  ASSERT_TRUE(workspace.Allocate(kCount));
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);
  for (CropBoxMode mode :
       {CropBoxMode::kKeepInside, CropBoxMode::kRemoveInside}) {
    for (bool reject_non_finite : {false, true}) {
      CropBoxFilterConfig config;
      config.min_x = -2.0f;
      config.max_x = 3.0f;
      config.min_y = -1.0f;
      config.max_y = 2.0f;
      config.min_z = -1.0f;
      config.max_z = 1.0f;
      config.mode = mode;
      config.reject_non_finite = reject_non_finite;
      std::set<uint32_t> expected;
      for (uint32_t i = 0U; i < kCount; ++i) {
        const float x = *source.field_element<float>(PointField::kX, i);
        const float y = *source.field_element<float>(PointField::kY, i);
        const float z = *source.field_element<float>(PointField::kZ, i);
        if (reject_non_finite && !std::isfinite(x)) {
          continue;
        }
        const bool inside = x >= -2.0f && x <= 3.0f && y >= -1.0f &&
                            y <= 2.0f && z >= -1.0f && z <= 1.0f;
        if (inside == (mode == CropBoxMode::kKeepInside)) {
          expected.insert(i);
        }
      }
      PointCloudBuffer output;
      const Status status = CropBoxFilter(config).FilterWithStatus(
          source.view(), &output, &workspace, stream);
      EXPECT_TRUE(status.ok()) << status.message;
      std::set<uint32_t> actual;
      for (uint32_t i = 0U; i < output.point_count(); ++i) {
        const uint32_t id =
            *output.field_element<uint32_t>(PointField::kRing, i);
        actual.insert(id);
        if (id < kCount) {
          EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kZ, i),
                          *source.field_element<float>(PointField::kZ, id));
        }
      }
      EXPECT_EQ(output.point_count(), expected.size());
      EXPECT_EQ(actual, expected);
    }
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(CropBoxFilterTest, ReusesMoveAssignedBufferAsOutput) {
  PointCloudBuffer output;
  ASSERT_TRUE(output.Allocate(3U, PointLayout::kPlanar, MemoryType::kUnified,
                              CoordinateFrame::kLidar,
                              {{PointField::kX, ScalarType::kFloat32},
                               {PointField::kY, ScalarType::kFloat32},
                               {PointField::kZ, ScalarType::kFloat32}}));
  for (uint32_t i = 0U; i < 3U; ++i) {
    *output.mutable_field_element<float>(PointField::kX, i) =
        static_cast<float>(i);
    *output.mutable_field_element<float>(PointField::kY, i) = 0.0f;
    *output.mutable_field_element<float>(PointField::kZ, i) = 0.0f;
  }
  PointCloudBuffer next;
  next = std::move(output);
  CropBoxFilterWorkspace workspace;
  ASSERT_TRUE(workspace.Allocate(3U));
  const Status status = CropBoxFilter({.min_x = 1.0f, .max_x = 1.0f})
                            .FilterWithStatus(next.view(), &output, &workspace);
  ASSERT_TRUE(status.ok()) << status.message;
  ASSERT_EQ(output.point_count(), 1U);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kX, 0U), 1.0f);
  EXPECT_EQ(next.point_count(), 3U);
}

}  // namespace
}  // namespace lidar
}  // namespace vision
}  // namespace wheel
