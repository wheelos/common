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

#include "pointcloud/filter/decimation_filter.h"

#include <utility>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace wheel {
namespace vision {
namespace lidar {
namespace {

TEST(DecimationFilterTest, DecimatesPlanarUnifiedPointCloud) {
  PointCloudBuffer source;
  ASSERT_TRUE(source.Allocate(
      6U, PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar,
      {{PointField::kX, ScalarType::kFloat32},
       {PointField::kY, ScalarType::kFloat32},
       {PointField::kZ, ScalarType::kFloat32},
       {PointField::kIntensity, ScalarType::kFloat32}}));

  for (uint32_t index = 0U; index < source.point_count(); ++index) {
    *source.mutable_field_element<float>(PointField::kX, index) =
        static_cast<float>(index);
    *source.mutable_field_element<float>(PointField::kY, index) =
        static_cast<float>(index + 10U);
    *source.mutable_field_element<float>(PointField::kZ, index) =
        static_cast<float>(index + 20U);
    *source.mutable_field_element<float>(PointField::kIntensity, index) =
        static_cast<float>(index + 30U);
  }

  DecimationFilter filter({.stride = 2U, .offset = 1U});
  PointCloudBuffer output;
  ASSERT_TRUE(filter.Filter(source.view(), &output));
  ASSERT_EQ(output.point_count(), 3U);
  ASSERT_EQ(output.layout(), PointLayout::kPlanar);

  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kX, 0U), 1.0F);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kX, 1U), 3.0F);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kX, 2U), 5.0F);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kIntensity, 2U),
                  35.0F);
}

TEST(DecimationFilterTest, DecimatesInterleavedInputToInterleavedOutput) {
  PointCloudBuffer source;
  ASSERT_TRUE(source.Allocate(5U, PointLayout::kInterleaved,
                              MemoryType::kUnified, CoordinateFrame::kVehicle,
                              {{PointField::kX, ScalarType::kFloat32},
                               {PointField::kY, ScalarType::kFloat32},
                               {PointField::kZ, ScalarType::kFloat32},
                               {PointField::kRing, ScalarType::kUint16}}));

  for (uint32_t index = 0U; index < source.point_count(); ++index) {
    *source.mutable_field_element<float>(PointField::kX, index) =
        static_cast<float>(index * 2U);
    *source.mutable_field_element<float>(PointField::kY, index) =
        static_cast<float>(index * 2U + 1U);
    *source.mutable_field_element<float>(PointField::kZ, index) =
        static_cast<float>(100U + index);
    *source.mutable_field_element<uint16_t>(PointField::kRing, index) =
        static_cast<uint16_t>(index + 7U);
  }

  DecimationFilter filter(
      {.stride = 2U, .offset = 0U, .output_layout = PointLayout::kInterleaved});
  PointCloudBuffer output;
  ASSERT_TRUE(filter.Filter(source.view(), &output));
  ASSERT_EQ(output.point_count(), 3U);
  ASSERT_EQ(output.layout(), PointLayout::kInterleaved);
  ASSERT_EQ(output.frame(), CoordinateFrame::kVehicle);

  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kY, 1U), 5.0F);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kZ, 2U), 104.0F);
  EXPECT_EQ(*output.field_element<uint16_t>(PointField::kRing, 2U), 11U);
}

TEST(DecimationFilterTest, ReportsInvalidArgumentsAndRejectsAliasing) {
  PointCloudBuffer source;
  ASSERT_TRUE(source.Allocate(3U, PointLayout::kPlanar, MemoryType::kUnified,
                              CoordinateFrame::kLidar,
                              {{PointField::kX, ScalarType::kFloat32}}));
  DecimationFilter filter({.stride = 2U});
  PointCloudBuffer output;
  EXPECT_EQ(filter.FilterWithStatus(source.view(), nullptr).code,
            StatusCode::kInvalidArgument);
  EXPECT_EQ(filter.FilterWithStatus(source.view(), &source).code,
            StatusCode::kInvalidArgument);
  PointCloudView copy = source.view();
  EXPECT_EQ(filter.FilterWithStatus(copy, &source).code,
            StatusCode::kInvalidArgument);
  EXPECT_EQ(source.point_count(), 3U);
  DecimationFilter invalid({.stride = 0U});
  EXPECT_EQ(invalid.FilterWithStatus(source.view(), &output).code,
            StatusCode::kInvalidArgument);
  copy.set_memory_type(MemoryType::kHost);
  EXPECT_EQ(filter.FilterWithStatus(copy, &output).code,
            StatusCode::kUnsupportedMemory);
}

TEST(DecimationFilterTest, HandlesOffsetsPastEndAndZeroPointSchemas) {
  PointCloudBuffer source;
  ASSERT_TRUE(source.Allocate(3U, PointLayout::kPlanar, MemoryType::kUnified,
                              CoordinateFrame::kLidar,
                              {{PointField::kX, ScalarType::kFloat32}}));
  PointCloudBuffer output;
  DecimationFilter past_end({.stride = 2U, .offset = UINT32_MAX});
  ASSERT_TRUE(past_end.Filter(source.view(), &output));
  EXPECT_EQ(output.point_count(), 0U);
  EXPECT_EQ(output.field_count(), 1U);
  PointCloudBuffer empty_output;
  ASSERT_TRUE(DecimationFilter({}).Filter(output.view(), &empty_output));
  EXPECT_EQ(empty_output.point_count(), 0U);
  EXPECT_TRUE(empty_output.view().IsValid());
}

TEST(DecimationFilterTest, AsyncMatchesReferenceAcrossBlocks) {
  constexpr uint32_t kCount = 2053U;
  PointCloudBuffer source;
  ASSERT_TRUE(source.Allocate(kCount, PointLayout::kInterleaved,
                              MemoryType::kUnified, CoordinateFrame::kVehicle,
                              {{PointField::kX, ScalarType::kFloat32},
                               {PointField::kIntensity, ScalarType::kUint8}}));
  for (uint32_t i = 0U; i < kCount; ++i) {
    *source.mutable_field_element<float>(PointField::kX, i) =
        static_cast<float>(i);
    *source.mutable_field_element<uint8_t>(PointField::kIntensity, i) =
        static_cast<uint8_t>(i % 251U);
  }
  PointCloudBuffer output;
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);
  const Status status =
      DecimationFilter({.stride = 3U, .offset = 2U})
          .FilterAsyncWithStatus(source.view(), &output, stream);
  EXPECT_TRUE(status.ok()) << status.message;
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(output.point_count(), 684U);
  for (uint32_t i = 0U; i < output.point_count(); ++i) {
    const uint32_t expected = 2U + i * 3U;
    EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kX, i),
                    static_cast<float>(expected));
    EXPECT_EQ(*output.field_element<uint8_t>(PointField::kIntensity, i),
              expected % 251U);
  }
}

TEST(DecimationFilterTest, ReusesMovedFromBufferAsOutput) {
  PointCloudBuffer output;
  ASSERT_TRUE(output.Allocate(3U, PointLayout::kInterleaved,
                              MemoryType::kUnified, CoordinateFrame::kLidar,
                              {{PointField::kX, ScalarType::kFloat32}}));
  for (uint32_t i = 0U; i < 3U; ++i) {
    *output.mutable_field_element<float>(PointField::kX, i) =
        static_cast<float>(i);
  }
  PointCloudBuffer next(std::move(output));
  const Status status =
      DecimationFilter({.stride = 2U}).FilterWithStatus(next.view(), &output);
  ASSERT_TRUE(status.ok()) << status.message;
  ASSERT_EQ(output.point_count(), 2U);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kX, 0U), 0.0f);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kX, 1U), 2.0f);
  EXPECT_EQ(next.point_count(), 3U);
}

}  // namespace
}  // namespace lidar
}  // namespace vision
}  // namespace wheel
