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

#include "pointcloud/core/point_cloud_buffer.h"

#include <utility>

#include "gtest/gtest.h"

namespace wheel {
namespace vision {
namespace lidar {
namespace {

TEST(PointCloudBufferTest, AllocatesPlanarBufferAndExposesFieldAccessors) {
  PointCloudBuffer buffer;
  ASSERT_TRUE(buffer.Allocate(
      3U, PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar,
      {
          {PointField::kX, ScalarType::kFloat32},
          {PointField::kY, ScalarType::kFloat32},
          {PointField::kZ, ScalarType::kFloat32},
          {PointField::kIntensity, ScalarType::kFloat32},
      }));

  ASSERT_NE(buffer.mutable_field_element<float>(PointField::kX, 0U), nullptr);
  *buffer.mutable_field_element<float>(PointField::kX, 0U) = 1.0f;
  *buffer.mutable_field_element<float>(PointField::kY, 0U) = 2.0f;
  *buffer.mutable_field_element<float>(PointField::kZ, 0U) = 3.0f;
  *buffer.mutable_field_element<float>(PointField::kIntensity, 0U) = 4.0f;

  EXPECT_EQ(buffer.layout(), PointLayout::kPlanar);
  EXPECT_EQ(buffer.point_count(), 3U);
  EXPECT_EQ(buffer.field_count(), 4U);
  EXPECT_EQ(buffer.point_stride_bytes(), 0U);
  EXPECT_TRUE(buffer.view().IsValid());
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kX, 0U), 1.0f);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kY, 0U), 2.0f);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kZ, 0U), 3.0f);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kIntensity, 0U),
                  4.0f);
}

TEST(PointCloudBufferTest, AllocatesInterleavedBufferWithExpectedStride) {
  PointCloudBuffer buffer;
  ASSERT_TRUE(buffer.Allocate(2U, PointLayout::kInterleaved,
                              MemoryType::kUnified, CoordinateFrame::kVehicle,
                              {
                                  {PointField::kX, ScalarType::kFloat32},
                                  {PointField::kY, ScalarType::kFloat32},
                                  {PointField::kZ, ScalarType::kFloat32},
                              }));

  ASSERT_NE(buffer.mutable_field_element<float>(PointField::kX, 1U), nullptr);
  *buffer.mutable_field_element<float>(PointField::kX, 1U) = 1.5f;
  *buffer.mutable_field_element<float>(PointField::kY, 1U) = -0.5f;
  *buffer.mutable_field_element<float>(PointField::kZ, 1U) = 0.25f;

  const PointFieldDescriptor* x = buffer.FindField(PointField::kX);
  const PointFieldDescriptor* y = buffer.FindField(PointField::kY);
  const PointFieldDescriptor* z = buffer.FindField(PointField::kZ);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);
  ASSERT_NE(z, nullptr);
  EXPECT_EQ(x->stride_bytes, sizeof(float) * 3U);
  EXPECT_EQ(y->stride_bytes, sizeof(float) * 3U);
  EXPECT_EQ(z->stride_bytes, sizeof(float) * 3U);
  EXPECT_EQ(buffer.point_stride_bytes(), sizeof(float) * 3U);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kX, 1U), 1.5f);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kY, 1U), -0.5f);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kZ, 1U), 0.25f);
}

TEST(PointCloudBufferTest, RejectsInvalidFieldSpecifications) {
  PointCloudBuffer duplicate_fields;
  PointCloudBuffer unknown_scalar_type;

  EXPECT_FALSE(duplicate_fields.Allocate(
      1U, PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar,
      {
          {PointField::kX, ScalarType::kFloat32},
          {PointField::kX, ScalarType::kFloat32},
      }));
  EXPECT_FALSE(unknown_scalar_type.Allocate(
      1U, PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar,
      {
          {PointField::kX, ScalarType::kUnknown},
      }));
}

TEST(PointCloudBufferTest, SupportsZeroPointSchemasWithoutBackingStorage) {
  PointCloudBuffer planar_buffer;
  PointCloudBuffer interleaved_buffer;

  ASSERT_TRUE(planar_buffer.Allocate(
      0U, PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar,
      {
          {PointField::kX, ScalarType::kFloat32},
          {PointField::kY, ScalarType::kFloat32},
          {PointField::kZ, ScalarType::kFloat32},
      }));
  ASSERT_TRUE(interleaved_buffer.Allocate(
      0U, PointLayout::kInterleaved, MemoryType::kUnified,
      CoordinateFrame::kVehicle,
      {
          {PointField::kX, ScalarType::kFloat32},
          {PointField::kIntensity, ScalarType::kUint8},
      }));

  EXPECT_TRUE(planar_buffer.view().IsValid());
  EXPECT_TRUE(interleaved_buffer.view().IsValid());
  EXPECT_EQ(planar_buffer.field_data<float>(PointField::kX), nullptr);
  EXPECT_EQ(interleaved_buffer.field_data<float>(PointField::kX), nullptr);
}

TEST(PointCloudBufferTest, SupportsMixedScalarTypesInInterleavedLayout) {
  PointCloudBuffer buffer;
  ASSERT_TRUE(buffer.Allocate(2U, PointLayout::kInterleaved,
                              MemoryType::kUnified, CoordinateFrame::kVehicle,
                              {
                                  {PointField::kX, ScalarType::kFloat32},
                                  {PointField::kIntensity, ScalarType::kUint8},
                                  {PointField::kRing, ScalarType::kUint16},
                                  {PointField::kY, ScalarType::kFloat32},
                              }));

  *buffer.mutable_field_element<float>(PointField::kX, 0U) = 2.0f;
  *buffer.mutable_field_element<uint8_t>(PointField::kIntensity, 0U) = 7U;
  *buffer.mutable_field_element<uint16_t>(PointField::kRing, 0U) = 12U;
  *buffer.mutable_field_element<float>(PointField::kY, 0U) = -1.0f;

  const PointFieldDescriptor* x = buffer.FindField(PointField::kX);
  const PointFieldDescriptor* intensity =
      buffer.FindField(PointField::kIntensity);
  const PointFieldDescriptor* ring = buffer.FindField(PointField::kRing);
  const PointFieldDescriptor* y = buffer.FindField(PointField::kY);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(intensity, nullptr);
  ASSERT_NE(ring, nullptr);
  ASSERT_NE(y, nullptr);
  EXPECT_EQ(buffer.point_stride_bytes(), 12U);
  EXPECT_EQ(x->stride_bytes, 12U);
  EXPECT_EQ(intensity->stride_bytes, 12U);
  EXPECT_EQ(ring->stride_bytes, 12U);
  EXPECT_EQ(y->stride_bytes, 12U);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kX, 0U), 2.0f);
  EXPECT_EQ(*buffer.field_element<uint8_t>(PointField::kIntensity, 0U), 7U);
  EXPECT_EQ(*buffer.field_element<uint16_t>(PointField::kRing, 0U), 12U);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kY, 0U), -1.0f);
}

TEST(PointCloudBufferTest, PadsTrailingByteToAlignEveryPoint) {
  PointCloudBuffer buffer;
  ASSERT_TRUE(buffer.Allocate(3U, PointLayout::kInterleaved,
                              MemoryType::kUnified, CoordinateFrame::kLidar,
                              {{PointField::kX, ScalarType::kFloat32},
                               {PointField::kY, ScalarType::kFloat32},
                               {PointField::kZ, ScalarType::kFloat32},
                               {PointField::kIntensity, ScalarType::kUint8}}));
  EXPECT_EQ(buffer.point_stride_bytes(), 16U);
  for (uint32_t i = 0U; i < 3U; ++i) {
    float* x = buffer.mutable_field_element<float>(PointField::kX, i);
    ASSERT_NE(x, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(x) % alignof(float), 0U);
    *x = static_cast<float>(i);
    *buffer.mutable_field_element<uint8_t>(PointField::kIntensity, i) =
        static_cast<uint8_t>(10U + i);
  }
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kX, 2U), 2.0f);
  EXPECT_EQ(*buffer.field_element<uint8_t>(PointField::kIntensity, 2U), 12U);
  EXPECT_TRUE(buffer.view().IsValid());
}

TEST(PointCloudBufferTest, InvalidAllocationResetsPreviousStorage) {
  PointCloudBuffer buffer;
  ASSERT_TRUE(buffer.Allocate(1U, PointLayout::kPlanar, MemoryType::kHost,
                              CoordinateFrame::kLidar,
                              {{PointField::kX, ScalarType::kFloat32}}));
  EXPECT_FALSE(buffer.Allocate(1U, static_cast<PointLayout>(255),
                               MemoryType::kHost, CoordinateFrame::kLidar,
                               {{PointField::kX, ScalarType::kFloat32}}));
  EXPECT_EQ(buffer.point_count(), 0U);
  EXPECT_EQ(buffer.field_count(), 0U);
}

TEST(PointCloudBufferTest, MovesStorageAndResetsSourceForBothLayouts) {
  for (PointLayout layout : {PointLayout::kPlanar, PointLayout::kInterleaved}) {
    PointCloudBuffer source;
    ASSERT_TRUE(source.Allocate(
        2U, layout, MemoryType::kHost, CoordinateFrame::kVehicle,
        {{PointField::kX, ScalarType::kFloat32},
         {PointField::kIntensity, ScalarType::kUint8}}));
    float* original = source.mutable_field_element<float>(PointField::kX, 1U);
    *original = 7.5f;
    const uint32_t stride = source.point_stride_bytes();

    PointCloudBuffer moved(std::move(source));
    EXPECT_EQ(source.point_count(), 0U);
    EXPECT_EQ(source.field_count(), 0U);
    EXPECT_EQ(source.point_stride_bytes(), 0U);
    EXPECT_TRUE(source.view().IsValid());
    EXPECT_EQ(source.FindField(PointField::kX), nullptr);
    EXPECT_EQ(moved.field_element<float>(PointField::kX, 1U), original);
    EXPECT_EQ(moved.point_stride_bytes(), stride);
    EXPECT_EQ(moved.layout(), layout);
    EXPECT_EQ(moved.frame(), CoordinateFrame::kVehicle);

    ASSERT_TRUE(source.Allocate(1U, PointLayout::kPlanar, MemoryType::kHost,
                                CoordinateFrame::kLidar,
                                {{PointField::kY, ScalarType::kFloat32}}));
    source = std::move(moved);
    EXPECT_EQ(moved.point_count(), 0U);
    EXPECT_EQ(moved.field_count(), 0U);
    EXPECT_EQ(moved.point_stride_bytes(), 0U);
    EXPECT_TRUE(moved.view().IsValid());
    EXPECT_EQ(source.field_element<float>(PointField::kX, 1U), original);
    EXPECT_FLOAT_EQ(*original, 7.5f);

    PointCloudBuffer& same = source;
    source = std::move(same);
    EXPECT_EQ(source.point_count(), 2U);
    EXPECT_EQ(source.field_element<float>(PointField::kX, 1U), original);
    EXPECT_FLOAT_EQ(*original, 7.5f);
  }
}

}  // namespace
}  // namespace lidar
}  // namespace vision
}  // namespace wheel
