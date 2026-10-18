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

#include <limits>

#include "gtest/gtest.h"

#include "pointcloud/core/point_cloud.h"
#include "pointcloud/core/rigid_transform.h"

namespace wheel {
namespace vision {
namespace lidar {
namespace {

struct InterleavedPoint {
  float x;
  float y;
  float z;
  float intensity;
};

TEST(PointCloudViewTest, AcceptsInterleavedAndPlanarLayouts) {
  InterleavedPoint interleaved_points[2] = {
      {1.0f, 0.0f, 0.0f, 10.0f},
      {0.0f, 1.0f, 0.0f, 20.0f},
  };

  PointCloudView interleaved(
      PointLayout::kInterleaved, MemoryType::kHost, CoordinateFrame::kLidar, 2U,
      {
          {PointField::kX, ScalarType::kFloat32, &interleaved_points[0].x,
           sizeof(InterleavedPoint)},
          {PointField::kY, ScalarType::kFloat32, &interleaved_points[0].y,
           sizeof(InterleavedPoint)},
          {PointField::kZ, ScalarType::kFloat32, &interleaved_points[0].z,
           sizeof(InterleavedPoint)},
          {PointField::kIntensity, ScalarType::kFloat32,
           &interleaved_points[0].intensity, sizeof(InterleavedPoint)},
      });

  float planar_x[2] = {1.0f, 2.0f};
  float planar_y[2] = {0.0f, 0.0f};
  float planar_z[2] = {0.0f, 1.0f};
  PointCloudView planar(
      PointLayout::kPlanar, MemoryType::kHost, CoordinateFrame::kLidar, 2U,
      {
          {PointField::kX, ScalarType::kFloat32, planar_x, sizeof(float)},
          {PointField::kY, ScalarType::kFloat32, planar_y, sizeof(float)},
          {PointField::kZ, ScalarType::kFloat32, planar_z, sizeof(float)},
      });

  EXPECT_TRUE(interleaved.IsValid());
  EXPECT_TRUE(interleaved.IsRangeImageCompatible());
  EXPECT_TRUE(planar.IsValid());
  EXPECT_TRUE(planar.IsRangeImageCompatible());
  ASSERT_NE(interleaved.FindField(PointField::kIntensity), nullptr);
  EXPECT_EQ(planar.FindField(PointField::kIntensity), nullptr);
}

TEST(PointCloudViewTest, RejectsDuplicateFieldsAndInvalidStride) {
  float points[2] = {1.0f, 2.0f};

  PointCloudView duplicate(
      PointLayout::kPlanar, MemoryType::kHost, CoordinateFrame::kLidar, 2U,
      {
          {PointField::kX, ScalarType::kFloat32, points, sizeof(float)},
          {PointField::kX, ScalarType::kFloat32, points, sizeof(float)},
      });
  PointCloudView invalid_stride(
      PointLayout::kPlanar, MemoryType::kHost, CoordinateFrame::kLidar, 2U,
      {
          {PointField::kX, ScalarType::kFloat32, points, 1U},
      });

  EXPECT_EQ(duplicate.field_count(), 0U);
  EXPECT_FALSE(duplicate.IsValid());
  EXPECT_FALSE(duplicate.IsRangeImageCompatible());
  EXPECT_FALSE(invalid_stride.IsValid());
}

TEST(PointCloudViewTest, RejectsMisalignedStorageAndStride) {
  alignas(float) unsigned char storage[32]{};
  PointCloudView misaligned_pointer(
      PointLayout::kPlanar, MemoryType::kHost, CoordinateFrame::kLidar, 2U,
      {{PointField::kX, ScalarType::kFloat32, storage + 1, sizeof(float)}});
  PointCloudView misaligned_stride(
      PointLayout::kInterleaved, MemoryType::kHost, CoordinateFrame::kLidar, 2U,
      {{PointField::kX, ScalarType::kFloat32, storage, 5U}});
  EXPECT_FALSE(misaligned_pointer.IsValid());
  EXPECT_FALSE(misaligned_stride.IsValid());
}

TEST(PointCloudViewTest, RevalidatesMutatedMetadataAndRejectsUnknownFields) {
  PointCloudView view;
  view.set_layout(static_cast<PointLayout>(255));
  EXPECT_FALSE(view.IsValid());
  view.set_layout(PointLayout::kPlanar);
  view.set_memory_type(static_cast<MemoryType>(255));
  EXPECT_FALSE(view.IsValid());
  view.set_memory_type(MemoryType::kHost);
  view.set_frame(static_cast<CoordinateFrame>(255));
  EXPECT_FALSE(view.IsValid());
  view.set_frame(CoordinateFrame::kLidar);
  EXPECT_FALSE(view.AddField(
      {static_cast<PointField>(255), ScalarType::kFloat32, nullptr, 4U}));
}

TEST(RigidTransformTest, RejectsFiniteNonRigidMatricesAndInvalidTolerance) {
  RigidTransform3f transform;
  transform.rotation << 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f;
  transform.translation << 1.0f, -2.0f, 3.0f;
  EXPECT_TRUE(transform.IsValid());
  transform.rotation(0, 1) = -2.0f;
  EXPECT_TRUE(transform.IsFinite());
  EXPECT_FALSE(transform.IsValid());
  transform.rotation = Eigen::Matrix3f::Identity();
  transform.rotation(0, 0) = -1.0f;
  EXPECT_FALSE(transform.IsValid());
  transform.rotation = Eigen::Matrix3f::Identity();
  transform.rotation(0, 1) = 0.1f;
  EXPECT_FALSE(transform.IsValid());
  transform.rotation = Eigen::Matrix3f::Identity();
  EXPECT_FALSE(transform.IsValid(0.0f));
  EXPECT_FALSE(transform.IsValid(std::numeric_limits<float>::infinity()));
  transform.translation.x() = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(transform.IsValid());
}

}  // namespace
}  // namespace lidar
}  // namespace vision
}  // namespace wheel
