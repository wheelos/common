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

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "gtest/gtest.h"

#include "pointcloud/core/point_cloud_buffer.h"
#include "pointcloud/range_image/range_image_builder.h"

namespace wheel {
namespace vision {
namespace lidar {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kQuarterPi = kPi / 4.0f;
constexpr float kTolerance = 1e-5f;

uint32_t PixelIndex(uint32_t row, uint32_t col, uint32_t cols) {
  return row * cols + col;
}

RangeImageConfig MakeTestConfig() {
  RangeImageConfig config;
  config.rows = 4U;
  config.cols = 8U;
  config.min_range_m = 0.1f;
  config.max_range_m = 10.0f;
  config.min_azimuth_rad = -kPi;
  config.max_azimuth_rad = kPi;
  config.min_elevation_rad = -kQuarterPi;
  config.max_elevation_rad = kQuarterPi;
  return config;
}

struct TestPoint {
  float x;
  float y;
  float z;
  float intensity;
};

PointCloudView MakePointView(const Buffer& buffer, uint32_t count) {
  const TestPoint* points = buffer.data<TestPoint>();
  return PointCloudView(
      PointLayout::kInterleaved, MemoryType::kUnified, CoordinateFrame::kLidar,
      count,
      {{PointField::kX, ScalarType::kFloat32, &points->x, sizeof(TestPoint)},
       {PointField::kY, ScalarType::kFloat32, &points->y, sizeof(TestPoint)},
       {PointField::kZ, ScalarType::kFloat32, &points->z, sizeof(TestPoint)},
       {PointField::kIntensity, ScalarType::kFloat32, &points->intensity,
        sizeof(TestPoint)}});
}

struct TestStream {
  cudaStream_t stream = nullptr;
  ~TestStream() {
    if (stream != nullptr) {
      EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
  }
};

TEST(RangeImageTest, RejectsOverflowAndNonFiniteOrUnsupportedAngularConfig) {
  RangeImageConfig config = MakeTestConfig();
  config.rows = 65536U;
  config.cols = 65536U;
  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  EXPECT_FALSE(config.IsValid());
  EXPECT_EQ(config.pixel_count(), 0U);
  EXPECT_FALSE(output.Allocate(config));
  EXPECT_FALSE(workspace.Allocate(config));

  using ConfigMember = float RangeImageConfig::*;
  const ConfigMember fields[] = {&RangeImageConfig::min_range_m,
                                 &RangeImageConfig::max_range_m,
                                 &RangeImageConfig::min_azimuth_rad,
                                 &RangeImageConfig::max_azimuth_rad,
                                 &RangeImageConfig::min_elevation_rad,
                                 &RangeImageConfig::max_elevation_rad};
  for (auto field : fields) {
    for (float value : {std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
      RangeImageConfig invalid = MakeTestConfig();
      invalid.*field = value;
      EXPECT_FALSE(invalid.IsValid());
    }
  }
  config = MakeTestConfig();
  config.min_azimuth_rad = -2.0f * kPi;
  EXPECT_FALSE(config.IsValid());
  config = MakeTestConfig();
  config.max_elevation_rad = kPi;
  EXPECT_FALSE(config.IsValid());
}

TEST(RangeImageTest, FailedReallocationPreservesOutputAndWorkspace) {
  const RangeImageConfig config = MakeTestConfig();
  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  ASSERT_TRUE(output.Allocate(config));
  ASSERT_TRUE(workspace.Allocate(config));
  float* range = output.range();
  uint64_t* packed = workspace.packed_range_intensity();
  EXPECT_FALSE(output.Allocate(config, static_cast<MemoryType>(255)));
  EXPECT_FALSE(workspace.Allocate(config, static_cast<MemoryType>(255)));
  EXPECT_EQ(output.range(), range);
  EXPECT_EQ(output.rows(), config.rows);
  EXPECT_EQ(workspace.packed_range_intensity(), packed);
  EXPECT_EQ(workspace.pixel_count(), config.pixel_count());
}

TEST(RangeImageTest, ReportsPreconditionsAndRejectsNonRigidTransform) {
  const RangeImageConfig config = MakeTestConfig();
  RangeImageBuilder builder(config);
  PointCloudView empty;
  empty.set_memory_type(MemoryType::kUnified);
  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  EXPECT_EQ(
      RangeImageBuilder({}).BuildWithStatus(empty, &output, &workspace).code,
      StatusCode::kInvalidArgument);
  EXPECT_EQ(builder.BuildWithStatus(empty, nullptr, &workspace).code,
            StatusCode::kInvalidArgument);
  EXPECT_EQ(builder.BuildWithStatus(empty, &output, &workspace).code,
            StatusCode::kInsufficientCapacity);
  ASSERT_TRUE(output.Allocate(config));
  ASSERT_TRUE(workspace.Allocate(config));
  empty.set_memory_type(MemoryType::kHost);
  EXPECT_EQ(builder.BuildWithStatus(empty, &output, &workspace).code,
            StatusCode::kUnsupportedMemory);
  empty.set_memory_type(MemoryType::kUnified);
  empty.set_frame(CoordinateFrame::kVehicle);
  EXPECT_EQ(builder.BuildWithStatus(empty, &output, &workspace).code,
            StatusCode::kInvalidArgument);
  RigidTransform3f transform;
  transform.rotation *= 2.0f;
  EXPECT_EQ(
      builder.BuildWithStatus(empty, &output, &workspace, &transform).code,
      StatusCode::kInvalidArgument);
  EXPECT_FALSE(builder.Build(empty, &output, &workspace, &transform));
}

TEST(RangeImageTest, DefinesAzimuthSeamAndRejectsZeroRange) {
  RangeImageConfig config = MakeTestConfig();
  config.min_range_m = 0.0f;
  Buffer points;
  ASSERT_TRUE(points.Allocate(3U * sizeof(TestPoint), MemoryType::kUnified));
  points.mutable_data<TestPoint>()[0] = {-1.0f, 0.0f, 0.0f, 3.0f};
  points.mutable_data<TestPoint>()[1] = {-1.0f, -0.0f, 0.0f, 4.0f};
  points.mutable_data<TestPoint>()[2] = {0.0f, 0.0f, 0.0f, 5.0f};
  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  ASSERT_TRUE(output.Allocate(config));
  ASSERT_TRUE(workspace.Allocate(config));
  ASSERT_TRUE(RangeImageBuilder(config).Build(MakePointView(points, 3U),
                                              &output, &workspace));
  EXPECT_FLOAT_EQ(output.intensity()[PixelIndex(2U, 7U, config.cols)], 3.0f);
  EXPECT_FLOAT_EQ(output.intensity()[PixelIndex(2U, 0U, config.cols)], 4.0f);
  EXPECT_EQ(output.valid_mask()[PixelIndex(2U, 4U, config.cols)], 0U);
}

TEST(RangeImageTest, ClipsAngularAndRangeBoundsAndSanitizesIntensity) {
  RangeImageConfig config = MakeTestConfig();
  config.min_range_m = 1.0f;
  config.max_range_m = 2.0f;
  config.min_azimuth_rad = 0.0f;
  config.max_azimuth_rad = kPi / 2.0f;
  Buffer points;
  ASSERT_TRUE(points.Allocate(6U * sizeof(TestPoint), MemoryType::kUnified));
  TestPoint* data = points.mutable_data<TestPoint>();
  data[0] = {1.0f, 0.0f, 0.0f, std::numeric_limits<float>::quiet_NaN()};
  data[1] = {0.0f, 2.0f, 0.0f, 6.0f};
  data[2] = {0.5f, 0.0f, 0.0f, 88.0f};
  data[3] = {3.0f / std::sqrt(2.0f), 3.0f / std::sqrt(2.0f), 0.0f, 99.0f};
  data[4] = {1.0f, -1.0f, 0.0f, 99.0f};
  data[5] = {0.25f, 0.25f, 1.1f, 99.0f};
  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  ASSERT_TRUE(output.Allocate(config));
  ASSERT_TRUE(workspace.Allocate(config));
  ASSERT_TRUE(RangeImageBuilder(config).Build(MakePointView(points, 6U),
                                              &output, &workspace));
  uint32_t valid_count = 0U;
  for (uint32_t i = 0U; i < config.pixel_count(); ++i) {
    valid_count += output.valid_mask()[i];
  }
  EXPECT_EQ(valid_count, 2U);
  const uint32_t first = PixelIndex(2U, 0U, config.cols);
  const uint32_t last = PixelIndex(2U, config.cols - 1U, config.cols);
  EXPECT_FLOAT_EQ(output.range()[first], 1.0f);
  EXPECT_FLOAT_EQ(output.intensity()[first], 0.0f);
  EXPECT_FLOAT_EQ(output.range()[last], 2.0f);
  EXPECT_FLOAT_EQ(output.intensity()[last], 6.0f);
}

TEST(RangeImageTest, MatchesCpuReferenceAcrossBlocksOnNonDefaultStream) {
  RangeImageConfig config = MakeTestConfig();
  config.rows = 12U;
  config.cols = 24U;
  constexpr uint32_t kCount = 1031U;
  Buffer points;
  ASSERT_TRUE(
      points.Allocate(kCount * sizeof(TestPoint), MemoryType::kUnified));
  std::vector<float> expected_range(config.pixel_count(),
                                    std::numeric_limits<float>::infinity());
  std::vector<float> expected_intensity(config.pixel_count(), 0.0f);
  for (uint32_t i = 0U; i < kCount; ++i) {
    const uint32_t row = (i / config.cols) % config.rows;
    const uint32_t col = i % config.cols;
    const float azimuth = -kPi + (col + 0.5f) * 2.0f * kPi / config.cols;
    const float elevation =
        kQuarterPi - (row + 0.5f) * 2.0f * kQuarterPi / config.rows;
    const float r = 0.5f + static_cast<float>(i % 80U) * 0.125f;
    TestPoint point{r * std::cos(elevation) * std::cos(azimuth),
                    r * std::cos(elevation) * std::sin(azimuth),
                    r * std::sin(elevation),
                    static_cast<float>(i % 11U) - 5.0f};
    if (i % 31U == 0U) {
      point.x = std::numeric_limits<float>::quiet_NaN();
    }
    points.mutable_data<TestPoint>()[i] = point;
    const double range = std::hypot(
        std::hypot(static_cast<double>(point.x), static_cast<double>(point.y)),
        static_cast<double>(point.z));
    if (!std::isfinite(range) || range < config.min_range_m ||
        range > config.max_range_m) {
      continue;
    }
    const double a = std::atan2(static_cast<double>(point.y), point.x);
    const double e = std::asin(point.z / range);
    const uint32_t reference_col = static_cast<uint32_t>(
        (a - config.min_azimuth_rad) /
        (config.max_azimuth_rad - config.min_azimuth_rad) * config.cols);
    const uint32_t reference_row = static_cast<uint32_t>(
        (config.max_elevation_rad - e) /
        (config.max_elevation_rad - config.min_elevation_rad) * config.rows);
    ASSERT_LT(reference_col, config.cols);
    ASSERT_LT(reference_row, config.rows);
    const uint32_t pixel =
        PixelIndex(reference_row, reference_col, config.cols);
    if (range < expected_range[pixel]) {
      expected_range[pixel] = static_cast<float>(range);
      expected_intensity[pixel] = point.intensity;
    }
  }
  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  ASSERT_TRUE(output.Allocate(config));
  ASSERT_TRUE(workspace.Allocate(config));
  TestStream stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream.stream, cudaStreamNonBlocking),
            cudaSuccess);
  RangeImageBuilder builder(config);
  ASSERT_TRUE(builder
                  .BuildAsyncWithStatus(MakePointView(points, kCount), &output,
                                        &workspace, nullptr, stream.stream)
                  .ok());
  ASSERT_EQ(cudaStreamSynchronize(stream.stream), cudaSuccess);
  for (uint32_t pixel = 0U; pixel < config.pixel_count(); ++pixel) {
    EXPECT_EQ(output.valid_mask()[pixel],
              std::isfinite(expected_range[pixel]) ? 1U : 0U);
    if (std::isfinite(expected_range[pixel])) {
      EXPECT_NEAR(output.range()[pixel], expected_range[pixel], 2e-5f);
      EXPECT_FLOAT_EQ(output.intensity()[pixel], expected_intensity[pixel]);
    } else {
      EXPECT_TRUE(std::isinf(output.range()[pixel]));
      EXPECT_FLOAT_EQ(output.intensity()[pixel], 0.0f);
    }
  }
}

TEST(RangeImageTest, ResolvesEqualRangeByMinimumIntensityAndClearsEmptyInput) {
  const RangeImageConfig config = MakeTestConfig();
  Buffer points;
  constexpr uint32_t kCount = 1024U;
  ASSERT_TRUE(
      points.Allocate(kCount * sizeof(TestPoint), MemoryType::kUnified));
  for (uint32_t i = 0U; i < kCount; ++i) {
    points.mutable_data<TestPoint>()[i] = {1.0f, 0.0f, 0.0f,
                                           i == kCount - 1U ? -3.0f : 9.0f};
  }
  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  ASSERT_TRUE(output.Allocate(config));
  ASSERT_TRUE(workspace.Allocate(config));
  RangeImageBuilder builder(config);
  for (int iteration = 0; iteration < 8; ++iteration) {
    ASSERT_TRUE(
        builder.Build(MakePointView(points, kCount), &output, &workspace));
    EXPECT_FLOAT_EQ(output.intensity()[PixelIndex(2U, 4U, config.cols)], -3.0f);
  }
  PointCloudView empty;
  empty.set_memory_type(MemoryType::kUnified);
  ASSERT_TRUE(builder.Build(empty, &output, &workspace));
  for (uint32_t i = 0U; i < config.pixel_count(); ++i) {
    EXPECT_EQ(output.valid_mask()[i], 0U);
    EXPECT_TRUE(std::isinf(output.range()[i]));
    EXPECT_FLOAT_EQ(output.intensity()[i], 0.0f);
  }
}

TEST(RangeImageTest, ProjectsPlanarPointsAndKeepsNearestHit) {
  const RangeImageConfig config = MakeTestConfig();
  RangeImageBuilder builder(config);

  Buffer x;
  Buffer y;
  Buffer z;
  Buffer intensity;
  ASSERT_TRUE(x.Allocate(4U * sizeof(float), MemoryType::kUnified));
  ASSERT_TRUE(y.Allocate(4U * sizeof(float), MemoryType::kUnified));
  ASSERT_TRUE(z.Allocate(4U * sizeof(float), MemoryType::kUnified));
  ASSERT_TRUE(intensity.Allocate(4U * sizeof(float), MemoryType::kUnified));

  float* x_data = x.mutable_data<float>();
  float* y_data = y.mutable_data<float>();
  float* z_data = z.mutable_data<float>();
  float* intensity_data = intensity.mutable_data<float>();
  x_data[0] = 1.0f;
  y_data[0] = 0.0f;
  z_data[0] = 0.0f;
  intensity_data[0] = 10.0f;

  x_data[1] = 2.0f;
  y_data[1] = 0.0f;
  z_data[1] = 0.0f;
  intensity_data[1] = 20.0f;

  x_data[2] = 0.0f;
  y_data[2] = 1.0f;
  z_data[2] = 0.0f;
  intensity_data[2] = 30.0f;

  x_data[3] = std::numeric_limits<float>::quiet_NaN();
  y_data[3] = 0.0f;
  z_data[3] = 0.0f;
  intensity_data[3] = 40.0f;

  PointCloudView point_cloud(
      PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar, 4U,
      {
          {PointField::kX, ScalarType::kFloat32, x.data(), sizeof(float)},
          {PointField::kY, ScalarType::kFloat32, y.data(), sizeof(float)},
          {PointField::kZ, ScalarType::kFloat32, z.data(), sizeof(float)},
          {PointField::kIntensity, ScalarType::kFloat32, intensity.data(),
           sizeof(float)},
      });

  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  ASSERT_TRUE(output.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(workspace.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(builder.Build(point_cloud, &output, &workspace));

  const uint32_t center_pixel = PixelIndex(2U, 4U, config.cols);
  const uint32_t left_pixel = PixelIndex(2U, 6U, config.cols);
  EXPECT_EQ(output.valid_mask()[center_pixel], 1U);
  EXPECT_NEAR(output.range()[center_pixel], 1.0f, kTolerance);
  EXPECT_NEAR(output.intensity()[center_pixel], 10.0f, kTolerance);

  EXPECT_EQ(output.valid_mask()[left_pixel], 1U);
  EXPECT_NEAR(output.range()[left_pixel], 1.0f, kTolerance);
  EXPECT_NEAR(output.intensity()[left_pixel], 30.0f, kTolerance);

  const uint32_t invalid_pixel = PixelIndex(0U, 0U, config.cols);
  EXPECT_EQ(output.valid_mask()[invalid_pixel], 0U);
  EXPECT_TRUE(std::isinf(output.range()[invalid_pixel]));
  EXPECT_NEAR(output.intensity()[invalid_pixel], 0.0f, kTolerance);
}

TEST(RangeImageTest, ProjectsInterleavedPointsWithRigidTransform) {
  const RangeImageConfig config = MakeTestConfig();
  RangeImageBuilder builder(config);

  struct Point {
    float x;
    float y;
    float z;
  };

  Buffer points;
  ASSERT_TRUE(points.Allocate(sizeof(Point), MemoryType::kUnified));
  Point* point = points.mutable_data<Point>();
  point[0] = {0.0f, 1.0f, 0.0f};

  PointCloudView point_cloud(
      PointLayout::kInterleaved, MemoryType::kUnified,
      CoordinateFrame::kVehicle, 1U,
      {
          {PointField::kX, ScalarType::kFloat32, &point[0].x, sizeof(Point)},
          {PointField::kY, ScalarType::kFloat32, &point[0].y, sizeof(Point)},
          {PointField::kZ, ScalarType::kFloat32, &point[0].z, sizeof(Point)},
      });

  RigidTransform3f lidar_from_vehicle;
  lidar_from_vehicle.rotation << 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f;

  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  ASSERT_TRUE(output.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(workspace.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(
      builder.Build(point_cloud, &output, &workspace, &lidar_from_vehicle));

  const uint32_t center_pixel = PixelIndex(2U, 4U, config.cols);
  EXPECT_EQ(output.valid_mask()[center_pixel], 1U);
  EXPECT_NEAR(output.range()[center_pixel], 1.0f, kTolerance);
  EXPECT_NEAR(output.intensity()[center_pixel], 0.0f, kTolerance);
}

TEST(RangeImageTest, MapsElevationBoundariesToImageRows) {
  const RangeImageConfig config = MakeTestConfig();
  RangeImageBuilder builder(config);

  Buffer x;
  Buffer y;
  Buffer z;
  ASSERT_TRUE(x.Allocate(2U * sizeof(float), MemoryType::kUnified));
  ASSERT_TRUE(y.Allocate(2U * sizeof(float), MemoryType::kUnified));
  ASSERT_TRUE(z.Allocate(2U * sizeof(float), MemoryType::kUnified));

  float* x_data = x.mutable_data<float>();
  float* y_data = y.mutable_data<float>();
  float* z_data = z.mutable_data<float>();
  x_data[0] = 1.0f;
  y_data[0] = 0.0f;
  z_data[0] = 1.0f;
  x_data[1] = 1.0f;
  y_data[1] = 0.0f;
  z_data[1] = -1.0f;

  PointCloudView point_cloud(
      PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar, 2U,
      {
          {PointField::kX, ScalarType::kFloat32, x.data(), sizeof(float)},
          {PointField::kY, ScalarType::kFloat32, y.data(), sizeof(float)},
          {PointField::kZ, ScalarType::kFloat32, z.data(), sizeof(float)},
      });

  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  ASSERT_TRUE(output.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(workspace.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(builder.Build(point_cloud, &output, &workspace));

  const uint32_t top_pixel = PixelIndex(0U, 4U, config.cols);
  const uint32_t bottom_pixel = PixelIndex(3U, 4U, config.cols);
  EXPECT_EQ(output.valid_mask()[top_pixel], 1U);
  EXPECT_EQ(output.valid_mask()[bottom_pixel], 1U);
  EXPECT_NEAR(output.range()[top_pixel], std::sqrt(2.0f), 1e-4f);
  EXPECT_NEAR(output.range()[bottom_pixel], std::sqrt(2.0f), 1e-4f);
}

TEST(RangeImageTest, ProjectsInterleavedPointCloudBufferView) {
  const RangeImageConfig config = MakeTestConfig();
  RangeImageBuilder builder(config);

  PointCloudBuffer point_cloud;
  ASSERT_TRUE(
      point_cloud.Allocate(2U, PointLayout::kInterleaved, MemoryType::kUnified,
                           CoordinateFrame::kLidar,
                           {
                               {PointField::kX, ScalarType::kFloat32},
                               {PointField::kY, ScalarType::kFloat32},
                               {PointField::kZ, ScalarType::kFloat32},
                               {PointField::kIntensity, ScalarType::kFloat32},
                           }));

  *point_cloud.mutable_field_element<float>(PointField::kX, 0U) = 1.0f;
  *point_cloud.mutable_field_element<float>(PointField::kY, 0U) = 0.0f;
  *point_cloud.mutable_field_element<float>(PointField::kZ, 0U) = 0.0f;
  *point_cloud.mutable_field_element<float>(PointField::kIntensity, 0U) = 5.0f;

  *point_cloud.mutable_field_element<float>(PointField::kX, 1U) = 0.0f;
  *point_cloud.mutable_field_element<float>(PointField::kY, 1U) = 1.0f;
  *point_cloud.mutable_field_element<float>(PointField::kZ, 1U) = 0.0f;
  *point_cloud.mutable_field_element<float>(PointField::kIntensity, 1U) = 8.0f;

  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  ASSERT_TRUE(output.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(workspace.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(builder.Build(point_cloud.view(), &output, &workspace));

  const uint32_t center_pixel = PixelIndex(2U, 4U, config.cols);
  const uint32_t left_pixel = PixelIndex(2U, 6U, config.cols);
  EXPECT_EQ(output.valid_mask()[center_pixel], 1U);
  EXPECT_EQ(output.valid_mask()[left_pixel], 1U);
  EXPECT_NEAR(output.range()[center_pixel], 1.0f, kTolerance);
  EXPECT_NEAR(output.intensity()[center_pixel], 5.0f, kTolerance);
  EXPECT_NEAR(output.range()[left_pixel], 1.0f, kTolerance);
  EXPECT_NEAR(output.intensity()[left_pixel], 8.0f, kTolerance);
}

}  // namespace
}  // namespace lidar
}  // namespace vision
}  // namespace wheel
