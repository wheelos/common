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

#include "image/cpu_ipm_transformer.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

#include "camera/omnidirectional_camera_model.h"
#include "camera/pinhole_camera_model.h"

namespace wheel {
namespace vision {
namespace image {
namespace {

std::shared_ptr<PinholeCameraModel> MakeCameraModel() {
  auto camera_model = std::make_shared<PinholeCameraModel>();
  Eigen::Matrix3f intrinsics = Eigen::Matrix3f::Identity();
  intrinsics(0, 0) = 2.0f;
  intrinsics(1, 1) = 2.0f;
  intrinsics(0, 2) = 3.0f;
  intrinsics(1, 2) = 3.0f;
  EXPECT_TRUE(camera_model->Init(8, 8, intrinsics));
  return camera_model;
}

Eigen::Isometry3d MakeVehicleFromCameraTransform() {
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0;
  return transform;
}

IpmConfig MakeConfig() {
  IpmConfig config;
  config.resolution = 1.0;
  config.bev_width = 2;
  config.bev_height = 2;
  config.x_min = -1.0;
  config.y_min = 2.0;
  return config;
}

TEST(CpuIpmTransformerTest, BuildsProjectionMapAndTransformsImage) {
  const auto camera_model = MakeCameraModel();
  const IpmConfig config = MakeConfig();
  const CpuIpmTransformer transformer(camera_model,
                                      MakeVehicleFromCameraTransform(), config);
  Eigen::Vector2f image_pixel;
  ASSERT_TRUE(transformer.MapPixel(0, 0, &image_pixel));
  EXPECT_NEAR(image_pixel.x(), 7.0f / 3.0f, 1e-5f);
  EXPECT_NEAR(image_pixel.y(), 3.0f, 1e-5f);

  std::vector<float> src_img(8 * 8);
  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 8; ++col) {
      src_img[static_cast<std::size_t>(row * 8 + col)] =
          static_cast<float>(10 * row + col);
    }
  }
  std::vector<float> dst_bev_img(config.bev_width * config.bev_height);
  transformer.Transform(
      ImageView<const float>{src_img.data(), 8, 8, 1, 8},
      ImageView<float>{dst_bev_img.data(), config.bev_width, config.bev_height,
                       1, static_cast<std::size_t>(config.bev_width)});

  EXPECT_NEAR(dst_bev_img[0], 10.0f * 3.0f + 7.0f / 3.0f, 2e-2f);
  EXPECT_NEAR(dst_bev_img[1], 33.0f, 1e-5f);
  EXPECT_NEAR(dst_bev_img[2], 32.0f, 1e-5f);
  EXPECT_NEAR(dst_bev_img[3], 33.0f, 1e-5f);
}

TEST(CpuIpmTransformerTest, InvalidGroundPlaneProjectionsAreMasked) {
  const CpuIpmTransformer transformer(
      MakeCameraModel(), Eigen::Isometry3d::Identity(), MakeConfig());
  Eigen::Vector2f image_pixel;
  EXPECT_FALSE(transformer.MapPixel(0, 0, &image_pixel));
  EXPECT_FALSE(transformer.MapPixel(-1, 0, &image_pixel));
  EXPECT_FALSE(transformer.MapPixel(0, 0, nullptr));
}

TEST(CpuIpmTransformerTest, RejectsInvalidCameraAndExtrinsics) {
  const IpmConfig config = MakeConfig();
  const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();

  EXPECT_THROW(
      CpuIpmTransformer(nullptr, MakeVehicleFromCameraTransform(), config),
      std::invalid_argument);
  EXPECT_THROW(CpuIpmTransformer(std::make_shared<PinholeCameraModel>(),
                                 MakeVehicleFromCameraTransform(), config),
               std::invalid_argument);

  Eigen::Isometry3d non_rigid = identity;
  non_rigid.linear()(0, 0) = 2.0;
  EXPECT_THROW(CpuIpmTransformer(MakeCameraModel(), non_rigid, config),
               std::invalid_argument);

  Eigen::Isometry3d left_handed = identity;
  left_handed.linear()(0, 0) = -1.0;
  EXPECT_THROW(CpuIpmTransformer(MakeCameraModel(), left_handed, config),
               std::invalid_argument);

  Eigen::Isometry3d non_finite = identity;
  non_finite.translation().x() = std::numeric_limits<double>::infinity();
  EXPECT_THROW(CpuIpmTransformer(MakeCameraModel(), non_finite, config),
               std::invalid_argument);

  Eigen::Isometry3d non_affine = identity;
  non_affine.matrix()(3, 0) = 1.0;
  EXPECT_THROW(CpuIpmTransformer(MakeCameraModel(), non_affine, config),
               std::invalid_argument);
}

TEST(CpuIpmTransformerTest, RejectsInvalidConfigAndMismatchedImageSize) {
  const auto camera_model = MakeCameraModel();
  const Eigen::Isometry3d transform = MakeVehicleFromCameraTransform();

  IpmConfig invalid_config = MakeConfig();
  invalid_config.resolution = 0.0;
  EXPECT_THROW(CpuIpmTransformer(camera_model, transform, invalid_config),
               std::invalid_argument);

  invalid_config = MakeConfig();
  invalid_config.ground_z = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(CpuIpmTransformer(camera_model, transform, invalid_config),
               std::invalid_argument);

  const CpuIpmTransformer transformer(camera_model, transform, MakeConfig());
  std::vector<unsigned char> src(8 * 8);
  std::vector<unsigned char> dst(4);
  EXPECT_THROW(transformer.Transform(
                   ImageView<const unsigned char>{src.data(), 8, 4, 1, 8},
                   ImageView<unsigned char>{dst.data(), 2, 2, 1, 2}),
               std::invalid_argument);
  EXPECT_THROW(
      transformer.Transform(ImageView<const unsigned char>{nullptr, 0, 0, 0, 0},
                            ImageView<unsigned char>{dst.data(), 2, 2, 1, 2}),
      std::invalid_argument);
}

TEST(CpuIpmTransformerTest, SupportsPaddedInterleavedImagesAndNearestSampling) {
  const auto camera_model = MakeCameraModel();
  const CpuIpmTransformer transformer(
      camera_model, MakeVehicleFromCameraTransform(), MakeConfig());

  constexpr int kChannels = 2;
  constexpr std::size_t kSourceStride = 8 * kChannels + 3;
  constexpr std::size_t kDestinationStride = 2 * kChannels + 1;
  std::vector<unsigned char> src(kSourceStride * 8, 0);
  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 8; ++col) {
      const std::size_t offset =
          static_cast<std::size_t>(row) * kSourceStride + col * kChannels;
      src[offset] = static_cast<unsigned char>(10 * row + col);
      src[offset + 1] = static_cast<unsigned char>(200 + row + col);
    }
  }
  std::vector<unsigned char> dst(kDestinationStride * 2, 0);

  transformer.Transform(
      ImageView<const unsigned char>{src.data(), 8, 8, kChannels,
                                     kSourceStride},
      ImageView<unsigned char>{dst.data(), 2, 2, kChannels, kDestinationStride},
      InterpolationMode::kNearest);

  const ImageProperties properties =
      ImageView<const unsigned char>{src.data(), 8, 8, kChannels, kSourceStride}
          .properties();
  EXPECT_EQ(properties.width, 8);
  EXPECT_EQ(properties.height, 8);
  EXPECT_EQ(properties.channels, kChannels);
  EXPECT_EQ(properties.data_type, ImageDataType::kUInt8);
  EXPECT_EQ(properties.layout, ImageLayout::kInterleaved);
  EXPECT_EQ(properties.memory_type, ImageMemoryType::kCpu);
  EXPECT_EQ(properties.row_stride_bytes, kSourceStride);

  EXPECT_EQ(dst[0], 32);
  EXPECT_EQ(dst[1], 205);
  EXPECT_EQ(dst[kChannels], 33);
  EXPECT_EQ(dst[kDestinationStride], 32);
  EXPECT_EQ(dst[kDestinationStride + kChannels], 33);
}

TEST(CpuIpmTransformerTest, SupportsOmnidirectionalRaysOutsidePinholeFront) {
  auto camera_model = std::make_shared<OmnidirectionalCameraModel>();
  Eigen::Matrix3f intrinsics = Eigen::Matrix3f::Identity();
  intrinsics(0, 0) = 2.0f;
  intrinsics(1, 1) = 2.0f;
  intrinsics(0, 2) = 3.0f;
  intrinsics(1, 2) = 3.0f;
  ASSERT_TRUE(camera_model->Init(8, 8, intrinsics, 1.0f,
                                 OmnidirectionalDistortionCoefficients()));

  IpmConfig config;
  config.resolution = 1.0;
  config.bev_width = 1;
  config.bev_height = 1;
  config.x_min = 0.0;
  config.y_min = -2.0;
  const CpuIpmTransformer transformer(camera_model,
                                      Eigen::Isometry3d::Identity(), config);

  Eigen::Vector2f image_pixel;
  ASSERT_TRUE(transformer.MapPixel(0, 0, &image_pixel));
  EXPECT_NEAR(image_pixel.x(), 3.0f, 1e-5f);
  EXPECT_NEAR(image_pixel.y(), 1.0f, 1e-5f);
}

TEST(CpuIpmTransformerTest, CapturesCameraGeometryAtConstruction) {
  const auto camera_model = MakeCameraModel();
  const CpuIpmTransformer transformer(
      camera_model, MakeVehicleFromCameraTransform(), MakeConfig());
  Eigen::Vector2f pixel_before;
  ASSERT_TRUE(transformer.MapPixel(0, 0, &pixel_before));

  Eigen::Matrix3f changed_intrinsics = Eigen::Matrix3f::Identity();
  changed_intrinsics(0, 0) = 4.0f;
  changed_intrinsics(1, 1) = 4.0f;
  changed_intrinsics(0, 2) = 1.0f;
  changed_intrinsics(1, 2) = 1.0f;
  ASSERT_TRUE(camera_model->Init(4, 4, changed_intrinsics));

  Eigen::Vector2f pixel_after;
  ASSERT_TRUE(transformer.MapPixel(0, 0, &pixel_after));
  EXPECT_FLOAT_EQ(pixel_after.x(), pixel_before.x());
  EXPECT_FLOAT_EQ(pixel_after.y(), pixel_before.y());

  std::vector<unsigned char> source(8 * 8, 1);
  std::vector<unsigned char> destination(4);
  EXPECT_NO_THROW(transformer.Transform(
      ImageView<const unsigned char>{source.data(), 8, 8, 1, 8},
      ImageView<unsigned char>{destination.data(), 2, 2, 1, 2}));
}

TEST(CpuIpmTransformerTest, RejectsOverlappingImageViews) {
  const CpuIpmTransformer transformer(
      MakeCameraModel(), MakeVehicleFromCameraTransform(), MakeConfig());
  std::vector<unsigned char> shared_storage(8 * 8);

  EXPECT_THROW(
      transformer.Transform(
          ImageView<const unsigned char>{shared_storage.data(), 8, 8, 1, 8},
          ImageView<unsigned char>{shared_storage.data() + 1, 2, 2, 1, 2}),
      std::invalid_argument);
}

}  // namespace
}  // namespace image
}  // namespace vision
}  // namespace wheel
