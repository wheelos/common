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

#include "image/opencv_ipm_adapter.h"

#include <memory>
#include <stdexcept>

#include <opencv2/core.hpp>

#include "gtest/gtest.h"

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

TEST(OpenCvIpmAdapterTest, SupportsNonContinuousSourceImages) {
  const CpuIpmTransformer transformer(
      MakeCameraModel(), MakeVehicleFromCameraTransform(), MakeConfig());
  cv::Mat padded_source(8, 10, CV_8UC1, cv::Scalar(0));
  cv::Mat source = padded_source(cv::Rect(1, 0, 8, 8));
  for (int row = 0; row < source.rows; ++row) {
    for (int col = 0; col < source.cols; ++col) {
      source.at<unsigned char>(row, col) =
          static_cast<unsigned char>(10 * row + col);
    }
  }

  cv::Mat bev;
  TransformOpenCv(transformer, source, &bev);

  ASSERT_EQ(bev.type(), CV_8UC1);
  ASSERT_EQ(bev.rows, 2);
  ASSERT_EQ(bev.cols, 2);
  EXPECT_EQ(bev.at<unsigned char>(0, 0), 32);
  EXPECT_EQ(bev.at<unsigned char>(0, 1), 33);
}

TEST(OpenCvIpmAdapterTest, RejectsUnsupportedDepthAndNullDestination) {
  const CpuIpmTransformer transformer(
      MakeCameraModel(), MakeVehicleFromCameraTransform(), MakeConfig());
  const cv::Mat unsupported(8, 8, CV_16UC1, cv::Scalar(1));
  cv::Mat output;

  EXPECT_THROW(TransformOpenCv(transformer, unsupported, &output),
               std::invalid_argument);
  EXPECT_THROW(TransformOpenCv(transformer, unsupported, nullptr),
               std::invalid_argument);
}

TEST(OpenCvIpmAdapterTest, RejectsOverlappingRegionsOfSharedStorage) {
  const CpuIpmTransformer transformer(
      MakeCameraModel(), MakeVehicleFromCameraTransform(), MakeConfig());
  cv::Mat storage(8, 10, CV_8UC1, cv::Scalar(0));
  const cv::Mat source = storage(cv::Rect(0, 0, 8, 8));
  cv::Mat destination = storage(cv::Rect(2, 2, 2, 2));

  EXPECT_THROW(TransformOpenCv(transformer, source, &destination),
               std::invalid_argument);
}

}  // namespace
}  // namespace image
}  // namespace vision
}  // namespace wheel
