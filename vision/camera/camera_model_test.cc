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
#include <memory>

#include "gtest/gtest.h"

#include "camera/brown_camera_model.h"
#include "camera/omnidirectional_camera_model.h"
#include "camera/pinhole_camera_model.h"

namespace wheel {
namespace vision {
namespace {

constexpr float kTolerance = 1e-4f;

Eigen::Matrix3f MakeIntrinsics(float fx, float fy, float cx, float cy,
                               float skew = 0.0f) {
  Eigen::Matrix3f intrinsics = Eigen::Matrix3f::Identity();
  intrinsics(0, 0) = fx;
  intrinsics(0, 1) = skew;
  intrinsics(0, 2) = cx;
  intrinsics(1, 1) = fy;
  intrinsics(1, 2) = cy;
  return intrinsics;
}

void ExpectVector2Near(const Eigen::Vector2f& actual,
                       const Eigen::Vector2f& expected, float tolerance) {
  EXPECT_NEAR(actual.x(), expected.x(), tolerance);
  EXPECT_NEAR(actual.y(), expected.y(), tolerance);
}

void ExpectDirectionNear(const Eigen::Vector3f& actual,
                         const Eigen::Vector3f& expected, float tolerance) {
  const Eigen::Vector3f actual_normalized = actual.normalized();
  const Eigen::Vector3f expected_normalized = expected.normalized();
  EXPECT_NEAR(actual_normalized.x(), expected_normalized.x(), tolerance);
  EXPECT_NEAR(actual_normalized.y(), expected_normalized.y(), tolerance);
  EXPECT_NEAR(actual_normalized.z(), expected_normalized.z(), tolerance);
}

TEST(PinholeCameraModelTest, ProjectsAndUnprojectsWithSkew) {
  PinholeCameraModel model;
  ASSERT_TRUE(model.Init(640, 480,
                         MakeIntrinsics(400.0f, 420.0f, 320.0f, 240.0f, 2.0f)));

  Eigen::Vector2f pixel;
  ASSERT_TRUE(model.RayToPixel(Eigen::Vector3f(0.2f, -0.1f, 1.0f), &pixel));
  ExpectVector2Near(pixel, Eigen::Vector2f(399.8f, 198.0f), 1e-3f);

  Eigen::Vector3f ray;
  ASSERT_TRUE(model.PixelToRay(pixel, &ray));
  ExpectVector2Near(ray.head<2>(), Eigen::Vector2f(0.2f, -0.1f), 1e-5f);
  EXPECT_NEAR(ray.z(), 1.0f, 1e-6f);
}

TEST(PinholeCameraModelTest, RejectsInvalidInitializationAndBackFacingPoints) {
  PinholeCameraModel invalid_model;
  Eigen::Matrix3f invalid_intrinsics =
      MakeIntrinsics(0.0f, 400.0f, 320.0f, 240.0f);
  EXPECT_FALSE(invalid_model.Init(640, 480, invalid_intrinsics));
  EXPECT_FALSE(invalid_model.is_initialized());

  PinholeCameraModel valid_model;
  ASSERT_TRUE(valid_model.Init(640, 480,
                               MakeIntrinsics(400.0f, 400.0f, 320.0f, 240.0f)));
  Eigen::Vector2f pixel;
  EXPECT_FALSE(
      valid_model.RayToPixel(Eigen::Vector3f(0.0f, 0.0f, 0.0f), &pixel));
  EXPECT_FALSE(
      valid_model.RayToPixel(Eigen::Vector3f(0.0f, 0.0f, -1.0f), &pixel));
}

TEST(PinholeCameraModelTest, RejectsMalformedAndNonFiniteIntrinsics) {
  PinholeCameraModel model;
  Eigen::Matrix3f intrinsics = MakeIntrinsics(400.0f, 400.0f, 320.0f, 240.0f);

  intrinsics(1, 0) = 0.1f;
  EXPECT_FALSE(model.Init(640, 480, intrinsics));
  EXPECT_FALSE(model.is_initialized());

  intrinsics = MakeIntrinsics(400.0f, 400.0f, 320.0f, 240.0f);
  intrinsics(0, 2) = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(model.Init(640, 480, intrinsics));
  EXPECT_FALSE(model.is_initialized());
}

TEST(PinholeCameraModelTest, ReportsImageSizeAndFieldOfView) {
  PinholeCameraModel model;
  ASSERT_TRUE(
      model.Init(640, 480, MakeIntrinsics(400.0f, 400.0f, 320.0f, 240.0f)));

  EXPECT_EQ(model.width(), 640U);
  EXPECT_EQ(model.height(), 480U);
  EXPECT_TRUE(model.IsRayInsideFov(Eigen::Vector3f(0.0f, 0.0f, 1.0f)));
  EXPECT_FALSE(model.IsRayInsideFov(Eigen::Vector3f(10.0f, 0.0f, 1.0f)));
}

TEST(BrownCameraModelTest, MatchesPinholeWhenDistortionIsZero) {
  const Eigen::Matrix3f intrinsics =
      MakeIntrinsics(450.0f, 430.0f, 320.0f, 240.0f, 1.5f);

  PinholeCameraModel pinhole;
  BrownCameraModel brown;
  ASSERT_TRUE(pinhole.Init(640, 480, intrinsics));
  ASSERT_TRUE(brown.Init(640, 480, intrinsics,
                         BrownDistortionCoefficients().ToVector()));

  const Eigen::Vector3f ray(0.15f, -0.05f, 1.0f);
  Eigen::Vector2f pinhole_pixel;
  Eigen::Vector2f brown_pixel;
  ASSERT_TRUE(pinhole.RayToPixel(ray, &pinhole_pixel));
  ASSERT_TRUE(brown.RayToPixel(ray, &brown_pixel));
  ExpectVector2Near(brown_pixel, pinhole_pixel, 1e-5f);
}

TEST(BrownCameraModelTest,
     RoundTripsWithDistortionAndReturnsIdealPinholeModel) {
  BrownCameraModel model;
  BrownDistortionCoefficients distortion;
  distortion.k1 = 0.05f;
  distortion.k2 = -0.02f;
  distortion.p1 = 0.001f;
  distortion.p2 = -0.0005f;
  distortion.k3 = 0.003f;

  ASSERT_TRUE(model.Init(
      640, 480, MakeIntrinsics(420.0f, 415.0f, 320.0f, 240.0f), distortion));

  const Eigen::Vector3f input_ray(0.12f, -0.08f, 1.0f);
  Eigen::Vector2f pixel;
  ASSERT_TRUE(model.RayToPixel(input_ray, &pixel));

  Eigen::Vector3f recovered_ray;
  ASSERT_TRUE(model.PixelToRay(pixel, &recovered_ray));
  ExpectVector2Near(recovered_ray.head<2>(), input_ray.head<2>(), 1e-4f);
  EXPECT_NEAR(recovered_ray.z(), 1.0f, 1e-6f);

  const std::shared_ptr<CameraModel> ideal_model = model.GetIdealModel();
  ASSERT_NE(ideal_model, nullptr);
  EXPECT_EQ(ideal_model->Type(), CameraModelType::kPinhole);
}

TEST(OmnidirectionalCameraModelTest, OpticalAxisProjectsToPrincipalPoint) {
  OmnidirectionalCameraModel model;
  ASSERT_TRUE(model.Init(1280, 720,
                         MakeIntrinsics(500.0f, 500.0f, 640.0f, 360.0f), 1.0f,
                         OmnidirectionalDistortionCoefficients()));

  Eigen::Vector2f pixel;
  ASSERT_TRUE(model.RayToPixel(Eigen::Vector3f(0.0f, 0.0f, 1.0f), &pixel));
  ExpectVector2Near(pixel, Eigen::Vector2f(640.0f, 360.0f), 1e-5f);
}

TEST(OmnidirectionalCameraModelTest, RoundTripsOpenCvOmnidirStyleParameters) {
  OmnidirectionalCameraModel model;
  OmnidirectionalDistortionCoefficients distortion;
  distortion.k1 = 0.01f;
  distortion.k2 = -0.002f;
  distortion.p1 = 0.0005f;
  distortion.p2 = -0.0003f;

  ASSERT_TRUE(model.Init(1280, 720,
                         MakeIntrinsics(480.0f, 470.0f, 640.0f, 360.0f, 1.0f),
                         1.1f, distortion));

  const Eigen::Vector3f input_ray(0.3f, -0.1f, 1.0f);
  Eigen::Vector2f pixel;
  ASSERT_TRUE(model.RayToPixel(input_ray, &pixel));

  Eigen::Vector3f recovered_ray;
  ASSERT_TRUE(model.PixelToRay(pixel, &recovered_ray));
  ExpectDirectionNear(recovered_ray, input_ray, kTolerance);

  const std::shared_ptr<CameraModel> ideal_model = model.GetIdealModel();
  ASSERT_NE(ideal_model, nullptr);
  EXPECT_EQ(ideal_model->Type(), CameraModelType::kOmni);
  const auto ideal_omni_model =
      std::dynamic_pointer_cast<OmnidirectionalCameraModel>(ideal_model);
  ASSERT_NE(ideal_omni_model, nullptr);
  EXPECT_FLOAT_EQ(ideal_omni_model->xi(), model.xi());
  EXPECT_TRUE(ideal_omni_model->distort_params().isZero());
}

TEST(OmnidirectionalCameraModelTest, RejectsSingularBackwardsDirection) {
  OmnidirectionalCameraModel model;
  ASSERT_TRUE(model.Init(1280, 720,
                         MakeIntrinsics(480.0f, 470.0f, 640.0f, 360.0f), 1.0f,
                         OmnidirectionalDistortionCoefficients()));

  Eigen::Vector2f pixel;
  EXPECT_FALSE(model.RayToPixel(Eigen::Vector3f(0.0f, 0.0f, -1.0f), &pixel));
}

}  // namespace
}  // namespace vision
}  // namespace wheel
