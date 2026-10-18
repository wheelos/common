#include "camera/camera_geometry.h"

#include <cmath>
#include <limits>

#include "gtest/gtest.h"

#include "camera/pinhole_camera_model.h"

namespace wheel {
namespace vision {
namespace {

constexpr double kPi = 3.14159265358979323846;

PinholeCameraModel MakeCameraModel() {
  PinholeCameraModel camera_model;
  Eigen::Matrix3f intrinsics = Eigen::Matrix3f::Identity();
  intrinsics(0, 0) = 2.0f;
  intrinsics(1, 1) = 2.0f;
  EXPECT_TRUE(camera_model.Init(8, 8, intrinsics));
  return camera_model;
}

Eigen::Isometry3d MakeDownwardCameraTransform() {
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() =
      Eigen::AngleAxisd(kPi, Eigen::Vector3d::UnitX()).toRotationMatrix();
  transform.translation().z() = 1.0;
  return transform;
}

TEST(CameraGeometryTest, IntersectsPixelRayWithGroundPlane) {
  const PinholeCameraModel camera_model = MakeCameraModel();
  const Eigen::Isometry3d T_vehicle_camera = MakeDownwardCameraTransform();
  const Plane3d ground_plane{1e-12 * Eigen::Vector3d::UnitZ(), 0.0};

  Eigen::Vector3d point_vehicle;
  ASSERT_TRUE(PixelToPlane(camera_model, Eigen::Vector2f(2.0f, 0.0f),
                           T_vehicle_camera, ground_plane, &point_vehicle));
  EXPECT_NEAR(point_vehicle.x(), 1.0, 1e-6);
  EXPECT_NEAR(point_vehicle.y(), 0.0, 1e-6);
  EXPECT_NEAR(point_vehicle.z(), 0.0, 1e-6);
}

TEST(CameraGeometryTest, TransformsRayAndProjectsTargetPoint) {
  const PinholeCameraModel camera_model = MakeCameraModel();
  const Eigen::Isometry3d T_vehicle_camera = MakeDownwardCameraTransform();

  Eigen::Vector3d ray_origin;
  Eigen::Vector3d ray_direction;
  ASSERT_TRUE(PixelToRayInFrame(camera_model, Eigen::Vector2f(2.0f, 0.0f),
                                T_vehicle_camera, &ray_origin, &ray_direction));
  EXPECT_NEAR(ray_origin.z(), 1.0, 1e-6);
  EXPECT_NEAR(ray_direction.x(), 1.0 / std::sqrt(2.0), 1e-6);
  EXPECT_NEAR(ray_direction.z(), -1.0 / std::sqrt(2.0), 1e-6);

  Eigen::Vector2f pixel;
  ASSERT_TRUE(PointToPixel(camera_model, Eigen::Vector3d(1.0, 0.0, 0.0),
                           T_vehicle_camera, &pixel));
  EXPECT_NEAR(pixel.x(), 2.0, 1e-6);
  EXPECT_NEAR(pixel.y(), 0.0, 1e-6);
}

TEST(CameraGeometryTest, RejectsParallelAndBehindPlaneIntersections) {
  const PinholeCameraModel camera_model = MakeCameraModel();
  const Eigen::Isometry3d T_target_camera = Eigen::Isometry3d::Identity();
  Eigen::Vector3d point;

  const Plane3d parallel_plane{Eigen::Vector3d::UnitX(), -1.0};
  EXPECT_FALSE(PixelToPlane(camera_model, Eigen::Vector2f::Zero(),
                            T_target_camera, parallel_plane, &point));

  const Plane3d behind_camera_plane{Eigen::Vector3d::UnitZ(), -2.0};
  EXPECT_FALSE(PixelToPlane(camera_model, Eigen::Vector2f::Zero(),
                            MakeDownwardCameraTransform(), behind_camera_plane,
                            &point));
}

TEST(CameraGeometryTest, RejectsInvalidInputsAndTransforms) {
  const PinholeCameraModel camera_model = MakeCameraModel();
  const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
  Eigen::Vector3d point;
  Eigen::Vector2f pixel;

  EXPECT_FALSE(PixelToPlane(camera_model, Eigen::Vector2f::Zero(), identity,
                            Plane3d{Eigen::Vector3d::Zero(), 0.0}, &point));
  EXPECT_FALSE(PixelToPlane(camera_model, Eigen::Vector2f::Zero(), identity,
                            Plane3d{Eigen::Vector3d::UnitZ(),
                                    std::numeric_limits<double>::quiet_NaN()},
                            &point));
  EXPECT_FALSE(PixelToRayInFrame(camera_model, Eigen::Vector2f::Zero(),
                                 identity, nullptr, &point));
  EXPECT_FALSE(PixelToRayInFrame(camera_model, Eigen::Vector2f::Zero(),
                                 identity, &point, &point));
  EXPECT_FALSE(
      PointToPixel(camera_model, Eigen::Vector3d::Zero(), identity, nullptr));

  Eigen::Isometry3d invalid_transform = identity;
  invalid_transform.linear()(0, 0) = 2.0;
  EXPECT_FALSE(PixelToRayInFrame(camera_model, Eigen::Vector2f::Zero(),
                                 invalid_transform, &point, &point));
  EXPECT_FALSE(PointToPixel(camera_model, Eigen::Vector3d::Zero(),
                            invalid_transform, &pixel));
}

}  // namespace
}  // namespace vision
}  // namespace wheel
