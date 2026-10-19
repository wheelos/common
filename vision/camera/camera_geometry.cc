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

#include "camera/camera_geometry.h"

#include <cmath>
#include <limits>

namespace wheel {
namespace vision {
namespace {

constexpr double kTransformTolerance = 1e-3;
constexpr double kRayEpsilon = 1e-8;

bool IsValidTransform(const Eigen::Isometry3d& transform) {
  if (!transform.matrix().allFinite()) {
    return false;
  }

  const Eigen::Matrix3d& rotation = transform.linear();
  return (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
                 .norm() <= kTransformTolerance &&
         std::abs(rotation.determinant() - 1.0) <= kTransformTolerance;
}

}  // namespace

bool PixelToRayInFrame(const CameraModel& camera_model,
                       const Eigen::Vector2f& pixel,
                       const Eigen::Isometry3d& T_target_camera,
                       Eigen::Vector3d* ray_origin,
                       Eigen::Vector3d* ray_direction) {
  if (ray_origin == nullptr || ray_direction == nullptr ||
      ray_origin == ray_direction || !camera_model.is_initialized() ||
      !pixel.allFinite() || !IsValidTransform(T_target_camera)) {
    return false;
  }

  Eigen::Vector3f ray_camera;
  if (!camera_model.PixelToRay(pixel, &ray_camera) || !ray_camera.allFinite() ||
      ray_camera.norm() <= kRayEpsilon) {
    return false;
  }

  const Eigen::Vector3d direction_target =
      T_target_camera.linear() * ray_camera.cast<double>().normalized();
  if (!direction_target.allFinite() || direction_target.norm() <= kRayEpsilon) {
    return false;
  }

  *ray_origin = T_target_camera.translation();
  *ray_direction = direction_target.normalized();
  return true;
}

bool PixelToPlane(const CameraModel& camera_model, const Eigen::Vector2f& pixel,
                  const Eigen::Isometry3d& T_target_camera,
                  const Plane3d& plane, Eigen::Vector3d* point) {
  if (point == nullptr || !plane.normal.allFinite() ||
      !std::isfinite(plane.offset)) {
    return false;
  }
  const double normal_norm = plane.normal.stableNorm();
  if (!std::isfinite(normal_norm) ||
      normal_norm <= std::numeric_limits<double>::min()) {
    return false;
  }
  const Eigen::Vector3d plane_normal = plane.normal / normal_norm;
  const double plane_offset = plane.offset / normal_norm;
  if (!plane_normal.allFinite() || !std::isfinite(plane_offset)) {
    return false;
  }

  Eigen::Vector3d ray_origin;
  Eigen::Vector3d ray_direction;
  if (!PixelToRayInFrame(camera_model, pixel, T_target_camera, &ray_origin,
                         &ray_direction)) {
    return false;
  }

  const double denominator = plane_normal.dot(ray_direction);
  if (!std::isfinite(denominator) || std::abs(denominator) <= kRayEpsilon) {
    return false;
  }

  const double distance =
      -(plane_normal.dot(ray_origin) + plane_offset) / denominator;
  if (!std::isfinite(distance) || distance <= kRayEpsilon) {
    return false;
  }

  const Eigen::Vector3d intersection = ray_origin + distance * ray_direction;
  if (!intersection.allFinite()) {
    return false;
  }
  *point = intersection;
  return true;
}

bool PointToPixel(const CameraModel& camera_model,
                  const Eigen::Vector3d& point_target,
                  const Eigen::Isometry3d& T_target_camera,
                  Eigen::Vector2f* pixel) {
  if (pixel == nullptr || !camera_model.is_initialized() ||
      !point_target.allFinite() || !IsValidTransform(T_target_camera)) {
    return false;
  }

  const Eigen::Vector3d point_camera = T_target_camera.inverse() * point_target;
  if (!point_camera.allFinite()) {
    return false;
  }

  return camera_model.RayToPixel(point_camera.cast<float>(), pixel);
}

}  // namespace vision
}  // namespace wheel
