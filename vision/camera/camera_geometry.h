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

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "camera/camera_model.h"

namespace wheel {
namespace vision {

// A plane in a target coordinate frame, represented as
// normal.dot(point) + offset = 0. The normal does not need to be normalized.
struct Plane3d {
  Eigen::Vector3d normal;
  double offset = 0.0;
};

// Back-projects a pixel into a ray in the target frame. T_target_camera maps
// camera-frame coordinates into the target frame. The returned direction is
// normalized; the ray origin is the camera origin expressed in the target
// frame.
bool PixelToRayInFrame(const CameraModel& camera_model,
                       const Eigen::Vector2f& pixel,
                       const Eigen::Isometry3d& T_target_camera,
                       Eigen::Vector3d* ray_origin,
                       Eigen::Vector3d* ray_direction);

// Intersects a pixel ray with a plane expressed in the target frame. Returns
// false for invalid input, a parallel ray, or an intersection behind the
// camera.
bool PixelToPlane(const CameraModel& camera_model, const Eigen::Vector2f& pixel,
                  const Eigen::Isometry3d& T_target_camera,
                  const Plane3d& plane, Eigen::Vector3d* point);

// Projects a point in the target frame into the camera image. This checks
// whether the point is projectable, but does not require it to fall inside the
// calibrated image bounds.
bool PointToPixel(const CameraModel& camera_model,
                  const Eigen::Vector3d& point_target,
                  const Eigen::Isometry3d& T_target_camera,
                  Eigen::Vector2f* pixel);

}  // namespace vision
}  // namespace wheel
