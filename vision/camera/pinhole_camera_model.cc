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

#include "camera/pinhole_camera_model.h"

#include <cmath>

namespace wheel {
namespace vision {

namespace {
constexpr float kEpsilon = 1e-6f;
}  // namespace

bool PinholeCameraModel::Init(uint32_t width, uint32_t height,
                              const Eigen::Matrix3f& intrinsics) {
  return InitializeBase(width, height, intrinsics);
}

bool PinholeCameraModel::RayToPixel(const Eigen::Vector3f& ray,
                                    Eigen::Vector2f* pixel) const {
  if (pixel == nullptr || !is_initialized() || !ray.allFinite() ||
      ray.z() <= kEpsilon) {
    return false;
  }

  const float inv_z = 1.0f / ray.z();
  const float x_norm = ray.x() * inv_z;
  const float y_norm = ray.y() * inv_z;

  pixel->x() = intrinsic_matrix_(0, 0) * x_norm +
               intrinsic_matrix_(0, 1) * y_norm + intrinsic_matrix_(0, 2);
  pixel->y() = intrinsic_matrix_(1, 1) * y_norm + intrinsic_matrix_(1, 2);

  return pixel->allFinite();
}

bool PinholeCameraModel::PixelToRay(const Eigen::Vector2f& pixel,
                                    Eigen::Vector3f* ray) const {
  if (ray == nullptr || !is_initialized() || !pixel.allFinite()) {
    return false;
  }

  const float fy = intrinsic_matrix_(1, 1);
  const float fx = intrinsic_matrix_(0, 0);
  if (fx <= kEpsilon || fy <= kEpsilon) {
    return false;
  }

  const float y_norm = (pixel.y() - intrinsic_matrix_(1, 2)) / fy;
  const float x_norm =
      (pixel.x() - intrinsic_matrix_(0, 2) - intrinsic_matrix_(0, 1) * y_norm) /
      fx;

  ray->x() = x_norm;
  ray->y() = y_norm;
  ray->z() = 1.0f;

  return ray->allFinite();
}

std::shared_ptr<CameraModel> PinholeCameraModel::GetIdealModel() const {
  if (!is_initialized()) {
    return nullptr;
  }
  auto model = std::make_shared<PinholeCameraModel>();
  return model->Init(width_, height_, intrinsic_matrix_) ? model : nullptr;
}

}  // namespace vision
}  // namespace wheel
