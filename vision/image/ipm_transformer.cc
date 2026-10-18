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

#include "image/ipm_transformer.h"

#include <cmath>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>

namespace wheel {
namespace vision {
namespace image {

IpmTransformer::IpmTransformer(std::shared_ptr<CameraModel> camera_model,
                               const Eigen::Isometry3d& T_vehicle_camera,
                               const IpmConfig& config)
    : camera_model_(std::move(camera_model)),
      T_vehicle_camera_(T_vehicle_camera),
      config_(config) {
  ValidateInputs();
  ValidateExtrinsics();
  T_camera_vehicle_ = T_vehicle_camera_.inverse();

  PrecomputeLut();
}

bool IpmTransformer::MapPixel(int bev_x, int bev_y,
                              Eigen::Vector2f* image_pixel) const {
  if (image_pixel == nullptr || bev_x < 0 || bev_x >= config_.bev_width ||
      bev_y < 0 || bev_y >= config_.bev_height) {
    return false;
  }
  const std::size_t index = static_cast<std::size_t>(bev_y) *
                                static_cast<std::size_t>(config_.bev_width) +
                            static_cast<std::size_t>(bev_x);
  if (map_valid_[index] == 0) {
    return false;
  }
  *image_pixel = Eigen::Vector2f(map_x_[index], map_y_[index]);
  return true;
}

void IpmTransformer::ValidateInputs() const {
  if (!camera_model_) {
    throw std::invalid_argument("IpmTransformer: camera_model is nullptr.");
  }
  if (!camera_model_->is_initialized()) {
    throw std::invalid_argument(
        "IpmTransformer: camera_model is not initialized.");
  }

  if (!std::isfinite(config_.resolution) || config_.resolution <= 0.0) {
    throw std::invalid_argument("IpmTransformer: resolution must be positive.");
  }

  if (config_.bev_width <= 0 || config_.bev_height <= 0) {
    throw std::invalid_argument("IpmTransformer: invalid BEV size.");
  }

  if (!std::isfinite(config_.x_min) || !std::isfinite(config_.y_min) ||
      !std::isfinite(config_.ground_z)) {
    throw std::invalid_argument("IpmTransformer: BEV bounds must be finite.");
  }

  if (!std::isfinite(config_.min_camera_z) || config_.min_camera_z <= 0.0) {
    throw std::invalid_argument(
        "IpmTransformer: min_camera_z must be positive.");
  }
}

void IpmTransformer::ValidateExtrinsics() const {
  if (!T_vehicle_camera_.matrix().allFinite()) {
    throw std::invalid_argument(
        "IpmTransformer: T_vehicle_camera must contain finite values.");
  }

  constexpr double kTransformTolerance = 1e-5;
  for (int col = 0; col < 3; ++col) {
    if (std::abs(T_vehicle_camera_.matrix()(3, col)) > kTransformTolerance) {
      throw std::invalid_argument(
          "IpmTransformer: T_vehicle_camera must be affine.");
    }
  }
  if (std::abs(T_vehicle_camera_.matrix()(3, 3) - 1.0) > kTransformTolerance) {
    throw std::invalid_argument(
        "IpmTransformer: T_vehicle_camera must be affine.");
  }

  const Eigen::Matrix3d& rotation = T_vehicle_camera_.linear();
  if ((rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
              .lpNorm<Eigen::Infinity>() > 1e-3 ||
      std::abs(rotation.determinant() - 1.0) > 1e-3) {
    throw std::invalid_argument(
        "IpmTransformer: rotation must be orthonormal and right-handed.");
  }
}

void IpmTransformer::PrecomputeLut() {
  const std::size_t map_size = static_cast<std::size_t>(config_.bev_width) *
                               static_cast<std::size_t>(config_.bev_height);
  map_x_.assign(map_size, -1.0f);
  map_y_.assign(map_size, -1.0f);
  map_valid_.assign(map_size, 0);

  const Eigen::Matrix3d& rotation = T_camera_vehicle_.linear();
  const Eigen::Vector3d& translation = T_camera_vehicle_.translation();

  for (int r = 0; r < config_.bev_height; ++r) {
    for (int c = 0; c < config_.bev_width; ++c) {
      // BEV pixel -> vehicle frame.
      //
      // Vehicle frame convention:
      // x: right positive
      // y: forward positive
      // z: up positive
      const double x_v =
          config_.x_min + static_cast<double>(c) * config_.resolution;
      const double y_v =
          config_.y_min +
          static_cast<double>(config_.bev_height - 1 - r) * config_.resolution;
      const double z_v = config_.ground_z;

      const Eigen::Vector3d point_camera =
          rotation * Eigen::Vector3d(x_v, y_v, z_v) + translation;

      const std::size_t map_index =
          static_cast<std::size_t>(r) *
              static_cast<std::size_t>(config_.bev_width) +
          static_cast<std::size_t>(c);

      if (point_camera.z() <= config_.min_camera_z) {
        continue;
      }

      Eigen::Vector2f pt_image;

      if (!camera_model_->RayToPixel(point_camera.cast<float>(), &pt_image)) {
        continue;
      }

      if (!pt_image.allFinite()) {
        continue;
      }
      map_x_[map_index] = pt_image.x();
      map_y_[map_index] = pt_image.y();
      map_valid_[map_index] = 1;
    }
  }
}

}  // namespace image
}  // namespace vision
}  // namespace wheel
