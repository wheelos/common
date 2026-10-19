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

#include "camera/brown_camera_model.h"

#include <cmath>

#include "camera/pinhole_camera_model.h"

namespace wheel {
namespace vision {

namespace {
constexpr float kEpsilon = 1e-6f;
constexpr int kMaxIterations = 20;
constexpr float kConvergenceEps = 1e-6f;

bool DistortBrownPoint(const Eigen::Vector2f& undistorted,
                       const Eigen::Matrix<float, 5, 1>& dist_coeffs,
                       Eigen::Vector2f* distorted) {
  if (distorted == nullptr || !undistorted.allFinite() ||
      !dist_coeffs.allFinite()) {
    return false;
  }

  const float x = undistorted.x();
  const float y = undistorted.y();
  const float r2 = x * x + y * y;
  const float r4 = r2 * r2;
  const float r6 = r4 * r2;

  const float k1 = dist_coeffs[0];
  const float k2 = dist_coeffs[1];
  const float p1 = dist_coeffs[2];
  const float p2 = dist_coeffs[3];
  const float k3 = dist_coeffs[4];

  const float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
  distorted->x() = x * radial + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
  distorted->y() = y * radial + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;

  return distorted->allFinite();
}
}  // namespace

bool BrownCameraModel::Init(uint32_t width, uint32_t height,
                            const Eigen::Matrix3f& intrinsics,
                            const Eigen::Matrix<float, 5, 1>& dist_coeffs) {
  if (!InitializeBase(width, height, intrinsics) || !dist_coeffs.allFinite()) {
    ResetBaseState();
    distort_params_.setZero();
    return false;
  }

  distort_params_ = dist_coeffs;
  return true;
}

bool BrownCameraModel::Init(uint32_t width, uint32_t height,
                            const Eigen::Matrix3f& intrinsics,
                            const BrownDistortionCoefficients& dist_coeffs) {
  if (!dist_coeffs.IsFinite()) {
    ResetBaseState();
    distort_params_.setZero();
    return false;
  }

  return Init(width, height, intrinsics, dist_coeffs.ToVector());
}

bool BrownCameraModel::RayToPixel(const Eigen::Vector3f& ray,
                                  Eigen::Vector2f* pixel) const {
  if (pixel == nullptr || !is_initialized() || !ray.allFinite() ||
      ray.z() <= kEpsilon) {
    return false;
  }

  Eigen::Vector2f distorted;
  if (!DistortBrownPoint(Eigen::Vector2f(ray.x() / ray.z(), ray.y() / ray.z()),
                         distort_params_, &distorted)) {
    return false;
  }

  pixel->x() = intrinsic_matrix_(0, 0) * distorted.x() +
               intrinsic_matrix_(0, 1) * distorted.y() +
               intrinsic_matrix_(0, 2);
  pixel->y() =
      intrinsic_matrix_(1, 1) * distorted.y() + intrinsic_matrix_(1, 2);

  return pixel->allFinite();
}

bool BrownCameraModel::PixelToRay(const Eigen::Vector2f& pixel,
                                  Eigen::Vector3f* ray) const {
  if (ray == nullptr || !is_initialized() || !pixel.allFinite()) {
    return false;
  }

  const float fx = intrinsic_matrix_(0, 0);
  const float fy = intrinsic_matrix_(1, 1);
  if (fx <= kEpsilon || fy <= kEpsilon) {
    return false;
  }

  const float y_dist = (pixel.y() - intrinsic_matrix_(1, 2)) / fy;
  const float x_dist =
      (pixel.x() - intrinsic_matrix_(0, 2) - intrinsic_matrix_(0, 1) * y_dist) /
      fx;

  const float k1 = distort_params_[0];
  const float k2 = distort_params_[1];
  const float p1 = distort_params_[2];
  const float p2 = distort_params_[3];
  const float k3 = distort_params_[4];

  float x_iter = x_dist;
  float y_iter = y_dist;
  bool converged = false;

  for (int i = 0; i < kMaxIterations; ++i) {
    const float r2 = x_iter * x_iter + y_iter * y_iter;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    const float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
    if (std::abs(radial) <= kEpsilon) {
      return false;
    }

    const float delta_x =
        2.0f * p1 * x_iter * y_iter + p2 * (r2 + 2.0f * x_iter * x_iter);
    const float delta_y =
        p1 * (r2 + 2.0f * y_iter * y_iter) + 2.0f * p2 * x_iter * y_iter;

    const float x_next = (x_dist - delta_x) / radial;
    const float y_next = (y_dist - delta_y) / radial;
    if (!std::isfinite(x_next) || !std::isfinite(y_next)) {
      return false;
    }

    if (std::abs(x_next - x_iter) < kConvergenceEps &&
        std::abs(y_next - y_iter) < kConvergenceEps) {
      x_iter = x_next;
      y_iter = y_next;
      converged = true;
      break;
    }

    x_iter = x_next;
    y_iter = y_next;
  }

  if (!converged) {
    return false;
  }

  ray->x() = x_iter;
  ray->y() = y_iter;
  ray->z() = 1.0f;
  return ray->allFinite();
}

std::shared_ptr<CameraModel> BrownCameraModel::GetIdealModel() const {
  if (!is_initialized()) {
    return nullptr;
  }

  auto model = std::make_shared<PinholeCameraModel>();
  return model->Init(width_, height_, intrinsic_matrix_) ? model : nullptr;
}

}  // namespace vision
}  // namespace wheel
