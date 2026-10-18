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

#include "camera/omnidirectional_camera_model.h"

#include <cmath>

#include "camera/pinhole_camera_model.h"

namespace wheel {
namespace vision {

namespace {
constexpr float kEpsilon = 1e-6f;
constexpr int kMaxIterations = 20;
constexpr float kConvergenceEps = 1e-6f;

bool DistortOmnidirectionalPoint(const Eigen::Vector2f& undistorted,
                                 const Eigen::Matrix<float, 4, 1>& dist_coeffs,
                                 Eigen::Vector2f* distorted) {
  if (distorted == nullptr || !undistorted.allFinite() ||
      !dist_coeffs.allFinite()) {
    return false;
  }

  const float x = undistorted.x();
  const float y = undistorted.y();
  const float r2 = x * x + y * y;
  const float r4 = r2 * r2;
  const float k1 = dist_coeffs[0];
  const float k2 = dist_coeffs[1];
  const float p1 = dist_coeffs[2];
  const float p2 = dist_coeffs[3];

  const float radial = 1.0f + k1 * r2 + k2 * r4;
  distorted->x() = x * radial + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
  distorted->y() = y * radial + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;

  return distorted->allFinite();
}
}  // namespace

bool OmnidirectionalCameraModel::Init(
    uint32_t width, uint32_t height, const Eigen::Matrix3f& intrinsics,
    float xi, const Eigen::Matrix<float, 4, 1>& dist_coeffs) {
  if (!InitializeBase(width, height, intrinsics) || !std::isfinite(xi) ||
      xi < 0.0f || !dist_coeffs.allFinite()) {
    ResetBaseState();
    xi_ = 0.0f;
    distort_params_.setZero();
    return false;
  }

  xi_ = xi;
  distort_params_ = dist_coeffs;
  return true;
}

bool OmnidirectionalCameraModel::Init(
    uint32_t width, uint32_t height, const Eigen::Matrix3f& intrinsics,
    float xi, const OmnidirectionalDistortionCoefficients& dist_coeffs) {
  if (!dist_coeffs.IsFinite()) {
    ResetBaseState();
    xi_ = 0.0f;
    distort_params_.setZero();
    return false;
  }

  return Init(width, height, intrinsics, xi, dist_coeffs.ToVector());
}

bool OmnidirectionalCameraModel::RayToPixel(const Eigen::Vector3f& ray,
                                            Eigen::Vector2f* pixel) const {
  if (pixel == nullptr || !is_initialized() || !ray.allFinite()) {
    return false;
  }

  const float ray_norm = ray.norm();
  if (ray_norm <= kEpsilon) {
    return false;
  }

  const Eigen::Vector3f sphere = ray / ray_norm;
  const float denominator = sphere.z() + xi_;
  if (denominator <= kEpsilon) {
    return false;
  }

  Eigen::Vector2f distorted;
  if (!DistortOmnidirectionalPoint(
          Eigen::Vector2f(sphere.x() / denominator, sphere.y() / denominator),
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

bool OmnidirectionalCameraModel::PixelToRay(const Eigen::Vector2f& pixel,
                                            Eigen::Vector3f* ray) const {
  if (ray == nullptr || !is_initialized() || !pixel.allFinite()) {
    return false;
  }

  const float fx = intrinsic_matrix_(0, 0);
  const float fy = intrinsic_matrix_(1, 1);
  if (fx <= kEpsilon || fy <= kEpsilon) {
    return false;
  }

  const float y_plane = (pixel.y() - intrinsic_matrix_(1, 2)) / fy;
  const float x_plane = (pixel.x() - intrinsic_matrix_(0, 2) -
                         intrinsic_matrix_(0, 1) * y_plane) /
                        fx;

  const float k1 = distort_params_[0];
  const float k2 = distort_params_[1];
  const float p1 = distort_params_[2];
  const float p2 = distort_params_[3];

  float x_undistorted = x_plane;
  float y_undistorted = y_plane;
  bool converged = false;
  for (int i = 0; i < kMaxIterations; ++i) {
    const float r2 =
        x_undistorted * x_undistorted + y_undistorted * y_undistorted;
    const float r4 = r2 * r2;
    const float radial = 1.0f + k1 * r2 + k2 * r4;
    if (std::abs(radial) <= kEpsilon) {
      return false;
    }

    const float next_x = (x_plane - 2.0f * p1 * x_undistorted * y_undistorted -
                          p2 * (r2 + 2.0f * x_undistorted * x_undistorted)) /
                         radial;
    const float next_y =
        (y_plane - p1 * (r2 + 2.0f * y_undistorted * y_undistorted) -
         2.0f * p2 * x_undistorted * y_undistorted) /
        radial;
    if (!std::isfinite(next_x) || !std::isfinite(next_y)) {
      return false;
    }

    if (std::abs(next_x - x_undistorted) < kConvergenceEps &&
        std::abs(next_y - y_undistorted) < kConvergenceEps) {
      x_undistorted = next_x;
      y_undistorted = next_y;
      converged = true;
      break;
    }

    x_undistorted = next_x;
    y_undistorted = next_y;
  }

  if (!converged) {
    return false;
  }

  const float r2 =
      x_undistorted * x_undistorted + y_undistorted * y_undistorted;
  const float a = r2 + 1.0f;
  const float b = 2.0f * xi_ * r2;
  const float c = r2 * xi_ * xi_ - 1.0f;
  const float discriminant = b * b - 4.0f * a * c;
  if (discriminant < 0.0f) {
    return false;
  }

  const float zs = (-b + std::sqrt(discriminant)) / (2.0f * a);
  const float scale = zs + xi_;
  if (scale <= kEpsilon) {
    return false;
  }

  ray->x() = x_undistorted * scale;
  ray->y() = y_undistorted * scale;
  ray->z() = zs;

  const float ray_norm = ray->norm();
  if (ray_norm <= kEpsilon) {
    return false;
  }

  *ray /= ray_norm;
  return true;
}

std::shared_ptr<CameraModel> OmnidirectionalCameraModel::GetIdealModel() const {
  if (!is_initialized()) {
    return nullptr;
  }

  auto model = std::make_shared<PinholeCameraModel>();
  return model->Init(width_, height_, intrinsic_matrix_) ? model : nullptr;
}

}  // namespace vision
}  // namespace wheel
