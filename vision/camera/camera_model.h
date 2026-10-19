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

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace wheel {
namespace vision {

// Camera model type definition
enum class CameraModelType : uint8_t {
  kUnknown = 0,
  kPinhole = 1,
  kBrown = 2,
  kOmni = 3,
};

// Unified image size structure
struct ImageSize {
  uint32_t width = 0;
  uint32_t height = 0;
};

// OpenCV-compatible Brown-Conrady distortion coefficients ordered as:
// (k1, k2, p1, p2, k3).
struct BrownDistortionCoefficients {
  float k1 = 0.0f;
  float k2 = 0.0f;
  float p1 = 0.0f;
  float p2 = 0.0f;
  float k3 = 0.0f;

  inline Eigen::Matrix<float, 5, 1> ToVector() const {
    Eigen::Matrix<float, 5, 1> vector;
    vector << k1, k2, p1, p2, k3;
    return vector;
  }

  inline bool IsFinite() const {
    return std::isfinite(k1) && std::isfinite(k2) && std::isfinite(p1) &&
           std::isfinite(p2) && std::isfinite(k3);
  }
};

// OpenCV omnidir-compatible distortion coefficients ordered as:
// (k1, k2, p1, p2).
struct OmnidirectionalDistortionCoefficients {
  float k1 = 0.0f;
  float k2 = 0.0f;
  float p1 = 0.0f;
  float p2 = 0.0f;

  inline Eigen::Matrix<float, 4, 1> ToVector() const {
    Eigen::Matrix<float, 4, 1> vector;
    vector << k1, k2, p1, p2;
    return vector;
  }

  inline bool IsFinite() const {
    return std::isfinite(k1) && std::isfinite(k2) && std::isfinite(p1) &&
           std::isfinite(p2);
  }
};

class CameraModel {
 public:
  virtual ~CameraModel() = default;

  // Project a camera-frame point/ray to a 2D pixel. Only its direction matters.
  virtual bool RayToPixel(const Eigen::Vector3f& ray,
                          Eigen::Vector2f* pixel) const = 0;

  // Back-project a pixel to a camera-frame direction. Ray scale is model
  // dependent; callers should use direction only.
  virtual bool PixelToRay(const Eigen::Vector2f& pixel,
                          Eigen::Vector3f* ray) const = 0;

  // Return this model's ideal (zero-distortion) projection, preserving its
  // projection geometry and intrinsics.
  virtual std::shared_ptr<CameraModel> GetIdealModel() const = 0;

  virtual CameraModelType Type() const = 0;
  virtual std::string Name() const = 0;

  inline bool is_initialized() const { return initialized_; }
  inline ImageSize image_size() const { return {width_, height_}; }
  inline uint32_t width() const { return width_; }
  inline uint32_t height() const { return height_; }
  inline const Eigen::Matrix3f& intrinsic_matrix() const {
    return intrinsic_matrix_;
  }

  // Check if the projected ray falls within the valid image field of view (FOV)
  inline bool IsRayInsideFov(const Eigen::Vector3f& ray) const {
    Eigen::Vector2f pixel;
    if (!RayToPixel(ray, &pixel)) {
      return false;
    }
    return pixel.x() >= 0.0f && pixel.y() >= 0.0f &&
           pixel.x() < static_cast<float>(width_) &&
           pixel.y() < static_cast<float>(height_);
  }

 protected:
  static constexpr float kDefaultEpsilon = 1e-6f;

  inline void ResetBaseState() {
    width_ = 0;
    height_ = 0;
    intrinsic_matrix_ = Eigen::Matrix3f::Identity();
    initialized_ = false;
  }

  inline bool InitializeBase(uint32_t width, uint32_t height,
                             const Eigen::Matrix3f& intrinsics) {
    ResetBaseState();
    if (width == 0 || height == 0 || !intrinsics.allFinite()) {
      return false;
    }

    const float fx = intrinsics(0, 0);
    const float fy = intrinsics(1, 1);
    if (fx <= kDefaultEpsilon || fy <= kDefaultEpsilon) {
      return false;
    }

    if (std::abs(intrinsics(1, 0)) > kDefaultEpsilon ||
        std::abs(intrinsics(2, 0)) > kDefaultEpsilon ||
        std::abs(intrinsics(2, 1)) > kDefaultEpsilon ||
        std::abs(intrinsics(2, 2) - 1.0f) > 1e-4f) {
      return false;
    }

    width_ = width;
    height_ = height;
    intrinsic_matrix_ = intrinsics;
    initialized_ = true;
    return true;
  }

  uint32_t width_ = 0;
  uint32_t height_ = 0;
  Eigen::Matrix3f intrinsic_matrix_ = Eigen::Matrix3f::Identity();
  bool initialized_ = false;
};

using CameraModelPtr = std::shared_ptr<CameraModel>;
using CameraModelConstPtr = std::shared_ptr<const CameraModel>;

}  // namespace vision
}  // namespace wheel
