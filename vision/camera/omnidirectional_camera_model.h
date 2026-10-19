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

#include <memory>

#include "camera/camera_model.h"

namespace wheel {
namespace vision {

class OmnidirectionalCameraModel : public CameraModel {
 public:
  OmnidirectionalCameraModel() = default;
  ~OmnidirectionalCameraModel() override = default;

  // Initializes a self-implemented OpenCV omnidir-compatible Mei model with:
  // K = [fx, skew, cx; 0, fy, cy; 0, 0, 1], xi, D = (k1, k2, p1, p2).
  bool Init(uint32_t width, uint32_t height, const Eigen::Matrix3f& intrinsics,
            float xi, const Eigen::Matrix<float, 4, 1>& dist_coeffs);
  bool Init(uint32_t width, uint32_t height, const Eigen::Matrix3f& intrinsics,
            float xi, const OmnidirectionalDistortionCoefficients& dist_coeffs);

  bool RayToPixel(const Eigen::Vector3f& ray,
                  Eigen::Vector2f* pixel) const override;

  bool PixelToRay(const Eigen::Vector2f& pixel,
                  Eigen::Vector3f* ray) const override;

  std::shared_ptr<CameraModel> GetIdealModel() const override;

  CameraModelType Type() const override { return CameraModelType::kOmni; }
  std::string Name() const override { return "OmnidirectionalCameraModel"; }

  inline float xi() const { return xi_; }
  inline const Eigen::Matrix<float, 4, 1>& distort_params() const {
    return distort_params_;
  }

 private:
  float xi_ = 0.0f;
  Eigen::Matrix<float, 4, 1> distort_params_ =
      Eigen::Matrix<float, 4, 1>::Zero();
};

}  // namespace vision
}  // namespace wheel
