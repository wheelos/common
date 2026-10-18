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

class BrownCameraModel : public CameraModel {
 public:
  BrownCameraModel() = default;
  ~BrownCameraModel() override = default;

  // Initializes the model with OpenCV-compatible Brown-Conrady coefficients:
  // (k1, k2, p1, p2, k3).
  bool Init(uint32_t width, uint32_t height, const Eigen::Matrix3f& intrinsics,
            const Eigen::Matrix<float, 5, 1>& dist_coeffs);
  bool Init(uint32_t width, uint32_t height, const Eigen::Matrix3f& intrinsics,
            const BrownDistortionCoefficients& dist_coeffs);

  bool RayToPixel(const Eigen::Vector3f& ray,
                  Eigen::Vector2f* pixel) const override;

  bool PixelToRay(const Eigen::Vector2f& pixel,
                  Eigen::Vector3f* ray) const override;

  std::shared_ptr<CameraModel> GetIdealModel() const override;

  CameraModelType Type() const override { return CameraModelType::kBrown; }
  std::string Name() const override { return "BrownCameraModel"; }

  inline const Eigen::Matrix<float, 5, 1>& distort_params() const {
    return distort_params_;
  }

 private:
  Eigen::Matrix<float, 5, 1> distort_params_ =
      Eigen::Matrix<float, 5, 1>::Zero();
};

}  // namespace vision
}  // namespace wheel
