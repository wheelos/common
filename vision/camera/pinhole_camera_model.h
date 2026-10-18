#pragma once

#include <memory>

#include "camera/camera_model.h"

namespace wheel {
namespace vision {

class PinholeCameraModel : public CameraModel {
 public:
  PinholeCameraModel() = default;
  ~PinholeCameraModel() override = default;

  // Initializes the model with an OpenCV-compatible camera matrix:
  // K = [fx, skew, cx; 0, fy, cy; 0, 0, 1].
  bool Init(uint32_t width, uint32_t height, const Eigen::Matrix3f& intrinsics);

  bool RayToPixel(const Eigen::Vector3f& ray,
                  Eigen::Vector2f* pixel) const override;

  bool PixelToRay(const Eigen::Vector2f& pixel,
                  Eigen::Vector3f* ray) const override;

  std::shared_ptr<CameraModel> GetIdealModel() const override;

  CameraModelType Type() const override { return CameraModelType::kPinhole; }
  std::string Name() const override { return "PinholeCameraModel"; }
};

}  // namespace vision
}  // namespace wheel
