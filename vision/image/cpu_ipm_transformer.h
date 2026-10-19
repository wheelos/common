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

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "camera/camera_model.h"
#include "image/image.h"

namespace wheel {
namespace vision {
namespace image {

struct IpmConfig {
  double resolution{0.05};

  int bev_width{400};
  int bev_height{600};

  // Vehicle frame:
  // x: right positive
  // y: forward positive
  // z: up positive
  double x_min{-10.0};
  double y_min{3.0};

  // Ground plane height in vehicle frame.
  // Usually z = 0 if vehicle frame origin is on ground.
  double ground_z{0.0};
};

enum class InterpolationMode {
  kNearest,
  kBilinear,
};

class CpuIpmTransformer {
 public:
  // Builds a CPU-side BEV-to-camera lookup table from the camera model.
  // T_vehicle_camera maps camera-frame coordinates into the right-handed
  // vehicle frame (x right, y forward, z up).
  CpuIpmTransformer(std::shared_ptr<const CameraModel> camera_model,
                    const Eigen::Isometry3d& T_vehicle_camera,
                    const IpmConfig& config);

  // Returns the projected source-image pixel for a BEV coordinate. Coordinates
  // outside the BEV or failed projections return false.
  bool MapPixel(int bev_x, int bev_y, Eigen::Vector2f* image_pixel) const;

  int bev_width() const { return config_.bev_width; }
  int bev_height() const { return config_.bev_height; }

  // Resamples CPU-resident, interleaved images using the precomputed mapping.
  // Source and destination storage must not overlap. Integral output is rounded
  // and saturated.
  template <typename Scalar>
  void Transform(
      const ImageView<const Scalar>& src_img,
      const ImageView<Scalar>& dst_bev_img,
      InterpolationMode interpolation = InterpolationMode::kBilinear) const;

 private:
  void ValidateInputs() const;
  void ValidateExtrinsics() const;
  void PrecomputeLut(const CameraModel& camera_model);

  ImageSize camera_image_size_;
  Eigen::Isometry3d T_vehicle_camera_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_camera_vehicle_ = Eigen::Isometry3d::Identity();
  IpmConfig config_;
  std::vector<float> map_x_;
  std::vector<float> map_y_;
  std::vector<unsigned char> map_valid_;
};

namespace internal {
template <typename Scalar>
Scalar SaturateSample(double value) {
  if constexpr (std::is_integral<Scalar>::value) {
    long double rounded = std::round(static_cast<long double>(value));
    rounded = std::max(rounded, static_cast<long double>(
                                    std::numeric_limits<Scalar>::lowest()));
    rounded = std::min(
        rounded, static_cast<long double>(std::numeric_limits<Scalar>::max()));
    return static_cast<Scalar>(rounded);
  }
  return static_cast<Scalar>(value);
}

template <typename Scalar>
double ReadSample(const ImageView<const Scalar>& image, int x, int y,
                  int channel) {
  if (x < 0 || x >= image.width || y < 0 || y >= image.height) {
    return 0.0;
  }
  const std::size_t index =
      static_cast<std::size_t>(y) * image.RowStride() +
      static_cast<std::size_t>(x) * static_cast<std::size_t>(image.channels) +
      static_cast<std::size_t>(channel);
  return static_cast<double>(image.data[index]);
}

}  // namespace internal

template <typename Scalar>
void CpuIpmTransformer::Transform(const ImageView<const Scalar>& src_img,
                                  const ImageView<Scalar>& dst_bev_img,
                                  InterpolationMode interpolation) const {
  static_assert(std::is_arithmetic<Scalar>::value,
                "Image scalar type must be arithmetic.");
  if (!internal::IsValidImageView(src_img) ||
      !internal::IsValidImageView(dst_bev_img)) {
    throw std::invalid_argument(
        "CpuIpmTransformer::Transform: invalid image view.");
  }
  if (internal::ImageViewsOverlap(src_img, dst_bev_img)) {
    throw std::invalid_argument(
        "CpuIpmTransformer::Transform: image views must not overlap.");
  }
  if (src_img.width != static_cast<int>(camera_image_size_.width) ||
      src_img.height != static_cast<int>(camera_image_size_.height)) {
    throw std::invalid_argument(
        "CpuIpmTransformer::Transform: source size does not match camera "
        "model.");
  }
  if (dst_bev_img.width != config_.bev_width ||
      dst_bev_img.height != config_.bev_height ||
      dst_bev_img.channels != src_img.channels) {
    throw std::invalid_argument(
        "CpuIpmTransformer::Transform: destination size or channels mismatch.");
  }
  if (interpolation != InterpolationMode::kNearest &&
      interpolation != InterpolationMode::kBilinear) {
    throw std::invalid_argument(
        "CpuIpmTransformer::Transform: unsupported interpolation mode.");
  }

  for (int y = 0; y < config_.bev_height; ++y) {
    Scalar* dst_row = dst_bev_img.data +
                      static_cast<std::size_t>(y) * dst_bev_img.RowStride();
    for (int x = 0; x < config_.bev_width; ++x) {
      const std::size_t map_index =
          static_cast<std::size_t>(y) *
              static_cast<std::size_t>(config_.bev_width) +
          static_cast<std::size_t>(x);
      const float source_x = map_x_[map_index];
      const float source_y = map_y_[map_index];
      const bool within_sample_bounds =
          interpolation == InterpolationMode::kNearest
              ? source_x >= -0.5f &&
                    source_x < static_cast<float>(src_img.width) - 0.5f &&
                    source_y >= -0.5f &&
                    source_y < static_cast<float>(src_img.height) - 0.5f
              : source_x > -1.0f &&
                    source_x < static_cast<float>(src_img.width) &&
                    source_y > -1.0f &&
                    source_y < static_cast<float>(src_img.height);
      for (int channel = 0; channel < src_img.channels; ++channel) {
        double value = 0.0;
        if (map_valid_[map_index] != 0 && within_sample_bounds) {
          if (interpolation == InterpolationMode::kNearest) {
            value = internal::ReadSample(
                src_img, static_cast<int>(std::floor(source_x + 0.5f)),
                static_cast<int>(std::floor(source_y + 0.5f)), channel);
          } else {
            const int x0 = static_cast<int>(std::floor(source_x));
            const int y0 = static_cast<int>(std::floor(source_y));
            const double dx = source_x - static_cast<float>(x0);
            const double dy = source_y - static_cast<float>(y0);
            const double top =
                internal::ReadSample(src_img, x0, y0, channel) * (1.0 - dx) +
                internal::ReadSample(src_img, x0 + 1, y0, channel) * dx;
            const double bottom =
                internal::ReadSample(src_img, x0, y0 + 1, channel) *
                    (1.0 - dx) +
                internal::ReadSample(src_img, x0 + 1, y0 + 1, channel) * dx;
            value = top * (1.0 - dy) + bottom * dy;
          }
        }
        dst_row[static_cast<std::size_t>(x) *
                    static_cast<std::size_t>(dst_bev_img.channels) +
                static_cast<std::size_t>(channel)] =
            internal::SaturateSample<Scalar>(value);
      }
    }
  }
}

}  // namespace image
}  // namespace vision
}  // namespace wheel
