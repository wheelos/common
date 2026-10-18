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
#include <cstddef>
#include <cstdint>
#include <limits>

#include "pointcloud/memory/buffer.h"

namespace wheel {
namespace vision {
namespace lidar {

struct RangeImageConfig {
  uint32_t rows = 0U;
  uint32_t cols = 0U;
  float min_range_m = 0.0f;
  float max_range_m = 0.0f;
  float min_azimuth_rad = 0.0f;
  float max_azimuth_rad = 0.0f;
  float min_elevation_rad = 0.0f;
  float max_elevation_rad = 0.0f;

  inline bool IsValid() const {
    const uint64_t pixels = static_cast<uint64_t>(rows) * cols;
    constexpr float kPi = 3.14159265358979323846f;
    return rows > 0U && cols > 0U &&
           pixels <= std::numeric_limits<uint32_t>::max() &&
           pixels <= std::numeric_limits<size_t>::max() / sizeof(uint64_t) &&
           std::isfinite(min_range_m) && std::isfinite(max_range_m) &&
           std::isfinite(min_azimuth_rad) && std::isfinite(max_azimuth_rad) &&
           std::isfinite(min_elevation_rad) &&
           std::isfinite(max_elevation_rad) && min_range_m >= 0.0f &&
           max_range_m > min_range_m && max_azimuth_rad > min_azimuth_rad &&
           min_azimuth_rad >= -kPi && max_azimuth_rad <= kPi &&
           max_elevation_rad > min_elevation_rad &&
           min_elevation_rad >= -kPi / 2.0f && max_elevation_rad <= kPi / 2.0f;
  }

  // Returns zero for unrepresentable dimensions; IsValid checks the full
  // config.
  inline uint32_t pixel_count() const {
    const uint64_t pixels = static_cast<uint64_t>(rows) * cols;
    return pixels <= std::numeric_limits<uint32_t>::max()
               ? static_cast<uint32_t>(pixels)
               : 0U;
  }
};

class RangeImageBuffer {
 public:
  // Failed allocation leaves the previous allocation unchanged.
  bool Allocate(const RangeImageConfig& config,
                MemoryType memory_type = MemoryType::kUnified);

  inline float* range() { return range_.mutable_data<float>(); }
  inline const float* range() const { return range_.data<float>(); }

  inline float* intensity() { return intensity_.mutable_data<float>(); }
  inline const float* intensity() const { return intensity_.data<float>(); }

  inline uint8_t* valid_mask() { return valid_mask_.mutable_data<uint8_t>(); }
  inline const uint8_t* valid_mask() const {
    return valid_mask_.data<uint8_t>();
  }

  inline uint32_t rows() const { return rows_; }
  inline uint32_t cols() const { return cols_; }
  inline uint32_t pixel_count() const { return rows_ * cols_; }
  inline MemoryType memory_type() const { return range_.memory_type(); }

 private:
  uint32_t rows_ = 0U;
  uint32_t cols_ = 0U;
  Buffer range_;
  Buffer intensity_;
  Buffer valid_mask_;
};

class RangeImageWorkspace {
 public:
  // Failed allocation leaves the previous allocation unchanged.
  bool Allocate(const RangeImageConfig& config,
                MemoryType memory_type = MemoryType::kUnified);

  inline uint64_t* packed_range_intensity() {
    return packed_range_intensity_.mutable_data<uint64_t>();
  }

  inline const uint64_t* packed_range_intensity() const {
    return packed_range_intensity_.data<uint64_t>();
  }

  inline uint32_t pixel_count() const { return pixel_count_; }
  inline MemoryType memory_type() const {
    return packed_range_intensity_.memory_type();
  }

 private:
  uint32_t pixel_count_ = 0U;
  Buffer packed_range_intensity_;
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
