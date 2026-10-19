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

#include <cstddef>
#include <cstdint>

namespace wheel {
namespace vision {
namespace lidar {

constexpr uint32_t kMaxPointFields = 8;

enum class MemoryType : uint8_t {
  kHost = 0,
  kPinnedHost = 1,
  kDevice = 2,
  kUnified = 3,
};

enum class PointLayout : uint8_t {
  kInterleaved = 0,
  kPlanar = 1,
};

enum class ScalarType : uint8_t {
  kUnknown = 0,
  kFloat32 = 1,
  kUint8 = 2,
  kUint16 = 3,
  kUint32 = 4,
};

enum class CoordinateFrame : uint8_t {
  kLidar = 0,
  kVehicle = 1,
  kWorld = 2,
};

enum class PointField : uint8_t {
  kX = 0,
  kY = 1,
  kZ = 2,
  kIntensity = 3,
  kRing = 4,
  kTimestamp = 5,
};

inline constexpr size_t ScalarTypeSize(ScalarType scalar_type) {
  switch (scalar_type) {
    case ScalarType::kFloat32:
      return sizeof(float);
    case ScalarType::kUint8:
      return sizeof(uint8_t);
    case ScalarType::kUint16:
      return sizeof(uint16_t);
    case ScalarType::kUint32:
      return sizeof(uint32_t);
    case ScalarType::kUnknown:
    default:
      return 0U;
  }
}

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
