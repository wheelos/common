// Copyright 2026 WheelOS. All Rights Reserved.
#pragma once

#include <string>

#include "modules/common/vehicle_calibration/registry.h"

namespace apollo {
namespace common {
namespace vehicle_calibration {

struct LidarHeight {
  double height = 0.0;
  double variance = 0.0;
  std::string frame_id;
};

// An empty sensor selects the only installed LiDAR; ambiguity is an error.
bool LoadLidarHeight(const Registry& registry, const std::string& sensor,
                     LidarHeight* height);

}  // namespace vehicle_calibration
}  // namespace common
}  // namespace apollo
