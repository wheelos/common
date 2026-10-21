// Copyright 2026 WheelOS. All Rights Reserved.
#include "modules/common/vehicle_calibration/lidar_height.h"

#include <cmath>
#include <string>

#include "yaml-cpp/yaml.h"

#include "cyber/common/log.h"

namespace apollo {
namespace common {
namespace vehicle_calibration {

bool LoadLidarHeight(const Registry& registry, const std::string& sensor,
                     LidarHeight* height) {
  const ResolvedSensorCalibration* selected = nullptr;
  if (!sensor.empty()) {
    selected = registry.FindSensor(sensor);
  } else {
    for (const auto& candidate : registry.sensors()) {
      if (candidate.calibration.sensor_type() == LIDAR) {
        if (selected != nullptr) {
          AERROR << "Multiple LiDARs require an explicit calibration sensor.";
          return false;
        }
        selected = &candidate;
      }
    }
  }
  if (height == nullptr || selected == nullptr ||
      selected->calibration.sensor_type() != LIDAR) {
    AERROR << "LiDAR height requires a selected LiDAR and output: " << sensor;
    return false;
  }
  const std::string& path = selected->resolved_lidar_height_file.empty() &&
                                    registry.manifest().schema_version() == 1
                                ? registry.resolved_lidar_height_file()
                                : selected->resolved_lidar_height_file;
  if (path.empty()) {
    AERROR << "Selected LiDAR has no ground-height calibration: " << sensor;
    return false;
  }
  try {
    const auto node = YAML::LoadFile(path)["vehicle"]["parameters"];
    LidarHeight candidate;
    candidate.frame_id = selected->calibration.frame_id();
    candidate.height = node["height"].as<double>();
    candidate.variance = node["height_var"].as<double>();
    if (!std::isfinite(candidate.height) || candidate.height <= 0.0 ||
        !std::isfinite(candidate.variance) || candidate.variance < 0.0) {
      AERROR << "Invalid LiDAR height/variance: " << path;
      return false;
    }
    *height = candidate;
    return true;
  } catch (const YAML::Exception& error) {
    AERROR << "Invalid LiDAR height " << path << ": " << error.what();
    return false;
  }
}

}  // namespace vehicle_calibration
}  // namespace common
}  // namespace apollo
