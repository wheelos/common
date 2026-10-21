// Copyright 2026 WheelOS All Rights Reserved.
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

#include <string>
#include <vector>

#include "modules/common/vehicle_calibration/proto/vehicle_calibration.pb.h"

namespace apollo {
namespace common {
namespace vehicle_calibration {

struct ResolvedSensorCalibration {
  SensorCalibration calibration;
  std::string resolved_extrinsic_file;
  std::string resolved_intrinsic_file;
  std::string resolved_optical_extrinsic_file;
  std::string resolved_metadata_file;
  std::string resolved_lidar_height_file;
};

struct ResolvedStaticTransform {
  StaticTransform calibration;
  std::string resolved_extrinsic_file;
};

class Registry {
 public:
  bool LoadFromEnvironment();
  bool LoadFromProfileKey(const std::string& profile_key);
  void Clear();

  const VehicleProfile& profile() const { return profile_; }
  const CalibrationManifest& manifest() const { return manifest_; }
  const std::string& profile_path() const { return profile_path_; }
  const std::string& bundle_path() const { return bundle_path_; }
  const std::vector<ResolvedSensorCalibration>& sensors() const {
    return sensors_;
  }
  const std::vector<ResolvedStaticTransform>& static_transforms() const {
    return static_transforms_;
  }
  const std::string& manifest_sha256() const { return manifest_sha256_; }
  const ResolvedSensorCalibration* FindSensor(
      const std::string& sensor_id_or_frame) const;
  bool ResolveFile(const std::string& relative_path,
                   std::string* resolved_path) const;
  static bool FileSha256(const std::string& path, std::string* digest);
  const std::string& resolved_lidar_height_file() const {
    return resolved_lidar_height_file_;
  }

  static bool ResolveBundleFile(const std::string& bundle_key,
                                const std::string& relative_path,
                                std::string* resolved_path);

 private:
  bool ValidateManifest();
  bool LoadProfile(const std::string& profile_key);
  bool ValidateStaticGraph();

  VehicleProfile profile_;
  CalibrationManifest manifest_;
  std::string profile_path_;
  std::string bundle_path_;
  std::vector<ResolvedSensorCalibration> sensors_;
  std::vector<ResolvedStaticTransform> static_transforms_;
  std::string manifest_sha256_;
  std::string resolved_lidar_height_file_;
};

}  // namespace vehicle_calibration
}  // namespace common
}  // namespace apollo
