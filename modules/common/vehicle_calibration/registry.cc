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

#include "modules/common/vehicle_calibration/registry.h"

#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <regex>
#include <string>
#include <unordered_set>
#include <utility>

#include "cyber/common/file.h"
#include "cyber/common/log.h"
#include "cyber/common/resource_manager.h"

namespace apollo {
namespace common {
namespace vehicle_calibration {
namespace {

namespace fs = std::filesystem;

constexpr char kProfileKeyEnv[] = "WHEELOS_VEHICLE_PROFILE_KEY";
constexpr char kAllowTestCalibrationEnv[] = "WHEELOS_ALLOW_TEST_CALIBRATION";
constexpr char kManifestFile[] = "calibration_manifest.pb.txt";
constexpr char kBaseFrame[] = "base_link";

bool IsSafeIdentifier(const std::string& value) {
  static const std::regex pattern("^[A-Za-z0-9_-]+$");
  return std::regex_match(value, pattern);
}

bool IsSafeRelativePath(const std::string& value) {
  if (value.empty()) {
    return false;
  }
  const fs::path path(value);
  if (path.is_absolute()) {
    return false;
  }
  for (const auto& component : path) {
    if (component == ".." || component == ".") {
      return false;
    }
  }
  return true;
}

bool IsReadableFile(const std::string& path) {
  std::error_code error;
  if (!fs::is_regular_file(path, error) || error) {
    return false;
  }
  std::ifstream file(path);
  return file.good();
}

bool IsCameraFrameId(const std::string& frame_id) {
  static const std::unordered_set<std::string> kCameraFrames = {
      "camera_front_link", "camera_front_left_link", "camera_front_right_link",
      "camera_left_link",  "camera_right_link",      "camera_rear_left_link",
      "camera_rear_link",  "camera_rear_right_link"};
  return kCameraFrames.count(frame_id) != 0;
}

bool IsCanonicalFrame(const SensorCalibration& sensor) {
  if (sensor.sensor_type() == IMU) {
    return sensor.frame_id() == "imu_link";
  }
  if (sensor.sensor_type() == GNSS) {
    return sensor.frame_id() == "gnss_antenna_link" ||
           sensor.frame_id() == "gnss_antenna_secondary_link";
  }
  if (sensor.sensor_type() == LIDAR) {
    return sensor.frame_id() == "lidar_top_link";
  }
  if (sensor.sensor_type() != CAMERA || !IsCameraFrameId(sensor.frame_id())) {
    return false;
  }
  return sensor.optical_frame_id() ==
         sensor.frame_id().substr(0, sensor.frame_id().size() - 5) +
             "_optical_frame";
}

bool IsStandardFrameId(const std::string& frame_id) {
  if (frame_id == "imu_link" || frame_id == "lidar_top_link" ||
      frame_id == "gnss_antenna_link" ||
      frame_id == "gnss_antenna_secondary_link") {
    return true;
  }
  return IsCameraFrameId(frame_id);
}

bool ResolveOptionalBundleFile(const std::string& bundle_key,
                               const std::string& relative_path,
                               std::string* resolved_path) {
  if (relative_path.empty()) {
    resolved_path->clear();
    return true;
  }
  return Registry::ResolveBundleFile(bundle_key, relative_path, resolved_path);
}

}  // namespace

void Registry::Clear() {
  profile_.Clear();
  manifest_.Clear();
  profile_path_.clear();
  bundle_path_.clear();
  sensors_.clear();
  resolved_lidar_height_file_.clear();
}

bool Registry::LoadFromEnvironment() {
  Clear();
  const char* profile_key = std::getenv(kProfileKeyEnv);
  if (profile_key == nullptr || *profile_key == '\0') {
    AERROR << kProfileKeyEnv << " is not set.";
    return false;
  }
  return LoadFromProfileKey(profile_key);
}

bool Registry::LoadFromProfileKey(const std::string& profile_key) {
  Clear();
  const auto fail = [this]() {
    Clear();
    return false;
  };
  if (!IsSafeRelativePath(profile_key)) {
    AERROR << "Invalid vehicle profile asset key: " << profile_key;
    return fail();
  }
  if (!cyber::common::ResourceManager::ResolveAssetPath(profile_key,
                                                        &profile_path_)) {
    AERROR << "Unable to resolve vehicle profile asset: " << profile_key;
    return fail();
  }
  if (!cyber::common::GetProtoFromASCIIFile(profile_path_, &profile_)) {
    AERROR << "Failed to parse vehicle profile: " << profile_path_;
    return fail();
  }
  const char* allow_test_calibration =
      std::getenv(kAllowTestCalibrationEnv);
  if (profile_.test_only() &&
      (allow_test_calibration == nullptr ||
       std::string(allow_test_calibration) != "1")) {
    AERROR << "Refusing test-only vehicle calibration profile: "
           << profile_path_;
    return fail();
  }
  if (!profile_.has_vehicle_type() ||
      !IsSafeIdentifier(profile_.vehicle_type()) ||
      !profile_.has_calibration_version() ||
      !IsSafeIdentifier(profile_.calibration_version())) {
    AERROR << "Vehicle profile must specify safe vehicle_type and "
              "calibration_version fields: "
           << profile_path_;
    return fail();
  }
  const std::string expected_profile_key =
      "vehicles/" + profile_.vehicle_type() + "/config/vehicle_profile.pb.txt";
  if (profile_key != expected_profile_key) {
    AERROR << "Vehicle profile key must match its vehicle_type: "
           << profile_key;
    return fail();
  }

  const std::string bundle_key = "vehicles/" + profile_.vehicle_type() +
                                 "/calibration/" +
                                 profile_.calibration_version();
  if (!cyber::common::ResourceManager::ResolveAssetPath(bundle_key,
                                                        &bundle_path_)) {
    AERROR << "Unable to resolve calibration bundle: " << bundle_key;
    return fail();
  }
  const std::string manifest_path =
      (fs::path(bundle_path_) / kManifestFile).string();
  if (!cyber::common::GetProtoFromASCIIFile(manifest_path, &manifest_)) {
    AERROR << "Failed to parse calibration manifest: " << manifest_path;
    return fail();
  }
  if (!manifest_.has_calibration_version() ||
      manifest_.calibration_version() != profile_.calibration_version()) {
    AERROR << "Calibration manifest version does not match selected profile: "
           << manifest_path;
    return fail();
  }
  if (!ValidateManifest()) {
    return fail();
  }

  AINFO << "[CALIBRATION] vehicle=" << profile_.vehicle_type()
        << " version=" << profile_.calibration_version()
        << " profile=" << profile_path_ << " bundle=" << bundle_path_;
  for (const auto& sensor : sensors_) {
    AINFO << "[CALIBRATION] sensor=" << sensor.calibration.sensor_id()
          << " frame=" << sensor.calibration.frame_id()
          << " extrinsic=" << sensor.resolved_extrinsic_file;
    if (!sensor.resolved_intrinsic_file.empty()) {
      AINFO << "[CALIBRATION] sensor=" << sensor.calibration.sensor_id()
            << " intrinsic=" << sensor.resolved_intrinsic_file;
    }
    if (!sensor.resolved_optical_extrinsic_file.empty()) {
      AINFO << "[CALIBRATION] sensor=" << sensor.calibration.sensor_id()
            << " optical_extrinsic="
            << sensor.resolved_optical_extrinsic_file;
    }
  }
  if (!resolved_lidar_height_file_.empty()) {
    AINFO << "[CALIBRATION] LiDAR height=" << resolved_lidar_height_file_;
  }
  return true;
}

bool Registry::ResolveBundleFile(const std::string& bundle_key,
                                 const std::string& relative_path,
                                 std::string* resolved_path) {
  if (resolved_path == nullptr || !IsSafeRelativePath(bundle_key) ||
      !IsSafeRelativePath(relative_path)) {
    AERROR << "Invalid calibration bundle/file key: " << bundle_key << "/"
           << relative_path;
    return false;
  }
  resolved_path->clear();
  const std::string key = (fs::path(bundle_key) / relative_path).string();
  if (!cyber::common::ResourceManager::ResolveAssetPath(key, resolved_path) ||
      !cyber::common::PathExists(*resolved_path) ||
      !IsReadableFile(*resolved_path)) {
    AERROR << "Calibration asset is missing or unreadable: " << key;
    resolved_path->clear();
    return false;
  }
  return true;
}

bool Registry::ValidateManifest() {
  std::unordered_set<std::string> ids;
  std::unordered_set<std::string> frames;
  std::unordered_set<std::string> optical_frames;
  std::unordered_set<std::string> required_frames;
  bool has_imu = false;
  bool has_primary_gnss = false;
  bool has_secondary_gnss = false;
  bool has_lidar = false;
  sensors_.reserve(manifest_.sensor_size());

  if (profile_.required_sensor_frame_id().empty()) {
    AERROR << "Vehicle profile must declare required_sensor_frame_id entries.";
    return false;
  }
  for (const auto& frame_id : profile_.required_sensor_frame_id()) {
    if (!IsStandardFrameId(frame_id) ||
        !required_frames.insert(frame_id).second) {
      AERROR << "Invalid or duplicate required sensor frame: " << frame_id;
      return false;
    }
  }

  const std::string bundle_key = "vehicles/" + profile_.vehicle_type() +
                                 "/calibration/" +
                                 profile_.calibration_version();
  for (const auto& sensor : manifest_.sensor()) {
    if (!sensor.has_sensor_id() || sensor.sensor_id().empty() ||
        !sensor.has_sensor_type() || !sensor.has_parent_frame_id() ||
        sensor.parent_frame_id() != kBaseFrame || !sensor.has_frame_id() ||
        !IsCanonicalFrame(sensor) || !sensor.has_extrinsic_file() ||
        !ids.insert(sensor.sensor_id()).second ||
        !frames.insert(sensor.frame_id()).second) {
      AERROR << "Invalid or duplicate sensor calibration entry: "
             << sensor.ShortDebugString();
      return false;
    }
    if (sensor.sensor_type() == IMU) {
      if (has_imu) {
        AERROR << "Calibration manifest contains multiple IMUs.";
        return false;
      }
      has_imu = true;
    } else if (sensor.sensor_type() == GNSS) {
      bool* seen = sensor.frame_id() == "gnss_antenna_link"
                       ? &has_primary_gnss
                       : &has_secondary_gnss;
      if (*seen) {
        AERROR << "Calibration manifest contains a duplicate GNSS antenna "
                  "frame: "
               << sensor.frame_id();
        return false;
      }
      *seen = true;
    } else if (sensor.sensor_type() == LIDAR) {
      if (has_lidar) {
        AERROR << "Calibration manifest contains multiple LiDARs.";
        return false;
      }
      has_lidar = true;
    } else if (sensor.sensor_type() == CAMERA) {
      if (!sensor.has_intrinsic_file() ||
          !sensor.has_optical_extrinsic_file() ||
          !optical_frames.insert(sensor.optical_frame_id()).second) {
        AERROR << "Camera calibration requires unique optical frame and "
                  "intrinsics and optical-frame extrinsics: "
               << sensor.sensor_id();
        return false;
      }
    }

    ResolvedSensorCalibration resolved;
    resolved.calibration = sensor;
    if (!ResolveBundleFile(bundle_key, sensor.extrinsic_file(),
                           &resolved.resolved_extrinsic_file) ||
        (sensor.sensor_type() == CAMERA &&
         !ResolveBundleFile(bundle_key, sensor.intrinsic_file(),
                            &resolved.resolved_intrinsic_file)) ||
        (sensor.sensor_type() == CAMERA &&
         !ResolveBundleFile(bundle_key, sensor.optical_extrinsic_file(),
                            &resolved.resolved_optical_extrinsic_file))) {
      return false;
    }
    sensors_.push_back(std::move(resolved));
  }

  if (manifest_.sensor_size() == 0) {
    AERROR << "Calibration manifest contains no sensor entries.";
    return false;
  }
  if (has_secondary_gnss && !has_primary_gnss) {
    AERROR << "Secondary GNSS antenna requires a primary antenna frame.";
    return false;
  }
  for (const auto& frame_id : required_frames) {
    if (frames.count(frame_id) == 0) {
      AERROR << "Required sensor frame is missing from calibration manifest: "
             << frame_id;
      return false;
    }
  }
  if (!ResolveOptionalBundleFile(bundle_key, manifest_.lidar_height_file(),
                                 &resolved_lidar_height_file_)) {
    return false;
  }
  return true;
}

}  // namespace vehicle_calibration
}  // namespace common
}  // namespace apollo
