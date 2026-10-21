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
#include <iomanip>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "openssl/sha.h"

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

bool IsFrameId(const std::string& frame) {
  static const std::regex pattern("^[A-Za-z0-9_][A-Za-z0-9_/-]*$");
  return std::regex_match(frame, pattern) &&
         frame.find("//") == std::string::npos && frame != "earth" &&
         frame != "world" && frame != "map" && frame != "odom";
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

}  // namespace

void Registry::Clear() {
  profile_.Clear();
  manifest_.Clear();
  profile_path_.clear();
  bundle_path_.clear();
  sensors_.clear();
  static_transforms_.clear();
  manifest_sha256_.clear();
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
  Registry candidate;
  if (!candidate.LoadProfile(profile_key)) {
    Clear();
    return false;
  }
  *this = std::move(candidate);
  return true;
}

bool Registry::LoadProfile(const std::string& profile_key) {
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
  const char* allow_test_calibration = std::getenv(kAllowTestCalibrationEnv);
  if (profile_.test_only() && (allow_test_calibration == nullptr ||
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
  std::string manifest_path;
  if (!ResolveFile(kManifestFile, &manifest_path)) {
    return fail();
  }
  if (!cyber::common::GetProtoFromASCIIFile(manifest_path, &manifest_)) {
    AERROR << "Failed to parse calibration manifest: " << manifest_path;
    return fail();
  }
  if (!FileSha256(manifest_path, &manifest_sha256_)) {
    return fail();
  }
  const char* expected_digest =
      std::getenv("WHEELOS_CALIBRATION_MANIFEST_SHA256");
  if ((profile_.has_manifest_sha256() &&
       profile_.manifest_sha256() != manifest_sha256_) ||
      (expected_digest != nullptr && manifest_sha256_ != expected_digest)) {
    AERROR << "Selected calibration manifest SHA256 mismatch: "
           << manifest_path;
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
        << " vehicle_id=" << profile_.vehicle_id()
        << " manifest_sha256=" << manifest_sha256_
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
            << " optical_extrinsic=" << sensor.resolved_optical_extrinsic_file;
    }
  }
  if (!resolved_lidar_height_file_.empty()) {
    AINFO << "[CALIBRATION] LiDAR height=" << resolved_lidar_height_file_;
  }
  return true;
}

bool Registry::FileSha256(const std::string& path, std::string* digest) {
  if (digest == nullptr) {
    AERROR << "Null calibration digest output.";
    return false;
  }
  digest->clear();
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    AERROR << "Unable to read calibration asset for SHA256: " << path;
    return false;
  }
  SHA256_CTX context;
  SHA256_Init(&context);
  char buffer[8192];
  while (input.read(buffer, sizeof(buffer)) || input.gcount() != 0) {
    SHA256_Update(&context, buffer, input.gcount());
  }
  if (!input.eof()) {
    AERROR << "Failed reading calibration asset: " << path;
    return false;
  }
  unsigned char bytes[SHA256_DIGEST_LENGTH];
  SHA256_Final(bytes, &context);
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned char byte : bytes) {
    output << std::setw(2) << static_cast<unsigned int>(byte);
  }
  *digest = output.str();
  return true;
}

bool Registry::ResolveFile(const std::string& relative_path,
                           std::string* resolved_path) const {
  if (resolved_path == nullptr || !IsSafeRelativePath(relative_path) ||
      bundle_path_.empty()) {
    AERROR << "Invalid selected calibration file: " << relative_path;
    return false;
  }
  resolved_path->clear();
  std::error_code error;
  const fs::path root = fs::canonical(bundle_path_, error);
  if (error) {
    AERROR << "Selected calibration bundle disappeared: " << bundle_path_;
    return false;
  }
  const fs::path file = fs::canonical(root / relative_path, error);
  const auto relative = file.lexically_relative(root);
  if (error || !IsSafeRelativePath(relative.string()) ||
      !IsReadableFile(file.string())) {
    AERROR << "Calibration file is missing or escapes selected bundle: "
           << relative_path;
    return false;
  }
  *resolved_path = file.string();
  return true;
}

const ResolvedSensorCalibration* Registry::FindSensor(
    const std::string& sensor_id_or_frame) const {
  for (const auto& sensor : sensors_) {
    if (sensor.calibration.sensor_id() == sensor_id_or_frame ||
        sensor.calibration.frame_id() == sensor_id_or_frame ||
        (!sensor.calibration.optical_frame_id().empty() &&
         sensor.calibration.optical_frame_id() == sensor_id_or_frame)) {
      return &sensor;
    }
  }
  AERROR << "Sensor not in selected calibration: " << sensor_id_or_frame;
  return nullptr;
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
  const bool version2 = manifest_.schema_version() == 2;
  if (manifest_.schema_version() != 1 && !version2) {
    AERROR << "Unsupported calibration schema: " << manifest_.schema_version();
    return false;
  }
  if (version2 &&
      (!IsSafeIdentifier(profile_.vehicle_id()) ||
       manifest_.vehicle_id() != profile_.vehicle_id() ||
       !profile_.has_manifest_sha256() || manifest_.provenance().empty() ||
       manifest_.has_lidar_height_file())) {
    AERROR << "Schema 2 requires matching vehicle identity, pinned manifest, "
              "provenance and per-sensor auxiliary calibration.";
    return false;
  }
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
    if (!(version2 ? IsFrameId(frame_id) : IsStandardFrameId(frame_id)) ||
        !required_frames.insert(frame_id).second) {
      AERROR << "Invalid or duplicate required sensor frame: " << frame_id;
      return false;
    }
  }

  std::unordered_set<std::string> referenced_assets;
  const auto resolve = [this, &referenced_assets](const std::string& path,
                                                  std::string* result) {
    referenced_assets.insert(path);
    return ResolveFile(path, result);
  };
  for (const auto& sensor : manifest_.sensor()) {
    if (!sensor.has_sensor_id() || sensor.sensor_id().empty() ||
        !sensor.has_sensor_type() ||
        sensor.sensor_type() == SENSOR_TYPE_UNSPECIFIED ||
        !sensor.has_frame_id() ||
        (version2
             ? (!IsFrameId(sensor.frame_id()) ||
                sensor.frame_id() == kBaseFrame ||
                sensor.has_parent_frame_id() || sensor.has_extrinsic_file() ||
                sensor.has_optical_extrinsic_file())
             : (!sensor.has_parent_frame_id() ||
                sensor.parent_frame_id() != kBaseFrame ||
                !IsCanonicalFrame(sensor) || !sensor.has_extrinsic_file())) ||
        !ids.insert(sensor.sensor_id()).second ||
        !frames.insert(sensor.frame_id()).second) {
      AERROR << "Invalid or duplicate sensor calibration entry: "
             << sensor.ShortDebugString();
      return false;
    }
    if (sensor.sensor_type() == IMU) {
      if (has_imu && !version2) {
        AERROR << "Calibration manifest contains multiple IMUs.";
        return false;
      }
      has_imu = true;
    } else if (sensor.sensor_type() == GNSS && !version2) {
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
      if (has_lidar && !version2) {
        AERROR << "Calibration manifest contains multiple LiDARs.";
        return false;
      }
      has_lidar = true;
    } else if (sensor.sensor_type() == CAMERA) {
      if (!sensor.has_intrinsic_file() ||
          (!version2 && !sensor.has_optical_extrinsic_file()) ||
          !IsFrameId(sensor.optical_frame_id()) ||
          !optical_frames.insert(sensor.optical_frame_id()).second) {
        AERROR << "Camera calibration requires unique optical frame and "
                  "intrinsics and optical-frame extrinsics: "
               << sensor.sensor_id();
        return false;
      }
    }

    ResolvedSensorCalibration resolved;
    resolved.calibration = sensor;
    if ((!version2 && !resolve(sensor.extrinsic_file(),
                               &resolved.resolved_extrinsic_file)) ||
        (sensor.sensor_type() == CAMERA &&
         !resolve(sensor.intrinsic_file(),
                  &resolved.resolved_intrinsic_file)) ||
        (!version2 && sensor.sensor_type() == CAMERA &&
         !resolve(sensor.optical_extrinsic_file(),
                  &resolved.resolved_optical_extrinsic_file)) ||
        (sensor.has_metadata_file() &&
         !resolve(sensor.metadata_file(), &resolved.resolved_metadata_file)) ||
        (sensor.has_lidar_height_file() &&
         (sensor.sensor_type() != LIDAR ||
          !resolve(sensor.lidar_height_file(),
                   &resolved.resolved_lidar_height_file)))) {
      return false;
    }
    if (version2 && sensor.sensor_type() == CAMERA &&
        (sensor.camera_model() != "brown" || sensor.image_width() == 0 ||
         sensor.image_height() == 0)) {
      AERROR << "Camera requires supported model and calibration resolution: "
             << sensor.sensor_id();
      return false;
    }
    if (version2 && sensor.sensor_type() != IMU &&
        sensor.sensor_type() != GNSS && !sensor.has_metadata_file()) {
      AERROR << "Perception sensor requires explicit metadata: "
             << sensor.sensor_id();
      return false;
    }
    sensors_.push_back(std::move(resolved));
  }

  if (manifest_.sensor_size() == 0) {
    AERROR << "Calibration manifest contains no sensor entries.";
    return false;
  }
  for (const auto& sensor : sensors_) {
    if ((frames.count(sensor.calibration.sensor_id()) != 0 &&
         sensor.calibration.sensor_id() != sensor.calibration.frame_id()) ||
        optical_frames.count(sensor.calibration.frame_id()) != 0 ||
        optical_frames.count(sensor.calibration.sensor_id()) != 0) {
      AERROR << "Ambiguous sensor ID or optical frame: "
             << sensor.calibration.sensor_id();
      return false;
    }
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
  if (!manifest_.lidar_height_file().empty() &&
      !resolve(manifest_.lidar_height_file(), &resolved_lidar_height_file_)) {
    return false;
  }
  if (!version2) {
    for (const auto& sensor : sensors_) {
      ResolvedStaticTransform edge;
      edge.calibration.set_parent_frame_id(
          sensor.calibration.parent_frame_id());
      edge.calibration.set_child_frame_id(sensor.calibration.frame_id());
      edge.calibration.set_extrinsic_file(sensor.calibration.extrinsic_file());
      edge.resolved_extrinsic_file = sensor.resolved_extrinsic_file;
      static_transforms_.push_back(edge);
      if (sensor.calibration.sensor_type() == CAMERA) {
        edge.calibration.set_parent_frame_id(sensor.calibration.frame_id());
        edge.calibration.set_child_frame_id(
            sensor.calibration.optical_frame_id());
        edge.calibration.set_extrinsic_file(
            sensor.calibration.optical_extrinsic_file());
        edge.resolved_extrinsic_file = sensor.resolved_optical_extrinsic_file;
        static_transforms_.push_back(edge);
      }
    }
    if (manifest_.static_transform_size() != 0) {
      AERROR << "Schema 1 cannot mix sensor extrinsics and static edges.";
      return false;
    }
  } else {
    for (const auto& transform : manifest_.static_transform()) {
      ResolvedStaticTransform edge;
      edge.calibration = transform;
      if (!resolve(transform.extrinsic_file(), &edge.resolved_extrinsic_file)) {
        return false;
      }
      static_transforms_.push_back(std::move(edge));
    }
  }
  if (!ValidateStaticGraph()) {
    return false;
  }
  if (version2) {
    std::unordered_set<std::string> verified;
    for (const auto& asset : manifest_.asset()) {
      std::string path;
      std::string digest;
      if (!verified.insert(asset.relative_path()).second ||
          !ResolveFile(asset.relative_path(), &path) ||
          !FileSha256(path, &digest) || digest != asset.sha256()) {
        AERROR << "Duplicate or invalid calibration asset SHA256: "
               << asset.relative_path();
        return false;
      }
    }
    for (const auto& path : referenced_assets) {
      if (verified.count(path) == 0) {
        AERROR << "Calibration asset has no checksum: " << path;
        return false;
      }
    }
  }
  return true;
}

bool Registry::ValidateStaticGraph() {
  std::unordered_map<std::string, std::string> parents;
  for (const auto& edge : static_transforms_) {
    const auto& parent = edge.calibration.parent_frame_id();
    const auto& child = edge.calibration.child_frame_id();
    if (!IsFrameId(parent) || !IsFrameId(child) || child == kBaseFrame ||
        !parents.emplace(child, parent).second) {
      AERROR << "Invalid static graph edge: " << parent << " -> " << child;
      return false;
    }
  }
  for (const auto& edge : parents) {
    std::unordered_set<std::string> visited;
    std::string frame = edge.first;
    while (frame != kBaseFrame) {
      const auto parent = parents.find(frame);
      if (!visited.insert(frame).second || parent == parents.end()) {
        AERROR << "Static graph is cyclic or disconnected at: " << frame;
        return false;
      }
      frame = parent->second;
    }
  }
  for (const auto& sensor : sensors_) {
    const auto& calibration = sensor.calibration;
    if (calibration.sensor_id() == kBaseFrame ||
        (parents.count(calibration.sensor_id()) != 0 &&
         calibration.sensor_id() != calibration.frame_id())) {
      AERROR << "Sensor ID aliases a different static graph frame: "
             << calibration.sensor_id();
      return false;
    }
    if (parents.count(calibration.frame_id()) == 0 ||
        (calibration.sensor_type() == CAMERA &&
         (parents.count(calibration.optical_frame_id()) == 0 ||
          parents.at(calibration.optical_frame_id()) !=
              calibration.frame_id()))) {
      AERROR << "Sensor frame is absent or optical parent is invalid: "
             << calibration.sensor_id();
      return false;
    }
  }
  return true;
}

}  // namespace vehicle_calibration
}  // namespace common
}  // namespace apollo
