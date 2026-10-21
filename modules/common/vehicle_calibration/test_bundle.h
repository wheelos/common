// Copyright 2026 WheelOS. All Rights Reserved.
#pragma once

#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "modules/common/vehicle_calibration/registry.h"

namespace apollo {
namespace common {
namespace vehicle_calibration {

class TestBundle {
 public:
  TestBundle() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "calibration-test-XXXXXX")
            .string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const char* root = mkdtemp(buffer.data());
    if (root == nullptr) {
      throw std::runtime_error("Unable to create calibration test directory");
    }
    root_ = root;
    for (const auto* name :
         {"WHEELOS_ASSET_ROOT", "WHEELOS_ALLOW_TEST_CALIBRATION",
          "WHEELOS_CALIBRATION_MANIFEST_SHA256",
          "WHEELOS_VEHICLE_PROFILE_KEY"}) {
      const char* value = std::getenv(name);
      environment_[name] = {value != nullptr, value == nullptr ? "" : value};
    }
    setenv("WHEELOS_ASSET_ROOT", root_.c_str(), 1);
    setenv("WHEELOS_ALLOW_TEST_CALIBRATION", "1", 1);
    unsetenv("WHEELOS_CALIBRATION_MANIFEST_SHA256");
    setenv("WHEELOS_VEHICLE_PROFILE_KEY", profile_key().c_str(), 1);
    manifest.set_schema_version(2);
    manifest.set_vehicle_id("test-vehicle");
    manifest.set_calibration_version("v2");
    manifest.set_provenance(
        "synthetic unit-test data, not vehicle measurements");
  }

  ~TestBundle() {
    for (const auto& entry : environment_) {
      if (entry.second.first) {
        setenv(entry.first.c_str(), entry.second.second.c_str(), 1);
      } else {
        unsetenv(entry.first.c_str());
      }
    }
    std::error_code error;
    std::filesystem::remove_all(root_, error);
  }

  static std::string profile_key() {
    return "vehicles/test/config/vehicle_profile.pb.txt";
  }

  std::string path(const std::string& relative) const {
    return (root_ / "vehicles/test/calibration/v2" / relative).string();
  }

  void Write(const std::string& relative, const std::string& content) {
    const std::filesystem::path file(path(relative));
    std::filesystem::create_directories(file.parent_path());
    std::ofstream(file) << content;
    files_[relative] = content;
  }

  void AddEdge(const std::string& parent, const std::string& child) {
    auto* edge = manifest.add_static_transform();
    edge->set_parent_frame_id(parent);
    edge->set_child_frame_id(child);
    edge->set_extrinsic_file(child + ".yaml");
    Write(child + ".yaml",
          "header: {frame_id: " + parent + "}\nchild_frame_id: " + child +
              "\ntransform:\n  translation: {x: 1, y: 0, z: 0}\n"
              "  rotation: {x: 0, y: 0, z: 0, w: 1}\n");
  }

  SensorCalibration* AddSensor(SensorType type, const std::string& id,
                               const std::string& frame) {
    auto* sensor = manifest.add_sensor();
    sensor->set_sensor_id(id);
    sensor->set_sensor_type(type);
    sensor->set_frame_id(frame);
    AddEdge("base_link", frame);
    if (type != IMU && type != GNSS) {
      sensor->set_metadata_file(id + ".pb.txt");
      const std::string perception_type = type == CAMERA  ? "MONOCULAR_CAMERA"
                                          : type == RADAR ? "LONG_RANGE_RADAR"
                                          : type == ULTRASONIC ? "ULTRASONIC"
                                                               : "VELODYNE_128";
      const auto measurement_frame =
          type == CAMERA ? frame + "_optical" : frame;
      Write(id + ".pb.txt",
            "name: \"" + id + "\"\nframe_id: \"" + measurement_frame +
                "\"\ntype: " + perception_type + "\norientation: FRONT\n");
    }
    if (type == CAMERA) {
      sensor->set_optical_frame_id(frame + "_optical");
      AddEdge(frame, sensor->optical_frame_id());
      sensor->set_intrinsic_file(id + "_intrinsic.yaml");
      sensor->set_camera_model("brown");
      sensor->set_image_width(640);
      sensor->set_image_height(480);
      Write(sensor->intrinsic_file(),
            "width: 640\nheight: 480\n"
            "K: [400, 0, 320, 0, 400, 240, 0, 0, 1]\nD: [0, 0, 0, 0, 0]\n");
    }
    if (type == LIDAR) {
      sensor->set_lidar_height_file(id + "_height.yaml");
      Write(sensor->lidar_height_file(),
            "vehicle:\n  parameters: {height: 1.8, height_var: 0.01}\n");
    }
    return sensor;
  }

  bool Publish() {
    manifest.clear_asset();
    for (const auto& file : files_) {
      auto* asset = manifest.add_asset();
      asset->set_relative_path(file.first);
      std::string digest;
      if (!Registry::FileSha256(path(file.first), &digest)) {
        return false;
      }
      asset->set_sha256(digest);
    }
    const std::string manifest_path = path("calibration_manifest.pb.txt");
    std::ofstream(manifest_path) << manifest.DebugString();
    std::string digest;
    if (!Registry::FileSha256(manifest_path, &digest)) {
      return false;
    }
    const auto profile = root_ / profile_key();
    std::filesystem::create_directories(profile.parent_path());
    std::ofstream(profile)
        << "vehicle_type: \"test\"\nvehicle_id: \"test-vehicle\"\n"
           "calibration_version: \"v2\"\ntest_only: true\n"
           "required_sensor_frame_id: \""
        << manifest.sensor(0).frame_id() << "\"\nmanifest_sha256: \"" << digest
        << "\"\n";
    return true;
  }

  CalibrationManifest manifest;

 private:
  std::filesystem::path root_;
  std::map<std::string, std::string> files_;
  std::map<std::string, std::pair<bool, std::string>> environment_;
};

}  // namespace vehicle_calibration
}  // namespace common
}  // namespace apollo
