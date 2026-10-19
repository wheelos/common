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
#include <string>

#include "gtest/gtest.h"

namespace apollo {
namespace common {
namespace vehicle_calibration {
namespace {

namespace fs = std::filesystem;

class RegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = fs::temp_directory_path() / "vehicle_calibration_registry_test";
    fs::remove_all(root_);
    fs::create_directories(root_ / "vehicles/cargo/config");
    fs::create_directories(root_ / "vehicles/cargo/calibration/v1");
    unsetenv("WHEELOS_ALLOW_TEST_CALIBRATION");
    setenv("WHEELOS_ASSET_ROOT", root_.c_str(), 1);
    Write("vehicles/cargo/config/vehicle_profile.pb.txt",
          "vehicle_type: \"cargo\"\n"
          "calibration_version: \"v1\"\n"
          "required_sensor_frame_id: \"imu_link\"\n");
  }

  void TearDown() override {
    unsetenv("WHEELOS_ASSET_ROOT");
    unsetenv("WHEELOS_VEHICLE_PROFILE_KEY");
    unsetenv("WHEELOS_ALLOW_TEST_CALIBRATION");
    std::error_code error;
    fs::remove_all(root_, error);
  }

  void Write(const std::string& key, const std::string& contents) {
    const fs::path path = root_ / key;
    fs::create_directories(path.parent_path());
    std::ofstream output(path);
    output << contents;
  }

  void WriteManifest(const std::string& entries) {
    Write("vehicles/cargo/calibration/v1/calibration_manifest.pb.txt",
          "calibration_version: \"v1\"\n" + entries);
  }

  fs::path root_;
};

TEST_F(RegistryTest, LoadsVersionedProfileAndCanonicalSensorFrames) {
  Write("vehicles/cargo/calibration/v1/imu.yaml", "transform: {}\n");
  WriteManifest(
      "sensor {\n"
      "  sensor_id: \"imu\"\n"
      "  sensor_type: IMU\n"
      "  parent_frame_id: \"base_link\"\n"
      "  frame_id: \"imu_link\"\n"
      "  extrinsic_file: \"imu.yaml\"\n"
      "}\n");

  Registry registry;
  ASSERT_TRUE(registry.LoadFromProfileKey(
      "vehicles/cargo/config/vehicle_profile.pb.txt"));
  EXPECT_EQ(registry.profile().vehicle_type(), "cargo");
  EXPECT_EQ(registry.profile().calibration_version(), "v1");
  ASSERT_EQ(registry.sensors().size(), 1u);
  EXPECT_EQ(registry.sensors()[0].calibration.frame_id(), "imu_link");
  EXPECT_EQ(registry.sensors()[0].resolved_extrinsic_file,
            (root_ / "vehicles/cargo/calibration/v1/imu.yaml").string());
}

TEST_F(RegistryTest, LoadsProfileSelectedByEnvironment) {
  Write("vehicles/cargo/calibration/v1/imu.yaml", "transform: {}\n");
  WriteManifest(
      "sensor {\n"
      "  sensor_id: \"imu\"\n"
      "  sensor_type: IMU\n"
      "  parent_frame_id: \"base_link\"\n"
      "  frame_id: \"imu_link\"\n"
      "  extrinsic_file: \"imu.yaml\"\n"
      "}\n");
  setenv("WHEELOS_VEHICLE_PROFILE_KEY",
         "vehicles/cargo/config/vehicle_profile.pb.txt", 1);

  Registry registry;
  ASSERT_TRUE(registry.LoadFromEnvironment());
  EXPECT_EQ(registry.profile().vehicle_type(), "cargo");
}

TEST_F(RegistryTest, LoadsCanonicalGnssAntennaFrame) {
  Write("vehicles/cargo/calibration/v1/imu.yaml", "transform: {}\n");
  Write("vehicles/cargo/calibration/v1/gnss.yaml", "transform: {}\n");
  Write("vehicles/cargo/calibration/v1/gnss_secondary.yaml",
        "transform: {}\n");
  Write("vehicles/cargo/config/vehicle_profile.pb.txt",
        "vehicle_type: \"cargo\"\n"
        "calibration_version: \"v1\"\n"
        "required_sensor_frame_id: \"imu_link\"\n"
        "required_sensor_frame_id: \"gnss_antenna_link\"\n");
  WriteManifest(
      "sensor {\n"
      "  sensor_id: \"imu\"\n"
      "  sensor_type: IMU\n"
      "  parent_frame_id: \"base_link\"\n"
      "  frame_id: \"imu_link\"\n"
      "  extrinsic_file: \"imu.yaml\"\n"
      "}\n"
      "sensor {\n"
      "  sensor_id: \"gnss_antenna\"\n"
      "  sensor_type: GNSS\n"
      "  parent_frame_id: \"base_link\"\n"
      "  frame_id: \"gnss_antenna_link\"\n"
      "  extrinsic_file: \"gnss.yaml\"\n"
      "}\n");
  const std::string secondary_entry =
      "sensor {\n"
      "  sensor_id: \"gnss_antenna_secondary\"\n"
      "  sensor_type: GNSS\n"
      "  parent_frame_id: \"base_link\"\n"
      "  frame_id: \"gnss_antenna_secondary_link\"\n"
      "  extrinsic_file: \"gnss_secondary.yaml\"\n"
      "}\n";

  Registry registry;
  ASSERT_TRUE(registry.LoadFromProfileKey(
      "vehicles/cargo/config/vehicle_profile.pb.txt"));
  ASSERT_EQ(registry.sensors().size(), 2u);
  EXPECT_EQ(registry.sensors()[1].calibration.sensor_type(), GNSS);
  EXPECT_EQ(registry.sensors()[1].calibration.frame_id(),
            "gnss_antenna_link");

  WriteManifest(
      "sensor {\n"
      "  sensor_id: \"imu\"\n"
      "  sensor_type: IMU\n"
      "  parent_frame_id: \"base_link\"\n"
      "  frame_id: \"imu_link\"\n"
      "  extrinsic_file: \"imu.yaml\"\n"
      "}\n"
      "sensor {\n"
      "  sensor_id: \"gnss_antenna\"\n"
      "  sensor_type: GNSS\n"
      "  parent_frame_id: \"base_link\"\n"
      "  frame_id: \"gnss_antenna_link\"\n"
      "  extrinsic_file: \"gnss.yaml\"\n"
      "}\n" + secondary_entry);
  ASSERT_TRUE(registry.LoadFromProfileKey(
      "vehicles/cargo/config/vehicle_profile.pb.txt"));
  ASSERT_EQ(registry.sensors().size(), 3u);
  EXPECT_EQ(registry.sensors()[2].calibration.frame_id(),
            "gnss_antenna_secondary_link");
}

TEST_F(RegistryTest, RejectsProfileTraversal) {
  Registry registry;
  EXPECT_FALSE(registry.LoadFromProfileKey(
      "vehicles/cargo/config/../config/profile.pb.txt"));
}

TEST_F(RegistryTest, RejectsBundleFileTraversal) {
  std::string resolved_path;
  EXPECT_FALSE(Registry::ResolveBundleFile("vehicles/cargo/calibration/v1",
                                           "../outside.yaml", &resolved_path));
}

TEST_F(RegistryTest, RejectsVersionMismatch) {
  WriteManifest("");
  Write("vehicles/cargo/config/vehicle_profile.pb.txt",
        "vehicle_type: \"cargo\"\ncalibration_version: \"v2\"\n");

  Registry registry;
  EXPECT_FALSE(registry.LoadFromProfileKey(
      "vehicles/cargo/config/vehicle_profile.pb.txt"));
}

TEST_F(RegistryTest, RejectsNonCanonicalFrameAndMissingCalibrationFile) {
  WriteManifest(
      "sensor {\n"
      "  sensor_id: \"imu\"\n"
      "  sensor_type: IMU\n"
      "  parent_frame_id: \"base_link\"\n"
      "  frame_id: \"imu\"\n"
      "  extrinsic_file: \"missing.yaml\"\n"
      "}\n");

  Registry registry;
  EXPECT_FALSE(registry.LoadFromProfileKey(
      "vehicles/cargo/config/vehicle_profile.pb.txt"));
}

TEST_F(RegistryTest, RejectsMissingRequiredSensor) {
  Write("vehicles/cargo/calibration/v1/lidar.yaml", "transform: {}\n");
  WriteManifest(
      "sensor {\n"
      "  sensor_id: \"lidar\"\n"
      "  sensor_type: LIDAR\n"
      "  parent_frame_id: \"base_link\"\n"
      "  frame_id: \"lidar_top_link\"\n"
      "  extrinsic_file: \"lidar.yaml\"\n"
      "}\n");

  Registry registry;
  EXPECT_FALSE(registry.LoadFromProfileKey(
      "vehicles/cargo/config/vehicle_profile.pb.txt"));
}

TEST_F(RegistryTest, LoadsCanonicalCameraFramesAndAllCalibrationFiles) {
  Write("vehicles/cargo/calibration/v1/camera.yaml", "transform: {}\n");
  Write("vehicles/cargo/calibration/v1/camera_optical.yaml",
        "transform: {}\n");
  Write("vehicles/cargo/calibration/v1/camera_intrinsics.yaml", "K: []\n");
  WriteManifest(
      "sensor {\n"
      "  sensor_id: \"front\"\n"
      "  sensor_type: CAMERA\n"
      "  parent_frame_id: \"base_link\"\n"
      "  frame_id: \"camera_front_link\"\n"
      "  optical_frame_id: \"camera_front_optical_frame\"\n"
      "  extrinsic_file: \"camera.yaml\"\n"
      "  optical_extrinsic_file: \"camera_optical.yaml\"\n"
      "  intrinsic_file: \"camera_intrinsics.yaml\"\n"
      "}\n"
      "sensor {\n"
      "  sensor_id: \"imu\"\n"
      "  sensor_type: IMU\n"
      "  parent_frame_id: \"base_link\"\n"
      "  frame_id: \"imu_link\"\n"
      "  extrinsic_file: \"camera.yaml\"\n"
      "}\n");

  Registry registry;
  ASSERT_TRUE(registry.LoadFromProfileKey(
      "vehicles/cargo/config/vehicle_profile.pb.txt"));
  ASSERT_EQ(registry.sensors().size(), 2u);
  EXPECT_EQ(registry.sensors()[0].resolved_intrinsic_file,
            (root_ / "vehicles/cargo/calibration/v1/camera_intrinsics.yaml")
                .string());
  EXPECT_EQ(registry.sensors()[0].resolved_optical_extrinsic_file,
            (root_ / "vehicles/cargo/calibration/v1/camera_optical.yaml")
                .string());
}

TEST_F(RegistryTest, RejectsCameraWithoutOpticalFrameOrIntrinsics) {
  WriteManifest(
      "sensor {\n"
      "  sensor_id: \"front\"\n"
      "  sensor_type: CAMERA\n"
      "  parent_frame_id: \"base_link\"\n"
      "  frame_id: \"camera_front_link\"\n"
      "  extrinsic_file: \"camera.yaml\"\n"
      "  intrinsic_file: \"camera_intrinsics.yaml\"\n"
      "}\n");

  Registry registry;
  EXPECT_FALSE(registry.LoadFromProfileKey(
      "vehicles/cargo/config/vehicle_profile.pb.txt"));
}

TEST_F(RegistryTest, RequiresProfileEnvironmentVariable) {
  unsetenv("WHEELOS_VEHICLE_PROFILE_KEY");
  Registry registry;
  EXPECT_FALSE(registry.LoadFromEnvironment());
}

TEST_F(RegistryTest, RejectsTestOnlyProfileUnlessExplicitlyAllowed) {
  Write("vehicles/cargo/config/vehicle_profile.pb.txt",
        "vehicle_type: \"cargo\"\n"
        "calibration_version: \"v1\"\n"
        "required_sensor_frame_id: \"imu_link\"\n"
        "test_only: true\n");
  Write("vehicles/cargo/calibration/v1/imu.yaml", "transform: {}\n");
  WriteManifest(
      "sensor {\n"
      "  sensor_id: \"imu\"\n"
      "  sensor_type: IMU\n"
      "  parent_frame_id: \"base_link\"\n"
      "  frame_id: \"imu_link\"\n"
      "  extrinsic_file: \"imu.yaml\"\n"
      "}\n");

  Registry registry;
  EXPECT_FALSE(registry.LoadFromProfileKey(
      "vehicles/cargo/config/vehicle_profile.pb.txt"));
  setenv("WHEELOS_ALLOW_TEST_CALIBRATION", "1", 1);
  EXPECT_TRUE(registry.LoadFromProfileKey(
      "vehicles/cargo/config/vehicle_profile.pb.txt"));
  unsetenv("WHEELOS_ALLOW_TEST_CALIBRATION");
}

}  // namespace
}  // namespace vehicle_calibration
}  // namespace common
}  // namespace apollo
