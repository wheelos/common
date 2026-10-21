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

#include "modules/common/vehicle_calibration/lidar_height.h"
#include "modules/common/vehicle_calibration/test_bundle.h"

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

TEST_F(RegistryTest, LegacyGlobalHeightUsesTheOnlyLidarAndRejectsBadValues) {
  Write("vehicles/cargo/calibration/v1/imu.yaml", "transform: {}\n");
  Write("vehicles/cargo/calibration/v1/lidar.yaml", "transform: {}\n");
  const std::string height_key = "vehicles/cargo/calibration/v1/height.yaml";
  Write(height_key,
        "vehicle:\n  parameters: {height: 1.8, height_var: 0.01}\n");
  WriteManifest(
      "lidar_height_file: \"height.yaml\"\n"
      "sensor { sensor_id: \"imu\" sensor_type: IMU "
      "parent_frame_id: \"base_link\" frame_id: \"imu_link\" "
      "extrinsic_file: \"imu.yaml\" }\n"
      "sensor { sensor_id: \"lidar\" sensor_type: LIDAR "
      "parent_frame_id: \"base_link\" frame_id: \"lidar_top_link\" "
      "extrinsic_file: \"lidar.yaml\" }\n");
  Registry registry;
  ASSERT_TRUE(registry.LoadFromProfileKey(
      "vehicles/cargo/config/vehicle_profile.pb.txt"));
  LidarHeight height;
  ASSERT_TRUE(LoadLidarHeight(registry, "", &height));
  EXPECT_DOUBLE_EQ(height.height, 1.8);
  EXPECT_DOUBLE_EQ(height.variance, 0.01);
  EXPECT_EQ(height.frame_id, "lidar_top_link");
  EXPECT_TRUE(LoadLidarHeight(registry, "lidar", &height));
  EXPECT_TRUE(LoadLidarHeight(registry, "lidar_top_link", &height));
  EXPECT_FALSE(LoadLidarHeight(registry, "imu", &height));
  Write(height_key, "vehicle:\n  parameters: {height: 0, height_var: 0}\n");
  EXPECT_FALSE(LoadLidarHeight(registry, "lidar", &height));
  EXPECT_DOUBLE_EQ(height.height, 1.8);
  Write(height_key, "vehicle:\n  parameters: {height: 1, height_var: -1}\n");
  EXPECT_FALSE(LoadLidarHeight(registry, "lidar", &height));
  Write(height_key, "vehicle:\n  parameters: {height: .nan, height_var: 0}\n");
  EXPECT_FALSE(LoadLidarHeight(registry, "lidar", &height));
}

TEST(RegistryHeightTest, SchemaTwoNeverUsesLegacyGlobalHeight) {
  TestBundle bundle;
  bundle.AddSensor(LIDAR, "lidar", "lidar_link")->clear_lidar_height_file();
  ASSERT_TRUE(bundle.Publish());
  Registry registry;
  ASSERT_TRUE(registry.LoadFromEnvironment());
  LidarHeight height;
  EXPECT_FALSE(LoadLidarHeight(registry, "lidar", &height));
  bundle.manifest.set_lidar_height_file("lidar_height.yaml");
  ASSERT_TRUE(bundle.Publish());
  EXPECT_FALSE(registry.LoadFromEnvironment());
}

TEST_F(RegistryTest, LoadsCanonicalGnssAntennaFrame) {
  Write("vehicles/cargo/calibration/v1/imu.yaml", "transform: {}\n");
  Write("vehicles/cargo/calibration/v1/gnss.yaml", "transform: {}\n");
  Write("vehicles/cargo/calibration/v1/gnss_secondary.yaml", "transform: {}\n");
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
  EXPECT_EQ(registry.sensors()[1].calibration.frame_id(), "gnss_antenna_link");

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
      "}\n" +
      secondary_entry);
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
  Write("vehicles/cargo/calibration/v1/camera_optical.yaml", "transform: {}\n");
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
  EXPECT_EQ(
      registry.sensors()[0].resolved_optical_extrinsic_file,
      (root_ / "vehicles/cargo/calibration/v1/camera_optical.yaml").string());
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

TEST(CalibrationBundleTest, LoadsMultiLidarRadarCameraAndHeight) {
  TestBundle bundle;
  bundle.AddSensor(LIDAR, "front_lidar", "lidar_front_link");
  bundle.AddSensor(LIDAR, "rear_lidar", "lidar_rear_link");
  bundle.AddSensor(RADAR, "radar", "radar_front_link");
  bundle.AddSensor(CAMERA, "camera", "camera_front_link");
  bundle.AddSensor(IMU, "imu_front", "imu_front_link");
  bundle.AddSensor(IMU, "imu_rear", "imu_rear_link");
  ASSERT_TRUE(bundle.Publish());
  Registry registry;
  ASSERT_TRUE(registry.LoadFromEnvironment());
  EXPECT_EQ(registry.sensors().size(), 6);
  EXPECT_EQ(registry.static_transforms().size(), 7);
  EXPECT_NE(registry.FindSensor("front_lidar"), nullptr);
  EXPECT_NE(registry.FindSensor("lidar_rear_link"), nullptr);
  LidarHeight height;
  EXPECT_FALSE(LoadLidarHeight(registry, "", &height));
  ASSERT_TRUE(LoadLidarHeight(registry, "rear_lidar", &height));
  EXPECT_DOUBLE_EQ(height.height, 1.8);
  EXPECT_DOUBLE_EQ(height.variance, 0.01);
}

TEST(CalibrationBundleTest, RejectsChangedAssetsAndClearsFailedSnapshot) {
  TestBundle bundle;
  bundle.AddSensor(IMU, "imu", "imu_link");
  ASSERT_TRUE(bundle.Publish());
  Registry registry;
  ASSERT_TRUE(registry.LoadFromEnvironment());
  bundle.Write("imu_link.yaml", "changed after publication\n");
  EXPECT_FALSE(registry.LoadFromEnvironment());
  EXPECT_TRUE(registry.sensors().empty());
  EXPECT_TRUE(registry.static_transforms().empty());
}

TEST(CalibrationBundleTest, RejectsCyclesDisconnectedAndDuplicateChildren) {
  TestBundle bundle;
  bundle.AddSensor(IMU, "imu", "imu_link");
  bundle.AddEdge("imu_link", "mount");
  bundle.manifest.mutable_static_transform(0)->set_parent_frame_id("mount");
  ASSERT_TRUE(bundle.Publish());
  Registry registry;
  EXPECT_FALSE(registry.LoadFromEnvironment());
  bundle.manifest.mutable_static_transform(0)->set_parent_frame_id("missing");
  ASSERT_TRUE(bundle.Publish());
  EXPECT_FALSE(registry.LoadFromEnvironment());
  bundle.manifest.mutable_static_transform(0)->set_parent_frame_id("base_link");
  bundle.AddEdge("base_link", "imu_link");
  ASSERT_TRUE(bundle.Publish());
  EXPECT_FALSE(registry.LoadFromEnvironment());
}

TEST(CalibrationBundleTest, RejectsWrongIdentityAndExpectedManifestDigest) {
  TestBundle bundle;
  bundle.AddSensor(IMU, "imu", "imu_link");
  bundle.manifest.set_vehicle_id("another-vehicle");
  ASSERT_TRUE(bundle.Publish());
  Registry registry;
  EXPECT_FALSE(registry.LoadFromEnvironment());
  bundle.manifest.set_vehicle_id("test-vehicle");
  ASSERT_TRUE(bundle.Publish());
  setenv("WHEELOS_CALIBRATION_MANIFEST_SHA256", "wrong", 1);
  EXPECT_FALSE(registry.LoadFromEnvironment());
}

TEST(CalibrationBundleTest, RejectsEscapingSymlinkAndInvalidHeight) {
  TestBundle bundle;
  bundle.AddSensor(LIDAR, "lidar", "lidar_link");
  ASSERT_TRUE(bundle.Publish());
  Registry registry;
  ASSERT_TRUE(registry.LoadFromEnvironment());
  std::filesystem::create_symlink("/etc/hosts", bundle.path("outside"));
  std::string resolved;
  EXPECT_FALSE(registry.ResolveFile("outside", &resolved));
  EXPECT_FALSE(registry.ResolveFile("../outside", &resolved));
  bundle.Write("lidar_height.yaml",
               "vehicle:\n  parameters: {height: .nan, height_var: -1}\n");
  ASSERT_TRUE(bundle.Publish());
  ASSERT_TRUE(registry.LoadFromEnvironment());
  LidarHeight height;
  EXPECT_FALSE(LoadLidarHeight(registry, "lidar", &height));
}

TEST(CalibrationBundleTest, ComputesStandardSha256) {
  TestBundle bundle;
  bundle.Write("sha.txt", "abc");
  std::string digest;
  ASSERT_TRUE(Registry::FileSha256(bundle.path("sha.txt"), &digest));
  EXPECT_EQ(digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(CalibrationBundleTest, RejectsSensorIdsAliasingBaseOrMountFrames) {
  for (const auto* id : {"base_link", "mount"}) {
    TestBundle bundle;
    bundle.AddSensor(IMU, id, "imu_link");
    bundle.AddEdge("base_link", "mount");
    ASSERT_TRUE(bundle.Publish());
    Registry registry;
    EXPECT_FALSE(registry.LoadFromEnvironment());
  }
}

}  // namespace
}  // namespace vehicle_calibration
}  // namespace common
}  // namespace apollo
