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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include "camera/pinhole_camera_model.h"
#include "image/opencv_ipm_adapter.h"

namespace wheel {
namespace vision {
namespace image {
namespace {

using Json = nlohmann::json;

constexpr std::array<const char*, 6> kCameraChannels = {
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
};
constexpr double kBevResolution = 0.1;
constexpr double kBevForwardRangeMeters = 60.0;
constexpr double kGroundRayEpsilon = 1e-9;

struct SampleData {
  std::string sample_token;
  std::string calibrated_sensor_token;
  std::string ego_pose_token;
  std::string filename;
};

struct CalibratedSensor {
  std::string sensor_token;
  Eigen::Matrix3f intrinsics = Eigen::Matrix3f::Identity();
  Eigen::Isometry3d T_ego_sensor = Eigen::Isometry3d::Identity();
  bool has_camera_intrinsics = false;
};

struct EgoPose {
  Eigen::Isometry3d T_global_ego;
};

Json ReadJsonArray(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open JSON file: " + path);
  }

  Json value;
  input >> value;
  if (!value.is_array()) {
    throw std::runtime_error("expected a JSON array in: " + path);
  }
  return value;
}

std::string ReadString(const Json& value, const char* key) {
  if (!value.contains(key) || !value.at(key).is_string()) {
    throw std::runtime_error(std::string("invalid or missing JSON field: ") +
                             key);
  }
  return value.at(key).get<std::string>();
}

template <typename Predicate, typename Consumer>
void ForEachRecord(const std::string& path, Predicate predicate,
                   Consumer consumer) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open JSON file: " + path);
  }

  bool root_is_array = false;
  const auto callback = [&](int depth, Json::parse_event_t event, Json& value) {
    if (depth == 0 && event == Json::parse_event_t::array_start) {
      root_is_array = true;
    }
    if (depth == 1 && event == Json::parse_event_t::object_end) {
      if (predicate(value)) {
        consumer(value);
      }
      return false;
    }
    return true;
  };
  const Json root = Json::parse(input, callback);
  if (!root_is_array || !root.is_array()) {
    throw std::runtime_error("expected a JSON array in: " + path);
  }
}

Eigen::Vector3d ReadVector3(const Json& value, const char* key) {
  const Json& raw = value.at(key);
  if (!raw.is_array() || raw.size() != 3) {
    throw std::runtime_error(std::string("expected a 3-vector in field: ") +
                             key);
  }

  const Eigen::Vector3d result(raw[0].get<double>(), raw[1].get<double>(),
                               raw[2].get<double>());
  if (!result.allFinite()) {
    throw std::runtime_error(std::string("non-finite values in field: ") + key);
  }
  return result;
}

Eigen::Quaterniond ReadQuaternion(const Json& value) {
  const Json& raw = value.at("rotation");
  if (!raw.is_array() || raw.size() != 4) {
    throw std::runtime_error("expected a 4-value rotation quaternion");
  }

  Eigen::Quaterniond quaternion(raw[0].get<double>(), raw[1].get<double>(),
                                raw[2].get<double>(), raw[3].get<double>());
  if (!quaternion.coeffs().allFinite() || quaternion.norm() < 1e-12) {
    throw std::runtime_error("invalid rotation quaternion");
  }
  quaternion.normalize();
  return quaternion;
}

Eigen::Isometry3d ReadTransform(const Json& value) {
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = ReadQuaternion(value).toRotationMatrix();
  transform.translation() = ReadVector3(value, "translation");
  return transform;
}

std::unordered_map<std::string, SampleData> ReadSampleData(
    const std::string& path, const std::string& sample_token) {
  std::unordered_map<std::string, SampleData> result;
  ForEachRecord(
      path,
      [&sample_token](const Json& value) {
        return ReadString(value, "sample_token") == sample_token;
      },
      [&result](const Json& value) {
        const std::string token = ReadString(value, "token");
        result.emplace(token,
                       SampleData{ReadString(value, "sample_token"),
                                  ReadString(value, "calibrated_sensor_token"),
                                  ReadString(value, "ego_pose_token"),
                                  ReadString(value, "filename")});
      });
  return result;
}

std::unordered_map<std::string, CalibratedSensor> ReadCalibratedSensors(
    const std::string& path,
    const std::unordered_set<std::string>& selected_tokens) {
  std::unordered_map<std::string, CalibratedSensor> result;
  ForEachRecord(
      path,
      [&selected_tokens](const Json& value) {
        return selected_tokens.count(ReadString(value, "token")) != 0;
      },
      [&result](const Json& value) {
        const std::string token = ReadString(value, "token");
        CalibratedSensor calibrated_sensor;
        calibrated_sensor.sensor_token = ReadString(value, "sensor_token");
        calibrated_sensor.T_ego_sensor = ReadTransform(value);

        if (value.contains("camera_intrinsic")) {
          const Json& raw_intrinsics = value.at("camera_intrinsic");
          if (!raw_intrinsics.is_array()) {
            throw std::runtime_error("invalid camera_intrinsic matrix");
          }
          if (!raw_intrinsics.empty()) {
            if (raw_intrinsics.size() != 3) {
              throw std::runtime_error("invalid camera_intrinsic matrix");
            }
            for (int row = 0; row < 3; ++row) {
              if (!raw_intrinsics[row].is_array() ||
                  raw_intrinsics[row].size() != 3) {
                throw std::runtime_error("invalid camera_intrinsic matrix");
              }
              for (int col = 0; col < 3; ++col) {
                calibrated_sensor.intrinsics(row, col) =
                    raw_intrinsics[row][col].get<float>();
              }
            }
            if (!calibrated_sensor.intrinsics.allFinite()) {
              throw std::runtime_error(
                  "camera_intrinsic contains non-finite values");
            }
            calibrated_sensor.has_camera_intrinsics = true;
          }
        }
        result.emplace(token, std::move(calibrated_sensor));
      });
  return result;
}

std::unordered_map<std::string, std::string> ReadSensorChannels(
    const std::string& path,
    const std::unordered_set<std::string>& selected_tokens) {
  std::unordered_map<std::string, std::string> result;
  ForEachRecord(
      path,
      [&selected_tokens](const Json& value) {
        return selected_tokens.count(ReadString(value, "token")) != 0;
      },
      [&result](const Json& value) {
        result.emplace(ReadString(value, "token"),
                       ReadString(value, "channel"));
      });
  return result;
}

std::unordered_map<std::string, EgoPose> ReadEgoPoses(
    const std::string& path,
    const std::unordered_set<std::string>& selected_tokens) {
  std::unordered_map<std::string, EgoPose> result;
  ForEachRecord(
      path,
      [&selected_tokens](const Json& value) {
        return selected_tokens.count(ReadString(value, "token")) != 0;
      },
      [&result](const Json& value) {
        result.emplace(ReadString(value, "token"),
                       EgoPose{ReadTransform(value)});
      });
  return result;
}

std::string JoinPath(const std::string& root, const std::string& relative) {
  if (relative.empty() || relative.front() == '/' ||
      relative.find("..") != std::string::npos) {
    throw std::runtime_error("invalid dataset-relative filename: " + relative);
  }
  return root + "/" + relative;
}

IpmConfig MakeAutoFitConfig(const CameraModel& camera_model,
                            const Eigen::Isometry3d& T_vehicle_camera) {
  const ImageSize image_size = camera_model.image_size();
  const Eigen::Vector3d camera_origin = T_vehicle_camera.translation();
  double min_x = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();

  const auto include_ground_intersection = [&](float pixel_x, float pixel_y) {
    Eigen::Vector3f ray_camera;
    if (!camera_model.PixelToRay(Eigen::Vector2f(pixel_x, pixel_y),
                                 &ray_camera)) {
      return;
    }
    const Eigen::Vector3d ray_vehicle =
        T_vehicle_camera.linear() * ray_camera.cast<double>();
    if (ray_vehicle.z() >= -kGroundRayEpsilon) {
      return;
    }
    const double distance = (0.0 - camera_origin.z()) / ray_vehicle.z();
    if (!std::isfinite(distance) || distance <= 0.0) {
      return;
    }
    const Eigen::Vector3d point_vehicle =
        camera_origin + distance * ray_vehicle;
    if (point_vehicle.y() >= 0.0 &&
        point_vehicle.y() <= kBevForwardRangeMeters) {
      min_x = std::min(min_x, point_vehicle.x());
      max_x = std::max(max_x, point_vehicle.x());
    }
  };

  const int last_x = static_cast<int>(image_size.width) - 1;
  const int last_y = static_cast<int>(image_size.height) - 1;
  for (int x = 0; x <= last_x; ++x) {
    include_ground_intersection(static_cast<float>(x), 0.0f);
    include_ground_intersection(static_cast<float>(x),
                                static_cast<float>(last_y));
  }
  for (int y = 1; y < last_y; ++y) {
    include_ground_intersection(0.0f, static_cast<float>(y));
    include_ground_intersection(static_cast<float>(last_x),
                                static_cast<float>(y));
  }
  if (!std::isfinite(min_x) || !std::isfinite(max_x)) {
    throw std::runtime_error(
        "camera image has no ground projection within the forward range");
  }

  IpmConfig config;
  config.resolution = kBevResolution;
  config.x_min = std::floor(min_x / config.resolution) * config.resolution;
  config.y_min = 0.0;
  const double x_max = std::ceil(max_x / config.resolution) * config.resolution;
  const double width =
      std::ceil((x_max - config.x_min) / config.resolution) + 1.0;
  const double height =
      std::ceil(kBevForwardRangeMeters / config.resolution) + 1.0;
  if (width > static_cast<double>(std::numeric_limits<int>::max()) ||
      height > static_cast<double>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("auto-fit BEV dimensions exceed supported size");
  }
  config.bev_width = std::max(1, static_cast<int>(width));
  config.bev_height = std::max(1, static_cast<int>(height));
  return config;
}

void RunDiagnostic(const std::string& dataset_root,
                   const std::string& metadata_dir,
                   const std::string& sample_token,
                   const std::string& selected_channel,
                   const std::string& output_path) {
  if (std::find(kCameraChannels.begin(), kCameraChannels.end(),
                selected_channel) == kCameraChannels.end()) {
    throw std::runtime_error("unsupported camera channel: " + selected_channel);
  }
  if (output_path.size() < 4 ||
      output_path.substr(output_path.size() - 4) != ".png") {
    throw std::runtime_error(
        "output path must end in .png for lossless image encoding");
  }

  const std::string sample_table_path = metadata_dir + "/sample.json";
  Json sample_table = ReadJsonArray(sample_table_path);

  Json sample;
  for (const Json& value : sample_table) {
    if (ReadString(value, "token") == sample_token) {
      sample = value;
      break;
    }
  }
  if (sample.is_null()) {
    throw std::runtime_error("sample token not found: " + sample_token);
  }
  sample_table = nullptr;

  const auto sample_data_by_token =
      ReadSampleData(metadata_dir + "/sample_data.json", sample_token);
  if (sample_data_by_token.empty()) {
    throw std::runtime_error("sample_data.json has no records for the sample");
  }

  std::unordered_set<std::string> calibrated_sensor_tokens;
  for (const auto& entry : sample_data_by_token) {
    calibrated_sensor_tokens.insert(entry.second.calibrated_sensor_token);
  }
  const auto calibrated_sensors = ReadCalibratedSensors(
      metadata_dir + "/calibrated_sensor.json", calibrated_sensor_tokens);
  if (calibrated_sensors.size() != calibrated_sensor_tokens.size()) {
    throw std::runtime_error("calibrated_sensor.json has missing records");
  }

  std::unordered_set<std::string> sensor_tokens;
  for (const auto& entry : calibrated_sensors) {
    sensor_tokens.insert(entry.second.sensor_token);
  }
  const auto sensor_channels =
      ReadSensorChannels(metadata_dir + "/sensor.json", sensor_tokens);
  if (sensor_channels.size() != sensor_tokens.size()) {
    throw std::runtime_error("sensor.json has missing records");
  }

  std::unordered_map<std::string, SampleData> camera_data_by_channel;
  std::unordered_set<std::string> ego_pose_tokens;
  for (const auto& entry : sample_data_by_token) {
    const SampleData& sample_data = entry.second;
    const CalibratedSensor& calibrated_sensor =
        calibrated_sensors.at(sample_data.calibrated_sensor_token);
    const std::string& channel =
        sensor_channels.at(calibrated_sensor.sensor_token);
    if (channel == selected_channel) {
      if (!calibrated_sensor.has_camera_intrinsics) {
        throw std::runtime_error("camera calibration has no intrinsics");
      }
      camera_data_by_channel.emplace(channel, sample_data);
      ego_pose_tokens.insert(sample_data.ego_pose_token);
    }
  }
  if (camera_data_by_channel.size() != 1) {
    throw std::runtime_error("sample does not contain camera channel: " +
                             selected_channel);
  }

  const auto ego_poses =
      ReadEgoPoses(metadata_dir + "/ego_pose.json", ego_pose_tokens);
  if (ego_poses.size() != ego_pose_tokens.size()) {
    throw std::runtime_error("ego_pose.json has missing records");
  }

  const SampleData& selected_data = camera_data_by_channel.at(selected_channel);
  const Eigen::Isometry3d T_global_ego_reference =
      ego_poses.at(selected_data.ego_pose_token).T_global_ego;

  Eigen::Isometry3d T_vehicle_ego = Eigen::Isometry3d::Identity();
  T_vehicle_ego.linear() << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  const Eigen::Isometry3d T_vehicle_global =
      T_vehicle_ego * T_global_ego_reference.inverse();

  const SampleData& sample_data = camera_data_by_channel.at(selected_channel);
  const CalibratedSensor& calibrated_sensor =
      calibrated_sensors.at(sample_data.calibrated_sensor_token);
  const EgoPose& ego_pose = ego_poses.at(sample_data.ego_pose_token);
  const std::string image_path = JoinPath(dataset_root, sample_data.filename);
  const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("cannot read camera image: " + image_path);
  }

  auto camera_model = std::make_shared<PinholeCameraModel>();
  if (!camera_model->Init(static_cast<uint32_t>(image.cols),
                          static_cast<uint32_t>(image.rows),
                          calibrated_sensor.intrinsics)) {
    throw std::runtime_error(std::string("invalid intrinsics for channel: ") +
                             selected_channel);
  }

  const Eigen::Isometry3d T_vehicle_camera =
      T_vehicle_global * ego_pose.T_global_ego * calibrated_sensor.T_ego_sensor;
  const IpmConfig config = MakeAutoFitConfig(*camera_model, T_vehicle_camera);
  const CpuIpmTransformer transformer(camera_model, T_vehicle_camera, config);
  cv::Mat camera_bev;
  TransformOpenCv(transformer, image, &camera_bev);
  if (!cv::imwrite(output_path, camera_bev, {cv::IMWRITE_PNG_COMPRESSION, 0})) {
    throw std::runtime_error("cannot write BEV image: " + output_path);
  }
  const cv::Mat verified_output = cv::imread(output_path, cv::IMREAD_UNCHANGED);
  if (verified_output.size() != camera_bev.size() ||
      verified_output.type() != camera_bev.type() ||
      cv::norm(verified_output, camera_bev, cv::NORM_INF) != 0.0) {
    throw std::runtime_error("PNG output failed lossless pixel verification: " +
                             output_path);
  }
  std::cout << "Wrote lossless single-camera BEV for " << selected_channel
            << " to " << output_path << " (source " << image.cols << "x"
            << image.rows << ", BEV " << camera_bev.cols << "x"
            << camera_bev.rows << ", x " << config.x_min << ".."
            << config.x_min + (config.bev_width - 1) * config.resolution
            << " m, y 0.." << kBevForwardRangeMeters << " m, "
            << config.resolution << " m/pixel; "
            << "PNG round-trip pixels identical).\n";
}

}  // namespace
}  // namespace image
}  // namespace vision
}  // namespace wheel

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: nuscenes_bev_diagnostic <dataset-root> "
                 "<metadata-dir> <sample-token> <camera-channel> "
                 "<output.png>\n";
    return EXIT_FAILURE;
  }

  try {
    wheel::vision::image::RunDiagnostic(argv[1], argv[2], argv[3], argv[4],
                                        argv[5]);
  } catch (const std::exception& error) {
    std::cerr << "nuScenes BEV diagnostic failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
