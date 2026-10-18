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

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
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
constexpr double kBevRadiusMeters = 30.0;

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

void RunDiagnostic(const std::string& dataset_root,
                   const std::string& metadata_dir,
                   const std::string& sample_token,
                   const std::string& output_path) {
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
    for (const char* camera_channel : kCameraChannels) {
      if (channel == camera_channel) {
        if (!calibrated_sensor.has_camera_intrinsics) {
          throw std::runtime_error("camera calibration has no intrinsics");
        }
        camera_data_by_channel.emplace(channel, sample_data);
        ego_pose_tokens.insert(sample_data.ego_pose_token);
        break;
      }
    }
  }
  if (camera_data_by_channel.size() != kCameraChannels.size()) {
    throw std::runtime_error("sample does not contain all six camera channels");
  }

  const auto ego_poses =
      ReadEgoPoses(metadata_dir + "/ego_pose.json", ego_pose_tokens);
  if (ego_poses.size() != ego_pose_tokens.size()) {
    throw std::runtime_error("ego_pose.json has missing records");
  }

  const SampleData& front_data = camera_data_by_channel.at("CAM_FRONT");
  const Eigen::Isometry3d T_global_ego_reference =
      ego_poses.at(front_data.ego_pose_token).T_global_ego;

  Eigen::Isometry3d T_vehicle_ego = Eigen::Isometry3d::Identity();
  T_vehicle_ego.linear() << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  const Eigen::Isometry3d T_vehicle_global =
      T_vehicle_ego * T_global_ego_reference.inverse();

  IpmConfig config;
  config.resolution = kBevResolution;
  config.bev_width = static_cast<int>(2.0 * kBevRadiusMeters / kBevResolution);
  config.bev_height = config.bev_width;
  config.x_min = -kBevRadiusMeters;
  config.y_min = -kBevRadiusMeters;

  const cv::Size bev_size(config.bev_width, config.bev_height);
  cv::Mat accumulated(bev_size, CV_32FC3, cv::Scalar::all(0));
  cv::Mat weights(bev_size, CV_32FC1, cv::Scalar::all(0));

  for (const char* channel : kCameraChannels) {
    const SampleData& sample_data = camera_data_by_channel.at(channel);
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
                               channel);
    }

    const Eigen::Isometry3d T_vehicle_camera = T_vehicle_global *
                                               ego_pose.T_global_ego *
                                               calibrated_sensor.T_ego_sensor;
    const IpmTransformer transformer(camera_model, T_vehicle_camera, config);

    cv::Mat camera_bev;
    TransformOpenCv(transformer, image, &camera_bev);
    std::vector<unsigned char> source_mask(
        static_cast<std::size_t>(image.rows) *
            static_cast<std::size_t>(image.cols),
        255);
    std::vector<unsigned char> bev_mask_data(
        static_cast<std::size_t>(config.bev_height) *
            static_cast<std::size_t>(config.bev_width),
        0);
    transformer.Transform(
        ImageView<const unsigned char>{source_mask.data(), image.cols,
                                       image.rows, 1,
                                       static_cast<std::size_t>(image.cols)},
        ImageView<unsigned char>{bev_mask_data.data(), config.bev_width,
                                 config.bev_height, 1,
                                 static_cast<std::size_t>(config.bev_width)},
        InterpolationMode::kNearest);
    cv::Mat bev_mask(config.bev_height, config.bev_width, CV_8UC1,
                     bev_mask_data.data());

    cv::Mat camera_bev_float;
    camera_bev.convertTo(camera_bev_float, CV_32FC3);
    cv::Mat mask_float;
    bev_mask.convertTo(mask_float, CV_32FC1, 1.0 / 255.0);
    cv::Mat mask_channels[] = {mask_float, mask_float, mask_float};
    cv::Mat mask_three_channel;
    cv::merge(mask_channels, 3, mask_three_channel);
    accumulated += camera_bev_float.mul(mask_three_channel);
    weights += mask_float;
  }

  cv::Mat safe_weights;
  cv::max(weights, 1.0, safe_weights);
  std::vector<cv::Mat> channels;
  cv::split(accumulated, channels);
  for (cv::Mat& channel : channels) {
    cv::divide(channel, safe_weights, channel);
  }
  cv::Mat stitched;
  cv::merge(channels, stitched);
  stitched.convertTo(stitched, CV_8UC3);
  if (!cv::imwrite(output_path, stitched)) {
    throw std::runtime_error("cannot write BEV image: " + output_path);
  }
  std::cout << "Wrote six-camera BEV diagnostic to " << output_path << " ("
            << config.bev_width << "x" << config.bev_height << ", "
            << config.resolution << " m/pixel).\n";
}

}  // namespace
}  // namespace image
}  // namespace vision
}  // namespace wheel

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: nuscenes_bev_diagnostic <dataset-root> "
                 "<metadata-dir> <sample-token> <output.png>\n";
    return EXIT_FAILURE;
  }

  try {
    wheel::vision::image::RunDiagnostic(argv[1], argv[2], argv[3], argv[4]);
  } catch (const std::exception& error) {
    std::cerr << "nuScenes BEV diagnostic failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
