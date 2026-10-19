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

#pragma once

#include "pointcloud/core/point_cloud.h"
#include "pointcloud/core/rigid_transform.h"
#include "pointcloud/range_image/range_image.h"
#include "pointcloud/runtime/cuda_stream.h"
#include "pointcloud/status.h"

namespace wheel {
namespace vision {
namespace lidar {

class RangeImageBuilder {
 public:
  explicit RangeImageBuilder(const RangeImageConfig& config)
      : config_(config) {}

  bool Build(const PointCloudView& point_cloud, RangeImageBuffer* output,
             RangeImageWorkspace* workspace,
             const RigidTransform3f* lidar_from_input = nullptr,
             CudaStream stream = kDefaultCudaStream) const;
  bool BuildAsync(const PointCloudView& point_cloud, RangeImageBuffer* output,
                  RangeImageWorkspace* workspace,
                  const RigidTransform3f* lidar_from_input = nullptr,
                  CudaStream stream = kDefaultCudaStream) const;

  Status BuildWithStatus(const PointCloudView& point_cloud,
                         RangeImageBuffer* output,
                         RangeImageWorkspace* workspace,
                         const RigidTransform3f* lidar_from_input = nullptr,
                         CudaStream stream = kDefaultCudaStream) const;
  // Enqueues work without allocating. Input storage, output, and workspace must
  // remain valid and unmodified until stream completion, even after an error
  // if earlier launches succeeded. Do not reuse workspace concurrently.
  // Success reports enqueue status; synchronize the stream for execution
  // errors.
  Status BuildAsyncWithStatus(
      const PointCloudView& point_cloud, RangeImageBuffer* output,
      RangeImageWorkspace* workspace,
      const RigidTransform3f* lidar_from_input = nullptr,
      CudaStream stream = kDefaultCudaStream) const;

  inline const RangeImageConfig& config() const { return config_; }

 private:
  RangeImageConfig config_;
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
