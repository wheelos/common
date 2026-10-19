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

#include <cstdint>

#include "pointcloud/core/point_cloud.h"
#include "pointcloud/core/point_cloud_buffer.h"
#include "pointcloud/runtime/cuda_stream.h"
#include "pointcloud/status.h"

namespace wheel {
namespace vision {
namespace lidar {

struct DecimationFilterConfig {
  uint32_t stride = 1U;
  uint32_t offset = 0U;
  PointLayout output_layout = PointLayout::kPlanar;

  inline bool IsValid() const {
    return stride > 0U && (output_layout == PointLayout::kPlanar ||
                           output_layout == PointLayout::kInterleaved);
  }
};

class DecimationFilter {
 public:
  explicit DecimationFilter(const DecimationFilterConfig& config)
      : config_(config) {}

  bool Filter(const PointCloudView& source, PointCloudBuffer* output,
              CudaStream stream = kDefaultCudaStream) const;
  bool FilterAsync(const PointCloudView& source, PointCloudBuffer* output,
                   CudaStream stream = kDefaultCudaStream) const;

  Status FilterWithStatus(const PointCloudView& source,
                          PointCloudBuffer* output,
                          CudaStream stream = kDefaultCudaStream) const;
  // Allocates output before enqueueing; allocation may block. Input and output
  // must not alias and must outlive stream completion. Ordering is stable.
  // A successful return does not report deferred CUDA execution errors.
  Status FilterAsyncWithStatus(const PointCloudView& source,
                               PointCloudBuffer* output,
                               CudaStream stream = kDefaultCudaStream) const;

  inline const DecimationFilterConfig& config() const { return config_; }

 private:
  DecimationFilterConfig config_;
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
