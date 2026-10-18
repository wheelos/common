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

#include <cuda_runtime.h>

#include "pointcloud/filter/decimation_filter.h"
#include "pointcloud/filter/internal/device_filter_utils.cuh"

namespace wheel {
namespace vision {
namespace lidar {

namespace {

bool IsDeviceAccessible(MemoryType memory_type) {
  return memory_type == MemoryType::kUnified ||
         memory_type == MemoryType::kDevice;
}

uint32_t ComputeOutputCount(uint32_t point_count,
                            const DecimationFilterConfig& config) {
  if (config.offset >= point_count) {
    return 0U;
  }

  return 1U + (point_count - 1U - config.offset) / config.stride;
}

__global__ void DecimatePointCloudKernel(
    internal::DevicePointCloudView source,
    internal::MutableDevicePointCloudView output, uint32_t stride,
    uint32_t offset) {
  const uint32_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= output.point_count) {
    return;
  }

  const uint32_t source_index = offset + output_index * stride;
  for (uint32_t field_index = 0U; field_index < source.field_count;
       ++field_index) {
    internal::CopyFieldElement(source.fields[field_index], source_index,
                               output.fields[field_index], output_index);
  }
}

}  // namespace

bool DecimationFilter::Filter(const PointCloudView& source,
                              PointCloudBuffer* output,
                              CudaStream stream) const {
  return FilterWithStatus(source, output, stream).ok();
}

bool DecimationFilter::FilterAsync(const PointCloudView& source,
                                   PointCloudBuffer* output,
                                   CudaStream stream) const {
  return FilterAsyncWithStatus(source, output, stream).ok();
}

Status DecimationFilter::FilterWithStatus(const PointCloudView& source,
                                          PointCloudBuffer* output,
                                          CudaStream stream) const {
  const Status status = FilterAsyncWithStatus(source, output, stream);
  if (!status.ok()) {
    return status;
  }
  const cudaError_t error = cudaStreamSynchronize(stream);
  return error == cudaSuccess
             ? Status{}
             : Status{StatusCode::kCudaError, "decimation synchronization",
                      static_cast<int>(error)};
}

Status DecimationFilter::FilterAsyncWithStatus(const PointCloudView& source,
                                               PointCloudBuffer* output,
                                               CudaStream stream) const {
  if (!config_.IsValid() || output == nullptr || !source.IsValid() ||
      source.field_count() == 0U) {
    return {StatusCode::kInvalidArgument,
            "invalid decimation config or schema"};
  }
  if (&source == &output->view() ||
      internal::OverlapsStorage(source, output->view())) {
    return {StatusCode::kInvalidArgument,
            "in-place filtering is not supported"};
  }
  if (!IsDeviceAccessible(source.memory_type())) {
    return {StatusCode::kUnsupportedMemory,
            "CUDA requires device/unified memory"};
  }

  const uint32_t output_count =
      ComputeOutputCount(source.point_count(), config_);
  if (!output->AllocateLike(source, output_count, config_.output_layout)) {
    return {StatusCode::kAllocationFailed, "decimation output allocation"};
  }

  if (output_count == 0U) {
    return {};
  }

  const internal::DevicePointCloudView device_source =
      internal::MakeDevicePointCloudView(source);
  const internal::MutableDevicePointCloudView device_output =
      internal::MakeMutableDevicePointCloudView(output->view());

  constexpr uint32_t kThreadsPerBlock = 256U;
  const uint32_t blocks = 1U + (output_count - 1U) / kThreadsPerBlock;
  DecimatePointCloudKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      device_source, device_output, config_.stride, config_.offset);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status{}
             : Status{StatusCode::kCudaError, "decimation launch",
                      static_cast<int>(error)};
}

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
