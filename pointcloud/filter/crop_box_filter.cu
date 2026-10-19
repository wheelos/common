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

#include <cmath>
#include <limits>

#include <cuda_runtime.h>

#include "pointcloud/filter/crop_box_filter.h"
#include "pointcloud/filter/internal/device_filter_utils.cuh"

namespace wheel {
namespace vision {
namespace lidar {

namespace {

bool IsDeviceAccessible(MemoryType memory_type) {
  return memory_type == MemoryType::kUnified ||
         memory_type == MemoryType::kDevice;
}

struct DeviceCropBoxFilterConfig {
  float min_x;
  float max_x;
  float min_y;
  float max_y;
  float min_z;
  float max_z;
  uint8_t reject_non_finite;
  uint8_t keep_inside;
};

DeviceCropBoxFilterConfig MakeDeviceConfig(const CropBoxFilterConfig& config) {
  DeviceCropBoxFilterConfig device_config;
  device_config.min_x = config.min_x;
  device_config.max_x = config.max_x;
  device_config.min_y = config.min_y;
  device_config.max_y = config.max_y;
  device_config.min_z = config.min_z;
  device_config.max_z = config.max_z;
  device_config.reject_non_finite = config.reject_non_finite ? 1U : 0U;
  device_config.keep_inside = config.mode == CropBoxMode::kKeepInside ? 1U : 0U;
  return device_config;
}

__device__ inline bool IsPointSelected(
    float x, float y, float z, const DeviceCropBoxFilterConfig& config) {
  if (config.reject_non_finite &&
      (!isfinite(x) || !isfinite(y) || !isfinite(z))) {
    return false;
  }

  const bool inside = x >= config.min_x && x <= config.max_x &&
                      y >= config.min_y && y <= config.max_y &&
                      z >= config.min_z && z <= config.max_z;
  return config.keep_inside != 0U ? inside : !inside;
}

__global__ void SelectCropBoxIndicesKernel(internal::DeviceFieldDescriptor x,
                                           internal::DeviceFieldDescriptor y,
                                           internal::DeviceFieldDescriptor z,
                                           uint32_t point_count,
                                           DeviceCropBoxFilterConfig config,
                                           uint32_t* selected_indices,
                                           uint32_t* selected_count) {
  const uint32_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index >= point_count) {
    return;
  }

  const float point_x = internal::LoadFloat(x, point_index);
  const float point_y = internal::LoadFloat(y, point_index);
  const float point_z = internal::LoadFloat(z, point_index);
  if (!IsPointSelected(point_x, point_y, point_z, config)) {
    return;
  }

  const uint32_t output_index = atomicAdd(selected_count, 1U);
  selected_indices[output_index] = point_index;
}

__global__ void GatherSelectedPointsKernel(
    internal::DevicePointCloudView source,
    internal::MutableDevicePointCloudView output,
    const uint32_t* selected_indices) {
  const uint32_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= output.point_count) {
    return;
  }

  const uint32_t source_index = selected_indices[output_index];
  for (uint32_t field_index = 0U; field_index < source.field_count;
       ++field_index) {
    internal::CopyFieldElement(source.fields[field_index], source_index,
                               output.fields[field_index], output_index);
  }
}

}  // namespace

bool CropBoxFilterWorkspace::Allocate(uint32_t max_points,
                                      MemoryType index_memory_type) {
  Reset();
  if (max_points == 0U ||
      max_points > std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
      !IsDeviceAccessible(index_memory_type)) {
    return false;
  }

  if (!selected_indices_.Allocate(
          static_cast<size_t>(max_points) * sizeof(uint32_t),
          index_memory_type) ||
      !selected_count_.Allocate(sizeof(uint32_t), MemoryType::kUnified)) {
    Reset();
    return false;
  }

  max_points_ = max_points;
  // Clear is asynchronous; allocation must finish initialization before a
  // caller can safely use this workspace on a non-default stream.
  if (!selected_count_.Memset(0)) {
    Reset();
    return false;
  }
  return true;
}

bool CropBoxFilterWorkspace::Clear(CudaStream stream) {
  if (selected_count_.size_bytes() != sizeof(uint32_t)) {
    return false;
  }

  return selected_count_.MemsetAsync(0, stream);
}

void CropBoxFilterWorkspace::Reset() {
  selected_indices_.Reset();
  selected_count_.Reset();
  max_points_ = 0U;
}

bool CropBoxFilter::Filter(const PointCloudView& source,
                           PointCloudBuffer* output,
                           CropBoxFilterWorkspace* workspace,
                           CudaStream stream) const {
  return FilterWithStatus(source, output, workspace, stream).ok();
}

Status CropBoxFilter::FilterWithStatus(const PointCloudView& source,
                                       PointCloudBuffer* output,
                                       CropBoxFilterWorkspace* workspace,
                                       CudaStream stream) const {
  if (!config_.IsValid() || output == nullptr || workspace == nullptr ||
      !source.IsValid()) {
    return {StatusCode::kInvalidArgument,
            "invalid crop config, buffers or schema"};
  }
  if (&source == &output->view() ||
      internal::OverlapsStorage(source, output->view())) {
    return {StatusCode::kInvalidArgument,
            "in-place filtering is not supported"};
  }
  if (source.point_count() > workspace->max_points() ||
      workspace->selected_count() == nullptr ||
      workspace->selected_indices() == nullptr) {
    return {StatusCode::kInsufficientCapacity, "crop workspace is too small"};
  }
  if (!IsDeviceAccessible(source.memory_type()) ||
      !IsDeviceAccessible(workspace->index_memory_type())) {
    return {StatusCode::kUnsupportedMemory,
            "CUDA requires device/unified memory"};
  }

  const PointFieldDescriptor* x = source.FindField(PointField::kX);
  const PointFieldDescriptor* y = source.FindField(PointField::kY);
  const PointFieldDescriptor* z = source.FindField(PointField::kZ);
  if (x == nullptr || y == nullptr || z == nullptr ||
      x->scalar_type != ScalarType::kFloat32 ||
      y->scalar_type != ScalarType::kFloat32 ||
      z->scalar_type != ScalarType::kFloat32) {
    return {StatusCode::kInvalidArgument, "crop requires float32 XYZ fields"};
  }

  cudaError_t error =
      cudaMemsetAsync(workspace->selected_count(), 0, sizeof(uint32_t), stream);
  if (error != cudaSuccess) {
    return {StatusCode::kCudaError, "crop count initialization",
            static_cast<int>(error)};
  }

  if (source.point_count() > 0U) {
    constexpr uint32_t kThreadsPerBlock = 256U;
    const uint32_t blocks = 1U + (source.point_count() - 1U) / kThreadsPerBlock;
    SelectCropBoxIndicesKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        internal::MakeDeviceField(*x), internal::MakeDeviceField(*y),
        internal::MakeDeviceField(*z), source.point_count(),
        MakeDeviceConfig(config_), workspace->selected_indices(),
        workspace->selected_count());
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      return {StatusCode::kCudaError, "crop selection launch",
              static_cast<int>(error)};
    }
  }

  error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    return {StatusCode::kCudaError, "crop selection synchronization",
            static_cast<int>(error)};
  }

  const uint32_t selected_count = *workspace->selected_count();
  if (!output->AllocateLike(source, selected_count, config_.output_layout)) {
    return {StatusCode::kAllocationFailed, "crop output allocation"};
  }

  if (selected_count == 0U) {
    return {};
  }

  const internal::DevicePointCloudView device_source =
      internal::MakeDevicePointCloudView(source);
  const internal::MutableDevicePointCloudView device_output =
      internal::MakeMutableDevicePointCloudView(output->view());
  constexpr uint32_t kThreadsPerBlock = 256U;
  const uint32_t blocks = 1U + (selected_count - 1U) / kThreadsPerBlock;
  GatherSelectedPointsKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      device_source, device_output, workspace->selected_indices());
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return {StatusCode::kCudaError, "crop gather launch",
            static_cast<int>(error)};
  }

  error = cudaStreamSynchronize(stream);
  return error == cudaSuccess
             ? Status{}
             : Status{StatusCode::kCudaError, "crop gather synchronization",
                      static_cast<int>(error)};
}

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
