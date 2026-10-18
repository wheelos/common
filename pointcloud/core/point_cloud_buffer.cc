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

#include "pointcloud/core/point_cloud_buffer.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace wheel {
namespace vision {
namespace lidar {

namespace {

size_t AlignTo(size_t value, size_t alignment) {
  if (alignment <= 1U) {
    return value;
  }
  const size_t remainder = value % alignment;
  return remainder == 0U ? value : value + alignment - remainder;
}

}  // namespace

PointCloudBuffer::PointCloudBuffer(PointCloudBuffer&& other) noexcept {
  *this = std::move(other);
}

PointCloudBuffer& PointCloudBuffer::operator=(
    PointCloudBuffer&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  planar_buffers_ = std::move(other.planar_buffers_);
  interleaved_buffer_ = std::move(other.interleaved_buffer_);
  view_ = other.view_;
  point_stride_bytes_ = other.point_stride_bytes_;
  other.Reset();
  return *this;
}

bool PointCloudBuffer::Allocate(
    uint32_t point_count, PointLayout layout, MemoryType memory_type,
    CoordinateFrame frame,
    std::initializer_list<PointFieldSpecification> fields) {
  return Allocate(point_count, layout, memory_type, frame, fields.begin(),
                  static_cast<uint32_t>(fields.size()));
}

bool PointCloudBuffer::Allocate(uint32_t point_count, PointLayout layout,
                                MemoryType memory_type, CoordinateFrame frame,
                                const PointFieldSpecification* fields,
                                uint32_t field_count) {
  Reset();
  if (fields == nullptr || field_count == 0U || field_count > kMaxPointFields) {
    return false;
  }

  view_.set_layout(layout);
  view_.set_memory_type(memory_type);
  view_.set_frame(frame);
  view_.set_point_count(point_count);
  view_.SetFields({});
  if (!view_.IsValid()) {
    Reset();
    return false;
  }

  if (layout == PointLayout::kPlanar) {
    return AllocatePlanar(point_count, memory_type, fields, field_count);
  }

  return AllocateInterleaved(point_count, memory_type, fields, field_count);
}

bool PointCloudBuffer::AllocateLike(const PointCloudView& source,
                                    uint32_t point_count,
                                    PointLayout output_layout) {
  if (!source.IsValid() || source.field_count() == 0U) {
    return false;
  }

  std::array<PointFieldSpecification, kMaxPointFields> fields{};
  for (uint32_t index = 0U; index < source.field_count(); ++index) {
    fields[index].field = source.fields()[index].field;
    fields[index].scalar_type = source.fields()[index].scalar_type;
  }

  return Allocate(point_count, output_layout, source.memory_type(),
                  source.frame(), fields.data(), source.field_count());
}

void PointCloudBuffer::Reset() {
  for (Buffer& buffer : planar_buffers_) {
    buffer.Reset();
  }
  interleaved_buffer_.Reset();
  point_stride_bytes_ = 0U;
  view_.set_layout(PointLayout::kInterleaved);
  view_.set_memory_type(MemoryType::kHost);
  view_.set_frame(CoordinateFrame::kLidar);
  view_.set_point_count(0U);
  view_.SetFields({});
}

const PointFieldDescriptor* PointCloudBuffer::FindField(
    PointField field) const {
  return view_.FindField(field);
}

bool PointCloudBuffer::AllocatePlanar(uint32_t point_count,
                                      MemoryType memory_type,
                                      const PointFieldSpecification* fields,
                                      uint32_t field_count) {
  for (uint32_t index = 0U; index < field_count; ++index) {
    const PointFieldSpecification& specification = fields[index];
    const size_t element_size = ScalarTypeSize(specification.scalar_type);
    if (element_size == 0U ||
        point_count > std::numeric_limits<size_t>::max() / element_size) {
      Reset();
      return false;
    }

    if (point_count > 0U &&
        !planar_buffers_[index].Allocate(
            static_cast<size_t>(point_count) * element_size, memory_type)) {
      Reset();
      return false;
    }

    PointFieldDescriptor descriptor;
    descriptor.field = specification.field;
    descriptor.scalar_type = specification.scalar_type;
    descriptor.data =
        point_count == 0U ? nullptr : planar_buffers_[index].data();
    descriptor.stride_bytes = static_cast<uint32_t>(element_size);
    if (!view_.AddField(descriptor)) {
      Reset();
      return false;
    }
  }

  point_stride_bytes_ = 0U;
  return view_.IsValid();
}

bool PointCloudBuffer::AllocateInterleaved(
    uint32_t point_count, MemoryType memory_type,
    const PointFieldSpecification* fields, uint32_t field_count) {
  size_t point_stride = 0U;
  size_t max_alignment = 1U;
  for (uint32_t index = 0U; index < field_count; ++index) {
    const PointFieldSpecification& specification = fields[index];
    const size_t element_size = ScalarTypeSize(specification.scalar_type);
    if (element_size == 0U) {
      Reset();
      return false;
    }
    point_stride = AlignTo(point_stride, element_size);
    point_stride += element_size;
    max_alignment = std::max(max_alignment, element_size);
  }
  // The next point must retain alignment for every scalar, including trailing
  // small fields such as uint8 intensity after float coordinates.
  point_stride = AlignTo(point_stride, max_alignment);

  if (point_stride == 0U || point_stride > UINT32_MAX ||
      point_count > std::numeric_limits<size_t>::max() / point_stride) {
    Reset();
    return false;
  }

  point_stride_bytes_ = static_cast<uint32_t>(point_stride);
  if (point_count > 0U &&
      !interleaved_buffer_.Allocate(
          static_cast<size_t>(point_count) * point_stride, memory_type)) {
    Reset();
    return false;
  }

  uint8_t* base = interleaved_buffer_.mutable_data<uint8_t>();
  size_t offset = 0U;
  for (uint32_t index = 0U; index < field_count; ++index) {
    const PointFieldSpecification& specification = fields[index];
    const size_t element_size = ScalarTypeSize(specification.scalar_type);
    offset = AlignTo(offset, element_size);

    PointFieldDescriptor descriptor;
    descriptor.field = specification.field;
    descriptor.scalar_type = specification.scalar_type;
    descriptor.data = point_count == 0U ? nullptr : base + offset;
    descriptor.stride_bytes = point_stride_bytes_;
    if (!view_.AddField(descriptor)) {
      Reset();
      return false;
    }

    offset += element_size;
  }

  return view_.IsValid();
}

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
