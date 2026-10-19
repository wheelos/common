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

#include "pointcloud/core/point_cloud.h"

#include <limits>

namespace wheel {
namespace vision {
namespace lidar {

namespace {

bool IsValidField(const PointFieldDescriptor& field, uint32_t point_count) {
  const size_t size = field.element_size();
  if (field.field < PointField::kX || field.field > PointField::kTimestamp ||
      size == 0U || field.stride_bytes < size ||
      field.stride_bytes % size != 0U) {
    return false;
  }
  if (point_count == 0U) {
    return true;
  }
  const uintptr_t address = reinterpret_cast<uintptr_t>(field.data);
  const uint64_t span =
      static_cast<uint64_t>(point_count - 1U) * field.stride_bytes + size;
  return field.data != nullptr && address % size == 0U &&
         span <= std::numeric_limits<uintptr_t>::max() - address;
}

}  // namespace

PointCloudView::PointCloudView(
    PointLayout layout, MemoryType memory_type, CoordinateFrame frame,
    uint32_t point_count, std::initializer_list<PointFieldDescriptor> fields)
    : layout_(layout),
      memory_type_(memory_type),
      frame_(frame),
      point_count_(point_count) {
  SetFields(fields);
}

bool PointCloudView::SetFields(
    std::initializer_list<PointFieldDescriptor> fields) {
  schema_valid_ = true;
  field_count_ = 0U;
  for (const PointFieldDescriptor& field : fields) {
    if (!AddField(field)) {
      schema_valid_ = false;
      field_count_ = 0U;
      return false;
    }
  }
  return true;
}

bool PointCloudView::AddField(const PointFieldDescriptor& field) {
  if (field_count_ >= kMaxPointFields || !IsValidField(field, point_count_)) {
    schema_valid_ = false;
    return false;
  }

  if (HasField(field.field)) {
    schema_valid_ = false;
    return false;
  }

  fields_[field_count_] = field;
  ++field_count_;
  return true;
}

const PointFieldDescriptor* PointCloudView::FindField(PointField field) const {
  for (uint32_t index = 0U; index < field_count_; ++index) {
    if (fields_[index].field == field) {
      return &fields_[index];
    }
  }
  return nullptr;
}

bool PointCloudView::HasField(PointField field) const {
  return FindField(field) != nullptr;
}

bool PointCloudView::IsValid() const {
  if (!schema_valid_ ||
      (layout_ != PointLayout::kPlanar &&
       layout_ != PointLayout::kInterleaved) ||
      memory_type_ < MemoryType::kHost || memory_type_ > MemoryType::kUnified ||
      frame_ < CoordinateFrame::kLidar || frame_ > CoordinateFrame::kWorld) {
    return false;
  }

  for (uint32_t index = 0U; index < field_count_; ++index) {
    const PointFieldDescriptor& field = fields_[index];
    if (!IsValidField(field, point_count_)) {
      return false;
    }
  }

  return true;
}

bool PointCloudView::IsRangeImageCompatible() const {
  if (!IsValid()) {
    return false;
  }

  if (point_count_ == 0U) {
    return true;
  }

  const PointFieldDescriptor* x = FindField(PointField::kX);
  const PointFieldDescriptor* y = FindField(PointField::kY);
  const PointFieldDescriptor* z = FindField(PointField::kZ);
  if (x == nullptr || y == nullptr || z == nullptr) {
    return false;
  }

  if (x->scalar_type != ScalarType::kFloat32 ||
      y->scalar_type != ScalarType::kFloat32 ||
      z->scalar_type != ScalarType::kFloat32) {
    return false;
  }

  const PointFieldDescriptor* intensity = FindField(PointField::kIntensity);
  return intensity == nullptr || intensity->scalar_type == ScalarType::kFloat32;
}

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
