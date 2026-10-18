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

#include "pointcloud/memory/buffer_pool.h"

#include <algorithm>

namespace wheel {
namespace vision {
namespace lidar {

bool BufferPool::Init(size_t buffer_size_bytes, uint32_t capacity,
                      MemoryType memory_type) {
  Reset();
  if (buffer_size_bytes == 0U || capacity == 0U) {
    return false;
  }

  buffers_.reserve(capacity);
  available_indices_.reserve(capacity);
  for (uint32_t index = 0U; index < capacity; ++index) {
    std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>();
    if (!buffer->Allocate(buffer_size_bytes, memory_type)) {
      Reset();
      return false;
    }
    buffers_.push_back(std::move(buffer));
    available_indices_.push_back(index);
  }

  buffer_size_bytes_ = buffer_size_bytes;
  memory_type_ = memory_type;
  return true;
}

Buffer* BufferPool::Acquire() {
  if (available_indices_.empty()) {
    return nullptr;
  }

  const size_t index = available_indices_.back();
  available_indices_.pop_back();
  return buffers_[index].get();
}

bool BufferPool::Release(Buffer* buffer) {
  if (buffer == nullptr) {
    return false;
  }

  for (size_t index = 0U; index < buffers_.size(); ++index) {
    if (buffers_[index].get() != buffer) {
      continue;
    }

    if (std::find(available_indices_.begin(), available_indices_.end(),
                  index) != available_indices_.end()) {
      return false;
    }

    available_indices_.push_back(index);
    return true;
  }

  return false;
}

void BufferPool::Reset() {
  available_indices_.clear();
  buffers_.clear();
  buffer_size_bytes_ = 0U;
  memory_type_ = MemoryType::kUnified;
}

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
