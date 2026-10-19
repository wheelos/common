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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "pointcloud/memory/buffer.h"

namespace wheel {
namespace vision {
namespace lidar {

class BufferPool {
 public:
  BufferPool() = default;

  bool Init(size_t buffer_size_bytes, uint32_t capacity,
            MemoryType memory_type = MemoryType::kUnified);
  Buffer* Acquire();
  bool Release(Buffer* buffer);
  void Reset();

  inline size_t buffer_size_bytes() const { return buffer_size_bytes_; }
  inline uint32_t capacity() const {
    return static_cast<uint32_t>(buffers_.size());
  }
  inline uint32_t available_count() const {
    return static_cast<uint32_t>(available_indices_.size());
  }

 private:
  size_t buffer_size_bytes_ = 0U;
  MemoryType memory_type_ = MemoryType::kUnified;
  std::vector<std::unique_ptr<Buffer>> buffers_;
  std::vector<size_t> available_indices_;
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
