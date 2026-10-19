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
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace wheel {
namespace vision {
namespace image {

enum class ImageDataType : uint8_t {
  kBool,
  kInt8,
  kUInt8,
  kInt16,
  kUInt16,
  kInt32,
  kUInt32,
  kInt64,
  kUInt64,
  kFloat32,
  kFloat64,
};

enum class ImageLayout : uint8_t {
  kInterleaved,
};

enum class ImageMemoryType : uint8_t {
  kCpu,
};

struct ImageProperties {
  int width = 0;
  int height = 0;
  int channels = 0;
  ImageDataType data_type = ImageDataType::kUInt8;
  ImageLayout layout = ImageLayout::kInterleaved;
  ImageMemoryType memory_type = ImageMemoryType::kCpu;
  std::size_t row_stride_bytes = 0;
};

namespace internal {

template <typename Scalar>
constexpr ImageDataType GetImageDataType() {
  using ValueType = typename std::remove_cv<Scalar>::type;
  static_assert(std::is_arithmetic<ValueType>::value,
                "Image element type must be arithmetic.");

  if constexpr (std::is_same<ValueType, bool>::value) {
    return ImageDataType::kBool;
  } else if constexpr (std::is_floating_point<ValueType>::value) {
    static_assert(sizeof(ValueType) == 4 || sizeof(ValueType) == 8,
                  "Image floating-point elements must be 32-bit or 64-bit.");
    return sizeof(ValueType) == 4 ? ImageDataType::kFloat32
                                  : ImageDataType::kFloat64;
  } else {
    static_assert(sizeof(ValueType) == 1 || sizeof(ValueType) == 2 ||
                      sizeof(ValueType) == 4 || sizeof(ValueType) == 8,
                  "Image integer elements must be 8, 16, 32, or 64 bits.");
    if constexpr (sizeof(ValueType) == 1) {
      return std::is_signed<ValueType>::value ? ImageDataType::kInt8
                                              : ImageDataType::kUInt8;
    } else if constexpr (sizeof(ValueType) == 2) {
      return std::is_signed<ValueType>::value ? ImageDataType::kInt16
                                              : ImageDataType::kUInt16;
    } else if constexpr (sizeof(ValueType) == 4) {
      return std::is_signed<ValueType>::value ? ImageDataType::kInt32
                                              : ImageDataType::kUInt32;
    } else {
      return std::is_signed<ValueType>::value ? ImageDataType::kInt64
                                              : ImageDataType::kUInt64;
    }
  }
}

}  // namespace internal

// A non-owning, CPU-accessible view of an interleaved image. row_stride is in
// elements; ImageProperties reports the equivalent stride in bytes.
template <typename Scalar>
struct ImageView {
  Scalar* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 0;
  std::size_t row_stride = 0;

  std::size_t RowStride() const {
    return row_stride == 0 ? static_cast<std::size_t>(width) *
                                 static_cast<std::size_t>(channels)
                           : row_stride;
  }

  ImageProperties properties() const {
    using ValueType = typename std::remove_const<Scalar>::type;
    if (width <= 0 || height <= 0 || channels <= 0) {
      throw std::invalid_argument("ImageView::properties: invalid dimensions.");
    }
    constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
    const std::size_t image_width = static_cast<std::size_t>(width);
    const std::size_t image_channels = static_cast<std::size_t>(channels);
    if (image_width > kMax / image_channels) {
      throw std::overflow_error("ImageView::properties: row size overflow.");
    }
    const std::size_t row_stride_elements = RowStride();
    if (row_stride_elements < image_width * image_channels) {
      throw std::invalid_argument("ImageView::properties: invalid row stride.");
    }
    if (row_stride_elements > kMax / sizeof(ValueType)) {
      throw std::overflow_error("ImageView::properties: row size overflow.");
    }
    return {width,
            height,
            channels,
            internal::GetImageDataType<ValueType>(),
            ImageLayout::kInterleaved,
            ImageMemoryType::kCpu,
            row_stride_elements * sizeof(ValueType)};
  }
};

namespace internal {

template <typename Scalar>
bool IsValidImageView(const ImageView<Scalar>& image) {
  using ValueType = typename std::remove_const<Scalar>::type;
  static_assert(std::is_arithmetic<ValueType>::value,
                "ImageView scalar type must be arithmetic.");
  if (image.data == nullptr || image.width <= 0 || image.height <= 0 ||
      image.channels <= 0) {
    return false;
  }

  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  const std::size_t width = static_cast<std::size_t>(image.width);
  const std::size_t channels = static_cast<std::size_t>(image.channels);
  if (width > kMax / channels) {
    return false;
  }
  const std::size_t packed_stride = width * channels;
  const std::size_t row_stride = image.RowStride();
  if (row_stride < packed_stride || static_cast<std::size_t>(image.height - 1) >
                                        (kMax - packed_stride) / row_stride) {
    return false;
  }
  const std::size_t span_elements =
      static_cast<std::size_t>(image.height - 1) * row_stride + packed_stride;
  return span_elements <= kMax / sizeof(ValueType);
}

template <typename Scalar>
std::size_t ImageStorageSpanBytes(const ImageView<Scalar>& image) {
  const std::size_t packed_stride = static_cast<std::size_t>(image.width) *
                                    static_cast<std::size_t>(image.channels);
  const std::size_t span_elements =
      static_cast<std::size_t>(image.height - 1) * image.RowStride() +
      packed_stride;
  return span_elements * sizeof(typename std::remove_const<Scalar>::type);
}

inline bool ImageStorageOverlaps(const void* first_data,
                                 std::size_t first_size_bytes,
                                 const void* second_data,
                                 std::size_t second_size_bytes) {
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first_data);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second_data);
  constexpr std::uintptr_t kMax = std::numeric_limits<std::uintptr_t>::max();
  if (first_size_bytes > kMax - first_begin ||
      second_size_bytes > kMax - second_begin) {
    return true;
  }
  const std::uintptr_t first_end = first_begin + first_size_bytes;
  const std::uintptr_t second_end = second_begin + second_size_bytes;
  return first_begin < second_end && second_begin < first_end;
}

template <typename FirstScalar, typename SecondScalar>
bool ImageViewsOverlap(const ImageView<FirstScalar>& first,
                       const ImageView<SecondScalar>& second) {
  return ImageStorageOverlaps(first.data, ImageStorageSpanBytes(first),
                              second.data, ImageStorageSpanBytes(second));
}

}  // namespace internal
}  // namespace image
}  // namespace vision
}  // namespace wheel
