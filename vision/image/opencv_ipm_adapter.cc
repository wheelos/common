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

#include "image/opencv_ipm_adapter.h"

#include <limits>
#include <stdexcept>

namespace wheel {
namespace vision {
namespace image {
namespace {

std::size_t MatStorageSpanBytes(const cv::Mat& image) {
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  if (image.cols <= 0 || image.rows <= 0 || image.elemSize() == 0) {
    throw std::invalid_argument("TransformOpenCv: invalid image storage.");
  }
  const std::size_t pixel_bytes = image.elemSize();
  if (static_cast<std::size_t>(image.cols) > kMax / pixel_bytes) {
    throw std::invalid_argument("TransformOpenCv: invalid image storage.");
  }
  const std::size_t row_bytes =
      static_cast<std::size_t>(image.cols) * pixel_bytes;
  if (image.step[0] < row_bytes || static_cast<std::size_t>(image.rows - 1) >
                                       (kMax - row_bytes) / image.step[0]) {
    throw std::invalid_argument("TransformOpenCv: invalid image storage.");
  }
  return static_cast<std::size_t>(image.rows - 1) * image.step[0] + row_bytes;
}

bool MatStorageOverlaps(const cv::Mat& first, const cv::Mat& second) {
  return internal::ImageStorageOverlaps(first.data, MatStorageSpanBytes(first),
                                        second.data,
                                        MatStorageSpanBytes(second));
}

template <typename Scalar>
void TransformTyped(const CpuIpmTransformer& transformer,
                    const cv::Mat& src_img, cv::Mat* dst_bev_img,
                    InterpolationMode interpolation) {
  const int channels = src_img.channels();
  const std::size_t src_stride = src_img.step1();
  dst_bev_img->create(transformer.bev_height(), transformer.bev_width(),
                      CV_MAKETYPE(cv::DataType<Scalar>::type, channels));
  transformer.Transform(
      ImageView<const Scalar>{src_img.ptr<Scalar>(), src_img.cols, src_img.rows,
                              channels, src_stride},
      ImageView<Scalar>{dst_bev_img->ptr<Scalar>(), dst_bev_img->cols,
                        dst_bev_img->rows, channels, dst_bev_img->step1()},
      interpolation);
}

}  // namespace

void TransformOpenCv(const CpuIpmTransformer& transformer,
                     const cv::Mat& src_img, cv::Mat* dst_bev_img,
                     InterpolationMode interpolation) {
  if (dst_bev_img == nullptr) {
    throw std::invalid_argument(
        "TransformOpenCv: destination image must not be null.");
  }
  if (src_img.empty()) {
    throw std::invalid_argument("TransformOpenCv: source image is empty.");
  }
  if (!dst_bev_img->empty() && MatStorageOverlaps(src_img, *dst_bev_img)) {
    throw std::invalid_argument(
        "TransformOpenCv: source and destination images must not overlap.");
  }

  if (src_img.depth() == CV_8U) {
    TransformTyped<unsigned char>(transformer, src_img, dst_bev_img,
                                  interpolation);
  } else if (src_img.depth() == CV_32F) {
    TransformTyped<float>(transformer, src_img, dst_bev_img, interpolation);
  } else {
    throw std::invalid_argument(
        "TransformOpenCv: image depth must be CV_8U or CV_32F.");
  }
}

}  // namespace image
}  // namespace vision
}  // namespace wheel
