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

#include <opencv2/core.hpp>

#include "image/cpu_ipm_transformer.h"

namespace wheel {
namespace vision {
namespace image {

// OpenCV convenience adapter for CV_8U and CV_32F images. OpenCV is required
// only by this adapter, not by CpuIpmTransformer.
void TransformOpenCv(
    const CpuIpmTransformer& transformer, const cv::Mat& src_img,
    cv::Mat* dst_bev_img,
    InterpolationMode interpolation = InterpolationMode::kBilinear);

}  // namespace image
}  // namespace vision
}  // namespace wheel
