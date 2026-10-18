#pragma once

#include <opencv2/core.hpp>

#include "image/ipm_transformer.h"

namespace wheel {
namespace vision {
namespace image {

// OpenCV convenience adapter for 8-bit and 32-bit floating-point images.
// OpenCV is required only by this adapter, not by IpmTransformer.
void TransformOpenCv(
    const IpmTransformer& transformer, const cv::Mat& src_img,
    cv::Mat* dst_bev_img,
    InterpolationMode interpolation = InterpolationMode::kBilinear);

}  // namespace image
}  // namespace vision
}  // namespace wheel
