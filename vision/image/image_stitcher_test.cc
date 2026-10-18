#include "image/image_stitcher.h"

#include "gtest/gtest.h"

namespace wheel {
namespace vision {
namespace image {
namespace {

TEST(ImageStitcherTest, StitchSucceeds) {
  const ImageStitcher stitcher;
  EXPECT_TRUE(stitcher.Stitch());
}

}  // namespace
}  // namespace image
}  // namespace vision
}  // namespace wheel
