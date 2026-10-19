# Camera Image Geometry

`CpuIpmTransformer` maps a calibrated, CPU-resident camera image onto a
ground-plane BEV. Its camera-to-vehicle transform must be a finite,
right-handed rigid transform passed as `Eigen::Isometry3d`. Image buffers use
non-owning interleaved `ImageView`s with explicit element type, layout, memory
type, and row stride metadata. The CPU implementation supports arithmetic
scalar types, padded rows, nearest-neighbor, and bilinear sampling. The source
image dimensions must match the camera model, and the source and destination
storage must not overlap. Projection validity is determined by the camera
model.

Consumers using OpenCV can opt into the separate `opencv_ipm_adapter` target.
It accepts `CV_8U` and `CV_32F` images and does not add OpenCV dependencies to
the CPU IPM target. The adapter creates the destination image and rejects
overlapping source and destination storage.

The `image` aggregate target currently exposes the CPU IPM capability only.
GPU image processing and image stitching are not implemented in this package;
there are no placeholder APIs that report success without doing work.

The nuScenes single-camera geometry diagnostic is a standalone application
under `vision/tools`; it auto-fits the lateral BEV bounds within a finite
forward ground range, without stitching camera images. Its usage and dataset
requirements are documented there. Unit tests for camera projection and IPM do
not require nuScenes data.
