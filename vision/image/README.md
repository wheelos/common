# Camera Image Geometry

`IpmTransformer` maps a calibrated camera image onto a ground-plane BEV. Its
camera-to-vehicle transform must be a finite, right-handed rigid transform
passed as `Eigen::Isometry3d`. The core target depends on Eigen and the camera
model only; image buffers use non-owning interleaved `ImageView`s and support
arithmetic scalar types, padded rows, nearest-neighbor, and bilinear sampling.
The source image dimensions must match the camera model.

Consumers using OpenCV can opt into the separate `opencv_ipm_adapter` target.
It accepts `CV_8U` and `CV_32F` images and does not add OpenCV dependencies to
the IPM core target.

The nuScenes six-camera geometry diagnostic is a standalone application under
`vision/tools`; its usage and dataset requirements are documented there. Unit
tests for camera projection and IPM do not require nuScenes data.
