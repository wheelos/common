# Point Cloud

`pointcloud` provides reusable lidar data, memory, and CUDA preprocessing
primitives in `wheelos_common`. It is independent of the camera packages under
`vision/`.

## Capabilities

- Non-owning planar/interleaved `PointCloudView` and owning `PointCloudBuffer`.
- Host, pinned-host, device, and unified-memory `Buffer`; fixed-capacity
  `BufferPool`.
- CUDA decimation and crop-box filters.
- Spherical range-image projection with optional rigid transform, nearest-hit
  selection, and range, intensity, and validity outputs.

## Use

Depend on the umbrella target:

```bzl
deps = ["@wheelos_common//pointcloud:pointcloud"]
```

Include the point-cloud view:

```cpp
#include "pointcloud/core/point_cloud.h"
```

Run the package tests in Apollo-lite's managed GPU development container:

```bash
bazel test --config=dev --config=gpu --lockfile_mode=off \
  @wheelos_common//pointcloud:all
```

## API notes

- CUDA operations require device-accessible input and output memory.
- Async calls enqueue work on the supplied stream. Keep referenced storage
  alive and unchanged until that stream completes.
- `BuildWithStatus` and the filter `WithStatus` methods provide per-call
  diagnostics; existing `bool` methods remain supported.
- Coordinates and translations use meters. Fields such as intensity, ring,
  and timestamp retain sensor-native meanings; the library does not normalize
  intensity, infer timestamp units, or deskew points.
- Range-image projection is spherical, not ring-organized. Wrapped azimuth
  sectors must be split by the caller. Pixel collisions keep the nearest point;
  equal-range ties select the smallest sanitized intensity.

This package is a GPU-native foundation, not a complete DriveWorks replacement:
organized-cloud conversion, ring-aware projection, voxel filtering,
registration, segmentation, and fusion are not implemented.
