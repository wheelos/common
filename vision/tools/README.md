# Vision Tools

## nuScenes single-camera BEV diagnostic

`nuscenes_bev_diagnostic` reads one camera image from a nuScenes sample with its
calibration and ego pose, then projects it into a single-camera BEV. It does not
stitch camera images. The lateral bounds are fit to the camera's ground-plane
projection within a 0–60 m forward range, at 0.1 m/pixel. Image rays that do not
intersect the ground plane within this range are not included. It writes PNG
with lossless encoding and compression level 0; it does not resize the source
before projection or use JPEG/lossy output compression. The BEV projection
resamples pixels geometrically, so its dimensions differ from the source. This
is a manual geometry diagnostic, not a calibration-quality metric. It assumes
a flat ground plane and does not perform photometric correction.

Run it from Apollo-lite's managed development container:

```bash
bazel run --config=dev --lockfile_mode=off \
  @wheelos_common//vision/tools:nuscenes_bev_diagnostic -- \
  <dataset-root> <metadata-dir> <sample-token> <camera-channel> <output.png>
```

`dataset-root` must contain the extracted `samples/` directory referenced by
`sample_data.json`; `metadata-dir` must contain `sample.json`,
`sample_data.json`, `calibrated_sensor.json`, `sensor.json`, and `ego_pose.json`
for the same nuScenes version. `camera-channel` must be one of `CAM_FRONT`,
`CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, or
`CAM_BACK_RIGHT`. Both paths must be readable from inside the development
container; a host-only NAS path is not automatically available there. The BEV
is expressed in the selected camera sample's ego-pose frame, converted to the
package's vehicle convention (x right, y forward, z up).
