# Vision Tools

## nuScenes six-camera BEV diagnostic

`nuscenes_bev_diagnostic` reads one nuScenes sample's six camera images,
calibrated sensor records, and ego poses, then projects and equally blends
valid pixels into a 60 m square BEV at 0.1 m/pixel. It is a manual geometry
diagnostic, not a seam-optimized panorama or a calibration-quality metric. It
assumes a flat ground plane and does not perform photometric correction.

Run it from Apollo-lite's managed development container:

```bash
bazel run --config=dev --lockfile_mode=off \
  @wheelos_common//vision/tools:nuscenes_bev_diagnostic -- \
  <dataset-root> <metadata-dir> <sample-token> <output.png>
```

`dataset-root` must contain the extracted `samples/` directory referenced by
`sample_data.json`; `metadata-dir` must contain `sample.json`,
`sample_data.json`, `calibrated_sensor.json`, `sensor.json`, and `ego_pose.json`
for the same nuScenes version. The sample must include all six camera channels.
Both paths must be readable from inside the development container; a host-only
NAS path is not automatically available there. The BEV is expressed in the
front camera's ego-pose frame, converted to the package's vehicle convention
(x right, y forward, z up).
