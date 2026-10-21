# Vehicle calibration

This `wheelos_common` module owns startup-time selection and structural
validation of a vehicle calibration bundle. It resolves only asset-relative paths through
`cyber::common::ResourceManager`; it does not search module source directories
or interpret calibration values.

Set `WHEELOS_VEHICLE_PROFILE_KEY` to the selected profile asset key, for
example `vehicles/cargo/config/vehicle_profile.pb.txt`. The profile selects a
`vehicle_type` and `calibration_version`; the registry then loads
`vehicles/<vehicle_type>/calibration/<calibration_version>/calibration_manifest.pb.txt`.
The profile also lists each required sensor frame, allowing hardware variants
to declare their actual calibrated inventory without fabricating entries.
Each process loads the profile once during initialization. Changing profiles
requires stopping and restarting all calibration consumers together.
Profiles marked `test_only: true` are rejected unless
`WHEELOS_ALLOW_TEST_CALIBRATION=1` is set explicitly; never set this variable
for a real vehicle.

Schema 2 separates `sensor` inventory from `static_transform` edges. It supports
multiple LiDARs, radar, ultrasonic, IMU, GNSS and cameras. Sensor frames must be
connected to `base_link`; intermediate mounting frames are allowed. Camera
optical frames must be children of their camera link. Dynamic world/map/odom
edges are owned elsewhere. A manifest lists only installed and calibrated
sensors; do not fabricate zero-valued calibration for absent hardware.

Schema 1 remains readable for existing API consumers, including its original
canonical frame restrictions and sensor-owned extrinsic fields. Do not mix
schema 1 extrinsic fields with schema 2 graph edges. New deployments should use
schema 2.
Its global `lidar_height_file` is still consumed for the single LiDAR when no
per-sensor height is specified. This compatibility does not invent missing
perception metadata; old bundles must supply it to use SensorManager.

Schema 2 requires matching profile/manifest `vehicle_id`, `provenance`,
`profile.manifest_sha256`, and an `asset { relative_path, sha256 }` entry for every
referenced file. Hashes are lowercase SHA256 of exact file bytes, not textproto
semantic hashes. The manifest does not hash itself: its hash lives in the
profile. `WHEELOS_CALIBRATION_MANIFEST_SHA256`, when set, additionally pins the
deployment-wide expected manifest. This establishes content identity, not the
physical accuracy of the calibration or an authenticity signature.

The registry checks unique IDs/frames, required sensors, graph connectivity,
cycles, duplicate children, file readability and digests. Sensor IDs must not
alias `base_link` or another graph frame. Files are resolved
under the canonical selected bundle, not by searching module directories or
re-reading the asset root for each file. Failed loads expose no partial state.
Bundles must remain immutable and read-only throughout process execution.

Perception-capable sensors reference a `metadata_file`: a single existing
perception `SensorMeta` textproto, not a `MultiSensorMeta` list. The adapter
validates name/frame/type agreement; camera metadata and perception
`SensorInfo.frame_id` use the optical frame, while other sensors use `frame_id`.
SensorManager frame queries normalize camera IDs and mechanical links to that
optical measurement frame.
Perception-specific enums do not enter the
common schema. Cameras explicitly select `camera_model: "brown"`,
`image_width`, `image_height`, `intrinsic_file` and `optical_frame_id`.
Additional model parsers require explicit support, not silent Brown fallback.

Transform validates numeric extrinsics and YAML/manifest frame agreement before
publishing the complete static graph. `T_parent_child` maps coordinates from
child to parent, in metres with unit quaternions. Perception reads the same
bundle's intrinsics. `lidar_height_file` belongs to a LiDAR sensor in schema 2;
`LoadLidarHeight` checks finite positive height and finite nonnegative variance.
Its YAML schema is `vehicle.parameters.{height,height_var}`. The loader returns
the selected LiDAR frame together with the values, so consumers do not repeat
sensor selection. Ground height is measured along nominal `base_link` +Z and
is not equivalent to the LiDAR's base-frame z translation or distance along a
tilted LiDAR axis. Transform owns derived frame/ground geometry; Common does
not depend on TF or perception types.

No vehicle asset files are installed by this module. The current cargo
measurements have not been confirmed against the new frame mapping, so they
must not be activated through this registry until their provenance is
verified.

The source consumer uses opt-in local Bzlmod overrides during development.
Changes here must be published and selected through the registry before claiming
release-mode acceptance; local override builds alone are not a release.
