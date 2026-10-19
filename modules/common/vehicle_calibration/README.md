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

Sensor frame IDs use the standard names `imu_link`, `gnss_antenna_link`,
optional `gnss_antenna_secondary_link`, `lidar_top_link`, and the
eight `camera_*_link` positions, each camera paired with its corresponding
`camera_*_optical_frame` and an explicit optical-frame extrinsic file. A
vehicle manifest lists only installed and calibrated sensors; absent hardware
must not be represented with fabricated zero-valued calibration. Static
sensor frames are children of `base_link`, while optical frames are children
of their camera link;
the dynamic `earth -> map -> odom -> base_link` chain is owned elsewhere.

The registry checks profile/version consistency, canonical sensor frames,
unique sensor IDs and frames, required per-sensor file references, and file
existence. Transform publishes all selected static extrinsics. Localization
reads IMU-to-GNSS antenna lever arms through static TF; only GNSS uncertainty
and ground-relative LiDAR height remain semantic calibration parameters outside
TF. Perception reads camera intrinsics from the same selected bundle.

No vehicle asset files are installed by this module. The current cargo
measurements have not been confirmed against the new frame mapping, so they
must not be activated through this registry until their provenance is
verified.
