load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "vehicle_calibration_cc_proto",
    deps = [":vehicle_calibration_proto"],
)

proto_library(
    name = "vehicle_calibration_proto",
    srcs = ["vehicle_calibration.proto"],
)
