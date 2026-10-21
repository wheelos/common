load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "registry",
    srcs = ["registry.cc"],
    hdrs = ["registry.h"],
    deps = [
        "//modules/common/vehicle_calibration/proto:vehicle_calibration_cc_proto",
        "@boringssl//:crypto",
        "@core//cyber/common:file",
        "@core//cyber/common:log",
        "@core//cyber/common:resource_manager",
    ],
)

cc_test(
    name = "registry_test",
    size = "small",
    srcs = ["registry_test.cc"],
    deps = [
        ":lidar_height",
        ":registry",
        ":test_bundle",
        "//modules/common/vehicle_calibration/proto:vehicle_calibration_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_bundle",
    testonly = True,
    hdrs = ["test_bundle.h"],
    deps = [":registry"],
)

cc_library(
    name = "lidar_height",
    srcs = ["lidar_height.cc"],
    hdrs = ["lidar_height.h"],
    deps = [
        ":registry",
        "@com_github_jbeder_yaml_cpp//:yaml-cpp",
        "@core//cyber/common:log",
    ],
)
