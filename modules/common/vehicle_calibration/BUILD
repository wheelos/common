load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "registry",
    srcs = ["registry.cc"],
    hdrs = ["registry.h"],
    deps = [
        "//modules/common/vehicle_calibration/proto:vehicle_calibration_cc_proto",
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
        ":registry",
        "//modules/common/vehicle_calibration/proto:vehicle_calibration_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)
