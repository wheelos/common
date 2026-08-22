load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])


cc_library(
    name = "vehicle_state_provider",
    srcs = [
        "vehicle_state_provider.cc",
    ],
    hdrs = ["vehicle_state_provider.h"],
    deps = [
        "@core//cyber",
        "@wheelos_msgs//wheelos_msgs/chassis_msgs:chassis_cc_proto",
        "@wheelos_msgs//wheelos_msgs/localization_msgs:localization_cc_proto",
        "//modules/common/configs:config_gflags",
        "//modules/common/math",
        "//modules/common/status",
        "//modules/common/vehicle_state/proto:vehicle_state_cc_proto",
        "@eigen",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "vehicle_state_provider_test",
    size = "small",
    srcs = ["vehicle_state_provider_test.cc"],
    data = [
        ":testdata",
    ],
    deps = [
        ":vehicle_state_provider",
        "@core//cyber",
        "@wheelos_msgs//wheelos_msgs/chassis_msgs:chassis_cc_proto",
        "@wheelos_msgs//wheelos_msgs/localization_msgs:localization_cc_proto",
        "//modules/common/configs:config_gflags",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)
