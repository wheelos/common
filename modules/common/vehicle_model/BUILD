load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "vehicle_model",
    srcs = [
        "vehicle_model.cc",
        "vehicle_model_factory.cc",
    ],
    hdrs = [
        "vehicle_model.h",
        "vehicle_model_factory.h",
        "vehicle_model_input.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@core//cyber",
        "//modules/common/configs:vehicle_config_helper",
        "//modules/common/math",
        "//modules/common/status",
        "//modules/common/vehicle_model/proto:vehicle_model_config_cc_proto",
        "//modules/common/vehicle_state:vehicle_reference",
        "//modules/common/vehicle_state/proto:vehicle_state_cc_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "vehicle_model_test",
    size = "small",
    srcs = ["vehicle_model_test.cc"],
    deps = [
        ":vehicle_model",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "vehicle_model_config_data",
    srcs = ["conf/vehicle_model_config.pb.txt"],
)
