load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "motion_model",
    srcs = ["motion_model.cc"],
    hdrs = ["motion_model.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@eigen",
    ],
)

cc_library(
    name = "multi_model_estimator",
    hdrs = ["multi_model_estimator.h"],
    deps = [
        "//tracking/core:interfaces",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "predictor",
    srcs = ["predictor.cc"],
    hdrs = ["predictor.h"],
    deps = [
        ":motion_model",
        ":multi_model_estimator",
        "//tracking/core:interfaces",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@eigen",
    ],
)
