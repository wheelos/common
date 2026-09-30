load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "types",
    hdrs = ["types.h"],
)

cc_library(
    name = "lifecycle",
    hdrs = ["lifecycle.h"],
    deps = [
        ":types",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "interfaces",
    hdrs = ["interfaces.h"],
    deps = [
        ":lifecycle",
        ":types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "multi_model_estimator",
    hdrs = ["multi_model_estimator.h"],
    deps = [
        ":interfaces",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "track_manager",
    hdrs = ["track_manager.h"],
    deps = [
        ":lifecycle",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "tracking",
    hdrs = ["multi_object_tracker.h"],
    deps = [
        ":associator",
        ":association_cost",
        ":interfaces",
        ":multi_model_estimator",
        ":track_manager",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "associator",
    srcs = ["associator.cc"],
    hdrs = ["associator.h"],
    deps = [
        ":types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

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
    name = "predictor",
    srcs = ["predictor.cc"],
    hdrs = ["predictor.h"],
    deps = [
        ":interfaces",
        ":motion_model",
        ":multi_model_estimator",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@eigen",
    ],
)

cc_library(
    name = "association_cost",
    hdrs = ["association_cost.h"],
    deps = [
        ":interfaces",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "gaussian_cost",
    hdrs = ["gaussian_cost.h"],
    deps = [
        ":predictor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@eigen",
    ],
)

cc_library(
    name = "geometry_cost",
    srcs = ["geometry_cost.cc"],
    hdrs = ["geometry_cost.h"],
    deps = [
        ":types",
        "//modules/common/math:geometry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@eigen",
    ],
)

cc_library(
    name = "embedding_cost",
    srcs = ["embedding_cost.cc"],
    hdrs = ["embedding_cost.h"],
    deps = [
        ":types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@eigen",
    ],
)

cc_library(
    name = "cost",
    hdrs = ["cost.h"],
    deps = [
        ":association_cost",
        ":embedding_cost",
        ":gaussian_cost",
        ":geometry_cost",
    ],
)

cc_test(
    name = "associator_test",
    size = "small",
    srcs = ["associator_test.cc"],
    deps = [
        ":associator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "generic_tracking_test",
    size = "small",
    srcs = ["generic_tracking_test.cc"],
    deps = [
        ":association_cost",
        ":tracking",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tracking_test",
    size = "small",
    srcs = ["tracking_test.cc"],
    deps = [
        ":association_cost",
        ":gaussian_cost",
        ":predictor",
        ":tracking",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "multi_model_test",
    size = "small",
    srcs = ["multi_model_test.cc"],
    deps = [
        ":association_cost",
        ":embedding_cost",
        ":predictor",
        ":tracking",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "predictor_cost_test",
    size = "small",
    srcs = ["predictor_cost_test.cc"],
    deps = [
        ":cost",
        ":predictor",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)
