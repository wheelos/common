load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "association_cost",
    hdrs = ["association_cost.h"],
    deps = [
        "//tracking/core:interfaces",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "gaussian_cost",
    hdrs = ["gaussian_cost.h"],
    deps = [
        "//tracking/estimation:predictor",
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
        "//modules/common/math:geometry",
        "//tracking/core:types",
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
        "//tracking/core:types",
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
