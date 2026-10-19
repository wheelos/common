load("@rules_cc//cc:defs.bzl", "cc_library")

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
        ":interfaces",
        ":track_manager",
        "//tracking/association:association_strategy",
        "//tracking/association:assignment_solver",
        "//tracking/costs:association_cost",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
