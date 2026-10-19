load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "assignment_solver",
    srcs = ["assignment_solver.cc"],
    hdrs = ["assignment_solver.h"],
    deps = [
        "//tracking/core:types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "association_strategy",
    srcs = ["association_strategy.cc"],
    hdrs = ["association_strategy.h"],
    deps = [
        ":assignment_solver",
        "//tracking/core:types",
        "@com_google_absl//absl/status:statusor",
    ],
)
