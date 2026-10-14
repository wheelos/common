load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "store",
    srcs = ["store.cc"],
    hdrs = ["store.h"],
    deps = ["@sqlite3//:sqlite3"],
)

cc_library(
    name = "worker",
    srcs = ["worker.cc"],
    hdrs = ["worker.h"],
    deps = [":store"],
)

cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    deps = [":worker"],
)

cc_library(
    name = "flags",
    srcs = ["flags.cc"],
    hdrs = ["flags.h"],
    deps = ["@com_github_gflags_gflags//:gflags"],
)

cc_test(
    name = "store_test",
    srcs = ["store_test.cc"],
    deps = [
        ":client",
        ":store",
        ":worker",
        "@com_google_googletest//:gtest_main",
        "@sqlite3//:sqlite3",
    ],
)
