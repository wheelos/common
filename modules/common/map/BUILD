load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "map_selection",
    srcs = ["map_selection.cc"],
    hdrs = ["map_selection.h"],
    deps = [
        "//modules/common/kv_db:kv_db",
        "@core//cyber/common:log",
        "@core//cyber/common:resource_manager",
    ],
)

cc_test(
    name = "map_selection_test",
    size = "small",
    srcs = ["map_selection_test.cc"],
    deps = [
        ":map_selection",
        "@com_google_googletest//:gtest_main",
    ],
)
