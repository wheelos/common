load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

cc_library(
    name = "kv_db",
    srcs = [
        "kv_db.cc",
    ],
    hdrs = ["kv_db.h"],
    deps = [
        "@core//cyber",
        "//modules/common/util",
        "//modules/common/util:util_tool",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/strings",
        "@sqlite3",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "kv_db_test",
    size = "small",
    srcs = ["kv_db_test.cc"],
    deps = [
        ":kv_db",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "kv_db_tool",
    srcs = ["kv_db_tool.cc"],
    deps = [
        ":kv_db",
        "@core//cyber",
        "@com_github_gflags_gflags//:gflags",
    ],
)
