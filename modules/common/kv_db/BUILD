load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

cc_library(
    name = "kv_db",
    srcs = [
        "kv_db.cc",
    ],
    hdrs = ["kv_db.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//modules/common/util",
        "//modules/common/util:util_tool",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/strings",
        "@core//cyber",
        "@sqlite3",
    ],
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
    visibility = ["//visibility:public"],
    srcs = ["kv_db_tool.cc"],
    deps = [
        ":kv_db",
        "@com_github_gflags_gflags//:gflags",
        "@core//cyber",
    ],
)
