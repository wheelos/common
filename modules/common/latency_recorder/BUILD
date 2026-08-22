load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "latency_recorder",
    srcs = [
        "latency_recorder.cc",
    ],
    hdrs = ["latency_recorder.h"],
    deps = [
        "@core//cyber",
        "//modules/common/adapters:adapter_gflags",
        "//modules/common/latency_recorder/proto:latency_record_cc_proto",
        "//modules/common/util:util_tool",
    ],
    visibility = ["//visibility:public"],
)
