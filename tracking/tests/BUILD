load("@rules_cc//cc:defs.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "assignment_solver_test",
    size = "small",
    srcs = ["assignment_solver_test.cc"],
    deps = [
        "//tracking/association:assignment_solver",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "association_strategy_test",
    size = "small",
    srcs = ["association_strategy_test.cc"],
    deps = [
        "//tracking/association:association_strategy",
        "//tracking/association:assignment_solver",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "generic_tracking_test",
    size = "small",
    srcs = ["generic_tracking_test.cc"],
    deps = [
        "//tracking/association:association_strategy",
        "//tracking/association:assignment_solver",
        "//tracking/core:tracking",
        "//tracking/costs:association_cost",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tracking_test",
    size = "small",
    srcs = ["tracking_test.cc"],
    deps = [
        "//tracking/association:assignment_solver",
        "//tracking/core:tracking",
        "//tracking/costs:association_cost",
        "//tracking/costs:gaussian_cost",
        "//tracking/estimation:predictor",
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
        "//tracking/association:assignment_solver",
        "//tracking/core:tracking",
        "//tracking/costs:association_cost",
        "//tracking/costs:embedding_cost",
        "//tracking/estimation:predictor",
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
        "//tracking/costs:cost",
        "//tracking/estimation:predictor",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)
