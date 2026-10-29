# WheelOS Perception Operators

This package is maintained inside the `wheelos_common` Bzlmod module.

This package contains reusable perception compute operators. Its first
capability is the CUDA sparse-convolution graph executor backed by the
repository's declared `spconv@2.3.8.bcr.3` dependency.

The package owns the model-neutral sparse graph data structures, execution
API, CUDA implementation, and their direct spconv dependency. It does not own
ONNX schemas/parsers, voxelizers, model weights, TensorRT heads, postprocessing,
or Cyber lifecycle. Consumers parse their model format into the sparse graph
API, validate graph attributes/weights and input/output contracts, and pass
the validated graph to the executor.

## Sparse-convolution target

Consumers declare a direct dependency on `wheelos_common` and use:

```text
@wheelos_common//perception/ops/spconv:spconv_backbone
```

The API accepts a validated sparse graph description and exposes stream-aware
forward execution. Output storage belongs to the executor instance until its
next forward call or destruction. An instance is not safe for concurrent use
or for use across CUDA devices. See `spconv/` for the public API and
operator-specific tests/constraints.

`//perception/ops/spconv:sparse_model` exposes only standard-C++ graph data. Format import
belongs to `//perception/model_io/onnx`, not this package. CPU graph consumers
do not configure a CUDA toolkit; execution targets explicitly depend on CUDA.
GPU tests live at `//perception/ops/spconv:spconv_backbone_test` and compare a small sparse
convolution/scatter graph against a CPU reference on two nondefault streams.
Run these tests with the common repository as the Bazel root.
`test_module/` is a separate consumer root that builds `//:executor_consumer`
and links `//:executor_consumer_link` against the public executor without
importing the library's test dependencies. The link target rejects undefined
symbols, so a host-only archive cannot pass as a complete CUDA executor.
It declares `rules_cuda` directly because its build flags use that module's
settings. Run it with the same SM89, C++17 and optimized-build flags.

The common module is currently consumed through Apollo's opt-in local Bzlmod override
in `.bazelrc.dev`; it is not yet published to a remote registry.

The local registry's spconv revision runs upstream PCCM generation as a Bazel
action, using pinned Python dependencies and an explicit, content-checked
output manifest for CUDA 12.8 / SM89. No generated-source archive or
`SPCONV_SOURCE_ROOT` is required. Generation currently requires a Linux x86_64
execution platform; Python and cumm's native helper are build tools only.
Other GPU architectures require separate generation and runtime validation.
The executor requires `--@rules_cuda//cuda:archs=sm_89` at analysis time,
CUDA 12.8 at compile time and an SM89 device at runtime.
