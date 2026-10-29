# Perception model I/O

The `wheelos_common` package `//perception/model_io/onnx:sparse_model` owns
supported model-format import, not execution. It imports the FP16 sparse ONNX
dialect used by BEVFusion and CenterPoint into
`//perception/ops/spconv:sparse_model`.

The importer validates operators, shapes, attributes, weights and graph
connectivity. It neither executes CUDA nor depends on CUDA headers, TensorRT,
Cyber, Apollo source paths or a detector module. Model feature/layout
constraints remain explicit; this is not a general-purpose ONNX runtime.
Errors are reported as exceptions with the existing `spconv ONNX:` prefix.

The schema snapshot under `onnx/schema` preserves the source and license
used before extraction. Tests live at
`//perception/model_io/onnx:sparse_model_test`.
Run tests with the common repository as the Bazel root.
`test_module/` is a separate consumer root that builds
`//:importer_consumer` using the public importer and neutral graph APIs.
The common module is developed through an explicit local Bzlmod override;
there is no independently published model-I/O module.
