# Perception model I/O

`wheelos_perception_model_io` owns supported model-format import, not execution.
`//onnx:sparse_model` imports the FP16 sparse ONNX dialect used by BEVFusion
and CenterPoint into `wheelos_perception_ops//spconv:sparse_model`.

The importer validates operators, shapes, attributes, weights and graph
connectivity. It neither executes CUDA nor depends on CUDA headers, TensorRT,
Cyber, Apollo source paths or a detector module. Model feature/layout
constraints remain explicit; this is not a general-purpose ONNX runtime.
Errors are reported as exceptions with the existing `spconv ONNX:` prefix.

The schema snapshot under `onnx/schema` preserves the source and license
used before extraction. Tests live at `//onnx:sparse_model_test`.
Run tests with this directory as the Bazel root and an explicit override for
the sibling ops module. Googletest is development-only, not a product
dependency. `test_module/` is a separate consumer root that builds
`//:importer_consumer` using the public importer and neutral graph APIs.
This module is developed through explicit local Bzlmod overrides and is not
yet published to a registry.
