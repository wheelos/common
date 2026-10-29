# Perception capabilities

This directory groups packages in the single `wheelos_common` Bzlmod module.
Keep compute and model-format responsibilities separate without publishing
independently versioned perception modules for now.

| Package group | Module | Ownership |
| --- | --- | --- |
| `ops/` | `wheelos_common` | Neutral sparse graph data, device API, CUDA executor and numerical tests |
| `model_io/` | `wheelos_common` | Supported sparse ONNX importer, schemas and validation tests |

Model I/O depends only on the neutral graph target, not the executor. Model
consumers declare `wheelos_common` and use its public package targets. BEVFusion and
CenterPoint retain their preprocessing, heads, decoding and lifecycle;
PointPillars retains its model-specific pillar/scatter/plugin implementation
and has no mandatory spconv dependency.

Use `@wheelos_common//perception/ops/spconv:spconv_backbone` and
`@wheelos_common//perception/model_io/onnx:sparse_model`. Package-level dependencies
keep the importer independent of CUDA compilation. Registry publication remains
separate from local override validation; the released `v0.1.3` archive needs
the registry's consolidation patch to expose this layout.
