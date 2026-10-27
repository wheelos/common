# Perception capabilities

This directory groups independent common-library modules. It is not a
detector runtime, Cyber integration package or an aggregate Bzlmod module.

| Module root | Identity | Ownership |
| --- | --- | --- |
| `ops/` | `wheelos_perception_ops` | Neutral sparse graph data, device API, CUDA executor and numerical tests |
| `model_io/` | `wheelos_perception_model_io` | Supported sparse ONNX importer, schemas and validation tests |

Model I/O depends only on the neutral graph target, not the executor. Model
consumers depend directly on the public targets they use. BEVFusion and
CenterPoint retain their preprocessing, heads, decoding and lifecycle;
PointPillars retains its model-specific pillar/scatter/plugin implementation
and has no mandatory spconv dependency.

There are no compatibility aliases for the old `perception_ops/ops/spconv`
source layout. Independent module source archives must include the module's
own LICENSE and schema/source provenance. Registry publication remains separate
from local override validation.

The legacy root module is now developed as `wheelos_common@0.1.3`; it still
owns existing framework-coupled packages and is not an aggregate for these
independent perception modules. Registry `wheelos_common@0.1.1.bcr.1` repairs
the older immutable source delivery, not this unpublished source snapshot.
