# ONNX schemas

These existing schemas were moved unchanged from `third_party/spconv/onnx`
when the NVIDIA spconv integration was removed. They describe the serialized
model, not a sparse-convolution implementation. Preserve the ONNX copyright
and license notices embedded in each file.

This snapshot was copied byte-for-byte from Apollo-lite's `third_party/onnx`
during common-library extraction. Both schema files retain their embedded
upstream license notices. Their earlier import provenance is described above;
no newer upstream schema revision is claimed.

`//onnx/schema:onnx_proto_cc` supplies host-side protobuf parsing for this
module's importer, independently of Apollo. This package does not execute
operators or depend on the spconv CUDA library.
