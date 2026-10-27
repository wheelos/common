# Perception Operators Agent Guide

- Keep reusable perception operators independent of model frameworks, ONNX
  schemas, Cyber, and detector-specific configuration.
- Declare every external build dependency in `MODULE.bazel`; consumers depend
  on public targets rather than reaching into source directories.
- Read `README.md` and relevant operator tests before changing an API.
- Keep neutral graph types independent of CUDA headers and model formats.
- Build and test in the managed development container using the current mapped
  non-root user and the repository's shared Bazel cache.
