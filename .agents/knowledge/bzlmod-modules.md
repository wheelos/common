# Bzlmod Modules in the Common Monorepo

## Goal

Keep one Git repository for `wheelos/common`, while exposing independently
usable and versioned Bazel modules such as `common.math`, `common.util`, and
`common.perception`. A consumer should be able to depend on only the modules
it needs. This separates Git repository ownership from Bzlmod dependency
boundaries; creating a module does not require creating another Git repository.

This follows the repository knowledge layout used by `wheelos/core`
(`AGENTS.md` and `.agents/knowledge/`), without copying core-specific
architecture details.

## Module Boundaries

- Split by cohesive public capability and dependency boundary, not by every
  Bazel package or source directory.
- Give each published module its own `MODULE.bazel`, public Bazel targets, and
  declared dependencies.
- Keep module-to-module dependencies explicit. For example,
  `common.perception` may depend on `common.math`; it should not reach into
  math's files through an implicit path within the monorepo.
- Preserve public C++ headers, namespaces, and include paths during the split
  unless an API migration is explicitly intended.
- Prefer a small number of stable modules first. Candidate boundaries include
  math, util, vehicle-related functionality, and perception; confirm actual
  dependency graphs before finalizing them.

## Example Layout

```text
common/                             # One Git repository
|-- MODULE.bazel                    # Monorepo development/integration root
|-- .agents/knowledge/
`-- modules/
    |-- common.math/
    |   |-- MODULE.bazel            # module(name = "common.math")
    |   |-- BUILD                   # Public targets such as :math
    |   `-- ...
    |-- common.util/
    |   |-- MODULE.bazel            # module(name = "common.util")
    |   |-- BUILD
    |   `-- ...
    `-- common.perception/
        |-- MODULE.bazel            # module(name = "common.perception")
        |-- BUILD
        `-- ...
```

Each nested module is a standalone Bazel module root. Its `MODULE.bazel`
declares only its direct external and WheelOS module dependencies. Its BUILD
files expose intentional public targets and use labels from its own module
(`//...`) or explicit external module labels (`@common.math//...`).

## Consumer Usage

Declare only required modules in a consumer's `MODULE.bazel`:

```starlark
bazel_dep(name = "common.math", version = "0.2.0")
bazel_dep(name = "common.perception", version = "0.2.0")
```

Reference their public targets from BUILD files:

```starlark
deps = [
    "@common.math//:math",
    "@common.perception//:perception",
]
```

The concrete target names should be chosen and documented as each module's
public API is defined.

## Monorepo Development and Publishing

- Keep the repository root `MODULE.bazel` as a development/integration entry
  point if useful, and configure it to use local module checkouts during
  monorepo development. Use `local_path_override` for module roots and ensure
  the root's dependencies match the module names in those roots.
- Build and test each module through its public targets, including its
  declared module dependencies; do not rely only on a root build that bypasses
  module boundaries.
- Publish one registry entry per module and version. Each entry contains that
  module's registry `MODULE.bazel` metadata and a `source.json` pointing to a
  source archive of the same Git repository. Set `strip_prefix` to the
  corresponding module subdirectory so that the extracted directory is the
  module root. Include a verified archive integrity hash and keep the URL
  stable.
- Use the configured WheelOS registry for internal modules. A module need not
  be published to Bazel Central Registry to be consumed with Bzlmod.
- Decide the release policy deliberately: a coordinated repository release
  can version all modules together; independent module versioning requires
  module-specific release tags/metadata and registry entries. In either case,
  make each published source immutable and keep registry metadata consistent
  with the source module's `MODULE.bazel`.
- Test a consumer against the published registry metadata as well as local
  overrides, so packaging and `strip_prefix` errors are caught before release.

## Migration from the Current Layout

The repository currently has a single `wheelos_common` module in the root
`MODULE.bazel`, with functionality under `modules/common/` and internal labels
such as `//modules/common/math`. Migrate incrementally:

1. Map public targets and cross-package dependencies; select a cohesive first
   module boundary.
2. Give the new module a root `MODULE.bazel` and public BUILD targets. Declare
   every external dependency it uses instead of inheriting incidental
   dependencies from the current root module.
3. Convert dependencies crossing module boundaries from in-repository labels
   such as `//modules/common/math` to explicit labels such as
   `@common.math//:math`.
4. Add local development overrides and module-scoped build/test coverage.
5. Publish the module's registry metadata and validate resolution from a small
   consumer before migrating the next boundary.
6. Migrate consumers module by module; retain compatibility aliases or the
   old aggregate module for an announced transition period if existing users
   need it.

Do not split the repository or move every package into a separate module just
to achieve names like `@common.math`. The intended result is one monorepo with
several deliberate, independently consumable Bzlmod modules.

## Current Repository References

- Root module and existing dependency declarations: `MODULE.bazel`
- Configured Bzlmod registries: `.bazelrc`
- Existing aggregate alias: `BUILD`
- Current shared package layout: `modules/common/`
