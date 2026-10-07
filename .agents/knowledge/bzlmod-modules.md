# Bzlmod Modules in the Common Monorepo

## Goal

Keep one Git repository for `wheelos/common`, while exposing independently
usable and versioned capability modules such as `estimation`, `vision`, and
`perception`. A consumer should be able to depend on only the modules it
needs. This separates Git repository ownership from Bzlmod dependency
boundaries; creating a module does not require creating another Git repository.

This follows the repository knowledge layout used by `wheelos/core`
(`AGENTS.md` and `.agents/knowledge/`), without copying core-specific
architecture details.

## Module Boundaries

- Split by cohesive public capability and dependency boundary, not by every
  Bazel package or source directory.
- Give each published module its own `MODULE.bazel`, public Bazel targets, and
  declared dependencies.
- Place genuinely reusable, low-level algorithms in shared capability
  modules, rather than copying them into each business module or putting all
  reusable code into one broad `common` target.
- Keep domain-specific types and integration layers with their owning domain.
  A camera geometry primitive may be reusable across perception stacks while
  still belonging to a vision/geometry capability rather than generic math.
- Keep module-to-module dependencies explicit. For example, `perception` may
  depend on `estimation`; it should not reach into estimation's files through
  an implicit path within the monorepo.
- Preserve public C++ headers, namespaces, and include paths during the split
  unless an API migration is explicitly intended.
- Prefer a small number of stable modules first. Candidate boundaries include
  math, util, vehicle-related functionality, and perception; confirm actual
  dependency graphs before finalizing them.

## Reuse Across Business Modules

Use capability names directly as Bzlmod module names; do not add a redundant
`common.` prefix. For example, consumers should be able to depend on
`estimation`, `vision`, or `optimization`. When multiple perception or control
implementations need the same algorithm, share it through a narrow capability
module with a stable API:

| Reusable capability | Example boundary | Keep out of the shared API |
| --- | --- | --- |
| Generic Kalman filters / state estimation | `estimation` | Perception object types, tracker lifecycle, sensor-framework integration |
| Camera intrinsics and coordinate transforms | `vision` or `geometry` | Camera drivers, perception pipeline stages, model/runtime integration |
| PID controller primitives | `pid` (or a broader `control` capability if it exists) | Vehicle-specific control policy and module orchestration |
| Quadratic-programming API or solver adapter | `optimization` | Making every consumer depend on OSQP when it does not solve a QP |

These are candidate boundaries, not a requirement to create one module per
algorithm. Combine capabilities when they have a cohesive API, similar
consumers, and compatible release needs. Keep separate modules when consumers
should be able to depend on or version them independently.

For solver libraries such as OSQP, distinguish the third-party dependency
from WheelOS's reusable API. A module that directly uses OSQP should declare
that dependency itself. Add a shared optimization module only when it provides
a useful stable abstraction or common solver integration; avoid a wrapper
that merely renames upstream targets.

Keep reusable algorithms independent of perception/control framework types
where practical. Put framework adapters in the consuming business module.
This keeps the dependency graph one-way: business modules depend on reusable
capabilities, while generic capabilities do not depend on a particular
perception or control implementation.

## Candidate Shared-Capability Inventory

Use function and dependency direction as the first principles for deciding
what belongs in `modules/common`:

1. **Function:** Is this a stable algorithm/capability, or application
   orchestration, runtime integration, configuration, or product policy?
2. **Consumers:** Do at least two independent modules need the same semantics,
   not merely similarly named code?
3. **Dependencies:** Can the capability avoid perception/control-specific
   types, message schemas, Cyber lifecycle, hardware SDKs, and global flags?
4. **API:** Can it expose a small, testable public API with clear units,
   coordinate frames, ownership, and error behavior?
5. **Lifecycle:** Does independent reuse/versioning justify another module,
   rather than simply a separate target/package in an existing module?

The table is a candidate inventory based on the current `modules/common/`
layout in this repository and observed Apollo-lite patterns. Apollo-lite's
`modules/perception/common`, `modules/control/common`, and similar directories
are domain-scoped Bazel packages inside its root `apollo` module, not published
standalone Bzlmod modules. The findings indicate extraction opportunities,
not proof that every candidate should become a Bzlmod module; verify consumers
and dependency closures before extraction.

| Capability candidate | Likely disposition | Current-code clues / boundary |
| --- | --- | --- |
| Estimation algorithms | Strong extraction candidate; consider module `estimation` with focused `kalman`, filters, and model/predictor targets | `modules/common/math/kalman_filter.h`, `modules/common/filters/`, and `modules/common/tracking/predictor.*`. Apollo-lite has Kalman implementations in perception fusion, camera tracking, and radar filtering. Compare state models and semantics before unifying; keep object tracking policy and perception-specific types in their domain modules. |
| Vision geometry | Candidate when another consumer needs the same API; consider `vision` or a focused `geometry` module | Apollo-lite has camera model, projection, distortion, and omnidirectional model targets under `modules/perception/base`, plus camera homography under `modules/perception/common/geometry`. Extract pure camera/coordinate math; leave perception base objects, camera drivers, calibration services, image pipelines, and runtime adapters with their owners. |
| General geometry / transforms | Strong candidate if APIs are domain-neutral | `modules/common/math` contains geometric primitives such as boxes, polygons, vectors, and coordinate math. Separate generic geometry from vehicle/map/message-specific transforms; document coordinate frames and units. |
| Control algorithms | Candidate after separating algorithm core from control application | Apollo-lite's `modules/control/common/pid_controller.*` uses `PidConf` from control proto and Cyber logging. Extract only the numerical PID state/update logic and a framework-neutral parameter API; a focused module could be named `pid`, avoiding confusion with a future WheelOS `control` business module. Keep proto conversion, controller variants, vehicle tuning, and runtime policy in control. |
| Optimization | Conditional candidate; use `optimization` only for a meaningful common API/adapter | Existing `modules/common/math/qp_solver` and `mpc_osqp.*` are clues. Apollo-lite has MPC in control/planning. Keep direct OSQP use as a direct dependency for consumers; share a solver-neutral QP API or WheelOS adapter only if multiple consumers benefit. |
| Generic tracking / data association | Candidate for reuse, but keep separate from estimation primitives if release and dependency needs differ | Existing `modules/common/tracking/` has generic interfaces, associators, costs, motion models, and estimators. Ensure public types do not encode perception messages, object lifecycle, or sensor/runtime policy. |
| Basic containers, string/time helpers, serialization-neutral utilities | Reusable targets, but not automatically separate modules | Existing `modules/common/util/` is a broad umbrella. Preserve small target boundaries and promote only cohesive, independently useful groups; avoid making all consumers depend on `util_tool`. |
| Digital and statistical filters | Candidate for `estimation` or a focused filtering target | Existing `modules/common/filters/` provides digital and mean filters but depends on Cyber. Extract a pure algorithm layer if Cyber is only used for incidental logging/types; otherwise keep the adapter separate. |
| Vehicle state, vehicle model, vehicle configs | Usually remain WheelOS vehicle-domain modules, not general algorithm modules | Existing `vehicle_state`, `vehicle_model`, and `configs` depend on WheelOS messages, configuration, and Cyber. Share generic geometry/math beneath them, not vehicle-specific schemas and runtime providers. |
| Status, adapters, monitor/logging, latency recorder, global config | Usually remain framework/infrastructure integration modules | Existing targets depend on WheelOS messages, protobuf, Cyber, flags, or runtime lifecycle. Keep them separate from low-level capability modules; extract only a truly framework-neutral API if there are independent consumers. |
| KV database / persistence | Conditional candidate | Promote only if there is a stable storage abstraction with multiple consumers and an intentional backend/dependency policy; do not classify it as an algorithm merely because it is under `common`. |

### Apollo-lite Signals

Apollo-lite has one root Bzlmod module (`apollo`). Its domain directories named
`common` are internal packages, not independently versioned modules. They
provide useful evidence about ownership and coupling:

- **Estimation is the clearest cross-domain extraction candidate.**
  Kalman-like code appears in `modules/perception/fusion/common/kalman_filter`,
  `modules/perception/camera/lib/obstacle/tracker/common/kalman_filter`, and
  `modules/perception/radar/lib/tracker/filter/adaptive_kalman_filter`.
  These are not interchangeable APIs: fusion's filter is matrix-generic,
  camera has constant-velocity/extended models plus mean/low-pass filters,
  and radar's filter consumes `perception::base::Object`. Extract a tested,
  domain-neutral predict/update core into module `estimation`; keep motion
  models, object conversion, likelihood/policy, and sensor-specific tuning as
  adapters in perception. First confirm actual use by localization/control.
- **Vision geometry is a credible candidate with a clean lower-level seam.**
  `modules/perception/base/camera.*` implements pinhole projection,
  `distortion_model.*` and `omnidirectional_model.*` implement camera models,
  and `modules/perception/common/geometry/camera_homography.*` implements
  cross-camera geometry. A module `vision` can own pure intrinsics,
  projection, distortion, and coordinate math. Keep the current perception
  interfaces/types and pipeline integrations in perception until a stable
  domain-neutral API is established.
- **PID is reusable only after decoupling.**
  `modules/control/common/pid_controller.*` consumes `PidConf` and Cyber
  logging, while `pid_BC_controller` and `pid_IC_controller` extend it. A
  `control` module should expose the numerical controller and parameter
  struct; keep protobuf conversion, Cyber diagnostics, and vehicle-specific
  controller variants in the control business module.
- **Optimization is conditional rather than an automatic extraction.**
  The existing common repository has `math/qp_solver` and `mpc_osqp`; Apollo
  has a control MPC implementation. Share a module `optimization` only if it
  defines a useful stable solver-neutral API or shared OSQP adapter. Otherwise
  let each actual solver user depend directly on OSQP.
- **Most other Apollo `*/common` code is domain-local.** For example,
  `perception/common/sensor_manager` depends on perception base, perception
  protos, config manager, and transform calibration; `control/common` includes
  config/runtime-oriented targets; `localization/common/pointcloud_io`
  depends on PCL and YAML. Keep these in their domains rather than extracting
  them based on the folder name.

### Naming and Module Granularity

Use direct capability module names such as `estimation`, `vision`, `pid`, and
`optimization`, without a `common.` prefix. Avoid overly broad names that may
collide with business modules; for example, `pid` can be clearer than `control`
when only the controller primitive is shared. Do not make each algorithm a
Bzlmod module by default: prefer module `estimation` with focused targets such
as `@estimation//kalman:kalman_filter` when Kalman filters, motion models, and
related estimators share consumers, dependencies, and release cadence. A
target/package namespace such as `kalman` is enough for organization. A
separate module named `estimation.kalman` is warranted only if Kalman has a
genuinely independent consumer set, API/release lifecycle, and small
dependency closure. Apply the same granularity rule to other capabilities.

An algorithm's name is not enough to establish interchangeability. Before
consolidating duplicated Kalman filters, compare state representation,
prediction/update semantics, covariance assumptions, numerical behavior,
threading, and API compatibility. Prefer one well-tested generic core plus
domain adapters over a forced common API that obscures meaningful differences.

## Example Layout

```text
common/                             # One Git repository
|-- MODULE.bazel                    # Monorepo development/integration root
|-- .agents/knowledge/
`-- modules/
    |-- estimation/
    |   |-- MODULE.bazel            # module(name = "estimation")
    |   |-- BUILD                   # Public capability targets
    |   `-- ...
    |-- vision/
    |   |-- MODULE.bazel            # module(name = "vision")
    |   |-- BUILD
    |   `-- ...
    `-- perception/
        |-- MODULE.bazel            # module(name = "perception")
        |-- BUILD
        `-- ...
```

Each nested module is a standalone Bazel module root. Its `MODULE.bazel`
declares only its direct external and WheelOS module dependencies. Its BUILD
files expose intentional public targets and use labels from its own module
(`//...`) or explicit external module labels (`@estimation//...`).

## Consumer Usage

Declare only required modules in a consumer's `MODULE.bazel`:

```starlark
bazel_dep(name = "estimation", version = "0.2.0")
bazel_dep(name = "perception", version = "0.2.0")
```

Reference their public targets from BUILD files:

```starlark
deps = [
    "@estimation//kalman:kalman_filter",
    "@perception//:perception",
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
   `@estimation//kalman:kalman_filter`.
4. Add local development overrides and module-scoped build/test coverage.
5. Publish the module's registry metadata and validate resolution from a small
   consumer before migrating the next boundary.
6. Migrate consumers module by module; retain compatibility aliases or the
   old aggregate module for an announced transition period if existing users
   need it.

Do not split the repository or move every package into a separate module just
to achieve names like `@estimation`. The intended result is one monorepo with
several deliberate, independently consumable Bzlmod modules.

## Current Repository References

- Root module and existing dependency declarations: `MODULE.bazel`
- Configured Bzlmod registries: `.bazelrc`
- Existing aggregate alias: `BUILD`
- Current shared package layout: `modules/common/`
