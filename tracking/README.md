# Tracking

`tracking` is a detector-agnostic, online multi-object tracking library. The
caller owns detection, timestamp alignment, feature extraction, and coordinate
transforms. The library consumes validated observations and runs prediction,
association scoring/gating, one-to-one assignment, track update, birth, and
expiration.

## Packages

| Package | Responsibility |
| --- | --- |
| `core/` | Observation/frame/result types, tracker orchestration, estimator/cost interfaces, lifecycle and track management |
| `association/` | Assignment solvers plus single-/multi-stage association strategies |
| `estimation/` | Motion models, Gaussian prediction/update, and per-track estimator selection |
| `costs/` | Composite, Mahalanobis, geometry, and appearance-embedding association costs |
| `tests/` | Cross-package behavior and solver tests |

The dependency direction is `core -> association/costs`, `estimation -> core`,
and feature-specific `costs -> core/estimation`. Detection and feature
extractors remain outside this library; callers pass their outputs through the
typed `Payload` and scoring interfaces.

## Public entrypoints

The lightweight generic tracker remains available as
`@wheelos_common//tracking:tracking`. Prefer capability-specific targets for
optional features:

```bzl
deps = [
    "@wheelos_common//tracking/core:tracking",
    "@wheelos_common//tracking/association:association_strategy",
    "@wheelos_common//tracking/association:assignment_solver",
    "@wheelos_common//tracking/costs:association_cost",
]
```

Include the generic tracker with:

```cpp
#include "tracking/core/multi_object_tracker.h"
```

`TrackingResult::tracking_events` reports per-frame `kMatched`, `kAdded`, and
`kExpired` outcomes. Persistent lifecycle states (`kTentative`, `kConfirmed`,
`kLost`, `kDeleted`) remain on `TrackSnapshot`; event kinds and track states
are distinct concepts. Default expiration uses elapsed seconds, not a fixed
frame count.

## Industry alignment and extension points

The implementation follows the tracking-by-detection pipeline used by SORT and
Deep SORT: predict track state, gate/score track-observation pairs, solve a
global assignment, update matched tracks, handle unmatched tracks and
observations, then apply lifecycle retention. Geometry, motion uncertainty,
and appearance similarity are separate scoring features; the library does
not require a particular detector, embedding model, or object schema.

ByteTrack demonstrates a useful alternative association policy: associate
high-confidence detections first, then use low-confidence detections to recover
unmatched confirmed tracks (not lost or tentative tracks), while restricting
births to unmatched detections over a separate threshold.
`ByteTrackAssociationStrategy` implements this optional policy.
`Observation::confidence` defaults to `1.0`, preserving ordinary single-stage
use; select the strategy explicitly to enable thresholds. Its default
low/high/new-track thresholds are `0.1`, `0.5`, and `0.6`; tune them to the
detector and validate tracking metrics on representative sequences.
Detections below the low threshold are not associated; low-score unmatched
detections never start tracks.

```cpp
auto strategy =
    std::make_shared<ByteTrackAssociationStrategy>(ByteTrackConfig{});
MultiObjectTracker<State, Detection> tracker(
    config, estimator, cost, std::make_shared<HungarianSolver>(), lifecycle,
    strategy);
```

`MinCostFlowSolver` uses successive shortest augmenting paths with reduced
cost potentials and Dijkstra. `TrackerConfig::max_solver_work` bounds residual
edge scans; exceeding the budget returns `ResourceExhausted` instead of
allowing an unexpectedly expensive association to run unbounded.

References:

- [SORT paper and reference implementation](https://arxiv.org/abs/1602.00763) · [code](https://github.com/abewley/sort)
- [Deep SORT paper and reference implementation](https://arxiv.org/abs/1703.07402) · [code](https://github.com/nwojke/deep_sort)
- [ByteTrack paper and reference implementation](https://arxiv.org/abs/2110.06864) · [code](https://github.com/FoundationVision/ByteTrack)
