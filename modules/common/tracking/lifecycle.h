#pragma once

#include "absl/status/statusor.h"

#include "modules/common/tracking/types.h"

namespace apollo {
namespace common {
namespace tracking {

template <typename State, typename Payload>
class TrackLifecyclePolicy {
 public:
  using Track = TrackSnapshot<State, Payload>;
  virtual ~TrackLifecyclePolicy() = default;

  virtual absl::StatusOr<BirthDecision> OnBirth(
      const Observation<Payload>&) const {
    return BirthDecision{};
  }

  virtual absl::StatusOr<bool> ShouldExpire(
      const Track& track, double timestamp,
      const LifecycleConfig& config) const {
    return timestamp - track.last_observed_time > config.lost_timeout_seconds;
  }

  virtual TrackStatus OnHit(const Track& track,
                            const LifecycleConfig& config) const {
    if (track.status == TrackStatus::kLost ||
        track.status == TrackStatus::kConfirmed ||
        track.consecutive_hits >= config.confirmation_hits) {
      return TrackStatus::kConfirmed;
    }
    return TrackStatus::kTentative;
  }

  virtual TrackStatus OnMiss(const Track& track, double,
                             const LifecycleConfig& config) const {
    if (track.status == TrackStatus::kTentative &&
        track.consecutive_misses > config.tentative_max_misses) {
      return TrackStatus::kDeleted;
    }
    return track.status == TrackStatus::kTentative ? track.status
                                                   : TrackStatus::kLost;
  }

  virtual bool ShouldPublish(const Track& track,
                             const LifecycleConfig& config) const {
    return track.status == TrackStatus::kConfirmed ||
           (config.publish_tentative &&
            track.status == TrackStatus::kTentative) ||
           (config.publish_lost && track.status == TrackStatus::kLost);
  }
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
