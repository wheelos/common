#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "modules/common/tracking/lifecycle.h"

namespace apollo {
namespace common {
namespace tracking {

template <typename State, typename Payload>
class TrackManager {
 public:
  using Track = TrackSnapshot<State, Payload>;

  TrackManager(
      LifecycleConfig config,
      std::shared_ptr<const TrackLifecyclePolicy<State, Payload>> policy)
      : config_(std::move(config)), policy_(std::move(policy)) {}

  absl::Status Validate() const {
    if (!policy_ || config_.confirmation_hits == 0 ||
        !std::isfinite(config_.lost_timeout_seconds) ||
        config_.lost_timeout_seconds < 0.0) {
      return absl::InvalidArgumentError(
          "invalid lifecycle configuration or policy");
    }
    return absl::OkStatus();
  }

  absl::StatusOr<BirthDecision> Admit(
      const Observation<Payload>& observation) const {
    const auto valid = Validate();
    if (!valid.ok()) {
      return valid;
    }
    auto decision = policy_->OnBirth(observation);
    if (!decision.ok()) {
      return decision.status();
    }
    if (!decision->allowed && decision->reason.empty()) {
      return absl::InvalidArgumentError("birth rejection requires a reason");
    }
    return decision;
  }

  absl::StatusOr<Track> Birth(uint64_t id, double timestamp, State state,
                              const Observation<Payload>& observation,
                              TrackingDiagnostics* diagnostics) const {
    if (diagnostics == nullptr) {
      return absl::InvalidArgumentError("null lifecycle diagnostics");
    }
    Track track{id, TrackStatus::kTentative, std::move(state)};
    track.first_seen_time = timestamp;
    track.state_time = timestamp;
    diagnostics->lifecycle_events.push_back(
        {id, TrackStatus::kDeleted, TrackStatus::kTentative, "birth"});
    const auto status = Hit(timestamp, observation, &track, diagnostics);
    if (!status.ok()) {
      return status;
    }
    return track;
  }

  absl::Status Expire(double timestamp, Track* track,
                      TrackingDiagnostics* diagnostics) const {
    const auto valid = Validate();
    if (!valid.ok()) {
      return valid;
    }
    if (track == nullptr || diagnostics == nullptr) {
      return absl::InvalidArgumentError("null lifecycle input");
    }
    auto expired = policy_->ShouldExpire(*track, timestamp, config_);
    if (!expired.ok()) {
      return expired.status();
    }
    return *expired ? Transition(TrackStatus::kDeleted, "expiration policy",
                                 track, diagnostics)
                    : absl::OkStatus();
  }

  absl::Status Hit(
      double timestamp, const Observation<Payload>& observation, Track* track,
      TrackingDiagnostics* diagnostics,
      std::optional<double> association_cost = std::nullopt) const {
    const auto valid = Validate();
    if (!valid.ok()) {
      return valid;
    }
    if (track == nullptr || diagnostics == nullptr || !observation.payload) {
      return absl::InvalidArgumentError("null lifecycle input");
    }
    if (association_cost &&
        (!std::isfinite(*association_cost) || *association_cost < 0.0)) {
      return absl::InvalidArgumentError("invalid lifecycle association cost");
    }
    if (track->hits == std::numeric_limits<size_t>::max() ||
        track->consecutive_hits == std::numeric_limits<size_t>::max()) {
      return absl::ResourceExhaustedError("track hit counters exhausted");
    }
    ++track->hits;
    ++track->consecutive_hits;
    track->consecutive_misses = 0;
    track->last_observed_time = timestamp;
    track->last_observation = observation.payload;
    track->last_association_cost = association_cost;
    track->observed = true;
    return Transition(policy_->OnHit(*track, config_), "observation matched",
                      track, diagnostics);
  }

  absl::Status Miss(double timestamp, Track* track,
                    TrackingDiagnostics* diagnostics) const {
    const auto valid = Validate();
    if (!valid.ok()) {
      return valid;
    }
    if (track == nullptr || diagnostics == nullptr) {
      return absl::InvalidArgumentError("null lifecycle input");
    }
    if (track->consecutive_misses == std::numeric_limits<size_t>::max()) {
      return absl::ResourceExhaustedError("track miss counter exhausted");
    }
    track->consecutive_hits = 0;
    ++track->consecutive_misses;
    track->observed = false;
    return Transition(policy_->OnMiss(*track, timestamp, config_),
                      "observation missing", track, diagnostics);
  }

  void Prune(std::vector<Track>* tracks) const {
    tracks->erase(std::remove_if(tracks->begin(), tracks->end(),
                                 [](const Track& track) {
                                   return track.status == TrackStatus::kDeleted;
                                 }),
                  tracks->end());
  }

  void Record(Track* track) const {
    if (config_.history_limit == 0) {
      return;
    }
    if (track->history.size() >= config_.history_limit) {
      track->history.erase(track->history.begin());
    }
    track->history.push_back({track->state_time, track->state, track->observed,
                              track->observed ? track->last_association_cost
                                              : std::optional<double>{}});
  }

  bool Publish(const Track& track) const {
    return track.status != TrackStatus::kDeleted &&
           policy_->ShouldPublish(track, config_);
  }

 private:
  absl::Status Transition(TrackStatus next, const char* reason, Track* track,
                          TrackingDiagnostics* diagnostics) const {
    bool allowed = false;
    switch (track->status) {
      case TrackStatus::kTentative:
        allowed = next == TrackStatus::kTentative ||
                  next == TrackStatus::kConfirmed ||
                  next == TrackStatus::kDeleted;
        break;
      case TrackStatus::kConfirmed:
      case TrackStatus::kLost:
        allowed = next == TrackStatus::kConfirmed ||
                  next == TrackStatus::kLost || next == TrackStatus::kDeleted;
        break;
      case TrackStatus::kDeleted:
        allowed = next == TrackStatus::kDeleted;
        break;
    }
    if (!allowed) {
      return absl::InternalError(
          "lifecycle policy returned an invalid transition");
    }
    if (track->status != next) {
      diagnostics->lifecycle_events.push_back(
          {track->id, track->status, next, reason});
      track->status = next;
    }
    return absl::OkStatus();
  }

  LifecycleConfig config_;
  std::shared_ptr<const TrackLifecyclePolicy<State, Payload>> policy_;
};

}  // namespace tracking
}  // namespace common
}  // namespace apollo
