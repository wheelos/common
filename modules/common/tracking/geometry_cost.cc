#include "modules/common/tracking/geometry_cost.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "modules/common/math/box2d.h"
#include "modules/common/math/polygon2d.h"
#include "modules/common/math/vec2d.h"

namespace apollo {
namespace common {
namespace tracking {
namespace {

absl::Status ValidatePair(const Box3D& a, const Box3D& b) {
  const auto first = ValidateBox(a);
  if (!first.ok()) {
    return first;
  }
  const auto second = ValidateBox(b);
  if (!second.ok()) {
    return second;
  }
  if (!(b.center - a.center).allFinite()) {
    return absl::InvalidArgumentError("box center difference overflow");
  }
  return absl::OkStatus();
}

class Footprint final : public math::Polygon2d {
 public:
  static absl::StatusOr<Footprint> Create(const Box3D& box,
                                          const Eigen::Vector2d& center) {
    auto corners = math::Box2d(math::Vec2d(center.x(), center.y()), box.yaw,
                               box.size.x(), box.size.y())
                       .GetAllCorners();
    auto area = Area(corners);
    if (!area.ok()) {
      return area.status();
    }
    if (*area <= math::kMathEpsilon) {
      return absl::InvalidArgumentError(
          "box footprint below geometry precision");
    }
    return Footprint(std::move(corners));
  }

  absl::StatusOr<double> IntersectionArea(const Footprint& other) const {
    auto vertices = other.points();
    for (const auto& edge : line_segments()) {
      if (!ClipConvexHull(edge, &vertices)) {
        return 0.0;
      }
    }
    // Reuse clipping, but do not construct a Polygon2d for a near-zero overlap.
    return Area(vertices);
  }

 private:
  explicit Footprint(std::vector<math::Vec2d> vertices)
      : math::Polygon2d(std::move(vertices)) {}

  static absl::StatusOr<double> Area(const std::vector<math::Vec2d>& vertices) {
    double doubled_area = 0.0;
    for (size_t index = 2; index < vertices.size(); ++index) {
      const auto first = vertices[index - 1] - vertices[0];
      const auto second = vertices[index] - vertices[0];
      doubled_area += first.x() * second.y() - first.y() * second.x();
    }
    if (!std::isfinite(doubled_area)) {
      return absl::InvalidArgumentError("nonfinite box intersection area");
    }
    return std::abs(doubled_area) * 0.5;
  }
};

absl::StatusOr<double> IntersectionArea(const Box3D& a, const Box3D& b) {
  // Translation invariance avoids precision loss at large absolute positions.
  const Eigen::Vector2d delta = (b.center - a.center).head<2>();
  const double radius =
      a.size.head<2>().stableNorm() * 0.5 + b.size.head<2>().stableNorm() * 0.5;
  if (!std::isfinite(radius)) {
    return absl::InvalidArgumentError("box footprint extent overflow");
  }
  auto first = Footprint::Create(a, Eigen::Vector2d::Zero());
  if (!first.ok()) {
    return first.status();
  }
  auto second_local = Footprint::Create(b, Eigen::Vector2d::Zero());
  if (!second_local.ok()) {
    return second_local.status();
  }
  if (delta.stableNorm() > radius) {
    return 0.0;
  }
  auto second = Footprint::Create(b, delta);
  if (!second.ok()) {
    return second.status();
  }
  return first->IntersectionArea(*second);
}

absl::StatusOr<CostEvaluation> IouCost(double intersection, double first,
                                       double second) {
  const double union_size = first + second - intersection;
  if (!std::isfinite(intersection) || !std::isfinite(union_size) ||
      union_size <= 0.0) {
    return absl::InvalidArgumentError("invalid IoU geometry");
  }
  return CostEvaluation::Accept(
      1.0 - std::clamp(intersection / union_size, 0.0, 1.0));
}

}  // namespace

absl::Status ValidateBox(const Box3D& box) {
  if (!box.center.allFinite() || !box.size.allFinite() ||
      !std::isfinite(box.yaw) || box.size.minCoeff() <= 1e-8 ||
      !std::isfinite(box.size.prod())) {
    return absl::InvalidArgumentError(
        "box requires finite positive dimensions");
  }
  return absl::OkStatus();
}

absl::StatusOr<CostEvaluation> CenterDistanceCost::Evaluate(
    const Box3D& predicted, const Box3D& observed) const {
  const auto valid = ValidatePair(predicted, observed);
  if (!valid.ok()) {
    return valid;
  }
  if (!std::isfinite(gate_) || gate_ <= 0.0) {
    return absl::InvalidArgumentError("center distance gate must be positive");
  }
  const Eigen::Vector3d difference = predicted.center - observed.center;
  const double distance =
      use_bev_ ? difference.head<2>().stableNorm() : difference.stableNorm();
  if (!std::isfinite(distance)) {
    return absl::InvalidArgumentError("center distance overflow");
  }
  return distance <= gate_ ? CostEvaluation::Accept(distance / gate_)
                           : CostEvaluation::Reject("center distance gate");
}

absl::StatusOr<CostEvaluation> BevIouCost::Evaluate(
    const Box3D& predicted, const Box3D& observed) const {
  const auto valid = ValidatePair(predicted, observed);
  if (!valid.ok()) {
    return valid;
  }
  auto intersection = IntersectionArea(predicted, observed);
  if (!intersection.ok()) {
    return intersection.status();
  }
  return IouCost(*intersection, predicted.size.x() * predicted.size.y(),
                 observed.size.x() * observed.size.y());
}

absl::StatusOr<CostEvaluation> Iou3dCost::Evaluate(
    const Box3D& predicted, const Box3D& observed) const {
  const auto valid = ValidatePair(predicted, observed);
  if (!valid.ok()) {
    return valid;
  }
  const double delta_z = observed.center.z() - predicted.center.z();
  const double height = std::max(
      0.0,
      std::min(predicted.size.z() * 0.5, delta_z + observed.size.z() * 0.5) -
          std::max(-predicted.size.z() * 0.5,
                   delta_z - observed.size.z() * 0.5));
  auto intersection = IntersectionArea(predicted, observed);
  if (!intersection.ok()) {
    return intersection.status();
  }
  return IouCost(*intersection * height, predicted.size.prod(),
                 observed.size.prod());
}

}  // namespace tracking
}  // namespace common
}  // namespace apollo
