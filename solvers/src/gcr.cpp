#include "autan/solvers/gcr.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace autan::solvers {

KrylovReport Gcr::solve(const Operator& apply, const Operator& precondition, std::span<double> r,
                        std::span<double> x, int max_iterations, double tolerance,
                        const Reductions& reductions) {
  const std::size_t m = r.size();
  auto dot = [&](std::span<const double> u, std::span<const double> v) {
    if (reductions.dot) return reductions.dot(u, v);
    double s = 0.0;
    for (std::size_t k = 0; k < u.size(); ++k) s += u[k] * v[k];
    return s;
  };
  // The iteration runs on the residual scaled by its largest entry, so that no norm underflows.
  double scale = 0.0;
  if (reductions.max_abs) {
    scale = reductions.max_abs(r);
  } else {
    for (const double v : r) scale = std::max(scale, std::abs(v));
    if (std::any_of(r.begin(), r.end(), [](double v) { return std::isnan(v); })) {
      scale = std::numeric_limits<double>::quiet_NaN();
    }
  }
  KrylovReport report;
  if (!std::isfinite(scale)) {
    report.residual_drop = std::numeric_limits<double>::quiet_NaN();
    return report;
  }
  if (scale == 0.0) {
    report.residual_drop = 0.0;
    return report;
  }
  for (double& v : r) v /= scale;
  const double r0 = std::sqrt(dot(r, r));
  const auto kmax = static_cast<std::size_t>(std::max(max_iterations, 0));
  if (directions_.size() < kmax) {
    directions_.resize(kmax);
    images_.resize(kmax);
  }
  double rr = r0 * r0;
  for (std::size_t k = 0; k < kmax; ++k) {
    std::vector<double>& z = directions_[k];
    std::vector<double>& w = images_[k];
    z.assign(m, 0.0);
    w.resize(m);
    precondition(r, z);
    apply(z, w);
    for (std::size_t j = 0; j < k; ++j) {
      const double beta = dot(w, images_[j]) / dot(images_[j], images_[j]);
      for (std::size_t i = 0; i < m; ++i) {
        z[i] -= beta * directions_[j][i];
        w[i] -= beta * images_[j][i];
      }
    }
    const double ww = dot(w, w);
    report.cycles = static_cast<int>(k) + 1;
    if (!(ww > 0.0)) {
      if (!std::isfinite(ww)) rr = ww;  // a direction that is not finite: the solve failed
      break;
    }
    const double alpha = dot(r, w) / ww;
    for (std::size_t i = 0; i < m; ++i) {
      x[i] += scale * alpha * z[i];
      r[i] -= alpha * w[i];
    }
    rr = dot(r, r);
    if (std::sqrt(rr) <= tolerance * r0) break;
  }
  for (double& v : r) v *= scale;
  report.residual_drop =
      std::isfinite(rr) ? std::sqrt(rr) / r0 : std::numeric_limits<double>::quiet_NaN();
  return report;
}

}  // namespace autan::solvers
