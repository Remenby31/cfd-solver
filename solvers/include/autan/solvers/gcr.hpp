#pragma once

#include <functional>
#include <span>
#include <vector>

namespace autan::solvers {

struct KrylovReport {
  int cycles = 0;              ///< preconditioner applications
  double residual_drop = 1.0;  ///< |b - A x| / |b - A x0|; NaN when not finite
};

/// Generalised conjugate residual method (Eisenstat, Elman and Schultz 1983, SIAM J. Numer.
/// Anal. 20(2):345-357) with a variable right preconditioner, so that the preconditioner may be
/// a multigrid cycle and the operator a finite-difference product. Directions are kept, not
/// restarted. FP64, loops in a fixed order: bitwise reproducible.
class Gcr {
 public:
  using Operator = std::function<void(std::span<const double> x, std::span<double> y)>;
  /// How the iteration reduces vectors: an inner product and the largest |entry|. Empty: over
  /// every entry, on one rank. On a partition, both cover the owned entries and reduce over the
  /// ranks (exactly, for the bits not to depend on the partition).
  struct Reductions {
    std::function<double(std::span<const double> u, std::span<const double> v)> dot;
    std::function<double(std::span<const double> u)> max_abs;
  };

  /// From the residual r = b - A x0 (overwritten with the final residual) improves x until
  /// |r| <= tolerance |r0| or `max_iterations` iterations; `apply` computes y = A x and
  /// `precondition` z ~ A^-1 r into a zeroed z.
  KrylovReport solve(const Operator& apply, const Operator& precondition, std::span<double> r,
                     std::span<double> x, int max_iterations, double tolerance,
                     const Reductions& reductions = {});

 private:
  std::vector<std::vector<double>> directions_, images_;
};

}  // namespace autan::solvers
