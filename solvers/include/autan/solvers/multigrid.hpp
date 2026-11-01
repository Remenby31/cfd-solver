#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/solvers/agglomeration.hpp"
#include "autan/solvers/gcr.hpp"

namespace autan::solvers {

/// V, W, or K: Krylov-accelerated coarse solves (Notay and Vassilevski 2008).
enum class Cycle : std::uint8_t { v, w, k };

/// Factor alpha of each coarse correction e: none (1), residual (minimises |r - alpha A e|) or
/// galerkin (r - alpha A e orthogonal to e).
enum class Scaling : std::uint8_t { none, residual, galerkin };

struct MultigridOptions {
  Cycle cycle = Cycle::k;
  int pre_sweeps = 1;   ///< symmetric smoothing sweeps before the coarse correction
  int post_sweeps = 1;  ///< and after it
  /// Sweeps on the coarsest level when it is too large for its dense factorisation.
  int coarsest_sweeps = 10;
  Scaling scaling = Scaling::galerkin;
  AgglomerationOptions agglomeration;
};

/// Agglomeration multigrid for systems of 5 x 5 blocks on a mesh's cell adjacency: line-implicit
/// block Gauss-Seidel smoothing, piecewise-constant transfers, Galerkin coarse operators
/// (docs/design/multigrid.md). The operator has one diagonal block per cell and, per interior
/// face f (owner o, neighbour n), the blocks A(o, n) (`upper`) and A(n, o) (`lower`), row-major;
/// vectors are structure of arrays, component c of cell i at c * n_cells + i. FP64 throughout;
/// every loop runs in a fixed order, so results are bitwise reproducible.
class Multigrid {
 public:
  /// Builds the hierarchy of `mesh` (no ghost cells; std::invalid_argument otherwise).
  Multigrid(const Mesh& mesh, const Geometry& geometry, const MultigridOptions& options = {});
  ~Multigrid();
  Multigrid(const Multigrid&) = delete;
  Multigrid& operator=(const Multigrid&) = delete;

  [[nodiscard]] const std::vector<Level>& levels() const noexcept { return levels_; }
  /// Whether the coarsest level is solved by dense LU; otherwise by `coarsest_sweeps` sweeps.
  [[nodiscard]] bool direct_coarsest() const noexcept { return direct_; }

  /// Sets the fine operator, builds the coarse operators P^T A P and factors every level's
  /// smoother. The fine arrays are referenced, not copied: they must stay unchanged and alive
  /// until the next set_operator. Throws std::domain_error on a singular block.
  void set_operator(std::span<const double> diagonal, std::span<const double> upper,
                    std::span<const double> lower);
  /// One multigrid cycle on A x = b from x.
  void cycle(std::span<const double> b, std::span<double> x);
  using Report = KrylovReport;
  /// Solves A x = b from x by the generalised conjugate residual method preconditioned by one
  /// multigrid cycle per iteration, until |b - A x| <= tolerance |b - A x0| or `max_cycles`
  /// iterations.
  Report solve(std::span<const double> b, std::span<double> x, int max_cycles, double tolerance);
  /// `sweeps` symmetric line Gauss-Seidel sweeps of the finest level alone on A x = b from x.
  void smooth(std::span<const double> b, std::span<double> x, int sweeps);

 private:
  struct Operator;
  struct Smoother;

  void cycle_level(std::size_t l, std::span<const double> b, std::span<double> x);
  void coarse_solve(std::size_t m, std::span<const double> b, std::span<double> x);

  MultigridOptions options_;
  std::vector<Level> levels_;
  std::vector<Operator> ops_;
  std::vector<Smoother> smoothers_;
  // Per level: residual, restricted right-hand side and correction of the next level.
  std::vector<std::vector<double>> r_, bc_, xc_, e_, q_;
  std::vector<std::vector<std::vector<double>>> krylov_;  // per level: v1, w1, v2, w2, r1
  Gcr gcr_;
  std::vector<double> residual_;
  bool direct_ = false;
};

}  // namespace autan::solvers
