#include "autan/solvers/multigrid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "blocks.hpp"

namespace autan::solvers {

namespace {

using detail::add5;
using detail::b2;
using detail::invert5;
using detail::multiply5;
// The coarsest level is factored densely up to this many unknowns.
constexpr std::size_t dense_unknowns = 1000;

}  // namespace

// A level's operator: blocks and the node -> edges adjacency.
struct Multigrid::Operator {
  std::size_t n = 0;
  const Level* level = nullptr;
  std::vector<offset_t> node_offsets;
  std::vector<index_t> node_edges;
  std::vector<double> diag_store, upper_store, lower_store;  // coarse levels
  std::span<const double> diag, upper, lower;

  explicit Operator(const Level& l) : n(l.n_nodes), level(&l) {
    node_offsets.assign(n + 1, 0);
    for (std::size_t e = 0; e < l.owner.size(); ++e) {
      ++node_offsets[static_cast<std::size_t>(l.owner[e]) + 1];
      ++node_offsets[static_cast<std::size_t>(l.neighbour[e]) + 1];
    }
    for (std::size_t i = 0; i < n; ++i) node_offsets[i + 1] += node_offsets[i];
    node_edges.resize(static_cast<std::size_t>(node_offsets.back()));
    std::vector<offset_t> fill(node_offsets.begin(), node_offsets.end() - 1);
    for (std::size_t e = 0; e < l.owner.size(); ++e) {
      node_edges[static_cast<std::size_t>(fill[static_cast<std::size_t>(l.owner[e])]++)] =
          static_cast<index_t>(e);
      node_edges[static_cast<std::size_t>(fill[static_cast<std::size_t>(l.neighbour[e])]++)] =
          static_cast<index_t>(e);
    }
  }

  // Block A(i, j) of edge e seen from node i, and j.
  [[nodiscard]] std::pair<const double*, std::size_t> off(std::size_t e, std::size_t i) const {
    const bool owned = static_cast<std::size_t>(level->owner[e]) == i;
    return {owned ? &upper[b2 * e] : &lower[b2 * e],
            static_cast<std::size_t>(owned ? level->neighbour[e] : level->owner[e])};
  }

  // r_i = b_i - sum over the edges of i whose other node is not skipped of A(i, j) x_j.
  template <class Skip>
  void off_residual(std::size_t i, std::span<const double> b, std::span<const double> x,
                    Skip&& skip, double* r) const {
    for (std::size_t c = 0; c < 5; ++c) r[c] = b[c * n + i];
    for (auto k = node_offsets[i]; k < node_offsets[i + 1]; ++k) {
      const auto [a, j] = off(static_cast<std::size_t>(node_edges[static_cast<std::size_t>(k)]), i);
      if (skip(j)) continue;
      for (std::size_t p = 0; p < 5; ++p) {
        for (std::size_t q = 0; q < 5; ++q) r[p] -= a[5 * p + q] * x[q * n + j];
      }
    }
  }

  void residual(std::span<const double> b, std::span<const double> x, std::span<double> r) const {
    std::array<double, 5> s{};
    for (std::size_t i = 0; i < n; ++i) {
      off_residual(i, b, x, [](std::size_t) { return false; }, s.data());
      const double* d = &diag[b2 * i];
      for (std::size_t p = 0; p < 5; ++p) {
        for (std::size_t q = 0; q < 5; ++q) s[p] -= d[5 * p + q] * x[q * n + i];
        r[p * n + i] = s[p];
      }
    }
  }

  void multiply(std::span<const double> x, std::span<double> y) const {
    for (std::size_t i = 0; i < n; ++i) {
      std::array<double, 5> s{};
      const double* d = &diag[b2 * i];
      for (std::size_t p = 0; p < 5; ++p) {
        for (std::size_t q = 0; q < 5; ++q) s[p] += d[5 * p + q] * x[q * n + i];
      }
      for (auto k = node_offsets[i]; k < node_offsets[i + 1]; ++k) {
        const auto [a, j] =
            off(static_cast<std::size_t>(node_edges[static_cast<std::size_t>(k)]), i);
        for (std::size_t p = 0; p < 5; ++p) {
          for (std::size_t q = 0; q < 5; ++q) s[p] += a[5 * p + q] * x[q * n + j];
        }
      }
      for (std::size_t p = 0; p < 5; ++p) y[p * n + i] = s[p];
    }
  }

  // Block sum of the coupling between nodes i and j (every edge joining them).
  void coupling(std::size_t i, std::size_t j, double* a) const {
    std::fill(a, a + b2, 0.0);
    for (auto k = node_offsets[i]; k < node_offsets[i + 1]; ++k) {
      const auto [blk, o] =
          off(static_cast<std::size_t>(node_edges[static_cast<std::size_t>(k)]), i);
      if (o == j) add5(a, blk);
    }
  }
};

// Line-implicit block Gauss-Seidel: each line is solved by block-tridiagonal elimination with
// its couplings to other lines lagged; lines of one colour share no edge, so their solves are
// independent, and a sweep visits the colours forwards then backwards. A line of one node is
// point block Gauss-Seidel.
struct Multigrid::Smoother {
  const Level* level = nullptr;
  std::vector<std::vector<std::size_t>> colours;  // lines per colour
  std::vector<index_t> prev, next;                // line neighbours, -1 at the ends
  std::vector<std::size_t> slot;                  // node -> link storage, line nodes only
  std::vector<double> dinv;                       // inverse of the eliminated diagonal, per node
  std::vector<double> lower, gain;                // A(i, prev), D'^-1 A(i, next), per slot
  std::vector<double> work;
  std::vector<double> dense;  // coarsest level: LU factors
  std::vector<std::size_t> pivot;

  explicit Smoother(const Level& l) : level(&l) {
    const std::size_t n = l.n_nodes;
    prev.assign(n, -1);
    next.assign(n, -1);
    slot.assign(n, 0);
    std::vector<std::size_t> line_of(n);
    std::size_t n_slots = 0, longest = 1;
    for (std::size_t li = 0; li < l.n_lines(); ++li) {
      const auto b = static_cast<std::size_t>(l.line_offsets[li]);
      const auto e = static_cast<std::size_t>(l.line_offsets[li + 1]);
      longest = std::max(longest, e - b);
      for (std::size_t k = b; k < e; ++k) {
        const auto i = static_cast<std::size_t>(l.line_nodes[k]);
        line_of[i] = li;
        if (k > b) prev[i] = l.line_nodes[k - 1];
        if (k + 1 < e) next[i] = l.line_nodes[k + 1];
        if (e - b > 1) slot[i] = n_slots++;
      }
    }
    lower.assign(b2 * n_slots, 0.0);
    gain.assign(b2 * n_slots, 0.0);
    dinv.assign(b2 * n, 0.0);
    work.assign(5 * longest, 0.0);
    // Greedy colouring of the lines in line order: lines of a colour share no edge.
    std::vector<std::vector<std::size_t>> adjacent(l.n_lines());
    for (std::size_t e = 0; e < l.owner.size(); ++e) {
      const std::size_t a = line_of[static_cast<std::size_t>(l.owner[e])];
      const std::size_t b = line_of[static_cast<std::size_t>(l.neighbour[e])];
      if (a == b) continue;
      adjacent[a].push_back(b);
      adjacent[b].push_back(a);
    }
    std::vector<int> colour(l.n_lines(), -1);
    std::vector<int> seen;
    int n_colours = 0;
    for (std::size_t li = 0; li < l.n_lines(); ++li) {
      seen.assign(static_cast<std::size_t>(n_colours) + 1, 0);
      for (const std::size_t o : adjacent[li]) {
        if (colour[o] >= 0) seen[static_cast<std::size_t>(colour[o])] = 1;
      }
      int k = 0;
      while (seen[static_cast<std::size_t>(k)] != 0) ++k;
      colour[li] = k;
      n_colours = std::max(n_colours, k + 1);
    }
    colours.assign(static_cast<std::size_t>(n_colours), {});
    for (std::size_t li = 0; li < l.n_lines(); ++li) {
      colours[static_cast<std::size_t>(colour[li])].push_back(li);
    }
  }

  // Block LU of every line: D'_0 = D_0, D'_k = D_k - A(k, k-1) D'_{k-1}^-1 A(k-1, k).
  void factor(const Operator& op) {
    std::array<double, b2> d{}, t{}, u{};
    for (std::size_t li = 0; li < level->n_lines(); ++li) {
      for (auto k = level->line_offsets[li]; k < level->line_offsets[li + 1]; ++k) {
        const auto i = static_cast<std::size_t>(level->line_nodes[static_cast<std::size_t>(k)]);
        std::copy_n(&op.diag[b2 * i], b2, d.data());
        if (prev[i] >= 0) {
          const auto p = static_cast<std::size_t>(prev[i]);
          double* lo = &lower[b2 * slot[i]];
          op.coupling(i, p, lo);
          multiply5(lo, &gain[b2 * slot[p]], t.data());
          for (std::size_t m = 0; m < b2; ++m) d[m] -= t[m];
        }
        invert5(d.data(), &dinv[b2 * i]);
        if (next[i] >= 0) {
          op.coupling(i, static_cast<std::size_t>(next[i]), u.data());
          multiply5(&dinv[b2 * i], u.data(), &gain[b2 * slot[i]]);
        }
      }
    }
  }

  void relax_line(const Operator& op, std::size_t li, std::span<const double> b,
                  std::span<double> x) {
    const std::size_t n = op.n;
    const auto first = static_cast<std::size_t>(level->line_offsets[li]);
    const auto last = static_cast<std::size_t>(level->line_offsets[li + 1]);
    for (std::size_t k = first; k < last; ++k) {
      const auto i = static_cast<std::size_t>(level->line_nodes[k]);
      const auto p = prev[i], q = next[i];
      op.off_residual(
          i, b, x,
          [&](std::size_t j) {
            return static_cast<index_t>(j) == p || static_cast<index_t>(j) == q;
          },
          &work[5 * (k - first)]);
    }
    // Forward elimination y_k = D'_k^-1 (r_k - A(k, k-1) y_{k-1}), into work.
    std::array<double, 5> s{};
    for (std::size_t k = first; k < last; ++k) {
      const auto i = static_cast<std::size_t>(level->line_nodes[k]);
      double* r = &work[5 * (k - first)];
      if (k > first) {
        const double* lo = &lower[b2 * slot[i]];
        const double* y = &work[5 * (k - 1 - first)];
        for (std::size_t a = 0; a < 5; ++a) {
          for (std::size_t c = 0; c < 5; ++c) r[a] -= lo[5 * a + c] * y[c];
        }
      }
      const double* inv = &dinv[b2 * i];
      for (std::size_t a = 0; a < 5; ++a) {
        s[a] = 0.0;
        for (std::size_t c = 0; c < 5; ++c) s[a] += inv[5 * a + c] * r[c];
      }
      std::copy_n(s.data(), 5, r);
    }
    // Back substitution x_k = y_k - D'_k^-1 A(k, k+1) x_{k+1}.
    for (std::size_t k = last; k-- > first;) {
      const auto i = static_cast<std::size_t>(level->line_nodes[k]);
      double* y = &work[5 * (k - first)];
      if (k + 1 < last) {
        const double* g = &gain[b2 * slot[i]];
        const double* xn = &work[5 * (k + 1 - first)];
        for (std::size_t a = 0; a < 5; ++a) {
          for (std::size_t c = 0; c < 5; ++c) y[a] -= g[5 * a + c] * xn[c];
        }
      }
      for (std::size_t a = 0; a < 5; ++a) x[a * n + i] = y[a];
    }
  }

  void sweep(const Operator& op, std::span<const double> b, std::span<double> x, int sweeps) {
    for (int s = 0; s < sweeps; ++s) {
      for (const auto& colour : colours) {
        for (const std::size_t li : colour) relax_line(op, li, b, x);
      }
      for (auto it = colours.rbegin(); it != colours.rend(); ++it) {
        for (const std::size_t li : *it) relax_line(op, li, b, x);
      }
    }
  }

  // Dense LU with partial pivoting of the whole level, unknown c of node i at c * n + i.
  void factor_dense(const Operator& op) {
    const std::size_t n = op.n, m = 5 * n;
    dense.assign(m * m, 0.0);
    auto put = [&](std::size_t i, std::size_t j, const double* a) {
      for (std::size_t p = 0; p < 5; ++p) {
        for (std::size_t q = 0; q < 5; ++q) dense[(p * n + i) * m + q * n + j] += a[5 * p + q];
      }
    };
    for (std::size_t i = 0; i < n; ++i) {
      put(i, i, &op.diag[b2 * i]);
      for (auto k = op.node_offsets[i]; k < op.node_offsets[i + 1]; ++k) {
        const auto [a, j] =
            op.off(static_cast<std::size_t>(op.node_edges[static_cast<std::size_t>(k)]), i);
        put(i, j, a);
      }
    }
    pivot.resize(m);
    for (std::size_t k = 0; k < m; ++k) {
      std::size_t p = k;
      for (std::size_t i = k + 1; i < m; ++i) {
        if (std::abs(dense[i * m + k]) > std::abs(dense[p * m + k])) p = i;
      }
      if (dense[p * m + k] == 0.0 || !std::isfinite(dense[p * m + k])) {
        throw std::domain_error("singular coarsest operator");
      }
      pivot[k] = p;
      if (p != k) std::swap_ranges(&dense[k * m], &dense[k * m] + m, &dense[p * m]);
      for (std::size_t i = k + 1; i < m; ++i) {
        const double f = dense[i * m + k] /= dense[k * m + k];
        if (f == 0.0) continue;
        for (std::size_t j = k + 1; j < m; ++j) dense[i * m + j] -= f * dense[k * m + j];
      }
    }
  }

  void solve_dense(std::span<const double> b, std::span<double> x) const {
    const std::size_t m = pivot.size();
    std::copy_n(b.begin(), m, x.begin());
    for (std::size_t k = 0; k < m; ++k) std::swap(x[k], x[pivot[k]]);
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < i; ++j) x[i] -= dense[i * m + j] * x[j];
    }
    for (std::size_t i = m; i-- > 0;) {
      for (std::size_t j = i + 1; j < m; ++j) x[i] -= dense[i * m + j] * x[j];
      x[i] /= dense[i * m + i];
    }
  }
};

Multigrid::Multigrid(const Mesh& mesh, const Geometry& geometry, const MultigridOptions& options)
    : options_(options), levels_(agglomerate(mesh, geometry, options.agglomeration)) {
  // Work vectors only where the cycle uses them: residual and correction above the coarsest
  // level, right-hand side, solution and Krylov vectors below the finest.
  for (std::size_t i = 0; i < levels_.size(); ++i) {
    const Level& l = levels_[i];
    const std::size_t above = i + 1 < levels_.size() ? 5 * l.n_nodes : 0;
    const std::size_t below = i > 0 ? 5 * l.n_nodes : 0;
    ops_.emplace_back(l);
    smoothers_.emplace_back(l);
    r_.emplace_back(above);
    e_.emplace_back(above);
    q_.emplace_back(options_.scaling != Scaling::none ? above : 0);
    bc_.emplace_back(below);
    xc_.emplace_back(below);
    krylov_.emplace_back(options_.cycle == Cycle::k && i + 1 < levels_.size() ? 5 : 0,
                         std::vector<double>(below));
  }
  direct_ = 5 * levels_.back().n_nodes <= dense_unknowns;
}

Multigrid::~Multigrid() = default;

void Multigrid::set_operator(std::span<const double> diagonal, std::span<const double> upper,
                             std::span<const double> lower) {
  ops_[0].diag = diagonal;
  ops_[0].upper = upper;
  ops_[0].lower = lower;
  // Galerkin coarse operators P^T A P with P piecewise constant: block sums.
  for (std::size_t l = 0; l + 1 < levels_.size(); ++l) {
    const Level& f = levels_[l];
    const Operator& a = ops_[l];
    Operator& c = ops_[l + 1];
    c.diag_store.assign(b2 * c.n, 0.0);
    c.upper_store.assign(b2 * levels_[l + 1].owner.size(), 0.0);
    c.lower_store.assign(b2 * levels_[l + 1].owner.size(), 0.0);
    for (std::size_t i = 0; i < f.n_nodes; ++i) {
      add5(&c.diag_store[b2 * static_cast<std::size_t>(f.coarse_node[i])], &a.diag[b2 * i]);
    }
    for (std::size_t e = 0; e < f.owner.size(); ++e) {
      const index_t ce = f.coarse_edge[e];
      if (ce < 0) {
        double* d = &c.diag_store[b2 * static_cast<std::size_t>(
                                           f.coarse_node[static_cast<std::size_t>(f.owner[e])])];
        add5(d, &a.upper[b2 * e]);
        add5(d, &a.lower[b2 * e]);
        continue;
      }
      const auto E = static_cast<std::size_t>(ce);
      const bool flip = f.coarse_flipped[e] != 0;
      add5(&c.upper_store[b2 * E], flip ? &a.lower[b2 * e] : &a.upper[b2 * e]);
      add5(&c.lower_store[b2 * E], flip ? &a.upper[b2 * e] : &a.lower[b2 * e]);
    }
    c.diag = c.diag_store;
    c.upper = c.upper_store;
    c.lower = c.lower_store;
  }
  for (std::size_t l = 0; l < levels_.size(); ++l) {
    const bool coarsest = l + 1 == levels_.size();
    // A coarsest level solved directly needs no smoother, unless it is also the finest.
    if (!(coarsest && direct_) || l == 0) smoothers_[l].factor(ops_[l]);
    if (coarsest && direct_) smoothers_[l].factor_dense(ops_[l]);
  }
}

void Multigrid::smooth(std::span<const double> b, std::span<double> x, int sweeps) {
  smoothers_[0].sweep(ops_[0], b, x, sweeps);
}

void Multigrid::cycle(std::span<const double> b, std::span<double> x) { cycle_level(0, b, x); }

Multigrid::Report Multigrid::solve(std::span<const double> b, std::span<double> x, int max_cycles,
                                   double tolerance) {
  const Operator& a = ops_[0];
  residual_.resize(5 * a.n);
  a.residual(b, x, residual_);
  return gcr_.solve([&](std::span<const double> v, std::span<double> y) { a.multiply(v, y); },
                    [&](std::span<const double> r, std::span<double> z) { cycle_level(0, r, z); },
                    residual_, x, max_cycles, tolerance);
}

void Multigrid::cycle_level(std::size_t l, std::span<const double> b, std::span<double> x) {
  Smoother& s = smoothers_[l];
  const Operator& a = ops_[l];
  if (l + 1 == levels_.size()) {
    if (!s.pivot.empty()) {
      s.solve_dense(b, x);
    } else {
      s.sweep(a, b, x, options_.coarsest_sweeps);
    }
    return;
  }
  s.sweep(a, b, x, options_.pre_sweeps);
  const Level& f = levels_[l];
  const std::size_t n = f.n_nodes, nc = levels_[l + 1].n_nodes;
  std::vector<double>& r = r_[l];
  std::vector<double>& bc = bc_[l + 1];
  std::vector<double>& xc = xc_[l + 1];
  a.residual(b, x, r);
  std::fill(bc.begin(), bc.end(), 0.0);
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t i = 0; i < n; ++i) {
      bc[c * nc + static_cast<std::size_t>(f.coarse_node[i])] += r[c * n + i];
    }
  }
  coarse_solve(l + 1, bc, xc);
  std::vector<double>& e = e_[l];
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t i = 0; i < n; ++i) {
      e[c * n + i] = xc[c * nc + static_cast<std::size_t>(f.coarse_node[i])];
    }
  }
  double alpha = 1.0;
  if (options_.scaling != Scaling::none) {
    std::vector<double>& q = q_[l];
    a.multiply(e, q);
    double num = 0.0, den = 0.0;
    for (std::size_t k = 0; k < 5 * n; ++k) {
      if (options_.scaling == Scaling::residual) {
        num += r[k] * q[k];
        den += q[k] * q[k];
      } else {
        num += r[k] * e[k];
        den += q[k] * e[k];
      }
    }
    if (den > 0.0) alpha = num / den;
  }
  for (std::size_t k = 0; k < 5 * n; ++k) x[k] += alpha * e[k];
  s.sweep(a, b, x, options_.post_sweeps);
}

namespace {

double dot(std::span<const double> a, std::span<const double> b) {
  double s = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) s += a[k] * b[k];
  return s;
}

}  // namespace

// Approximate solve of level m's A x = b from x = 0: one (V) or two (W) cycles, or (K) two
// steps of the generalised conjugate residual method preconditioned by one cycle each, the
// second only when the first left more than a quarter of the residual (Notay and Vassilevski
// 2008, Numer. Linear Algebra Appl. 15:123-145, algorithm 3.1).
void Multigrid::coarse_solve(std::size_t m, std::span<const double> b, std::span<double> x) {
  std::fill(x.begin(), x.end(), 0.0);
  const bool coarsest = m + 1 == levels_.size();
  if (options_.cycle != Cycle::k || coarsest) {
    const int visits = options_.cycle != Cycle::v && !coarsest ? 2 : 1;
    for (int k = 0; k < visits; ++k) cycle_level(m, b, x);
    return;
  }
  const Operator& a = ops_[m];
  auto& kv = krylov_[m];
  std::vector<double>&v1 = kv[0], &w1 = kv[1], &v2 = kv[2], &w2 = kv[3], &r1 = kv[4];
  std::fill(v1.begin(), v1.end(), 0.0);
  cycle_level(m, b, v1);
  a.multiply(v1, w1);
  const double rho1 = dot(w1, w1);
  if (!(rho1 > 0.0)) return;
  const double a1 = dot(w1, b) / rho1;
  for (std::size_t k = 0; k < x.size(); ++k) {
    x[k] = a1 * v1[k];
    r1[k] = b[k] - a1 * w1[k];
  }
  if (dot(r1, r1) <= 0.0625 * dot(b, b)) return;
  std::fill(v2.begin(), v2.end(), 0.0);
  cycle_level(m, r1, v2);
  a.multiply(v2, w2);
  const double beta = dot(w2, w1) / rho1;
  for (std::size_t k = 0; k < x.size(); ++k) {
    v2[k] -= beta * v1[k];
    w2[k] -= beta * w1[k];
  }
  const double rho2 = dot(w2, w2);
  if (!(rho2 > 0.0)) return;
  const double a2 = dot(w2, r1) / rho2;
  for (std::size_t k = 0; k < x.size(); ++k) x[k] += a2 * v2[k];
}

}  // namespace autan::solvers
