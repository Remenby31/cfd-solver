#include "autan/solvers/line_smoother.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

#include "blocks.hpp"

namespace autan::solvers {

using detail::b2;

LineSmoother::LineSmoother(const Mesh& mesh, std::span<const offset_t> line_offsets,
                           std::span<const index_t> line_cells,
                           std::span<const std::int32_t> cell_colours, std::int32_t n_colours,
                           std::span<const std::int64_t> face_global_id)
    : n_cells_(static_cast<std::size_t>(mesh.n_cells())),
      n_owned_(static_cast<std::size_t>(mesh.n_owned_cells())),
      n_colours_(n_colours) {
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  owner_.assign(mesh.owner().begin(), mesh.owner().begin() + static_cast<std::ptrdiff_t>(ni));
  neighbour_.assign(mesh.neighbour().begin(), mesh.neighbour().end());
  cell_offsets_.assign(n_owned_ + 1, 0);
  for (std::size_t f = 0; f < ni; ++f) {
    for (const index_t c : {owner_[f], neighbour_[f]}) {
      if (static_cast<std::size_t>(c) < n_owned_) ++cell_offsets_[static_cast<std::size_t>(c) + 1];
    }
  }
  for (std::size_t c = 0; c < n_owned_; ++c) cell_offsets_[c + 1] += cell_offsets_[c];
  cell_faces_.resize(static_cast<std::size_t>(cell_offsets_.back()));
  std::vector<offset_t> fill(cell_offsets_.begin(), cell_offsets_.end() - 1);
  for (std::size_t f = 0; f < ni; ++f) {
    for (const index_t c : {owner_[f], neighbour_[f]}) {
      const auto cu = static_cast<std::size_t>(c);
      if (cu < n_owned_)
        cell_faces_[static_cast<std::size_t>(fill[cu]++)] = static_cast<index_t>(f);
    }
  }
  if (!face_global_id.empty()) {
    if (face_global_id.size() < ni) {
      throw std::invalid_argument("line smoother: a global id per interior face");
    }
    for (std::size_t c = 0; c < n_owned_; ++c) {
      std::sort(cell_faces_.begin() + cell_offsets_[c], cell_faces_.begin() + cell_offsets_[c + 1],
                [&](index_t a, index_t b) {
                  return face_global_id[static_cast<std::size_t>(a)] <
                         face_global_id[static_cast<std::size_t>(b)];
                });
    }
  }
  if (cell_colours.size() < n_owned_ || n_colours < 0) {
    throw std::invalid_argument("line smoother: a colour per owned cell");
  }
  auto adjacent = [&](std::size_t a, std::size_t b) {
    for (auto k = cell_offsets_[a]; k < cell_offsets_[a + 1]; ++k) {
      const auto f = static_cast<std::size_t>(cell_faces_[static_cast<std::size_t>(k)]);
      if (static_cast<std::size_t>(owner_[f]) == b ||
          static_cast<std::size_t>(neighbour_[f]) == b) {
        return true;
      }
    }
    return false;
  };
  std::vector<std::uint8_t> in_line(n_owned_, 0);
  for (std::size_t l = 0; l + 1 < line_offsets.size(); ++l) {
    const auto b = static_cast<std::size_t>(line_offsets[l]);
    const auto e = static_cast<std::size_t>(line_offsets[l + 1]);
    for (std::size_t k = b; k < e; ++k) {
      const auto c = static_cast<std::size_t>(line_cells[k]);
      if (c >= n_owned_ || in_line[c] != 0) {
        throw std::invalid_argument("line smoother: a line cell is a ghost or in two lines");
      }
      if (k > b && !adjacent(c, static_cast<std::size_t>(line_cells[k - 1]))) {
        throw std::invalid_argument("line smoother: consecutive line cells share no face");
      }
      if (cell_colours[c] != cell_colours[static_cast<std::size_t>(line_cells[b])]) {
        throw std::invalid_argument("line smoother: a line spans two colours");
      }
      in_line[c] = 1;
      cells_.push_back(line_cells[k]);
    }
    units_.push_back({cells_.size() - (e - b), e - b});
  }
  for (std::size_t c = 0; c < n_owned_; ++c) {
    if (in_line[c] != 0) continue;
    cells_.push_back(static_cast<index_t>(c));
    units_.push_back({cells_.size() - 1, 1});
  }
  by_colour_.assign(static_cast<std::size_t>(n_colours), {});
  for (std::size_t u = 0; u < units_.size(); ++u) {
    const std::int32_t colour = cell_colours[static_cast<std::size_t>(cells_[units_[u].first])];
    if (colour < 0 || colour >= n_colours) {
      throw std::invalid_argument("line smoother: a colour out of range");
    }
    by_colour_[static_cast<std::size_t>(colour)].push_back(u);
  }
}

void LineSmoother::coupling(std::size_t c, std::size_t m, double* out) const {
  std::fill(out, out + b2, 0.0);
  for (auto k = cell_offsets_[c]; k < cell_offsets_[c + 1]; ++k) {
    const auto f = static_cast<std::size_t>(cell_faces_[static_cast<std::size_t>(k)]);
    if (static_cast<std::size_t>(owner_[f]) == c && static_cast<std::size_t>(neighbour_[f]) == m) {
      detail::add5(out, &upper_[b2 * f]);
    } else if (static_cast<std::size_t>(neighbour_[f]) == c &&
               static_cast<std::size_t>(owner_[f]) == m) {
      detail::add5(out, &lower_[b2 * f]);
    }
  }
}

// Block-tridiagonal elimination along each line (Thomas): D'_0 = D_0,
// D'_k = D_k - L_k (D'_{k-1})^-1 U_{k-1}, with L_k = A(c_k, c_{k-1}) and U_k = A(c_k, c_{k+1}).
void LineSmoother::set_operator(std::span<const double> diagonal, std::span<const double> upper,
                                std::span<const double> lower) {
  diag_ = diagonal;
  upper_ = upper;
  lower_ = lower;
  g_.assign(b2 * cells_.size(), 0.0);
  gu_.assign(b2 * cells_.size(), 0.0);
  sub_.assign(b2 * cells_.size(), 0.0);
  std::array<double, b2> d{}, t{}, u{};
  for (const Unit& unit : units_) {
    for (std::size_t k = 0; k < unit.size; ++k) {
      const std::size_t i = unit.first + k;
      const auto c = static_cast<std::size_t>(cells_[i]);
      std::copy_n(&diag_[b2 * c], b2, d.begin());
      if (k > 0) {
        coupling(c, static_cast<std::size_t>(cells_[i - 1]), &sub_[b2 * i]);
        detail::multiply5(&sub_[b2 * i], &gu_[b2 * (i - 1)], t.data());
        for (std::size_t q = 0; q < b2; ++q) d[q] -= t[q];
      }
      detail::invert5(d.data(), &g_[b2 * i]);
      if (k + 1 < unit.size) {
        coupling(c, static_cast<std::size_t>(cells_[i + 1]), u.data());
        detail::multiply5(&g_[b2 * i], u.data(), &gu_[b2 * i]);
      }
    }
  }
}

void LineSmoother::solve_unit(const Unit& unit, std::span<const double> b,
                              std::span<double> x) const {
  const std::size_t n = n_cells_;
  std::array<double, 5> r{}, s{};
  std::vector<double> w(5 * unit.size);
  for (std::size_t k = 0; k < unit.size; ++k) {
    const std::size_t i = unit.first + k;
    const auto c = static_cast<std::size_t>(cells_[i]);
    const std::size_t prev = k > 0 ? static_cast<std::size_t>(cells_[i - 1]) : n;
    const std::size_t next = k + 1 < unit.size ? static_cast<std::size_t>(cells_[i + 1]) : n;
    for (std::size_t q = 0; q < 5; ++q) r[q] = b[q * n + c];
    // The couplings outside the line, from the current values.
    for (auto p = cell_offsets_[c]; p < cell_offsets_[c + 1]; ++p) {
      const auto f = static_cast<std::size_t>(cell_faces_[static_cast<std::size_t>(p)]);
      const bool own = static_cast<std::size_t>(owner_[f]) == c;
      const auto m = static_cast<std::size_t>(own ? neighbour_[f] : owner_[f]);
      if (m == prev || m == next) continue;
      const double* a = own ? &upper_[b2 * f] : &lower_[b2 * f];
      for (std::size_t q = 0; q < 5; ++q) {
        double v = 0.0;
        for (std::size_t j = 0; j < 5; ++j) v += a[5 * q + j] * x[j * n + m];
        r[q] -= v;
      }
    }
    if (k > 0) {
      const double* l = &sub_[b2 * i];
      for (std::size_t q = 0; q < 5; ++q) {
        double v = 0.0;
        for (std::size_t j = 0; j < 5; ++j) v += l[5 * q + j] * w[5 * (k - 1) + j];
        r[q] -= v;
      }
    }
    const double* g = &g_[b2 * i];
    for (std::size_t q = 0; q < 5; ++q) {
      double v = 0.0;
      for (std::size_t j = 0; j < 5; ++j) v += g[5 * q + j] * r[j];
      w[5 * k + q] = v;
    }
  }
  for (std::size_t k = unit.size; k-- > 0;) {
    const std::size_t i = unit.first + k;
    const auto c = static_cast<std::size_t>(cells_[i]);
    if (k + 1 < unit.size) {
      const double* gu = &gu_[b2 * i];
      for (std::size_t q = 0; q < 5; ++q) {
        double v = 0.0;
        for (std::size_t j = 0; j < 5; ++j) v += gu[5 * q + j] * s[j];
        w[5 * k + q] -= v;
      }
    }
    for (std::size_t q = 0; q < 5; ++q) {
      s[q] = w[5 * k + q];
      x[q * n + c] = s[q];
    }
  }
}

void LineSmoother::relax(std::int32_t colour, std::span<const double> b,
                         std::span<double> x) const {
  for (const std::size_t u : by_colour_[static_cast<std::size_t>(colour)]) {
    solve_unit(units_[u], b, x);
  }
}

void LineSmoother::sweep(std::span<const double> b, std::span<double> x, int sweeps,
                         const std::function<void(std::span<double>)>& exchange) const {
  for (int s = 0; s < sweeps; ++s) {
    for (std::int32_t c = 0; c < n_colours_; ++c) {
      relax(c, b, x);
      exchange(x);
    }
    for (std::int32_t c = n_colours_; c-- > 0;) {
      relax(c, b, x);
      exchange(x);
    }
  }
}

}  // namespace autan::solvers
