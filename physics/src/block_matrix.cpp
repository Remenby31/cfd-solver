#include "autan/physics/block_matrix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace autan::physics {

namespace {

// Inverse of a 5 x 5 row-major block by Gauss-Jordan elimination with partial pivoting.
void invert5(const double* a, double* inv) {
  std::array<double, 50> m{};
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < 5; ++j) {
      m[10 * i + j] = a[5 * i + j];
      m[10 * i + 5 + j] = i == j ? 1.0 : 0.0;
    }
  }
  for (std::size_t k = 0; k < 5; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < 5; ++i) {
      if (std::abs(m[10 * i + k]) > std::abs(m[10 * p + k])) p = i;
    }
    if (m[10 * p + k] == 0.0 || !std::isfinite(m[10 * p + k])) {
      throw std::domain_error("singular diagonal block");
    }
    if (p != k) {
      for (std::size_t j = 0; j < 10; ++j) std::swap(m[10 * k + j], m[10 * p + j]);
    }
    const double d = 1.0 / m[10 * k + k];
    for (std::size_t j = 0; j < 10; ++j) m[10 * k + j] *= d;
    for (std::size_t i = 0; i < 5; ++i) {
      if (i == k) continue;
      const double f = m[10 * i + k];
      for (std::size_t j = 0; j < 10; ++j) m[10 * i + j] -= f * m[10 * k + j];
    }
  }
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < 5; ++j) inv[5 * i + j] = m[10 * i + 5 + j];
  }
}

}  // namespace

BlockMatrix::BlockMatrix(const Mesh& mesh)
    : n_cells_(static_cast<std::size_t>(mesh.n_cells())),
      owner_(mesh.owner().begin(), mesh.owner().begin() + mesh.n_interior_faces()),
      neighbour_(mesh.neighbour().begin(), mesh.neighbour().end()) {
  const std::size_t ni = neighbour_.size();
  cell_offsets_.assign(n_cells_ + 1, 0);
  for (std::size_t f = 0; f < ni; ++f) {
    ++cell_offsets_[static_cast<std::size_t>(owner_[f]) + 1];
    ++cell_offsets_[static_cast<std::size_t>(neighbour_[f]) + 1];
  }
  for (std::size_t c = 0; c < n_cells_; ++c) cell_offsets_[c + 1] += cell_offsets_[c];
  cell_faces_.resize(static_cast<std::size_t>(cell_offsets_.back()));
  std::vector<offset_t> fill(cell_offsets_.begin(), cell_offsets_.end() - 1);
  for (std::size_t f = 0; f < ni; ++f) {
    cell_faces_[static_cast<std::size_t>(fill[static_cast<std::size_t>(owner_[f])]++)] =
        static_cast<index_t>(f);
    cell_faces_[static_cast<std::size_t>(fill[static_cast<std::size_t>(neighbour_[f])]++)] =
        static_cast<index_t>(f);
  }
  // Greedy distance-1 colouring in cell order: cells of a colour share no face.
  std::vector<int> colour(n_cells_, -1);
  std::vector<int> seen;
  int n_colours = 0;
  for (std::size_t c = 0; c < n_cells_; ++c) {
    seen.assign(static_cast<std::size_t>(n_colours) + 1, 0);
    for (auto k = cell_offsets_[c]; k < cell_offsets_[c + 1]; ++k) {
      const auto f = static_cast<std::size_t>(cell_faces_[static_cast<std::size_t>(k)]);
      const auto o = static_cast<std::size_t>(owner_[f] == static_cast<index_t>(c) ? neighbour_[f]
                                                                                   : owner_[f]);
      if (colour[o] >= 0) seen[static_cast<std::size_t>(colour[o])] = 1;
    }
    int k = 0;
    while (seen[static_cast<std::size_t>(k)] != 0) ++k;
    colour[c] = k;
    n_colours = std::max(n_colours, k + 1);
  }
  colours_.assign(static_cast<std::size_t>(n_colours), {});
  for (std::size_t c = 0; c < n_cells_; ++c)
    colours_[static_cast<std::size_t>(colour[c])].push_back(c);
  diag_.assign(25 * n_cells_, 0.0);
  inverse_.assign(25 * n_cells_, 0.0);
  upper_.assign(25 * ni, 0.0);
  lower_.assign(25 * ni, 0.0);
}

void BlockMatrix::zero() {
  std::fill(diag_.begin(), diag_.end(), 0.0);
  std::fill(upper_.begin(), upper_.end(), 0.0);
  std::fill(lower_.begin(), lower_.end(), 0.0);
}

void BlockMatrix::multiply(std::span<const double> x, std::span<double> y) const {
  const std::size_t n = n_cells_;
  auto add = [&](const double* a, std::size_t row, std::size_t col) {
    for (std::size_t i = 0; i < 5; ++i) {
      double s = 0.0;
      for (std::size_t j = 0; j < 5; ++j) s += a[5 * i + j] * x[j * n + col];
      y[i * n + row] += s;
    }
  };
  std::fill(y.begin(), y.end(), 0.0);
  for (std::size_t c = 0; c < n; ++c) add(&diag_[25 * c], c, c);
  for (std::size_t f = 0; f < neighbour_.size(); ++f) {
    const auto o = static_cast<std::size_t>(owner_[f]);
    const auto nb = static_cast<std::size_t>(neighbour_[f]);
    add(&upper_[25 * f], o, nb);
    add(&lower_[25 * f], nb, o);
  }
}

void BlockMatrix::factor() { factor(n_cells_); }

void BlockMatrix::factor(std::size_t rows) {
  for (std::size_t c = 0; c < rows; ++c) invert5(&diag_[25 * c], &inverse_[25 * c]);
}

void BlockMatrix::relax(std::size_t c, std::span<const double> b, std::span<double> x) const {
  const std::size_t n = n_cells_;
  std::array<double, 5> r{};
  for (std::size_t i = 0; i < 5; ++i) r[i] = b[i * n + c];
  for (auto k = cell_offsets_[c]; k < cell_offsets_[c + 1]; ++k) {
    const auto f = static_cast<std::size_t>(cell_faces_[static_cast<std::size_t>(k)]);
    const bool owned = static_cast<std::size_t>(owner_[f]) == c;
    const double* a = owned ? &upper_[25 * f] : &lower_[25 * f];
    const auto other = static_cast<std::size_t>(owned ? neighbour_[f] : owner_[f]);
    for (std::size_t i = 0; i < 5; ++i) {
      for (std::size_t j = 0; j < 5; ++j) r[i] -= a[5 * i + j] * x[j * n + other];
    }
  }
  const double* inv = &inverse_[25 * c];
  for (std::size_t i = 0; i < 5; ++i) {
    double s = 0.0;
    for (std::size_t j = 0; j < 5; ++j) s += inv[5 * i + j] * r[j];
    x[i * n + c] = s;
  }
}

void BlockMatrix::gauss_seidel(std::span<const double> b, std::span<double> x, int sweeps) const {
  for (int s = 0; s < sweeps; ++s) {
    for (const auto& colour : colours_) {
      for (const std::size_t c : colour) relax(c, b, x);
    }
    for (auto it = colours_.rbegin(); it != colours_.rend(); ++it) {
      for (const std::size_t c : *it) relax(c, b, x);
    }
  }
}

void BlockMatrix::gauss_seidel(std::span<const double> b, std::span<double> x, int sweeps,
                               std::span<const std::vector<std::size_t>> colours,
                               const std::function<void(std::span<double>)>& exchange) const {
  for (int s = 0; s < sweeps; ++s) {
    for (const auto& colour : colours) {
      for (const std::size_t c : colour) relax(c, b, x);
      exchange(x);
    }
    for (auto it = colours.rbegin(); it != colours.rend(); ++it) {
      for (const std::size_t c : *it) relax(c, b, x);
      exchange(x);
    }
  }
}

}  // namespace autan::physics
