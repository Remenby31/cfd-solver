#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "autan/core/mesh.hpp"

namespace autan::physics {

/// A sparse matrix of 5 x 5 blocks with the pattern of a mesh's cell adjacency: one diagonal
/// block per cell, and per interior face f (owner o, neighbour n) the blocks A(o, n) (`upper`) and
/// A(n, o) (`lower`). Blocks are row-major; vectors are structure of arrays, component c of cell
/// i at c * n_cells + i, as the residual. Accumulates in FP64.
class BlockMatrix {
 public:
  explicit BlockMatrix(const Mesh& mesh);

  [[nodiscard]] std::size_t n_cells() const noexcept { return n_cells_; }
  void zero();
  /// Block of cell c / of interior face f; 25 values, row-major.
  [[nodiscard]] double* diagonal(std::size_t c) { return &diag_[25 * c]; }
  [[nodiscard]] double* upper(std::size_t f) { return &upper_[25 * f]; }
  [[nodiscard]] double* lower(std::size_t f) { return &lower_[25 * f]; }
  /// Every diagonal / upper / lower block, in cell / interior face order.
  [[nodiscard]] std::span<const double> diagonals() const noexcept { return diag_; }
  [[nodiscard]] std::span<const double> uppers() const noexcept { return upper_; }
  [[nodiscard]] std::span<const double> lowers() const noexcept { return lower_; }

  /// y = A x.
  void multiply(std::span<const double> x, std::span<double> y) const;
  /// Inverts every diagonal block (Gaussian elimination with partial pivoting); throws
  /// std::domain_error on a singular block.
  void factor();
  /// As factor(), for the first `rows` cells only (the owned cells of a partition, whose rows
  /// are complete).
  void factor(std::size_t rows);
  /// `sweeps` symmetric multicolour block Gauss-Seidel sweeps on A x = b from x, after factor().
  /// Cells of one colour share no face, so their updates are independent; the order is fixed,
  /// so the result is deterministic.
  void gauss_seidel(std::span<const double> b, std::span<double> x, int sweeps) const;
  /// `sweeps` symmetric block Gauss-Seidel sweeps over the given colours (each a list of owned
  /// cells sharing no face), in increasing then decreasing colour order, calling `exchange(x)`
  /// after each colour so that ghost cells take the values their owners just computed. With
  /// colours that do not depend on the partition, neither does the result.
  void gauss_seidel(std::span<const double> b, std::span<double> x, int sweeps,
                    std::span<const std::vector<std::size_t>> colours,
                    const std::function<void(std::span<double>)>& exchange) const;

 private:
  void relax(std::size_t c, std::span<const double> b, std::span<double> x) const;

  std::size_t n_cells_;
  std::vector<index_t> owner_, neighbour_;
  std::vector<offset_t> cell_offsets_;  // cell -> interior faces CSR
  std::vector<index_t> cell_faces_;
  std::vector<std::vector<std::size_t>> colours_;
  std::vector<double> diag_, upper_, lower_, inverse_;
};

}  // namespace autan::physics
