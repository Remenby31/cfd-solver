#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "autan/core/mesh.hpp"

namespace autan::solvers {

/// Line-implicit block Gauss-Seidel on one rank's share of a partitioned mesh, for systems of
/// 5 x 5 blocks with the pattern of the mesh's cell adjacency (BlockMatrix's layout: one diagonal
/// block per cell, A(o, n) `upper` and A(n, o) `lower` per interior face, row-major; vectors in
/// structure of arrays, component c of cell i at c * n_cells + i, ghosts included). Each line is
/// solved exactly (block-tridiagonal elimination along it, the other couplings taken from the
/// current values); a cell outside every line is relaxed by its diagonal block. Lines and cells
/// are swept by colour, no two of one colour coupled, so the result does not depend on the order
/// within a colour; with colours and lines that do not depend on the partition, neither does the
/// result, bit for bit.
class LineSmoother {
 public:
  /// `mesh` is the local mesh: owned cells first, then ghosts. `line_offsets`/`line_cells` are
  /// the lines of owned cells (CSR, at least two cells each, in sweep order); `cell_colours`
  /// gives each owned cell the colour of its line or its own, every cell of a line the same,
  /// in [0, n_colours). Each cell sums its couplings in increasing `face_global_id` (one per
  /// interior face; empty: local face order), so that the sums do not depend on the partition.
  /// Throws std::invalid_argument on a line holding a ghost, a cell in two lines, consecutive
  /// line cells that share no face, or a line whose cells differ in colour.
  LineSmoother(const Mesh& mesh, std::span<const offset_t> line_offsets,
               std::span<const index_t> line_cells, std::span<const std::int32_t> cell_colours,
               std::int32_t n_colours, std::span<const std::int64_t> face_global_id = {});

  /// Sets the operator and factors every line and every lone cell's diagonal block. The arrays
  /// are referenced, not copied: they must stay unchanged and alive until the next call. Only
  /// owned rows are read. Throws std::domain_error on a singular block.
  void set_operator(std::span<const double> diagonal, std::span<const double> upper,
                    std::span<const double> lower);

  [[nodiscard]] std::int32_t n_colours() const noexcept { return n_colours_; }
  /// Relaxes the lines and lone cells of one colour on A x = b, from the current x, ghosts
  /// included.
  void relax(std::int32_t colour, std::span<const double> b, std::span<double> x) const;
  /// `sweeps` symmetric sweeps: colours in increasing then decreasing order, `exchange(x)` after
  /// each colour so that ghosts take the values their owners just computed.
  void sweep(std::span<const double> b, std::span<double> x, int sweeps,
             const std::function<void(std::span<double>)>& exchange) const;

 private:
  struct Unit {
    std::size_t first, size;  // cells cells_[first, first + size)
  };
  // The coupling A(c, m) of cell c to cell m summed over the faces they share, into out.
  void coupling(std::size_t c, std::size_t m, double* out) const;
  void solve_unit(const Unit& u, std::span<const double> b, std::span<double> x) const;

  std::size_t n_cells_, n_owned_;
  std::int32_t n_colours_;
  std::vector<index_t> owner_, neighbour_;
  std::vector<offset_t> cell_offsets_;  // owned cell -> interior faces
  std::vector<index_t> cell_faces_;
  std::vector<index_t> cells_;  // units' cells, unit after unit
  std::vector<Unit> units_;
  std::vector<std::vector<std::size_t>> by_colour_;  // colour -> units
  std::span<const double> diag_, upper_, lower_;
  // Per unit cell (index into cells_): G = (D')^-1 and G U, U the coupling to the next cell.
  std::vector<double> g_, gu_, sub_;  // sub_: coupling to the previous cell
};

}  // namespace autan::solvers
