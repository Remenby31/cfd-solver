#pragma once

#include <span>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"

namespace autan {

/// Reverse Cuthill-McKee order of the owned cells (Cuthill & McKee 1969;
/// reversal: George 1971), each connected component started from a
/// pseudo-peripheral cell (Gibbs, Poole & Stockmeyer 1976). Ghost cells keep
/// their order after the owned cells. Returns new_to_old: the cell placed at
/// position i is new_to_old[i]. Deterministic.
std::vector<index_t> rcm_order(const Mesh& mesh);

/// Returns `mesh` with its cells permuted (cell i of the result is cell
/// new_to_old[i] of `mesh`) and its faces re-sorted by assemble(). Any
/// permutation that maps owned cells to owned positions is accepted, so other
/// orderings (space-filling curves) plug in here. Every cell keeps its global
/// id. `map`, when given, receives assemble()'s map, whose face_source gives
/// each face of the result as a face of `mesh`. Throws E0102 for an invalid
/// permutation.
Mesh renumber_cells(const Mesh& mesh, std::span<const index_t> new_to_old,
                    AssemblyMap* map = nullptr);

/// The geometry of renumber_cells(mesh, new_to_old, &map) taken from
/// `geometry`, the geometry of `mesh`: cells and faces permuted, the area
/// vector of a flipped face negated. Its bits are those of `mesh`'s, whereas
/// compute_geometry on the renumbered mesh sums each cell's faces in another
/// order. Throws E0102 for sizes that do not match.
Geometry renumber_geometry(const Geometry& geometry, std::span<const index_t> new_to_old,
                           const AssemblyMap& map);

}  // namespace autan
