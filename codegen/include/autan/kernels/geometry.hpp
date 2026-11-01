#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/kernels/runtime.hpp"

namespace autan::kernels {

/// FP64 geometry in the layout of the generated `geometry.*` datasets: structure of arrays,
/// component c of entity i at c * n + i, matrices row-major.
struct PackedGeometry {
  std::vector<double> face_area;      ///< 3 x n_faces, owner -> neighbour
  std::vector<double> face_centroid;  ///< 3 x n_faces
  std::vector<double> face_normal;    ///< 3 x n_faces, unit
  std::vector<double> face_area_magnitude;
  std::vector<double> face_r_left;  ///< 3 x n_faces: face centroid - owner centroid
  std::vector<double>
      face_r_right;  ///< 3 x n_faces: face centroid - neighbour centroid, 0 on boundary
  /// 9 x n_faces: sum over the face's fan triangles t of (c_t - x_f) (x) S_t. Zero on a planar
  /// face; with it the flux of a linear field u through any face is u(x_f) S_f + M_f^T grad(u).
  std::vector<double> face_moment;
  std::vector<double> cell_centroid;  ///< 3 x n_cells
  std::vector<double> cell_volume;    ///< n_cells
  /// 3 x n_incidences: weighted least-squares gradient weights w_k = M^-1 d_k / |d_k|^2 with
  /// M = sum_k d_k d_k^T / |d_k|^2, d_k from the cell centroid to the neighbour centroid
  /// (interior face) or the face centroid (boundary face). Zero on ghost cells.
  std::vector<double> lsq_weight;
  /// n_incidences: +1 where the cell is the face's owner (left), and on boundary faces; -1 where
  /// it is the neighbour (right).
  std::vector<double> incidence_side;
  /// 3 x stencil size: LSQ weights over the cell stencil, when one is set (CellStencil).
  std::vector<double> stencil_lsq_weight;
};

/// The global numbering of a partition (LocalPartition: Mesh::cell_global_id, face_global_id,
/// face_reversed), so that what a cell sums or a face computes does not depend on how the mesh
/// was partitioned: bit-for-bit results across rank counts (runtime.md, Ranks).
struct GlobalKeys {
  std::span<const std::int64_t> cell;      ///< global id of each local cell
  std::span<const std::int64_t> face;      ///< global id of each local face
  std::span<const std::uint8_t> reversed;  ///< 1 where a local face runs against its global one
  /// Translation of each ghost cell (Mesh::ghost_translation; zero for an ordinary ghost): a
  /// periodic image shares its cell's global id, so the pair (id, translation) is the key.
  std::span<const std::array<double, 3>> shift;
};

/// Without `keys`, as the mesh is numbered. With them: face moments computed in each face's
/// global orientation from a canonical fan start (the node of smallest coordinates) then signed
/// for the local orientation, and lsq_weight in the incidence order of global_order(mesh, keys).
PackedGeometry pack_geometry(const Mesh& mesh, const Geometry& geometry,
                             const GlobalKeys* keys = nullptr);

/// Each cell's faces ordered by global face id (Topology::cell_faces) and their incidence twins
/// (Topology::incidence_twin): a gather over a cell's faces then adds them in the same order on
/// every rank count.
struct GlobalOrder {
  std::vector<index_t> cell_faces;
  std::vector<offset_t> twin;
};
GlobalOrder global_order(const Mesh& mesh, const GlobalKeys& keys);

/// Topology of `mesh`; `incidence_twin` stays null (set it from incidence_twins()).
[[nodiscard]] Topology topology(const MeshView& mesh) noexcept;

/// For each incidence k (entry of the cell -> face CSR) of an interior face f, the incidence of f
/// in the other cell's list; -1 for boundary faces.
std::vector<offset_t> incidence_twins(const Mesh& mesh);

/// Cell stencils for least-squares gradients wider than the face neighbours, which make the
/// unlimited second-order scheme linearly stable on tetrahedra (Diskin and Thomas, AIAA J.
/// 49(4), 2011): `vertex` holds the cells sharing a node with the cell and the boundary faces
/// sharing a node with it; `face2` the face neighbours and their face neighbours, and the
/// boundary faces of the cell and of its face neighbours.
enum class StencilKind : std::uint8_t { vertex, face2 };

/// A cell -> entries CSR (layout of Topology::stencil) with its twins and the LSQ weights over
/// it (3 x size, structure of arrays, as PackedGeometry::lsq_weight). Entries of a cell are its
/// stencil cells in increasing order, then its boundary faces in increasing order. Ghost cells get
/// the stencil their known faces give, so every owned -> ghost entry has its twin (the adjoint's
/// partial sums for the reverse halo exchange); LSQ weights are those of owned cells. A ghost
/// layer as deep as the stencil is the partitioner's job.
struct CellStencil {
  std::vector<offset_t> offsets;
  std::vector<index_t> entries;
  std::vector<offset_t> twin;
  std::vector<double> lsq_weight;
  /// 3 x stencil size: LSQ weights over the cell stencil, when one is set (CellStencil).
  std::vector<double> stencil_lsq_weight;
};

/// With `keys`, a cell's entries are its stencil cells by global cell id, then its boundary faces
/// by global face id, and the LSQ weights are summed in that order.
CellStencil cell_stencil(const Mesh& mesh, const PackedGeometry& geometry, StencilKind kind,
                         const GlobalKeys* keys = nullptr);

/// A cell stencil in sliced layout (Topology::stencil_stride): cells in slices of `slice`
/// consecutive cells, entry j of the cell in lane l of slice s stored at start(s) + j * slice + l,
/// start(s) the sum of slice * (longest row) over the slices before. Each cell keeps its entries
/// in the same order, so a reduction gives the same bits. `position[k]` is where CSR entry k
/// went; twins point at sliced positions; padding holds entry 0, twin -1, weights 0.
struct SlicedStencil {
  CellStencil stencil;  ///< offsets: n_cells + 1 starts (the last the total size)
  std::vector<index_t> length;
  std::vector<offset_t> position;
  offset_t stride = 1;
};
SlicedStencil sliced(const CellStencil& csr, int slice);

/// The cell -> face incidences of `csr` (a Topology in CSR layout, with incidence_twin) in the
/// sliced layout of SlicedStencil (Topology::incidence_stride): cell_face_offsets holds n_cells + 1
/// starts (the last the total size, padding included), `cell` the cell across each incidence's
/// face (-1 on boundary faces and padding), twins sliced positions. Each cell keeps its faces in
/// the same order, so a reduction gives the same bits. Padding holds face 0.
struct SlicedIncidences {
  std::vector<offset_t> cell_face_offsets;
  std::vector<index_t> length;
  std::vector<offset_t> position;
  std::vector<index_t> cell_faces;
  std::vector<index_t> cell;
  std::vector<offset_t> twin;
  offset_t stride = 1;
};
SlicedIncidences sliced_incidences(const Topology& csr, int slice);

/// Per-incidence data (`comps` x n_incidences, component-major) moved to the sliced positions of
/// `s` (`comps` x its total size), padding 0.
std::vector<double> sliced_values(const SlicedIncidences& s, const std::vector<double>& csr,
                                  int comps);

/// Copy of `v` converted to T (the storage type of a precision role).
template <class T>
std::vector<T> converted(const std::vector<double>& v) {
  return std::vector<T>(v.begin(), v.end());
}

}  // namespace autan::kernels
