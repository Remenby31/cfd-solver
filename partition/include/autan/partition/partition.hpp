#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"

namespace autan {

/// Graph partitioner behind partition_cells. METIS is the default; Scotch is
/// the option for builds that carry it. Both are seeded, so the same mesh and
/// options give the same parts on every run.
enum class PartitionMethod : std::uint8_t { metis, scotch };

struct PartitionOptions {
  PartitionMethod method = PartitionMethod::metis;
  /// Cost of each cell, ≥ 1; empty for uniform cost. cell_cost_weights gives
  /// the face-count cost of polyhedral meshes.
  std::vector<std::int32_t> cell_weights;
  /// Allowed load imbalance: the heaviest part may weigh (1 + tolerance)
  /// times the mean.
  double imbalance_tolerance = 0.03;
  std::int32_t seed = 0;
  /// Groups of cells that must share a part (a smoother's lines), as CSR: group g holds
  /// group_cells[group_offsets[g] .. group_offsets[g + 1]). Each group is one node of the graph
  /// the method partitions, weighing the sum of its cells. Groups are non-empty and disjoint,
  /// or the request is refused (E0140) whatever the part count. Both empty: no group.
  std::span<const offset_t> group_offsets = {};
  std::span<const index_t> group_cells = {};
};

/// Number of faces of each cell: the cost of a face loop, which dominates the
/// solver, for cells with from 4 to hundreds of faces.
std::vector<std::int32_t> cell_cost_weights(const Mesh& mesh);

/// Part in [0, n_parts) of every cell of an unpartitioned mesh, from its dual
/// graph (cells joined by interior faces). Throws E0140 for n_parts < 1, more
/// parts than cells or weights of the wrong size or below 1, E0141 if the mesh
/// already has ghost cells, E0142 if the method is not built or fails, and
/// E0143 if a part comes out empty.
std::vector<std::int32_t> partition_cells(const Mesh& mesh, std::int32_t n_parts,
                                          const PartitionOptions& options = {});

/// Greedy distance-1 colouring of a graph: nodes visited in increasing
/// `global_id` (distinct), each taking the least colour that none of its
/// already coloured neighbours has. `offsets` (n + 1) and `neighbours` give
/// each node's neighbours (CSR, symmetric). The colours depend on the graph
/// and the ids only, not on the order of the nodes, so every rank of a run
/// and every cell order compute the same ones. Throws E0140 for sizes that
/// do not match or a neighbour out of range.
std::vector<std::int32_t> colour_greedy(std::span<const offset_t> offsets,
                                        std::span<const index_t> neighbours,
                                        std::span<const std::int64_t> global_id);

/// colour_greedy on the cells of an unpartitioned mesh: neighbours across
/// interior faces and across periodic pairs, ids cell_global_id. Two cells
/// that a smoother updates from each other's values never share a colour.
/// With groups (CSR, as PartitionOptions::group_offsets and group_cells), each
/// group is one node, its id the least cell_global_id of its cells, and every
/// cell takes its group's colour: a line smoother's colouring of its lines.
/// Throws E0141 if the mesh has ghosts, E0140 for invalid groups.
std::vector<std::int32_t> colour_cells(const Mesh& global,
                                       std::span<const offset_t> group_offsets = {},
                                       std::span<const index_t> group_cells = {});

/// Quality of a partition, as each part would hold it with one ghost layer.
struct PartitionQuality {
  std::int64_t cut_faces = 0;              ///< interior faces whose cells lie in two parts
  double imbalance = 0.0;                  ///< heaviest part weight / mean part weight
  double ghost_ratio = 0.0;                ///< total ghost cells / total owned cells
  double max_ghost_ratio = 0.0;            ///< largest ghost/owned ratio of one part
  std::int32_t max_neighbours = 0;         ///< largest number of parts one part exchanges with
  std::vector<std::int64_t> owned_cells;   ///< per part
  std::vector<std::int64_t> ghost_cells;   ///< per part
  std::vector<std::int64_t> part_weights;  ///< per part, with the weights given
};

/// Measures `parts` on `mesh` (unpartitioned). `cell_weights` empty means
/// uniform. Throws E0140 if a part index or a size is out of range.
PartitionQuality partition_quality(const Mesh& mesh, std::span<const std::int32_t> parts,
                                   std::int32_t n_parts,
                                   std::span<const std::int32_t> cell_weights = {});

/// What one rank exchanges with the others, its own rank included when it
/// holds periodic images of its cells. Ghost cells are ordered by owning
/// rank, then by index in the global mesh, then by periodic shift, so the ghosts received from
/// neighbours[i] are the contiguous local cells n_owned_cells + [recv_offsets[i],
/// recv_offsets[i+1]). send_cells[send_offsets[i] .. send_offsets[i+1]) are the owned cells whose
/// copies are those ghosts on neighbours[i], in the same order.
struct HaloPattern {
  std::int32_t rank = 0;
  std::int32_t n_ranks = 1;
  std::vector<std::int32_t> neighbours;  ///< ascending
  std::vector<index_t> send_offsets{0};
  std::vector<index_t> send_cells;
  std::vector<index_t> recv_offsets{0};
  /// Reverse accumulation in a fixed order: owned cell reverse_cells[u]
  /// receives the send slots reverse_slots[reverse_offsets[u] ..
  /// reverse_offsets[u+1]), ascending, so by ascending neighbour rank.
  std::vector<index_t> reverse_cells;
  std::vector<index_t> reverse_offsets{0};
  std::vector<index_t> reverse_slots;
  /// Interior faces whose neighbour is a ghost, ascending: the faces a kernel
  /// must defer until the forward exchange completes.
  std::vector<index_t> interface_faces;
};

/// Where the fields an exchange moves live. `cuda` packs and unpacks on the
/// GPU and hands device buffers to a CUDA-aware MPI.
enum class MemorySpace : std::uint8_t { host, cuda };

/// One rank's share of a partitioned mesh.
struct LocalPartition {
  Mesh mesh;
  /// Owned cells and every face carry the global geometry bit for bit (area
  /// vectors negated on faces whose orientation flipped); ghost cells hold
  /// NaN until a forward halo exchange fills them.
  Geometry geometry;
  std::vector<std::int64_t> face_global_id;  ///< face of the global mesh of each local face
  /// 1 where the local face runs opposite to its global face: its owner is
  /// the global face's neighbour (or, for a periodic face, the cell across the
  /// pair), and its area vector is negated. Boundary faces never are.
  std::vector<std::uint8_t> face_reversed;
  HaloPattern halo;
};

/// Builds the part `rank` of `parts` with `layers` (1 or 2) layers of ghost
/// cells: owned cells in the order of `global`, then ghosts ordered as
/// HaloPattern states; every cell keeps the cell_global_id it has in
/// `global`. With two layers the first ghost layer carries all its faces,
/// so the face-of-face stencil of every owned cell is local. A periodic face
/// becomes an interior face to an image ghost: the cell across it, shifted
/// by the pair's translation (Mesh::ghost_translation), even on one part.
/// `global_geometry` must be compute_geometry(global). Throws E0141 if
/// `global` has ghosts and E0140 if `parts`, `n_parts`, `rank` or `layers`
/// are out of range, `rank` owns no cell, or the periodic translations of
/// `global` span more than three directions or are not linearly independent.
LocalPartition extract_partition(const Mesh& global, const Geometry& global_geometry,
                                 std::span<const std::int32_t> parts, std::int32_t n_parts,
                                 std::int32_t rank, int layers = 2);

/// One level of a graph hierarchy (a multigrid level): n_nodes nodes, edges owner -> neighbour
/// (indices of nodes), and each node's global id.
struct GraphLevel {
  std::size_t n_nodes = 0;
  std::span<const index_t> owner, neighbour;
  std::span<const std::int64_t> global_id;
};

/// The part of each coarse node: that of its child with the least global id, so that every
/// rank, holding the same hierarchy, agrees. `coarse_node` gives each fine node's coarse node
/// (n_coarse of them). Throws E0140 for sizes that do not match or a coarse node without child.
std::vector<std::int32_t> coarse_parts(std::span<const std::int32_t> parts,
                                       std::span<const index_t> coarse_node,
                                       std::span<const std::int64_t> global_id,
                                       std::size_t n_coarse);

/// The fine nodes that `rank` needs from other parts to restrict onto its coarse nodes: the
/// children, owned elsewhere, of the coarse nodes it owns. Ascending node index.
std::vector<index_t> remote_children(std::int32_t rank, std::span<const std::int32_t> parts,
                                     std::span<const index_t> coarse_node,
                                     std::span<const std::int32_t> coarse_parts);

/// One rank's share of a graph level, as LocalPartition is a mesh's.
struct LocalGraph {
  /// Owned nodes in increasing global id, then ghost nodes ordered by owning part, then
  /// global id: the edge neighbours of the owned nodes, the remote children (with coarse
  /// parts given) and the edge neighbours of those.
  std::size_t n_owned = 0;
  std::vector<std::int64_t> node_global_id;
  std::vector<index_t> node_source;  ///< node of the level of each local node
  /// Local edges: those with an end owned or a remote child. An edge belongs to the part of its
  /// owner node; owned edges first in level order, then ghost edges by owning part, then level
  /// order. Local owner < local neighbour; `edge_reversed` 1 where that swaps the level's.
  std::size_t n_owned_edges = 0;
  std::vector<index_t> owner, neighbour;
  std::vector<std::int64_t> edge_global_id;  ///< edge of the level
  std::vector<std::uint8_t> edge_reversed;
  /// HaloExchange patterns for per-node and per-edge data (owned items, then ghosts).
  HaloPattern node_halo, edge_halo;
};

/// The part `rank` of a graph level partitioned by `parts` (each node's part), for exchanges
/// with HaloExchange. With `coarse_node` and `coarse_parts`, it also holds the remote children
/// (remote_children) and their edges. Every rank of a run, holding the same level, builds
/// matching patterns. Throws E0140 for sizes that do not match, a part or node out of range.
LocalGraph extract_graph(const GraphLevel& level, std::span<const std::int32_t> parts,
                         std::int32_t n_parts, std::int32_t rank,
                         std::span<const index_t> coarse_node = {},
                         std::span<const std::int32_t> coarse_parts = {});

/// The part of the forward exchange a rank does alone: copies each owned
/// cell's values into the ghosts that are periodic images of it (the block
/// of `halo` whose neighbour is `halo.rank`), for every component (n_cells
/// values each, host memory). On one part, the whole forward exchange.
template <class T>
void forward_self(const HaloPattern& halo, index_t n_owned_cells, std::span<T* const> components);

/// Fills the ghost volumes and centroids of a one-part partition from its
/// owned cells, centroids moved by ghost_translation, as the exchange does
/// on several. Throws E0140 when `part` is one of several.
void fill_ghost_geometry(LocalPartition& part);

}  // namespace autan
