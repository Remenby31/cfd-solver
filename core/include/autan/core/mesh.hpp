#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace autan {

/// Index of a point, face, cell or zone inside one mesh (one partition).
using index_t = std::int32_t;
/// Offset into a CSR array; 64-bit because face->node lists of large
/// polyhedral partitions exceed 2^31 entries.
using offset_t = std::int64_t;

inline constexpr index_t no_cell = -1;

/// Boundary-condition family of a face zone, normalised from the type the
/// source file declares with the table of docs/contracts/mesh-info.md.
/// `unspecified` when the format carries no type or a type without a fixed
/// meaning (OpenFOAM `patch`, Fluent `interior`, every Gmsh group).
enum class BcKind : std::uint8_t {
  unspecified,
  wall,          ///< no thermal condition: adiabatic unless the case says otherwise
  thermal_wall,  ///< the source declares a thermal condition whose value the case must give
  slip,
  symmetry,
  far_field,
  inlet,
  outlet,
  periodic,  ///< one side of a periodic pair (Mesh::periodic_pairs()); takes no condition
};

enum class CellKind : std::uint8_t { unspecified, fluid, solid };

/// Where a zone comes from. `source_id` is the identifier in the source file
/// (Fluent zone id, Gmsh physical tag, OpenFOAM patch or zone index), or -1
/// for a zone synthesised by the reader; `detail` says which.
struct Provenance {
  std::int64_t source_id = -1;
  std::string detail;
};

struct FaceZone {
  std::string name;
  std::string bc_type;  ///< type exactly as written in the source; empty if none
  BcKind kind = BcKind::unspecified;
  bool boundary = true;  ///< role: holds boundary faces (true) or interior faces (false)
  Provenance provenance;
};

struct CellZone {
  std::string name;
  std::string type;  ///< type exactly as written in the source; empty if none
  CellKind kind = CellKind::unspecified;
  Provenance provenance;
};

struct MeshSource {
  std::string
      format;  ///< "gmsh", "fluent", "polymesh", "cgns", or "memory" for meshes built in code
  std::string path;
};

/// Two boundary zones of kind `periodic` whose faces coincide under a
/// translation: x_b = x_a + translation for every node of zone a's faces.
struct PeriodicPair {
  index_t zone_a = 0;
  index_t zone_b = 0;
  std::array<double, 3> translation{};
};

/// Face-based mesh as a reader or partitioner produces it: faces in any order,
/// each oriented so that its right-hand-rule normal points away from
/// `face_cell0` and into `face_cell1` (`no_cell` on the boundary). Every array
/// is indexed by the entity it describes; `cell_zone` defines the number of
/// cells. The last `n_ghost_cells` cells are halo copies of cells owned by
/// a partition, translated by `ghost_translation` (empty: none) when they are
/// periodic images; `cell_global_id` is empty (identity) or gives the
/// global index of every cell.
struct MeshInput {
  std::vector<double> x, y, z;
  std::vector<offset_t> face_node_offsets;  ///< size n_faces + 1, starts at 0
  std::vector<index_t> face_nodes;
  std::vector<index_t> face_cell0;
  std::vector<index_t> face_cell1;
  std::vector<index_t> face_zone;
  std::vector<index_t> cell_zone;
  std::vector<std::int64_t> cell_global_id;
  index_t n_ghost_cells = 0;
  std::vector<std::array<double, 3>> ghost_translation;
  std::vector<FaceZone> face_zones;
  std::vector<CellZone> cell_zones;
  std::vector<PeriodicPair> periodic;
  MeshSource source;
};

/// Where each entity of an assembled mesh comes from in its MeshInput, for
/// callers that hold data attached to the input numbering.
struct AssemblyMap {
  std::vector<index_t> face_source;         ///< input face of each mesh face
  std::vector<std::uint8_t> face_reversed;  ///< 1 if the face was flipped (owner <-> neighbour)
  std::vector<index_t> point_source;        ///< input point of each mesh point
};

/// Non-owning view of a mesh's arrays for kernels. Plain pointers so that the
/// same struct can describe host or device copies. Layout and invariants:
/// docs/contracts/mesh.md.
struct MeshView {
  index_t n_points;
  index_t n_cells;
  index_t n_owned_cells;
  index_t n_faces;
  index_t n_interior_faces;
  index_t n_owned_interior_faces;
  index_t n_owned_faces;
  index_t n_face_zones;
  index_t n_cell_zones;
  const double* x;
  const double* y;
  const double* z;
  const offset_t* face_node_offsets;
  const index_t* face_nodes;
  const index_t* owner;
  const index_t* neighbour;
  const index_t* face_zone;
  const index_t* cell_zone;
  const std::int64_t* cell_global_id;
  const index_t* boundary_zone_offsets;
  const index_t* halo_boundary_zone_offsets;
  const offset_t* cell_face_offsets;
  const index_t* cell_faces;
};

/// Immutable face-based polyhedral mesh in canonical order (see
/// docs/contracts/mesh.md). Built only by assemble(), which validates it.
class Mesh {
 public:
  [[nodiscard]] index_t n_points() const noexcept { return static_cast<index_t>(x_.size()); }
  [[nodiscard]] index_t n_cells() const noexcept { return static_cast<index_t>(cell_zone_.size()); }
  [[nodiscard]] index_t n_owned_cells() const noexcept { return n_owned_cells_; }
  [[nodiscard]] index_t n_faces() const noexcept { return static_cast<index_t>(owner_.size()); }
  [[nodiscard]] index_t n_interior_faces() const noexcept {
    return static_cast<index_t>(neighbour_.size());
  }
  [[nodiscard]] index_t n_boundary_faces() const noexcept { return n_faces() - n_interior_faces(); }

  [[nodiscard]] std::span<const double> x() const noexcept { return x_; }
  [[nodiscard]] std::span<const double> y() const noexcept { return y_; }
  [[nodiscard]] std::span<const double> z() const noexcept { return z_; }
  [[nodiscard]] std::span<const offset_t> face_node_offsets() const noexcept {
    return face_node_offsets_;
  }
  [[nodiscard]] std::span<const index_t> face_nodes() const noexcept { return face_nodes_; }
  [[nodiscard]] std::span<const index_t> face_nodes(index_t face) const noexcept;
  [[nodiscard]] std::span<const index_t> owner() const noexcept { return owner_; }
  [[nodiscard]] std::span<const index_t> neighbour() const noexcept { return neighbour_; }
  [[nodiscard]] std::span<const index_t> face_zone() const noexcept { return face_zone_; }
  [[nodiscard]] std::span<const index_t> cell_zone() const noexcept { return cell_zone_; }
  [[nodiscard]] std::span<const std::int64_t> cell_global_id() const noexcept {
    return cell_global_id_;
  }
  /// Interior faces [0, n_owned_interior_faces) touch an owned cell; the
  /// others join two ghost cells.
  [[nodiscard]] index_t n_owned_interior_faces() const noexcept { return n_owned_interior_faces_; }
  /// Boundary faces [n_interior_faces, n_owned_faces) belong to owned cells;
  /// [n_owned_faces, n_faces) to ghost cells.
  [[nodiscard]] index_t n_owned_faces() const noexcept { return n_owned_faces_; }
  /// Owned boundary faces of face zone z: [off[z], off[z+1]), from
  /// n_interior_faces to n_owned_faces.
  [[nodiscard]] std::span<const index_t> boundary_zone_offsets() const noexcept {
    return boundary_zone_offsets_;
  }
  /// Ghost boundary faces of face zone z, from n_owned_faces to n_faces.
  [[nodiscard]] std::span<const index_t> halo_boundary_zone_offsets() const noexcept {
    return halo_boundary_zone_offsets_;
  }
  /// Translation of each ghost cell (index c - n_owned_cells): zero, or the
  /// shift of a periodic image.
  [[nodiscard]] std::span<const std::array<double, 3>> ghost_translation() const noexcept {
    return ghost_translation_;
  }
  [[nodiscard]] std::span<const offset_t> cell_face_offsets() const noexcept {
    return cell_face_offsets_;
  }
  [[nodiscard]] std::span<const index_t> cell_faces() const noexcept { return cell_faces_; }
  [[nodiscard]] std::span<const index_t> cell_faces(index_t cell) const noexcept;
  [[nodiscard]] std::span<const FaceZone> face_zones() const noexcept { return face_zones_; }
  [[nodiscard]] std::span<const CellZone> cell_zones() const noexcept { return cell_zones_; }
  [[nodiscard]] const MeshSource& source() const noexcept { return source_; }
  [[nodiscard]] std::span<const PeriodicPair> periodic_pairs() const noexcept {
    return periodic_pairs_;
  }
  /// Faces of pair `pair`'s zone b matched to the faces of its zone a, in
  /// the order of zone a's faces.
  [[nodiscard]] std::span<const index_t> periodic_partners(std::size_t pair) const noexcept {
    return periodic_partners_[pair];
  }

  [[nodiscard]] MeshView view() const noexcept;

 private:
  friend Mesh assemble(MeshInput input, AssemblyMap* map);

  std::vector<double> x_, y_, z_;
  std::vector<offset_t> face_node_offsets_;
  std::vector<index_t> face_nodes_;
  std::vector<index_t> owner_;
  std::vector<index_t> neighbour_;
  std::vector<index_t> face_zone_;
  std::vector<index_t> cell_zone_;
  std::vector<std::int64_t> cell_global_id_;
  index_t n_owned_cells_ = 0;
  std::vector<index_t> boundary_zone_offsets_;
  std::vector<index_t> halo_boundary_zone_offsets_;
  std::vector<std::array<double, 3>> ghost_translation_;
  index_t n_owned_interior_faces_ = 0;
  index_t n_owned_faces_ = 0;
  std::vector<offset_t> cell_face_offsets_;
  std::vector<index_t> cell_faces_;
  std::vector<FaceZone> face_zones_;
  std::vector<CellZone> cell_zones_;
  std::vector<PeriodicPair> periodic_pairs_;
  std::vector<std::vector<index_t>> periodic_partners_;
  MeshSource source_;
};

/// Matches the faces of each periodic pair of `mesh` (zone a's faces onto
/// zone b's) and returns, per pair, the face of zone b for each face of zone
/// a. Throws E0118 for a malformed declaration and E0119 for faces that do
/// not match one to one. Called by assemble().
std::vector<std::vector<index_t>> match_periodic_faces(const Mesh& mesh,
                                                       std::span<const PeriodicPair> pairs);

/// Validates `input` and returns it in canonical order: owner < neighbour on
/// interior faces, interior faces first sorted by (owner, neighbour), then
/// boundary faces grouped by zone index and sorted by owner; points that no
/// face references are removed. Cell order is kept. Fills `map` when given.
/// Throws autan::Error with an E01xx code on the first invariant violation.
Mesh assemble(MeshInput input, AssemblyMap* map = nullptr);

/// `mesh` with every point coordinate, periodic translation and ghost
/// translation multiplied by `scale` (a mesh in other length units brought to
/// metres), its faces, cells and zones unchanged. Throws E0102 for a scale
/// that is not finite and positive.
Mesh scaled(const Mesh& mesh, double scale);

}  // namespace autan
