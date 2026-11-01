#include "autan/core/mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>

#include "autan/core/error.hpp"

namespace autan {
namespace {

std::size_t to_size(index_t value) { return static_cast<std::size_t>(value); }
std::size_t to_size(offset_t value) { return static_cast<std::size_t>(value); }

[[noreturn]] void fail(ErrorCode code, const std::string& message) { throw Error(code, message); }

void check_sizes(const MeshInput& in) {
  const std::size_t n_points = in.x.size();
  if (in.y.size() != n_points || in.z.size() != n_points) {
    fail(ErrorCode::mesh_inconsistent_sizes, "x, y and z have different sizes");
  }
  const std::size_t n_faces = in.face_cell0.size();
  if (in.face_cell1.size() != n_faces || in.face_zone.size() != n_faces ||
      in.face_node_offsets.size() != n_faces + 1) {
    fail(ErrorCode::mesh_inconsistent_sizes,
         "face_cell0, face_cell1, face_zone and face_node_offsets disagree on the face count");
  }
  if (in.cell_zone.empty() || n_faces == 0 || n_points == 0) {
    fail(ErrorCode::mesh_empty, "mesh has no cells, faces or points");
  }
  constexpr auto max_index = static_cast<std::size_t>(std::numeric_limits<index_t>::max());
  if (n_points > max_index || n_faces > max_index || in.cell_zone.size() > max_index ||
      in.face_zones.size() > max_index || in.cell_zones.size() > max_index) {
    fail(ErrorCode::mesh_too_large, "entity count exceeds the 32-bit index range");
  }
  if (in.face_node_offsets.front() != 0 ||
      to_size(in.face_node_offsets.back()) != in.face_nodes.size()) {
    fail(ErrorCode::mesh_inconsistent_sizes,
         "face_node_offsets must start at 0 and end at face_nodes.size()");
  }
  for (std::size_t f = 0; f < n_faces; ++f) {
    if (in.face_node_offsets[f + 1] < in.face_node_offsets[f]) {
      fail(ErrorCode::mesh_inconsistent_sizes,
           "face_node_offsets decreases at face " + std::to_string(f));
    }
  }
  if (!in.cell_global_id.empty() && in.cell_global_id.size() != in.cell_zone.size()) {
    fail(ErrorCode::mesh_inconsistent_sizes,
         "cell_global_id must be empty or have one id per cell");
  }
  if (in.n_ghost_cells < 0 || to_size(in.n_ghost_cells) >= in.cell_zone.size()) {
    fail(ErrorCode::mesh_inconsistent_sizes, "n_ghost_cells must leave at least one owned cell");
  }
  if (!in.ghost_translation.empty() && in.ghost_translation.size() != to_size(in.n_ghost_cells)) {
    fail(ErrorCode::mesh_inconsistent_sizes,
         "ghost_translation must be empty or have one translation per ghost cell");
  }
}

template <typename Zone>
void check_unique_names(const std::vector<Zone>& zones, const char* what) {
  std::unordered_set<std::string> seen;
  for (const Zone& zone : zones) {
    if (!seen.insert(zone.name).second) {
      fail(ErrorCode::mesh_duplicate_zone_name,
           std::string(what) + " name '" + zone.name + "' is used twice");
    }
  }
}

void check_entities(const MeshInput& in) {
  const auto n_points = static_cast<index_t>(in.x.size());
  const auto n_cells = static_cast<index_t>(in.cell_zone.size());
  const auto n_face_zones = static_cast<index_t>(in.face_zones.size());
  const auto n_cell_zones = static_cast<index_t>(in.cell_zones.size());
  const std::size_t n_faces = in.face_cell0.size();
  const index_t n_owned = n_cells - in.n_ghost_cells;

  std::vector<index_t> sorted;
  for (std::size_t f = 0; f < n_faces; ++f) {
    const std::string where = " (face " + std::to_string(f) + ")";
    const auto begin = in.face_nodes.begin() + in.face_node_offsets[f];
    const auto end = in.face_nodes.begin() + in.face_node_offsets[f + 1];
    if (end - begin < 3) {
      fail(ErrorCode::mesh_face_too_few_nodes, "face has fewer than 3 nodes" + where);
    }
    sorted.assign(begin, end);
    std::ranges::sort(sorted);
    if (sorted.front() < 0 || sorted.back() >= n_points) {
      fail(ErrorCode::mesh_index_out_of_range, "node index out of range" + where);
    }
    if (std::ranges::adjacent_find(sorted) != sorted.end()) {
      fail(ErrorCode::mesh_face_repeated_node, "face lists a node twice" + where);
    }
    const index_t c0 = in.face_cell0[f];
    const index_t c1 = in.face_cell1[f];
    if (c0 < 0 || c0 >= n_cells || (c1 != no_cell && (c1 < 0 || c1 >= n_cells))) {
      fail(ErrorCode::mesh_index_out_of_range, "cell index out of range" + where);
    }
    if (c0 == c1) {
      fail(ErrorCode::mesh_face_same_cell_both_sides,
           "face has the same cell on both sides" + where);
    }
    if (in.face_zone[f] < 0 || in.face_zone[f] >= n_face_zones) {
      fail(ErrorCode::mesh_index_out_of_range, "face zone index out of range" + where);
    }
  }
  for (std::size_t c = 0; c < in.cell_zone.size(); ++c) {
    if (in.cell_zone[c] < 0 || in.cell_zone[c] >= n_cell_zones) {
      fail(ErrorCode::mesh_index_out_of_range,
           "cell zone index out of range (cell " + std::to_string(c) + ")");
    }
  }

  check_unique_names(in.face_zones, "face zone");
  check_unique_names(in.cell_zones, "cell zone");

  for (std::size_t f = 0; f < n_faces; ++f) {
    const FaceZone& zone = in.face_zones[to_size(in.face_zone[f])];
    if (zone.boundary != (in.face_cell1[f] == no_cell)) {
      fail(ErrorCode::mesh_face_zone_role,
           std::string(zone.boundary ? "interior" : "boundary") + " face " + std::to_string(f) +
               " in " + (zone.boundary ? "boundary" : "interior") + " zone '" + zone.name + "'");
    }
  }

  for (const auto& t : in.ghost_translation) {
    if (!std::isfinite(t[0]) || !std::isfinite(t[1]) || !std::isfinite(t[2])) {
      fail(ErrorCode::mesh_inconsistent_sizes, "a ghost translation is not finite");
    }
  }
  // A cell is its global id and its translation: a periodic image repeats
  // the id of the cell it copies, elsewhere.
  if (!in.cell_global_id.empty()) {
    using Identity = std::pair<std::int64_t, std::array<double, 3>>;
    std::vector<Identity> ids(in.cell_global_id.size());
    for (std::size_t c = 0; c < ids.size(); ++c) {
      ids[c].first = in.cell_global_id[c];
      if (c >= to_size(n_owned) && !in.ghost_translation.empty())
        ids[c].second = in.ghost_translation[c - to_size(n_owned)];
    }
    std::ranges::sort(ids);
    if (ids.front().first < 0 || std::ranges::adjacent_find(ids) != ids.end()) {
      fail(ErrorCode::mesh_duplicate_global_id,
           "cells must differ by global id or translation, and ids must be >= 0");
    }
  }
}

// Every edge of a closed, consistently oriented cell surface is traversed
// exactly twice, once in each direction, when each face is walked with its
// normal pointing out of the cell.
void check_cells_closed(const Mesh& mesh) {
  struct Edge {
    std::uint64_t key;
    bool forward;
  };
  std::vector<Edge> edges;
  for (index_t c = 0; c < mesh.n_owned_cells(); ++c) {
    edges.clear();
    for (const index_t f : mesh.cell_faces(c)) {
      const auto nodes = mesh.face_nodes(f);
      const bool outward = mesh.owner()[to_size(f)] == c;
      for (std::size_t i = 0; i < nodes.size(); ++i) {
        index_t a = nodes[i];
        index_t b = nodes[(i + 1) % nodes.size()];
        if (!outward) std::swap(a, b);
        const auto lo = static_cast<std::uint64_t>(std::min(a, b));
        const auto hi = static_cast<std::uint64_t>(std::max(a, b));
        edges.push_back({(lo << 32U) | hi, a < b});
      }
    }
    std::ranges::sort(edges, [](const Edge& l, const Edge& r) {
      return l.key != r.key ? l.key < r.key : l.forward < r.forward;
    });
    bool closed = edges.size() % 2 == 0;
    for (std::size_t i = 0; closed && i < edges.size(); i += 2) {
      closed = edges[i].key == edges[i + 1].key && !edges[i].forward && edges[i + 1].forward &&
               (i + 2 == edges.size() || edges[i + 2].key != edges[i].key);
    }
    if (!closed) {
      fail(ErrorCode::mesh_cell_not_closed,
           "cell " + std::to_string(c) +
               " is not a closed, consistently oriented surface: an edge is not shared by exactly "
               "two of its faces with opposite directions");
    }
  }
}

}  // namespace

std::span<const index_t> Mesh::face_nodes(index_t face) const noexcept {
  const auto f = to_size(face);
  return std::span<const index_t>(face_nodes_)
      .subspan(to_size(face_node_offsets_[f]),
               to_size(face_node_offsets_[f + 1] - face_node_offsets_[f]));
}

std::span<const index_t> Mesh::cell_faces(index_t cell) const noexcept {
  const auto c = to_size(cell);
  return std::span<const index_t>(cell_faces_)
      .subspan(to_size(cell_face_offsets_[c]),
               to_size(cell_face_offsets_[c + 1] - cell_face_offsets_[c]));
}

MeshView Mesh::view() const noexcept {
  return MeshView{
      .n_points = n_points(),
      .n_cells = n_cells(),
      .n_owned_cells = n_owned_cells(),
      .n_faces = n_faces(),
      .n_interior_faces = n_interior_faces(),
      .n_owned_interior_faces = n_owned_interior_faces_,
      .n_owned_faces = n_owned_faces_,
      .n_face_zones = static_cast<index_t>(face_zones_.size()),
      .n_cell_zones = static_cast<index_t>(cell_zones_.size()),
      .x = x_.data(),
      .y = y_.data(),
      .z = z_.data(),
      .face_node_offsets = face_node_offsets_.data(),
      .face_nodes = face_nodes_.data(),
      .owner = owner_.data(),
      .neighbour = neighbour_.data(),
      .face_zone = face_zone_.data(),
      .cell_zone = cell_zone_.data(),
      .cell_global_id = cell_global_id_.data(),
      .boundary_zone_offsets = boundary_zone_offsets_.data(),
      .halo_boundary_zone_offsets = halo_boundary_zone_offsets_.data(),
      .cell_face_offsets = cell_face_offsets_.data(),
      .cell_faces = cell_faces_.data(),
  };
}

Mesh assemble(MeshInput in, AssemblyMap* map) {
  check_sizes(in);
  check_entities(in);
  if (map != nullptr) *map = AssemblyMap{};

  const std::size_t n_faces = in.face_cell0.size();
  const std::size_t n_cells = in.cell_zone.size();

  // Orient interior faces from the lower to the higher cell index.
  std::vector<std::uint8_t> reversed(n_faces, 0);
  for (std::size_t f = 0; f < n_faces; ++f) {
    if (in.face_cell1[f] != no_cell && in.face_cell1[f] < in.face_cell0[f]) {
      reversed[f] = 1;
      std::swap(in.face_cell0[f], in.face_cell1[f]);
      std::reverse(in.face_nodes.begin() + in.face_node_offsets[f] + 1,
                   in.face_nodes.begin() + in.face_node_offsets[f + 1]);
    }
  }

  std::vector<index_t> order(n_faces);
  std::iota(order.begin(), order.end(), 0);
  const index_t first_ghost = static_cast<index_t>(n_cells) - in.n_ghost_cells;
  std::ranges::stable_sort(order, [&in, first_ghost](index_t a, index_t b) {
    const auto ia = to_size(a);
    const auto ib = to_size(b);
    const bool boundary_a = in.face_cell1[ia] == no_cell;
    const bool boundary_b = in.face_cell1[ib] == no_cell;
    if (boundary_a != boundary_b) return boundary_b;
    if (boundary_a) {
      // Owned cells' boundary faces first, then the ghosts' (halo faces).
      const bool halo_a = in.face_cell0[ia] >= first_ghost;
      const bool halo_b = in.face_cell0[ib] >= first_ghost;
      if (halo_a != halo_b) return halo_b;
      if (in.face_zone[ia] != in.face_zone[ib]) return in.face_zone[ia] < in.face_zone[ib];
      return in.face_cell0[ia] < in.face_cell0[ib];
    }
    if (in.face_cell0[ia] != in.face_cell0[ib]) return in.face_cell0[ia] < in.face_cell0[ib];
    return in.face_cell1[ia] < in.face_cell1[ib];
  });

  std::vector<index_t> point_map(in.x.size(), -1);
  for (const index_t node : in.face_nodes) point_map[to_size(node)] = 0;
  Mesh mesh;
  index_t n_used = 0;
  for (std::size_t p = 0; p < point_map.size(); ++p) {
    if (point_map[p] == 0) {
      point_map[p] = n_used++;
      if (map != nullptr) map->point_source.push_back(static_cast<index_t>(p));
      mesh.x_.push_back(in.x[p]);
      mesh.y_.push_back(in.y[p]);
      mesh.z_.push_back(in.z[p]);
    }
  }

  mesh.face_node_offsets_.reserve(n_faces + 1);
  mesh.face_node_offsets_.push_back(0);
  mesh.face_nodes_.reserve(in.face_nodes.size());
  mesh.owner_.reserve(n_faces);
  mesh.face_zone_.reserve(n_faces);
  for (const index_t face : order) {
    const auto f = to_size(face);
    for (offset_t k = in.face_node_offsets[f]; k < in.face_node_offsets[f + 1]; ++k) {
      mesh.face_nodes_.push_back(point_map[to_size(in.face_nodes[to_size(k)])]);
    }
    mesh.face_node_offsets_.push_back(static_cast<offset_t>(mesh.face_nodes_.size()));
    mesh.owner_.push_back(in.face_cell0[f]);
    mesh.face_zone_.push_back(in.face_zone[f]);
    if (in.face_cell1[f] != no_cell) mesh.neighbour_.push_back(in.face_cell1[f]);
  }

  const std::size_t n_zones = in.face_zones.size();
  const std::size_t n_interior = mesh.neighbour_.size();
  std::size_t n_owned_interior = 0;
  while (n_owned_interior < n_interior && mesh.owner_[n_owned_interior] < first_ghost)
    ++n_owned_interior;
  std::size_t n_owned_faces = n_interior;
  while (n_owned_faces < n_faces && mesh.owner_[n_owned_faces] < first_ghost) ++n_owned_faces;
  mesh.n_owned_interior_faces_ = static_cast<index_t>(n_owned_interior);
  mesh.n_owned_faces_ = static_cast<index_t>(n_owned_faces);
  const auto zone_offsets = [&](std::size_t begin, std::size_t end) {
    std::vector<index_t> off(n_zones + 1, 0);
    for (std::size_t f = begin; f < end; ++f) ++off[to_size(mesh.face_zone_[f]) + 1];
    off[0] = static_cast<index_t>(begin);
    std::partial_sum(off.begin(), off.end(), off.begin());
    return off;
  };
  mesh.boundary_zone_offsets_ = zone_offsets(n_interior, n_owned_faces);
  mesh.halo_boundary_zone_offsets_ = zone_offsets(n_owned_faces, n_faces);
  mesh.ghost_translation_ = std::move(in.ghost_translation);
  if (mesh.ghost_translation_.empty()) mesh.ghost_translation_.resize(to_size(in.n_ghost_cells));

  mesh.cell_face_offsets_.assign(n_cells + 1, 0);
  for (std::size_t f = 0; f < n_faces; ++f) {
    ++mesh.cell_face_offsets_[to_size(mesh.owner_[f]) + 1];
    if (f < mesh.neighbour_.size()) ++mesh.cell_face_offsets_[to_size(mesh.neighbour_[f]) + 1];
  }
  const auto n_owned = n_cells - to_size(in.n_ghost_cells);
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (mesh.cell_face_offsets_[c + 1] < (c < n_owned ? 4 : 1)) {
      fail(ErrorCode::mesh_cell_too_few_faces,
           "cell " + std::to_string(c) + " has too few faces (4 for an owned cell, 1 for a ghost)");
    }
  }
  std::partial_sum(mesh.cell_face_offsets_.begin(), mesh.cell_face_offsets_.end(),
                   mesh.cell_face_offsets_.begin());
  mesh.cell_faces_.resize(to_size(mesh.cell_face_offsets_.back()));
  std::vector<offset_t> fill(mesh.cell_face_offsets_.begin(), mesh.cell_face_offsets_.end() - 1);
  for (std::size_t f = 0; f < n_faces; ++f) {
    const auto face = static_cast<index_t>(f);
    mesh.cell_faces_[to_size(fill[to_size(mesh.owner_[f])]++)] = face;
    if (f < mesh.neighbour_.size()) {
      mesh.cell_faces_[to_size(fill[to_size(mesh.neighbour_[f])]++)] = face;
    }
  }

  mesh.n_owned_cells_ = static_cast<index_t>(n_owned);
  if (in.cell_global_id.empty()) {
    mesh.cell_global_id_.resize(n_cells);
    std::iota(mesh.cell_global_id_.begin(), mesh.cell_global_id_.end(), std::int64_t{0});
  } else {
    mesh.cell_global_id_ = std::move(in.cell_global_id);
  }
  if (map != nullptr) {
    map->face_source = order;
    map->face_reversed.resize(n_faces);
    for (std::size_t f = 0; f < n_faces; ++f) map->face_reversed[f] = reversed[to_size(order[f])];
  }
  mesh.cell_zone_ = std::move(in.cell_zone);
  mesh.face_zones_ = std::move(in.face_zones);
  mesh.cell_zones_ = std::move(in.cell_zones);
  mesh.source_ = std::move(in.source);

  check_cells_closed(mesh);
  mesh.periodic_partners_ = match_periodic_faces(mesh, in.periodic);
  mesh.periodic_pairs_ = std::move(in.periodic);
  return mesh;
}

Mesh scaled(const Mesh& mesh, double scale) {
  if (!(scale > 0.0) || !std::isfinite(scale)) {
    throw Error(ErrorCode::mesh_index_out_of_range, "a mesh scale must be finite and positive");
  }
  const auto times = [scale](std::span<const double> v) {
    std::vector<double> out(v.begin(), v.end());
    for (double& x : out) x *= scale;
    return out;
  };
  MeshInput in;
  in.x = times(mesh.x());
  in.y = times(mesh.y());
  in.z = times(mesh.z());
  in.face_node_offsets.assign(mesh.face_node_offsets().begin(), mesh.face_node_offsets().end());
  in.face_nodes.assign(mesh.face_nodes().begin(), mesh.face_nodes().end());
  in.face_zone.assign(mesh.face_zone().begin(), mesh.face_zone().end());
  in.face_cell0.assign(mesh.owner().begin(), mesh.owner().end());
  in.face_cell1.assign(static_cast<std::size_t>(mesh.n_faces()), no_cell);
  std::copy(mesh.neighbour().begin(), mesh.neighbour().end(), in.face_cell1.begin());
  in.cell_zone.assign(mesh.cell_zone().begin(), mesh.cell_zone().end());
  in.cell_global_id.assign(mesh.cell_global_id().begin(), mesh.cell_global_id().end());
  in.n_ghost_cells = mesh.n_cells() - mesh.n_owned_cells();
  for (auto t : mesh.ghost_translation()) {
    for (double& c : t) c *= scale;
    in.ghost_translation.push_back(t);
  }
  in.face_zones.assign(mesh.face_zones().begin(), mesh.face_zones().end());
  in.cell_zones.assign(mesh.cell_zones().begin(), mesh.cell_zones().end());
  in.periodic.assign(mesh.periodic_pairs().begin(), mesh.periodic_pairs().end());
  for (auto& pair : in.periodic) {
    for (double& c : pair.translation) c *= scale;
  }
  in.source = mesh.source();
  return assemble(std::move(in));
}

}  // namespace autan
