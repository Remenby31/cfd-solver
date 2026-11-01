#include "autan/core/renumber.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "autan/core/error.hpp"

namespace autan {
namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

// Owned-cell adjacency without duplicate edges (two cells may share faces).
struct Graph {
  std::vector<offset_t> offsets;
  std::vector<index_t> neighbours;

  [[nodiscard]] std::span<const index_t> of(index_t c) const {
    return std::span<const index_t>(neighbours)
        .subspan(static_cast<std::size_t>(offsets[at(c)]),
                 static_cast<std::size_t>(offsets[at(c) + 1] - offsets[at(c)]));
  }
  [[nodiscard]] std::size_t degree(index_t c) const { return of(c).size(); }
};

Graph owned_graph(const Mesh& mesh) {
  const index_t n = mesh.n_owned_cells();
  std::vector<offset_t> count(at(n) + 1, 0);
  for (index_t f = 0; f < mesh.n_interior_faces(); ++f) {
    const index_t b = mesh.neighbour()[at(f)];
    if (b >= n) continue;
    ++count[at(mesh.owner()[at(f)]) + 1];
    ++count[at(b) + 1];
  }
  std::partial_sum(count.begin(), count.end(), count.begin());
  std::vector<index_t> all(static_cast<std::size_t>(count.back()));
  std::vector<offset_t> fill(count.begin(), count.end() - 1);
  for (index_t f = 0; f < mesh.n_interior_faces(); ++f) {
    const index_t a = mesh.owner()[at(f)];
    const index_t b = mesh.neighbour()[at(f)];
    if (b >= n) continue;
    all[static_cast<std::size_t>(fill[at(a)]++)] = b;
    all[static_cast<std::size_t>(fill[at(b)]++)] = a;
  }
  Graph g;
  g.offsets.reserve(at(n) + 1);
  g.offsets.push_back(0);
  g.neighbours.reserve(all.size());
  for (index_t c = 0; c < n; ++c) {
    const auto begin = all.begin() + count[at(c)];
    const auto end = all.begin() + count[at(c) + 1];
    std::sort(begin, end);
    g.neighbours.insert(g.neighbours.end(), begin, std::unique(begin, end));
    g.offsets.push_back(static_cast<offset_t>(g.neighbours.size()));
  }
  return g;
}

// Breadth-first levels from `root` within its component; returns the cells of
// the last level and the number of levels.
std::pair<std::vector<index_t>, int> last_level(const Graph& g, index_t root,
                                                std::vector<int>& level) {
  std::vector<index_t> frontier{root};
  std::vector<index_t> touched{root};
  level[at(root)] = 0;
  int depth = 0;
  std::vector<index_t> next;
  for (;;) {
    next.clear();
    for (const index_t c : frontier) {
      for (const index_t d : g.of(c)) {
        if (level[at(d)] < 0) {
          level[at(d)] = depth + 1;
          next.push_back(d);
          touched.push_back(d);
        }
      }
    }
    if (next.empty()) break;
    frontier.swap(next);
    ++depth;
  }
  for (const index_t c : touched) level[at(c)] = -1;
  return {frontier, depth + 1};
}

index_t pseudo_peripheral(const Graph& g, index_t start, std::vector<int>& level) {
  index_t root = start;
  auto [last, depth] = last_level(g, root, level);
  for (;;) {
    const index_t candidate = *std::ranges::min_element(last, [&](index_t a, index_t b) {
      return g.degree(a) != g.degree(b) ? g.degree(a) < g.degree(b) : a < b;
    });
    auto [candidate_last, candidate_depth] = last_level(g, candidate, level);
    if (candidate_depth <= depth) return root;
    root = candidate;
    last = std::move(candidate_last);
    depth = candidate_depth;
  }
}

}  // namespace

std::vector<index_t> rcm_order(const Mesh& mesh) {
  const index_t n = mesh.n_owned_cells();
  const Graph g = owned_graph(mesh);
  std::vector<int> level(at(n), -1);
  std::vector<char> placed(at(n), 0);
  std::vector<index_t> order;
  order.reserve(at(mesh.n_cells()));
  // Components are started in increasing order of their lowest-degree cell.
  std::vector<index_t> by_degree(at(n));
  std::iota(by_degree.begin(), by_degree.end(), 0);
  std::ranges::stable_sort(by_degree,
                           [&](index_t a, index_t b) { return g.degree(a) < g.degree(b); });
  std::vector<index_t> sorted_neighbours;
  for (const index_t seed : by_degree) {
    if (placed[at(seed)] != 0) continue;
    const index_t root = pseudo_peripheral(g, seed, level);
    std::size_t head = order.size();
    order.push_back(root);
    placed[at(root)] = 1;
    while (head < order.size()) {
      const index_t c = order[head++];
      sorted_neighbours.clear();
      for (const index_t d : g.of(c)) {
        if (placed[at(d)] == 0) sorted_neighbours.push_back(d);
      }
      std::ranges::sort(sorted_neighbours, [&](index_t a, index_t b) {
        return g.degree(a) != g.degree(b) ? g.degree(a) < g.degree(b) : a < b;
      });
      for (const index_t d : sorted_neighbours) {
        placed[at(d)] = 1;
        order.push_back(d);
      }
    }
  }
  std::ranges::reverse(order);
  for (index_t c = n; c < mesh.n_cells(); ++c) order.push_back(c);
  return order;
}

Mesh renumber_cells(const Mesh& mesh, std::span<const index_t> new_to_old, AssemblyMap* map) {
  const auto n_cells = at(mesh.n_cells());
  if (new_to_old.size() != n_cells) {
    throw Error(ErrorCode::mesh_index_out_of_range, "permutation size differs from the cell count");
  }
  std::vector<index_t> old_to_new(n_cells, -1);
  for (std::size_t i = 0; i < n_cells; ++i) {
    const index_t old = new_to_old[i];
    if (old < 0 || at(old) >= n_cells || old_to_new[at(old)] != -1 ||
        ((old < mesh.n_owned_cells()) != (i < at(mesh.n_owned_cells())))) {
      throw Error(ErrorCode::mesh_index_out_of_range,
                  "not a permutation that keeps owned cells before ghosts (position " +
                      std::to_string(i) + ")");
    }
    old_to_new[at(old)] = static_cast<index_t>(i);
  }

  MeshInput in;
  in.x.assign(mesh.x().begin(), mesh.x().end());
  in.y.assign(mesh.y().begin(), mesh.y().end());
  in.z.assign(mesh.z().begin(), mesh.z().end());
  in.face_node_offsets.assign(mesh.face_node_offsets().begin(), mesh.face_node_offsets().end());
  in.face_nodes.assign(mesh.face_nodes().begin(), mesh.face_nodes().end());
  in.face_zone.assign(mesh.face_zone().begin(), mesh.face_zone().end());
  in.face_cell0.resize(at(mesh.n_faces()));
  in.face_cell1.assign(at(mesh.n_faces()), no_cell);
  for (index_t f = 0; f < mesh.n_faces(); ++f) {
    in.face_cell0[at(f)] = old_to_new[at(mesh.owner()[at(f)])];
    if (f < mesh.n_interior_faces()) in.face_cell1[at(f)] = old_to_new[at(mesh.neighbour()[at(f)])];
  }
  // Every cell keeps its global id, so ids stay traceable to the source.
  const auto ids = mesh.cell_global_id();
  in.cell_zone.resize(n_cells);
  in.cell_global_id.resize(n_cells);
  for (std::size_t i = 0; i < n_cells; ++i) {
    in.cell_zone[i] = mesh.cell_zone()[at(new_to_old[i])];
    in.cell_global_id[i] = ids[at(new_to_old[i])];
  }
  in.n_ghost_cells = mesh.n_cells() - mesh.n_owned_cells();
  const auto n_owned = at(mesh.n_owned_cells());
  in.ghost_translation.resize(n_cells - n_owned);
  for (std::size_t i = n_owned; i < n_cells; ++i) {
    in.ghost_translation[i - n_owned] = mesh.ghost_translation()[at(new_to_old[i]) - n_owned];
  }
  in.face_zones.assign(mesh.face_zones().begin(), mesh.face_zones().end());
  in.cell_zones.assign(mesh.cell_zones().begin(), mesh.cell_zones().end());
  in.periodic.assign(mesh.periodic_pairs().begin(), mesh.periodic_pairs().end());
  in.source = mesh.source();
  // Input face f is face f of `mesh`, so the map's face_source names them.
  return assemble(std::move(in), map);
}

Geometry renumber_geometry(const Geometry& geometry, std::span<const index_t> new_to_old,
                           const AssemblyMap& map) {
  const std::size_t n_faces = map.face_source.size();
  if (new_to_old.size() != geometry.cell_volume.size() || n_faces != geometry.face_area.size() ||
      map.face_reversed.size() != n_faces) {
    throw Error(ErrorCode::mesh_index_out_of_range,
                "the renumbering does not match the geometry's cells and faces");
  }
  Geometry out;
  const auto cells = [&](const std::vector<double>& from, std::vector<double>& to) {
    to.resize(new_to_old.size());
    for (std::size_t i = 0; i < to.size(); ++i) to[i] = from[at(new_to_old[i])];
  };
  const auto faces = [&](const std::vector<double>& from, std::vector<double>& to, bool vector) {
    to.resize(n_faces);
    for (std::size_t f = 0; f < n_faces; ++f) {
      const double v = from[at(map.face_source[f])];
      to[f] = vector && map.face_reversed[f] != 0 ? -v : v;
    }
  };
  faces(geometry.face_area_x, out.face_area_x, true);
  faces(geometry.face_area_y, out.face_area_y, true);
  faces(geometry.face_area_z, out.face_area_z, true);
  faces(geometry.face_area, out.face_area, false);
  faces(geometry.face_centroid_x, out.face_centroid_x, false);
  faces(geometry.face_centroid_y, out.face_centroid_y, false);
  faces(geometry.face_centroid_z, out.face_centroid_z, false);
  cells(geometry.cell_volume, out.cell_volume);
  cells(geometry.cell_centroid_x, out.cell_centroid_x);
  cells(geometry.cell_centroid_y, out.cell_centroid_y);
  cells(geometry.cell_centroid_z, out.cell_centroid_z);
  return out;
}

}  // namespace autan
