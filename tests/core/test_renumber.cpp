#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/renumber.hpp"
#include "fixtures.hpp"

using autan::index_t;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

index_t bandwidth(const autan::Mesh& mesh) {
  index_t b = 0;
  for (index_t f = 0; f < mesh.n_interior_faces(); ++f) {
    b = std::max(b, mesh.neighbour()[at(f)] - mesh.owner()[at(f)]);
  }
  return b;
}

std::vector<index_t> shuffled(index_t n, unsigned seed) {
  std::vector<index_t> p(at(n));
  std::iota(p.begin(), p.end(), 0);
  std::mt19937 rng(seed);
  for (std::size_t i = p.size(); i > 1; --i) std::swap(p[i - 1], p[rng() % i]);
  return p;
}

}  // namespace

TEST_CASE("RCM orders a chain of cells with bandwidth 1", "[renumber]") {
  // An n x 1 x 1 row of hexahedra is a path graph: its optimal bandwidth is 1
  // and Cuthill-McKee from an end (a pseudo-peripheral cell) reaches it.
  autan::CellwiseMesh cw;
  for (int i = 0; i <= 12; ++i)
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 2; ++j) {
        cw.x.push_back(i);
        cw.y.push_back(j);
        cw.z.push_back(k);
      }
  auto p = [](int i, int j, int k) { return static_cast<index_t>(4 * i + 2 * k + j); };
  for (int i = 0; i < 12; ++i) {
    cw.add_cell(autan::CellShape::hexa,
                std::array<index_t, 8>{p(i, 0, 0), p(i + 1, 0, 0), p(i + 1, 1, 0), p(i, 1, 0),
                                       p(i, 0, 1), p(i + 1, 0, 1), p(i + 1, 1, 1), p(i, 1, 1)},
                0);
  }
  cw.cell_zones.push_back(autan::CellZone{
      .name = "fluid", .type = "", .kind = autan::CellKind::unspecified, .provenance = {}});
  const autan::Mesh chain = autan::assemble(autan::to_face_based(cw));
  const autan::Mesh mixed = autan::renumber_cells(chain, shuffled(chain.n_cells(), 3));
  REQUIRE(bandwidth(mixed) > 1);
  const autan::Mesh ordered = autan::renumber_cells(mixed, autan::rcm_order(mixed));
  CHECK(bandwidth(ordered) == 1);
}

TEST_CASE("Renumbering permutes cells and keeps the mesh", "[renumber]") {
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(4, autan::test::BoxFill::hybrid, 0.3)));
  const std::vector<index_t> order = autan::rcm_order(mesh);
  std::vector<index_t> sorted(order);
  std::sort(sorted.begin(), sorted.end());
  std::vector<index_t> identity(sorted.size());
  std::iota(identity.begin(), identity.end(), 0);
  REQUIRE(sorted == identity);

  const autan::Mesh shuffled_mesh = autan::renumber_cells(mesh, shuffled(mesh.n_cells(), 11));
  const autan::Mesh renumbered =
      autan::renumber_cells(shuffled_mesh, autan::rcm_order(shuffled_mesh));
  CHECK(bandwidth(renumbered) < bandwidth(shuffled_mesh));
  CHECK(autan::rcm_order(shuffled_mesh) == autan::rcm_order(shuffled_mesh));

  autan::AssemblyMap map;
  const autan::Mesh again = autan::renumber_cells(mesh, order, &map);
  const autan::Geometry g0 = autan::compute_geometry(mesh);
  const autan::Geometry g1 = autan::compute_geometry(again);
  REQUIRE(again.n_faces() == mesh.n_faces());
  for (std::size_t i = 0; i < order.size(); ++i) {
    CHECK_THAT(g1.cell_volume[i], WithinRel(g0.cell_volume[at(order[i])], 1e-13));
    CHECK(again.cell_zone()[i] == mesh.cell_zone()[at(order[i])]);
    CHECK(again.cell_global_id()[i] == mesh.cell_global_id()[at(order[i])]);
  }
  for (std::size_t z = 0; z + 1 < mesh.boundary_zone_offsets().size(); ++z) {
    CHECK(again.boundary_zone_offsets()[z + 1] - again.boundary_zone_offsets()[z] ==
          mesh.boundary_zone_offsets()[z + 1] - mesh.boundary_zone_offsets()[z]);
  }  // The map names each face's source: the same nodes, zone and cells (by
  // global id), each source once.
  REQUIRE(map.face_source.size() == at(again.n_faces()));
  std::vector<int> seen(at(mesh.n_faces()), 0);
  const auto nodes = [](const autan::Mesh& m, index_t f) {
    const auto n = m.face_nodes(f);
    std::vector<index_t> v(n.begin(), n.end());
    std::sort(v.begin(), v.end());
    return v;
  };
  const auto cells = [](const autan::Mesh& m, index_t f) {
    std::vector<std::int64_t> c{m.cell_global_id()[at(m.owner()[at(f)])]};
    if (f < m.n_interior_faces()) c.push_back(m.cell_global_id()[at(m.neighbour()[at(f)])]);
    std::sort(c.begin(), c.end());
    return c;
  };
  for (index_t f = 0; f < again.n_faces(); ++f) {
    const index_t source = map.face_source[at(f)];
    ++seen[at(source)];
    CHECK(nodes(again, f) == nodes(mesh, source));
    CHECK(again.face_zone()[at(f)] == mesh.face_zone()[at(source)]);
    CHECK(cells(again, f) == cells(mesh, source));
  }
  CHECK(std::all_of(seen.begin(), seen.end(), [](int k) { return k == 1; }));

  // The geometry carried over: the source's bits, and to round-off what
  // compute_geometry gives on the renumbered mesh, area vectors oriented
  // from its owners as there.
  const autan::Geometry carried = autan::renumber_geometry(g0, order, map);
  for (std::size_t i = 0; i < order.size(); ++i) {
    CHECK(carried.cell_volume[i] == g0.cell_volume[at(order[i])]);
    CHECK_THAT(carried.cell_centroid_y[i], WithinAbs(g1.cell_centroid_y[i], 1e-14));
  }
  for (std::size_t f = 0; f < at(again.n_faces()); ++f) {
    CHECK(carried.face_area[f] == g0.face_area[at(map.face_source[f])]);
    for (const auto& [a, b] : {std::pair{&carried.face_area_x, &g1.face_area_x},
                               std::pair{&carried.face_area_y, &g1.face_area_y},
                               std::pair{&carried.face_area_z, &g1.face_area_z}}) {
      CHECK_THAT((*a)[f], WithinAbs((*b)[f], 1e-14 * g1.face_area[f]));
    }
  }
  CHECK(std::ranges::any_of(map.face_reversed, [](std::uint8_t r) { return r != 0; }));
  try {
    (void)autan::renumber_geometry(g0, std::span(order).first(order.size() - 1), map);
    FAIL("expected E0102");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::mesh_index_out_of_range);
  }
}

TEST_CASE("Renumbering rejects an invalid permutation", "[renumber]") {
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(1, autan::test::BoxFill::tetra)));
  std::vector<index_t> bad(at(mesh.n_cells()), 0);
  try {
    (void)autan::renumber_cells(mesh, bad);
    FAIL("expected E0102");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::mesh_index_out_of_range);
  }
}

TEST_CASE("Renumbering a partition keeps the global id of every cell", "[renumber]") {
  // Three cubes in a row: cells 0 and 1 owned, cell 2 a ghost carrying only
  // the face it shares with cell 1; global ids 0, 1, 2 look like an identity.
  autan::CellwiseMesh cw;
  for (int i = 0; i <= 3; ++i)
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 2; ++j) {
        cw.x.push_back(i);
        cw.y.push_back(j);
        cw.z.push_back(k);
      }
  auto p = [](int i, int j, int k) { return static_cast<index_t>(4 * i + 2 * k + j); };
  for (int i = 0; i < 3; ++i) {
    cw.add_cell(autan::CellShape::hexa,
                std::array<index_t, 8>{p(i, 0, 0), p(i + 1, 0, 0), p(i + 1, 1, 0), p(i, 1, 0),
                                       p(i, 0, 1), p(i + 1, 0, 1), p(i + 1, 1, 1), p(i, 1, 1)},
                0);
  }
  cw.cell_zones.push_back(autan::CellZone{
      .name = "fluid", .type = "", .kind = autan::CellKind::unspecified, .provenance = {}});
  const autan::MeshInput in = autan::to_face_based(cw);
  autan::MeshInput part = in;
  part.face_node_offsets = {0};
  part.face_nodes.clear();
  part.face_cell0.clear();
  part.face_cell1.clear();
  part.face_zone.clear();
  for (std::size_t f = 0; f + 1 < in.face_node_offsets.size(); ++f) {
    if (in.face_cell0[f] == 2 && in.face_cell1[f] == autan::no_cell) continue;
    part.face_nodes.insert(part.face_nodes.end(), in.face_nodes.begin() + in.face_node_offsets[f],
                           in.face_nodes.begin() + in.face_node_offsets[f + 1]);
    part.face_node_offsets.push_back(static_cast<autan::offset_t>(part.face_nodes.size()));
    part.face_cell0.push_back(in.face_cell0[f]);
    part.face_cell1.push_back(in.face_cell1[f]);
    part.face_zone.push_back(in.face_zone[f]);
  }
  part.n_ghost_cells = 1;
  part.cell_global_id = {0, 1, 2};
  const autan::Mesh mesh = autan::assemble(part);
  const std::array<index_t, 3> swap{1, 0, 2};
  const autan::Mesh renumbered = autan::renumber_cells(mesh, swap);
  CHECK(renumbered.cell_global_id()[0] == 1);
  CHECK(renumbered.cell_global_id()[1] == 0);
  CHECK(renumbered.cell_global_id()[2] == 2);
}
