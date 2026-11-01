#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/core/renumber.hpp"
#include "fixtures.hpp"

using Catch::Matchers::WithinRel;

using autan::index_t;
using autan::offset_t;

namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

void require_error(autan::ErrorCode expected, const std::function<void()>& action) {
  try {
    action();
    FAIL("expected " << autan::error_code_string(expected));
  } catch (const autan::Error& e) {
    CHECK(e.code() == expected);
  }
}

// Two unit cubes side by side along x, written directly in face-based form.
autan::MeshInput two_cubes() {
  autan::MeshInput in;
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 3; ++i) {
        in.x.push_back(i);
        in.y.push_back(j);
        in.z.push_back(k);
      }
  auto p = [](int i, int j, int k) { return static_cast<index_t>(i + 3 * (j + 2 * k)); };
  auto face = [&](std::array<index_t, 4> nodes, index_t c0, index_t c1, index_t zone) {
    in.face_nodes.insert(in.face_nodes.end(), nodes.begin(), nodes.end());
    in.face_node_offsets.push_back(static_cast<offset_t>(in.face_nodes.size()));
    in.face_cell0.push_back(c0);
    in.face_cell1.push_back(c1);
    in.face_zone.push_back(zone);
  };
  in.face_node_offsets.push_back(0);
  // Interior face listed from cell 1 so that assemble must flip it.
  face({p(1, 0, 0), p(1, 0, 1), p(1, 1, 1), p(1, 1, 0)}, 1, 0, 1);
  for (int c = 0; c < 2; ++c) {
    const auto ci = static_cast<index_t>(c);
    if (c == 0) face({p(0, 0, 0), p(0, 0, 1), p(0, 1, 1), p(0, 1, 0)}, ci, autan::no_cell, 0);
    if (c == 1) face({p(2, 0, 0), p(2, 1, 0), p(2, 1, 1), p(2, 0, 1)}, ci, autan::no_cell, 0);
    face({p(c, 0, 0), p(c + 1, 0, 0), p(c + 1, 0, 1), p(c, 0, 1)}, ci, autan::no_cell, 0);
    face({p(c, 1, 0), p(c, 1, 1), p(c + 1, 1, 1), p(c + 1, 1, 0)}, ci, autan::no_cell, 0);
    face({p(c, 0, 0), p(c, 1, 0), p(c + 1, 1, 0), p(c + 1, 0, 0)}, ci, autan::no_cell, 0);
    face({p(c, 0, 1), p(c + 1, 0, 1), p(c + 1, 1, 1), p(c, 1, 1)}, ci, autan::no_cell, 0);
  }
  in.cell_zone = {0, 0};
  in.face_zones = {
      autan::FaceZone{
          .name = "walls", .bc_type = "wall", .kind = autan::BcKind::wall, .provenance = {}},
      autan::FaceZone{.name = "interior",
                      .bc_type = "",
                      .kind = autan::BcKind::unspecified,
                      .boundary = false,
                      .provenance = {}}};
  in.cell_zones = {autan::CellZone{
      .name = "fluid", .type = "", .kind = autan::CellKind::unspecified, .provenance = {}}};
  in.source = autan::MeshSource{.format = "memory", .path = ""};
  return in;
}

}  // namespace

TEST_CASE("Assembled meshes satisfy the canonical ordering of the mesh contract", "[mesh]") {
  const auto fill =
      GENERATE(autan::test::BoxFill::hexa, autan::test::BoxFill::tetra, autan::test::BoxFill::prism,
               autan::test::BoxFill::pyramid, autan::test::BoxFill::hybrid);
  CAPTURE(autan::test::to_string(fill));
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(autan::test::box(4, fill, 0.0)));
  const autan::Geometry g = autan::compute_geometry(mesh);
  const auto owner = mesh.owner();
  const auto neighbour = mesh.neighbour();
  const index_t n_interior = mesh.n_interior_faces();

  for (index_t f = 0; f < n_interior; ++f) {
    CHECK(owner[at(f)] < neighbour[at(f)]);
    if (f > 0) {
      const auto prev = std::array{owner[at(f - 1)], neighbour[at(f - 1)]};
      CHECK(prev <= (std::array{owner[at(f)], neighbour[at(f)]}));
    }
    // Convex cells: the area vector points from the owner's centroid to the
    // neighbour's.
    const std::size_t o = at(owner[at(f)]);
    const std::size_t n = at(neighbour[at(f)]);
    const double d = g.face_area_x[at(f)] * (g.cell_centroid_x[n] - g.cell_centroid_x[o]) +
                     g.face_area_y[at(f)] * (g.cell_centroid_y[n] - g.cell_centroid_y[o]) +
                     g.face_area_z[at(f)] * (g.cell_centroid_z[n] - g.cell_centroid_z[o]);
    CHECK(d > 0.0);
  }

  const auto zone_offsets = mesh.boundary_zone_offsets();
  REQUIRE(zone_offsets.size() == mesh.face_zones().size() + 1);
  CHECK(zone_offsets.front() == n_interior);
  CHECK(zone_offsets.back() == mesh.n_faces());
  for (std::size_t z = 0; z + 1 < zone_offsets.size(); ++z) {
    for (index_t f = zone_offsets[z]; f < zone_offsets[z + 1]; ++f) {
      CHECK(mesh.face_zone()[at(f)] == static_cast<index_t>(z));
      if (f > zone_offsets[z]) CHECK(owner[at(f - 1)] <= owner[at(f)]);
      const std::size_t o = at(owner[at(f)]);
      const double d = g.face_area_x[at(f)] * (g.face_centroid_x[at(f)] - g.cell_centroid_x[o]) +
                       g.face_area_y[at(f)] * (g.face_centroid_y[at(f)] - g.cell_centroid_y[o]) +
                       g.face_area_z[at(f)] * (g.face_centroid_z[at(f)] - g.cell_centroid_z[o]);
      CHECK(d > 0.0);
    }
  }

  // Sides of the box keep their labels; interior faces get the synthesised zone.
  const std::array<std::string, 6> sides{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
  for (std::size_t z = 0; z < sides.size(); ++z) CHECK(mesh.face_zones()[z].name == sides[z]);
  REQUIRE(mesh.face_zones().size() == 7);
  CHECK(mesh.face_zones()[6].name == "interior");
  CHECK(mesh.face_zones()[6].provenance.source_id == -1);
  for (index_t f = 0; f < n_interior; ++f) CHECK(mesh.face_zone()[at(f)] == 6);

  // Cell->face CSR lists exactly the faces whose owner or neighbour is the cell.
  std::vector<int> seen(at(mesh.n_faces()), 0);
  for (index_t c = 0; c < mesh.n_cells(); ++c) {
    const auto faces = mesh.cell_faces(c);
    CHECK(std::is_sorted(faces.begin(), faces.end()));
    for (const index_t f : faces) {
      CHECK((owner[at(f)] == c || (f < n_interior && neighbour[at(f)] == c)));
      ++seen[at(f)];
    }
  }
  for (index_t f = 0; f < mesh.n_faces(); ++f) CHECK(seen[at(f)] == (f < n_interior ? 2 : 1));
}

TEST_CASE("Analytic entity counts of box meshes", "[mesh]") {
  struct Expected {
    autan::test::BoxFill fill;
    int cells_per_cube;
    int faces_per_cell;
    int boundary_faces_per_side_square;
  };
  const auto e = GENERATE(Expected{autan::test::BoxFill::hexa, 1, 6, 1},
                          Expected{autan::test::BoxFill::tetra, 6, 4, 2},
                          Expected{autan::test::BoxFill::pyramid, 6, 5, 1});
  CAPTURE(autan::test::to_string(e.fill));
  constexpr int n = 3;
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(autan::test::box(n, e.fill)));
  const int cells = e.cells_per_cube * n * n * n;
  const int boundary = 6 * n * n * e.boundary_faces_per_side_square;
  CHECK(mesh.n_cells() == cells);
  CHECK(mesh.n_boundary_faces() == boundary);
  CHECK(mesh.n_interior_faces() == (e.faces_per_cell * cells - boundary) / 2);
  const int corner_points = (n + 1) * (n + 1) * (n + 1);
  CHECK(mesh.n_points() ==
        corner_points + (e.fill == autan::test::BoxFill::pyramid ? n * n * n : 0));
}

TEST_CASE("Assemble flips interior faces to owner < neighbour and keeps zones", "[mesh]") {
  const autan::Mesh mesh = autan::assemble(two_cubes());
  REQUIRE(mesh.n_interior_faces() == 1);
  CHECK(mesh.owner()[0] == 0);
  CHECK(mesh.neighbour()[0] == 1);
  const autan::Geometry g = autan::compute_geometry(mesh);
  CHECK(g.face_area_x[0] == 1.0);
  CHECK(mesh.face_zones()[0].kind == autan::BcKind::wall);
  CHECK(mesh.face_zones()[0].bc_type == "wall");
  CHECK(mesh.boundary_zone_offsets()[1] - mesh.boundary_zone_offsets()[0] == 10);
}

TEST_CASE("Points referenced by no face are removed", "[mesh]") {
  autan::MeshInput in = two_cubes();
  in.x.insert(in.x.begin(), 42.0);
  in.y.insert(in.y.begin(), 42.0);
  in.z.insert(in.z.begin(), 42.0);
  for (index_t& node : in.face_nodes) ++node;
  const autan::Mesh mesh = autan::assemble(in);
  CHECK(mesh.n_points() == 12);
  CHECK(std::find(mesh.x().begin(), mesh.x().end(), 42.0) == mesh.x().end());
}

TEST_CASE("Invalid meshes are rejected with their documented code", "[mesh]") {
  using autan::ErrorCode;
  SECTION("E0101 inconsistent sizes") {
    auto in = two_cubes();
    in.face_zone.pop_back();
    require_error(ErrorCode::mesh_inconsistent_sizes, [&] { (void)autan::assemble(in); });
  }
  SECTION("E0102 node index out of range") {
    auto in = two_cubes();
    in.face_nodes[5] = 12;
    require_error(ErrorCode::mesh_index_out_of_range, [&] { (void)autan::assemble(in); });
  }
  SECTION("E0103 face with fewer than 3 nodes") {
    auto in = two_cubes();
    in.face_nodes.erase(in.face_nodes.begin() + 2, in.face_nodes.begin() + 4);
    for (std::size_t f = 1; f < in.face_node_offsets.size(); ++f) in.face_node_offsets[f] -= 2;
    require_error(ErrorCode::mesh_face_too_few_nodes, [&] { (void)autan::assemble(in); });
  }
  SECTION("E0104 repeated node") {
    auto in = two_cubes();
    in.face_nodes[1] = in.face_nodes[0];
    require_error(ErrorCode::mesh_face_repeated_node, [&] { (void)autan::assemble(in); });
  }
  SECTION("E0105 same cell on both sides") {
    auto in = two_cubes();
    in.face_cell1[0] = 1;
    require_error(ErrorCode::mesh_face_same_cell_both_sides, [&] { (void)autan::assemble(in); });
  }
  SECTION("E0106 cell with fewer than 4 faces") {
    auto in = two_cubes();
    in.cell_zone.push_back(0);
    require_error(ErrorCode::mesh_cell_too_few_faces, [&] { (void)autan::assemble(in); });
  }
  SECTION("E0107 open cell") {
    auto in = two_cubes();
    in.face_node_offsets.pop_back();
    in.face_nodes.resize(static_cast<std::size_t>(in.face_node_offsets.back()));
    in.face_cell0.pop_back();
    in.face_cell1.pop_back();
    in.face_zone.pop_back();
    require_error(ErrorCode::mesh_cell_not_closed, [&] { (void)autan::assemble(in); });
  }
  SECTION("E0107 inconsistently oriented cell") {
    auto in = two_cubes();
    std::reverse(in.face_nodes.end() - 4, in.face_nodes.end());
    require_error(ErrorCode::mesh_cell_not_closed, [&] { (void)autan::assemble(in); });
  }
  SECTION("E0108 face shared by three cells") {
    auto cw = autan::test::affine_cell(autan::CellShape::tetra, {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
                                       {0, 0, 0})
                  .mesh;
    cw.x.push_back(0.3);
    cw.y.push_back(0.3);
    cw.z.push_back(-1.0);
    cw.x.push_back(0.3);
    cw.y.push_back(0.3);
    cw.z.push_back(-2.0);
    cw.add_cell(autan::CellShape::tetra, std::array<index_t, 4>{0, 1, 2, 4}, 0);
    cw.add_cell(autan::CellShape::tetra, std::array<index_t, 4>{0, 1, 2, 5}, 0);
    require_error(ErrorCode::mesh_face_shared_by_more_than_two_cells,
                  [&] { (void)autan::to_face_based(cw); });
  }
  SECTION("E0109 zone with interior and boundary faces") {
    auto in = two_cubes();
    in.face_zone[0] = 0;
    require_error(ErrorCode::mesh_face_zone_role, [&] { (void)autan::assemble(in); });
  }
  SECTION("E0110 duplicate zone name") {
    auto in = two_cubes();
    in.face_zones[1].name = "walls";
    require_error(ErrorCode::mesh_duplicate_zone_name, [&] { (void)autan::assemble(in); });
  }
  SECTION("E0111 labelled face matching no cell face") {
    auto cw = autan::test::box(1, autan::test::BoxFill::hexa, 0.0, false);
    cw.face_zones.push_back(autan::FaceZone{
        .name = "stray", .bc_type = "", .kind = autan::BcKind::unspecified, .provenance = {}});
    cw.add_labelled_face(std::array<index_t, 3>{0, 1, 2}, 0);
    require_error(ErrorCode::mesh_labelled_face_not_found, [&] { (void)autan::to_face_based(cw); });
  }
  SECTION("E0101 cellwise offsets out of bounds") {
    auto cw = autan::test::affine_cell(autan::CellShape::tetra, {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
                                       {0, 0, 0})
                  .mesh;
    auto overflow = cw;
    cw.face_node_offsets[1] = 1000;
    require_error(ErrorCode::mesh_inconsistent_sizes, [&] { (void)autan::to_face_based(cw); });
    overflow.face_node_offsets[1] = std::numeric_limits<offset_t>::max();
    overflow.face_node_offsets[2] = -6;
    require_error(ErrorCode::mesh_inconsistent_sizes,
                  [&] { (void)autan::to_face_based(overflow); });
  }
  SECTION("E0102 labelled face with an invalid zone") {
    auto cw = autan::test::box(1, autan::test::BoxFill::hexa, 0.0, false);
    cw.face_zones.push_back(autan::FaceZone{
        .name = "side", .bc_type = "", .kind = autan::BcKind::unspecified, .provenance = {}});
    cw.add_labelled_face(std::array<index_t, 4>{0, 2, 6, 4}, -1);
    require_error(ErrorCode::mesh_index_out_of_range, [&] { (void)autan::to_face_based(cw); });
  }
  SECTION("E0109 labels on both interior and boundary faces") {
    auto cw = autan::test::box(2, autan::test::BoxFill::hexa, 0.0, false);
    cw.face_zones.push_back(autan::FaceZone{
        .name = "mixed", .bc_type = "", .kind = autan::BcKind::unspecified, .provenance = {}});
    cw.add_labelled_face(std::array<index_t, 4>{0, 3, 12, 9}, 0);   // x = 0 side
    cw.add_labelled_face(std::array<index_t, 4>{1, 4, 13, 10}, 0);  // x = 0.5, interior
    require_error(ErrorCode::mesh_face_zone_role, [&] { (void)autan::to_face_based(cw); });
  }
  SECTION("E0117 duplicate global id") {
    auto in = two_cubes();
    in.cell_global_id = {7, 7};
    require_error(ErrorCode::mesh_duplicate_global_id, [&] { (void)autan::assemble(in); });
    // A translation that is not a number is rejected before any comparison.
    in.n_ghost_cells = 1;
    in.ghost_translation = {{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}};
    require_error(ErrorCode::mesh_inconsistent_sizes, [&] { (void)autan::assemble(in); });
    // A ghost with the same id and the same (zero) translation is the same cell.
    in.n_ghost_cells = 1;
    in.ghost_translation = {{0.0, 0.0, 0.0}};
    require_error(ErrorCode::mesh_duplicate_global_id, [&] { (void)autan::assemble(in); });
  }
  SECTION("E0112 empty mesh") {
    autan::MeshInput in;
    in.face_node_offsets.push_back(0);
    require_error(ErrorCode::mesh_empty, [&] { (void)autan::assemble(in); });
  }
  SECTION("E0114 inverted neighbour in a cellwise mesh") {
    auto cw = autan::test::affine_cell(autan::CellShape::tetra, {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
                                       {0, 0, 0})
                  .mesh;
    cw.x.push_back(0.3);
    cw.y.push_back(0.3);
    cw.z.push_back(-1.0);
    // Inverted tetra: it lists the shared face (0, 1, 2) with the same
    // orientation as the first one.
    cw.add_cell(autan::CellShape::tetra, std::array<index_t, 4>{0, 1, 2, 4}, 0);
    require_error(ErrorCode::mesh_shared_face_orientation, [&] { (void)autan::to_face_based(cw); });
  }
  SECTION("E0115 face labelled twice") {
    auto cw = autan::test::box(1, autan::test::BoxFill::hexa);
    cw.add_labelled_face(std::array<index_t, 4>{0, 2, 6, 4}, 1);
    require_error(ErrorCode::mesh_face_labelled_twice, [&] { (void)autan::to_face_based(cw); });
  }
}

TEST_CASE("Unlabelled boundary faces land in the synthesised zone 'unassigned'", "[mesh]") {
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(autan::test::hanging_node_box()));
  const auto zones = mesh.face_zones();
  REQUIRE(zones.size() == 3);
  CHECK(zones[0].name == "inlet");
  CHECK(zones[0].kind == autan::BcKind::inlet);
  const auto offsets = mesh.boundary_zone_offsets();
  CHECK(offsets[1] - offsets[0] == 1);
  const auto unassigned = std::find_if(
      zones.begin(), zones.end(), [](const autan::FaceZone& z) { return z.name == "unassigned"; });
  REQUIRE(unassigned != zones.end());
  CHECK(unassigned->provenance.source_id == -1);
  const auto u = static_cast<std::size_t>(unassigned - zones.begin());
  // 4 sides of the big cell, 100 hex faces at x = 2 and 4 x 10 hex sides.
  CHECK(offsets[u + 1] - offsets[u] == 4 + 100 + 40);
  CHECK(mesh.n_interior_faces() == 100 + 2 * 10 * 9);
  CHECK(mesh.cell_zones()[at(mesh.cell_zone()[0])].name == "coarse");
}

TEST_CASE("The assembly map links every mesh entity to its input", "[mesh]") {
  autan::MeshInput in = two_cubes();
  in.x.insert(in.x.begin(), 42.0);
  in.y.insert(in.y.begin(), 42.0);
  in.z.insert(in.z.begin(), 42.0);
  for (index_t& node : in.face_nodes) ++node;
  const autan::MeshInput copy = in;
  autan::AssemblyMap map;
  const autan::Mesh mesh = autan::assemble(in, &map);

  REQUIRE(map.face_source.size() == static_cast<std::size_t>(mesh.n_faces()));
  REQUIRE(map.point_source.size() == static_cast<std::size_t>(mesh.n_points()));
  for (index_t p = 0; p < mesh.n_points(); ++p) {
    CHECK(mesh.x()[at(p)] == copy.x[at(map.point_source[at(p)])]);
  }
  for (index_t f = 0; f < mesh.n_faces(); ++f) {
    const auto src = at(map.face_source[at(f)]);
    const index_t owner = map.face_reversed[at(f)] ? copy.face_cell1[src] : copy.face_cell0[src];
    CHECK(mesh.owner()[at(f)] == owner);
    CHECK(mesh.face_zone()[at(f)] == copy.face_zone[src]);
    CHECK(map.point_source[at(mesh.face_nodes(f)[0])] ==
          copy.face_nodes[static_cast<std::size_t>(copy.face_node_offsets[src])]);
  }
  CHECK(map.face_source[0] == 0);
  CHECK(map.face_reversed[0] == 1);
}

TEST_CASE("A partition with ghost cells keeps owned geometry and global ids", "[mesh]") {
  // Partition of a 2x2x2 hex box: the four cells with x < 1/2 are owned, the
  // four across x = 1/2 are ghosts that only carry their shared faces.
  const autan::Mesh full =
      autan::assemble(autan::to_face_based(autan::test::box(2, autan::test::BoxFill::hexa)));
  const autan::Geometry gf = autan::compute_geometry(full);
  std::vector<index_t> owned, ghosts;
  for (index_t c = 0; c < full.n_cells(); ++c)
    (gf.cell_centroid_x[at(c)] < 0.5 ? owned : ghosts).push_back(c);
  std::vector<index_t> local(at(full.n_cells()), -1);
  autan::MeshInput in;
  in.x.assign(full.x().begin(), full.x().end());
  in.y.assign(full.y().begin(), full.y().end());
  in.z.assign(full.z().begin(), full.z().end());
  for (const index_t c : owned) {
    local[at(c)] = static_cast<index_t>(in.cell_zone.size());
    in.cell_zone.push_back(0);
    in.cell_global_id.push_back(100 + c);
  }
  for (const index_t c : ghosts) {
    local[at(c)] = static_cast<index_t>(in.cell_zone.size());
    in.cell_zone.push_back(0);
    in.cell_global_id.push_back(100 + c);
  }
  in.n_ghost_cells = static_cast<index_t>(ghosts.size());
  in.face_node_offsets.push_back(0);
  for (index_t f = 0; f < full.n_faces(); ++f) {
    const index_t o = local[at(full.owner()[at(f)])];
    const index_t n =
        f < full.n_interior_faces() ? local[at(full.neighbour()[at(f)])] : autan::no_cell;
    const bool o_owned = o < static_cast<index_t>(owned.size());
    const bool n_owned = n != autan::no_cell && n < static_cast<index_t>(owned.size());
    if (!o_owned && !n_owned) continue;
    const auto nodes = full.face_nodes(f);
    in.face_nodes.insert(in.face_nodes.end(), nodes.begin(), nodes.end());
    in.face_node_offsets.push_back(static_cast<offset_t>(in.face_nodes.size()));
    in.face_cell0.push_back(o);
    in.face_cell1.push_back(n);
    in.face_zone.push_back(full.face_zone()[at(f)]);
  }
  in.face_zones.assign(full.face_zones().begin(), full.face_zones().end());
  in.cell_zones.assign(full.cell_zones().begin(), full.cell_zones().end());
  const autan::Mesh part = autan::assemble(in);
  const autan::Geometry gp = autan::compute_geometry(part);

  CHECK(part.n_owned_cells() == 4);
  CHECK(part.n_cells() == 8);
  for (index_t f = 0; f < part.n_faces(); ++f) CHECK(part.owner()[at(f)] < part.n_owned_cells());
  for (index_t c = 0; c < part.n_cells(); ++c) {
    const auto global = static_cast<index_t>(part.cell_global_id()[at(c)] - 100);
    if (c < part.n_owned_cells()) {
      CHECK(gp.cell_volume[at(c)] == gf.cell_volume[at(global)]);
      CHECK(gp.cell_centroid_x[at(c)] == gf.cell_centroid_x[at(global)]);
    } else {
      CHECK(std::isnan(gp.cell_volume[at(c)]));
      CHECK(part.cell_faces(c).size() == 1);
    }
  }
  const auto closure = autan::cell_closure_error(part, gp);
  CHECK(closure.size() == 4);
  CHECK(*std::max_element(closure.begin(), closure.end()) < 1e-12);
}

TEST_CASE("A zone that labels no face keeps its declared role", "[mesh]") {
  auto cw = autan::test::box(1, autan::test::BoxFill::hexa);
  cw.face_zones.push_back(autan::FaceZone{
      .name = "spare-outlet", .bc_type = "", .kind = autan::BcKind::outlet, .provenance = {}});
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(cw));
  const auto& zone = mesh.face_zones()[6];
  CHECK(zone.name == "spare-outlet");
  CHECK(zone.boundary);
  CHECK(mesh.boundary_zone_offsets()[7] == mesh.boundary_zone_offsets()[6]);
}

TEST_CASE("Faces of owned cells come first, then the halo faces of ghost cells", "[mesh]") {
  // two_cubes with cube 1 a ghost that carries all its faces: one interior
  // face, five boundary faces per cube.
  auto in = two_cubes();
  in.n_ghost_cells = 1;
  // An image of cube 0 one unit further along x: same id, other place.
  in.cell_global_id = {7, 7};
  in.ghost_translation = {{1.0, 0.0, 0.0}};
  const autan::Mesh mesh = autan::assemble(in);
  CHECK(mesh.n_interior_faces() == 1);
  CHECK(mesh.n_owned_interior_faces() == 1);
  CHECK(mesh.n_owned_faces() == 1 + 5);
  CHECK(mesh.n_faces() == 1 + 5 + 5);
  CHECK(mesh.boundary_zone_offsets()[0] == 1);
  CHECK(mesh.boundary_zone_offsets()[1] == 6);  // zone "walls", owned half
  CHECK(mesh.halo_boundary_zone_offsets()[0] == 6);
  CHECK(mesh.halo_boundary_zone_offsets()[1] == 11);
  for (index_t f = mesh.n_interior_faces(); f < mesh.n_faces(); ++f) {
    CHECK((mesh.owner()[static_cast<std::size_t>(f)] == 1) == (f >= mesh.n_owned_faces()));
  }
  REQUIRE(mesh.ghost_translation().size() == 1);
  CHECK(mesh.ghost_translation()[0] == std::array<double, 3>{1.0, 0.0, 0.0});
  CHECK(mesh.view().n_owned_faces == 6);
  CHECK(mesh.view().halo_boundary_zone_offsets[1] == 11);
  // Renumbering keeps each ghost's translation with it.
  const autan::Mesh same = autan::renumber_cells(mesh, std::array<index_t, 2>{0, 1});
  CHECK(same.ghost_translation()[0] == std::array<double, 3>{1.0, 0.0, 0.0});
}

TEST_CASE("A scaled mesh keeps its topology, scales lengths and periodic translations", "[mesh]") {
  autan::CellwiseMesh cells = autan::test::box(3, autan::test::BoxFill::hybrid, 0.2);
  cells.face_zones[0].kind = autan::BcKind::periodic;
  cells.face_zones[1].kind = autan::BcKind::periodic;
  cells.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(std::move(cells)));
  const double scale = 0.0254;
  const autan::Mesh s = autan::scaled(mesh, scale);
  REQUIRE(s.n_points() == mesh.n_points());
  for (std::size_t i = 0; i < static_cast<std::size_t>(mesh.n_points()); ++i) {
    CHECK(s.x()[i] == mesh.x()[i] * scale);
    CHECK(s.z()[i] == mesh.z()[i] * scale);
  }
  CHECK(std::ranges::equal(s.face_nodes(), mesh.face_nodes()));
  CHECK(std::ranges::equal(s.owner(), mesh.owner()));
  CHECK(std::ranges::equal(s.neighbour(), mesh.neighbour()));
  CHECK(std::ranges::equal(s.cell_global_id(), mesh.cell_global_id()));
  REQUIRE(s.periodic_pairs().size() == 1);
  CHECK(s.periodic_pairs()[0].translation[0] == scale);
  const autan::Geometry g0 = autan::compute_geometry(mesh), g1 = autan::compute_geometry(s);
  for (std::size_t c = 0; c < g0.cell_volume.size(); ++c)
    CHECK_THAT(g1.cell_volume[c], WithinRel(g0.cell_volume[c] * scale * scale * scale, 1e-13));
  for (const double bad : {0.0, -1.0, std::numeric_limits<double>::infinity()}) {
    CAPTURE(bad);
    try {
      (void)autan::scaled(mesh, bad);
      FAIL("expected E0102");
    } catch (const autan::Error& e) {
      CHECK(e.code() == autan::ErrorCode::mesh_index_out_of_range);
    }
  }
}
