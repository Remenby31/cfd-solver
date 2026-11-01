// Agglomeration hierarchy and multigrid solver of solvers/.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/core/renumber.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/partition/partition.hpp"
#include "autan/solvers/agglomeration.hpp"
#include "autan/solvers/line_smoother.hpp"
#include "autan/solvers/multigrid.hpp"

namespace {

namespace fs = std::filesystem;
using autan::index_t;
using autan::Mesh;
using autan::solvers::Level;
using autan::solvers::Multigrid;
using autan::solvers::MultigridOptions;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
const std::array<const char*, 4> meshes = {"tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh",
                                           "poly/polyMesh", "hybrid/hybrid.ascii.msh"};

enum class Fill { hexa, prism };

// Box [0,1]^2 x [0,h] of nx x nx x nz cells: hexahedra, or prisms extruded along z. The
// spacing in z grows geometrically from `first` at the wall z = 0 (zone "wall") up to
// `largest` (the in-plane spacing by default), then stays uniform up to z = 1/2. Cells with
// x < 1/2 are in a second cell zone when `two_zones`. With `slab` > 0 the box is one cell of
// that thickness in y, as the meshes of two-dimensional cases.
Mesh layered_box(int nx, double first, Fill fill, bool two_zones = false, double largest = 0.0,
                 double slab = 0.0) {
  autan::CellwiseMesh m;
  for (const char* name : {"a", "b"}) {
    autan::CellZone z;
    z.name = name;
    z.kind = autan::CellKind::fluid;
    m.cell_zones.push_back(z);
  }
  autan::FaceZone wall;
  wall.name = "wall";
  wall.kind = autan::BcKind::wall;
  m.face_zones.push_back(wall);
  const double dx = 1.0 / nx;
  const double top = largest > 0.0 ? largest : dx;
  std::vector<double> z{0.0};
  for (double d = first; z.size() < 400; d = std::min(top, 1.2 * d)) {
    z.push_back(z.back() + d);
    if (d == top && z.back() >= 0.5) break;
  }
  const int nz = static_cast<int>(z.size()) - 1;
  const int ny = slab > 0.0 ? 1 : nx;
  const double dy = slab > 0.0 ? slab : dx;
  const int p = nx + 1, q = ny + 1;
  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= ny; ++j) {
      for (int i = 0; i <= nx; ++i) {
        m.x.push_back(i * dx);
        m.y.push_back(j * dy);
        m.z.push_back(z[static_cast<std::size_t>(k)]);
      }
    }
  }
  auto v = [&](int i, int j, int k) { return static_cast<index_t>(i + p * (j + q * k)); };
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const index_t zone = two_zones && 2 * i < nx ? 1 : 0;
        if (fill == Fill::hexa) {
          m.add_cell(autan::CellShape::hexa,
                     std::array<index_t, 8>{v(i, j, k), v(i + 1, j, k), v(i + 1, j + 1, k),
                                            v(i, j + 1, k), v(i, j, k + 1), v(i + 1, j, k + 1),
                                            v(i + 1, j + 1, k + 1), v(i, j + 1, k + 1)},
                     zone);
        } else {
          m.add_cell(
              autan::CellShape::prism,
              std::array<index_t, 6>{v(i, j, k), v(i + 1, j, k), v(i + 1, j + 1, k), v(i, j, k + 1),
                                     v(i + 1, j, k + 1), v(i + 1, j + 1, k + 1)},
              zone);
          m.add_cell(
              autan::CellShape::prism,
              std::array<index_t, 6>{v(i, j, k), v(i + 1, j + 1, k), v(i, j + 1, k), v(i, j, k + 1),
                                     v(i + 1, j + 1, k + 1), v(i, j + 1, k + 1)},
              zone);
        }
        if (k == 0) {
          // The wall faces, seen from outside the cells.
          if (fill == Fill::hexa) {
            m.add_labelled_face(std::array<index_t, 4>{v(i, j, 0), v(i, j + 1, 0),
                                                       v(i + 1, j + 1, 0), v(i + 1, j, 0)},
                                0);
          } else {
            m.add_labelled_face(
                std::array<index_t, 3>{v(i, j, 0), v(i + 1, j + 1, 0), v(i + 1, j, 0)}, 0);
            m.add_labelled_face(
                std::array<index_t, 3>{v(i, j, 0), v(i, j + 1, 0), v(i + 1, j + 1, 0)}, 0);
          }
        }
      }
    }
  }
  return autan::assemble(autan::to_face_based(std::move(m)));
}

// One-cell-thick slab in the x-z plane: the x spacing grows by 1.3 from 1e-3 on either side of
// x = 0, the z spacing by 1.3 from 1e-3 at the wall z = 0 (zone "wall"). Above the wall, cells
// are stretched along z near x = 0, so lines run along x and, in each row, begin where the cells
// become anisotropic enough: they start at different columns, as above a flat plate's leading
// edge.
Mesh staggered_slab(int n) {
  autan::CellwiseMesh m;
  autan::CellZone zone;
  zone.name = "a";
  zone.kind = autan::CellKind::fluid;
  m.cell_zones.push_back(zone);
  autan::FaceZone wall;
  wall.name = "wall";
  wall.kind = autan::BcKind::wall;
  m.face_zones.push_back(wall);
  std::vector<double> half{0.0}, z{0.0};
  for (double d = 1e-3; static_cast<int>(half.size()) <= n; d *= 1.3)
    half.push_back(half.back() + d);
  for (double d = 1e-3; static_cast<int>(z.size()) <= n; d *= 1.3) z.push_back(z.back() + d);
  std::vector<double> x;
  for (auto it = half.rbegin(); it != half.rend(); ++it) x.push_back(-*it);
  for (std::size_t i = 1; i < half.size(); ++i) x.push_back(half[i]);
  const int nx = static_cast<int>(x.size()) - 1, nz = static_cast<int>(z.size()) - 1;
  const int p = nx + 1;
  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= 1; ++j) {
      for (int i = 0; i <= nx; ++i) {
        m.x.push_back(x[static_cast<std::size_t>(i)]);
        m.y.push_back(j * 1e-2);
        m.z.push_back(z[static_cast<std::size_t>(k)]);
      }
    }
  }
  auto v = [&](int i, int j, int k) { return static_cast<index_t>(i + p * (j + 2 * k)); };
  for (int k = 0; k < nz; ++k) {
    for (int i = 0; i < nx; ++i) {
      m.add_cell(autan::CellShape::hexa,
                 std::array<index_t, 8>{v(i, 0, k), v(i + 1, 0, k), v(i + 1, 1, k), v(i, 1, k),
                                        v(i, 0, k + 1), v(i + 1, 0, k + 1), v(i + 1, 1, k + 1),
                                        v(i, 1, k + 1)},
                 0);
      if (k == 0) {
        m.add_labelled_face(
            std::array<index_t, 4>{v(i, 0, 0), v(i, 1, 0), v(i + 1, 1, 0), v(i + 1, 0, 0)}, 0);
      }
    }
  }
  return autan::assemble(autan::to_face_based(std::move(m)));
}

// One column of cells of aspect ratio 4 and more along z, as layered_box(1, 1e-5, Fill::hexa,
// false, 0.25), with a short line on its side: four cells thin in x against the middle cell.
Mesh column_with_branch() {
  autan::CellwiseMesh m;
  autan::CellZone zone;
  zone.name = "a";
  zone.kind = autan::CellKind::fluid;
  m.cell_zones.push_back(zone);
  std::vector<double> z{0.0};
  for (double d = 1e-5; z.size() < 400; d = std::min(0.25, 1.2 * d)) {
    z.push_back(z.back() + d);
    if (d == 0.25 && z.back() >= 0.5) break;
  }
  const int nz = static_cast<int>(z.size()) - 1;
  const int mid = nz / 2;
  std::map<std::array<double, 3>, index_t> nodes;  // shared vertices
  auto node = [&](double x, double y, double zz) {
    const auto [it, added] = nodes.try_emplace({x, y, zz}, static_cast<index_t>(m.x.size()));
    if (added) {
      m.x.push_back(x);
      m.y.push_back(y);
      m.z.push_back(zz);
    }
    return it->second;
  };
  auto hexa = [&](double x0, double x1, double z0, double z1) {
    m.add_cell(
        autan::CellShape::hexa,
        std::array<index_t, 8>{node(x0, 0, z0), node(x1, 0, z0), node(x1, 1, z0), node(x0, 1, z0),
                               node(x0, 0, z1), node(x1, 0, z1), node(x1, 1, z1), node(x0, 1, z1)},
        0);
  };
  for (int k = 0; k < nz; ++k) {
    hexa(0.0, 1.0, z[static_cast<std::size_t>(k)], z[static_cast<std::size_t>(k) + 1]);
  }
  const double h = 0.01 * (z[static_cast<std::size_t>(mid) + 1] - z[static_cast<std::size_t>(mid)]);
  for (int i = 0; i < 4; ++i) {
    hexa(1.0 + i * h, 1.0 + (i + 1) * h, z[static_cast<std::size_t>(mid)],
         z[static_cast<std::size_t>(mid) + 1]);
  }
  return autan::assemble(autan::to_face_based(std::move(m)));
}

// Four cells 1 x 1 x 0.01 stacked along z, one line, whose boundary patches are A (bottom
// face), B (a side face of each middle cell) and C (top face): only the middle pair may share an
// agglomerate.
Mesh four_cell_line() {
  autan::CellwiseMesh m;
  autan::CellZone zone;
  zone.name = "a";
  zone.kind = autan::CellKind::fluid;
  m.cell_zones.push_back(zone);
  for (const char* name : {"A", "B", "C"}) {
    autan::FaceZone f;
    f.name = name;
    f.kind = autan::BcKind::wall;
    m.face_zones.push_back(f);
  }
  for (int k = 0; k <= 4; ++k) {
    for (const auto& [x, y] : {std::pair{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}) {
      m.x.push_back(x);
      m.y.push_back(y);
      m.z.push_back(0.01 * k);
    }
  }
  auto v = [](int k, int i) { return static_cast<index_t>(4 * k + i); };
  for (int k = 0; k < 4; ++k) {
    m.add_cell(autan::CellShape::hexa,
               std::array<index_t, 8>{v(k, 0), v(k, 1), v(k, 2), v(k, 3), v(k + 1, 0), v(k + 1, 1),
                                      v(k + 1, 2), v(k + 1, 3)},
               0);
  }
  m.add_labelled_face(std::array<index_t, 4>{v(0, 0), v(0, 3), v(0, 2), v(0, 1)}, 0);
  for (int k = 1; k <= 2; ++k) {
    m.add_labelled_face(std::array<index_t, 4>{v(k, 1), v(k, 2), v(k + 1, 2), v(k + 1, 1)}, 1);
  }
  m.add_labelled_face(std::array<index_t, 4>{v(4, 0), v(4, 1), v(4, 2), v(4, 3)}, 2);
  return autan::assemble(autan::to_face_based(std::move(m)));
}

// Operator of the tests: a diffusion stencil (V shift + sum over faces of A_f / d_f) on every
// component, coupled through a random face perturbation, so that the 5 x 5 blocks are full.
struct System {
  std::vector<double> diag, upper, lower;
};

System diffusion(const Mesh& mesh, const autan::Geometry& g, double shift, double coupling,
                 bool directional = false, unsigned seed = 7) {
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  System s;
  s.diag.assign(25 * n, 0.0);
  s.upper.assign(25 * ni, 0.0);
  s.lower.assign(25 * ni, 0.0);
  for (std::size_t c = 0; c < n; ++c) {
    for (std::size_t i = 0; i < 5; ++i) s.diag[25 * c + 6 * i] = shift * g.cell_volume[c];
  }
  for (std::size_t f = 0; f < ni; ++f) {
    const auto o = static_cast<std::size_t>(mesh.owner()[f]);
    const auto nb = static_cast<std::size_t>(mesh.neighbour()[f]);
    const double dx = g.cell_centroid_x[nb] - g.cell_centroid_x[o];
    const double dy = g.cell_centroid_y[nb] - g.cell_centroid_y[o];
    const double dz = g.cell_centroid_z[nb] - g.cell_centroid_z[o];
    const double w = g.face_area[f] / std::sqrt(dx * dx + dy * dy + dz * dz);
    // Row o takes w a (x_o - x_n), row n takes w b (x_n - x_o); `directional`: b != a.
    for (std::size_t i = 0; i < 5; ++i) {
      for (std::size_t j = 0; j < 5; ++j) {
        const double a = (i == j ? 1.0 : 0.0) + coupling * u(rng);
        const double b = directional ? (i == j ? 1.0 : 0.0) + coupling * u(rng) : a;
        s.upper[25 * f + 5 * i + j] = -w * a;
        s.lower[25 * f + 5 * i + j] = -w * b;
        s.diag[25 * o + 5 * i + j] += w * a;
        s.diag[25 * nb + 5 * i + j] += w * b;
      }
    }
  }
  // Boundary faces: a Dirichlet-like A_f / d_f on the diagonal.
  for (auto f = static_cast<std::size_t>(mesh.n_interior_faces());
       f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
    const auto c = static_cast<std::size_t>(mesh.owner()[f]);
    const double dx = g.face_centroid_x[f] - g.cell_centroid_x[c];
    const double dy = g.face_centroid_y[f] - g.cell_centroid_y[c];
    const double dz = g.face_centroid_z[f] - g.cell_centroid_z[c];
    const double w = g.face_area[f] / std::sqrt(dx * dx + dy * dy + dz * dz);
    for (std::size_t i = 0; i < 5; ++i) s.diag[25 * c + 6 * i] += w;
  }
  return s;
}

// y = A x for the operator layout of Multigrid.
std::vector<double> apply(const Mesh& mesh, const System& s, const std::vector<double>& x) {
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  std::vector<double> y(5 * n, 0.0);
  auto add = [&](const double* a, std::size_t row, std::size_t col) {
    for (std::size_t i = 0; i < 5; ++i) {
      for (std::size_t j = 0; j < 5; ++j) y[i * n + row] += a[5 * i + j] * x[j * n + col];
    }
  };
  for (std::size_t c = 0; c < n; ++c) add(&s.diag[25 * c], c, c);
  for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_interior_faces()); ++f) {
    const auto o = static_cast<std::size_t>(mesh.owner()[f]);
    const auto nb = static_cast<std::size_t>(mesh.neighbour()[f]);
    add(&s.upper[25 * f], o, nb);
    add(&s.lower[25 * f], nb, o);
  }
  return y;
}

double norm(const std::vector<double>& v) {
  double s = 0.0;
  for (const double x : v) s += x * x;
  return std::sqrt(s);
}

void check_hierarchy(const Mesh& mesh, const std::vector<Level>& levels) {
  REQUIRE(!levels.empty());
  REQUIRE(levels[0].n_nodes == static_cast<std::size_t>(mesh.n_cells()));
  std::vector<index_t> zone(mesh.cell_zone().begin(), mesh.cell_zone().end());
  for (std::size_t l = 0; l < levels.size(); ++l) {
    const Level& lv = levels[l];
    INFO("level " << l);
    // Lines partition the nodes, consecutive nodes adjacent.
    std::set<std::pair<index_t, index_t>> edges;
    for (std::size_t e = 0; e < lv.owner.size(); ++e) {
      edges.emplace(std::min(lv.owner[e], lv.neighbour[e]), std::max(lv.owner[e], lv.neighbour[e]));
    }
    std::vector<int> seen(lv.n_nodes, 0);
    for (std::size_t li = 0; li < lv.n_lines(); ++li) {
      for (auto k = lv.line_offsets[li]; k < lv.line_offsets[li + 1]; ++k) {
        const index_t a = lv.line_nodes[static_cast<std::size_t>(k)];
        ++seen[static_cast<std::size_t>(a)];
        if (k > lv.line_offsets[li]) {
          const index_t b = lv.line_nodes[static_cast<std::size_t>(k - 1)];
          CHECK(edges.count({std::min(a, b), std::max(a, b)}) == 1);
        }
      }
    }
    CHECK(std::all_of(seen.begin(), seen.end(), [](int s) { return s == 1; }));
    if (l + 1 == levels.size()) {
      CHECK(lv.coarse_node.empty());
      break;
    }
    const Level& c = levels[l + 1];
    // Every coarse node holds fine nodes of one zone, connected through its inner edges.
    REQUIRE(lv.coarse_node.size() == lv.n_nodes);
    std::vector<std::vector<index_t>> members(c.n_nodes);
    for (std::size_t i = 0; i < lv.n_nodes; ++i) {
      members[static_cast<std::size_t>(lv.coarse_node[i])].push_back(static_cast<index_t>(i));
    }
    std::vector<index_t> coarse_zone(c.n_nodes);
    for (std::size_t C = 0; C < c.n_nodes; ++C) {
      REQUIRE(!members[C].empty());
      coarse_zone[C] = zone[static_cast<std::size_t>(members[C][0])];
      for (const index_t i : members[C]) CHECK(zone[static_cast<std::size_t>(i)] == coarse_zone[C]);
      std::set<index_t> reached{members[C][0]};
      for (bool grew = true; grew;) {
        grew = false;
        for (std::size_t e = 0; e < lv.owner.size(); ++e) {
          if (lv.coarse_edge[e] >= 0 ||
              lv.coarse_node[static_cast<std::size_t>(lv.owner[e])] != static_cast<index_t>(C)) {
            continue;
          }
          const bool a = reached.count(lv.owner[e]) != 0, b = reached.count(lv.neighbour[e]) != 0;
          if (a != b) {
            reached.insert(a ? lv.neighbour[e] : lv.owner[e]);
            grew = true;
          }
        }
      }
      CHECK(reached.size() == members[C].size());
    }
    // Coarse edges: one per adjacent pair, owner < neighbour, each fine edge mapped to its pair.
    for (std::size_t e = 0; e < c.owner.size(); ++e) CHECK(c.owner[e] < c.neighbour[e]);
    for (std::size_t e = 0; e < lv.owner.size(); ++e) {
      const index_t a = lv.coarse_node[static_cast<std::size_t>(lv.owner[e])];
      const index_t b = lv.coarse_node[static_cast<std::size_t>(lv.neighbour[e])];
      if (a == b) {
        CHECK(lv.coarse_edge[e] == -1);
        continue;
      }
      const auto E = static_cast<std::size_t>(lv.coarse_edge[e]);
      CHECK(c.owner[E] == (lv.coarse_flipped[e] != 0 ? b : a));
      CHECK(c.neighbour[E] == (lv.coarse_flipped[e] != 0 ? a : b));
    }
    zone = coarse_zone;
  }
}

}  // namespace

TEST_CASE("agglomerates are connected, stay in one cell zone and lines join adjacent nodes",
          "[multigrid]") {
  SECTION("reference meshes") {
    const char* file = GENERATE(from_range(meshes));
    INFO(file);
    const auto mesh = autan::io::read_mesh(mesh_dir / file);
    const auto g = autan::compute_geometry(mesh);
    autan::solvers::AgglomerationOptions opt;
    opt.coarsest_nodes = 4;
    const auto levels = autan::solvers::agglomerate(mesh, g, opt);
    CHECK(levels.size() >= 2);
    check_hierarchy(mesh, levels);
  }
  SECTION("wall layers in two cell zones") {
    const Fill fill = GENERATE(Fill::hexa, Fill::prism);
    const auto mesh = layered_box(8, 1e-4, fill, true);
    const auto g = autan::compute_geometry(mesh);
    const auto levels = autan::solvers::agglomerate(mesh, g);
    check_hierarchy(mesh, levels);
  }
}

TEST_CASE("wall layers form one line per wall cell, kept through the coarse levels",
          "[multigrid]") {
  const Fill fill = GENERATE(Fill::hexa, Fill::prism);
  const auto mesh = layered_box(8, 1e-4, fill);
  const auto g = autan::compute_geometry(mesh);
  const auto levels = autan::solvers::agglomerate(mesh, g);
  const std::size_t wall_cells = fill == Fill::hexa ? 64 : 128;
  // Layers whose aspect ratio is at least 2 (line_ratio 4 = aspect ratio squared).
  std::size_t lines = 0, first_layer = 0;
  for (std::size_t li = 0; li < levels[0].n_lines(); ++li) {
    const auto len = levels[0].line_offsets[li + 1] - levels[0].line_offsets[li];
    if (len < 2) continue;
    ++lines;
    const auto c = static_cast<std::size_t>(
        levels[0].line_nodes[static_cast<std::size_t>(levels[0].line_offsets[li])]);
    if (g.cell_centroid_z[c] < 1e-4) {
      ++first_layer;
      CHECK(len >= 20);
    }
  }
  CHECK(first_layer == wall_cells);
  CHECK(lines >= wall_cells);
  // The lines a partition keeps whole: the same, without the single cells.
  const auto groups = autan::solvers::smoother_lines(mesh, g);
  CHECK(groups.offsets.size() == lines + 1);
  std::size_t from_wall = 0;
  for (std::size_t li = 0; li + 1 < groups.offsets.size(); ++li) {
    const auto b = static_cast<std::size_t>(groups.offsets[li]);
    CHECK(groups.offsets[li + 1] - groups.offsets[li] >= 2);
    if (g.cell_centroid_z[static_cast<std::size_t>(groups.cells[b])] < 1e-4) ++from_wall;
  }
  CHECK(from_wall == wall_cells);
  // The coarse levels agglomerate in the wall plane while neighbouring lines remain (8 x 8 wall
  // cells, groups of 4: two levels), keeping the line length; then lines coarsen along
  // themselves down to a direct solve.
  REQUIRE(levels.size() >= 3);
  CHECK(5 * levels.back().n_nodes <= 1000);
  for (std::size_t l = 1; l < 3; ++l) {
    std::size_t longest = 0;
    for (std::size_t li = 0; li < levels[l].n_lines(); ++li) {
      longest = std::max(longest, static_cast<std::size_t>(levels[l].line_offsets[li + 1] -
                                                           levels[l].line_offsets[li]));
    }
    CHECK(longest >= 20);
  }
}

TEST_CASE(
    "wall layers of a one-cell-thick mesh form wall-normal lines and coarsen to a direct solve",
    "[multigrid]") {
  // The two side planes of a two-dimensional case couple each cell through faces much larger
  // than its others: they must neither make every cell a line seed nor block agglomeration.
  const int nx = 32;
  const auto mesh = layered_box(nx, 1e-5, Fill::hexa, false, 0.0, 1.0);
  const auto g = autan::compute_geometry(mesh);
  const auto levels = autan::solvers::agglomerate(mesh, g);
  const auto& l0 = levels[0];
  std::size_t wall_lines = 0;
  for (std::size_t li = 0; li < l0.n_lines(); ++li) {
    const auto b = static_cast<std::size_t>(l0.line_offsets[li]);
    const auto e = static_cast<std::size_t>(l0.line_offsets[li + 1]);
    if (e - b < 2) continue;
    // A line stays in one column: every node has the x of its first node.
    const auto first = static_cast<std::size_t>(l0.line_nodes[b]);
    for (std::size_t k = b; k < e; ++k) {
      const auto c = static_cast<std::size_t>(l0.line_nodes[k]);
      CHECK(std::abs(g.cell_centroid_x[c] - g.cell_centroid_x[first]) < 0.5 / nx);
    }
    if (g.cell_centroid_z[first] < 1e-5) {
      ++wall_lines;
      CHECK(e - b >= 20);
    }
  }
  CHECK(wall_lines == static_cast<std::size_t>(nx));
  REQUIRE(levels.size() >= 3);
  CHECK(5 * levels.back().n_nodes <= 1000);
  check_hierarchy(mesh, levels);
}

TEST_CASE("a line with no neighbouring line coarsens along itself", "[multigrid]") {
  // One column of cells of aspect ratio 4 and more: a single line, no line to group with.
  const auto mesh = layered_box(1, 1e-5, Fill::hexa, false, 0.25);
  const auto g = autan::compute_geometry(mesh);
  autan::solvers::AgglomerationOptions options;
  options.coarsest_nodes = 8;
  const auto levels = autan::solvers::agglomerate(mesh, g, options);
  REQUIRE(levels[0].n_lines() == 1);
  REQUIRE(levels.size() >= 2);
  // Two layers per coarse node, the coarse level still one line.
  CHECK(levels[1].n_nodes == (levels[0].n_nodes + 1) / 2);
  CHECK(levels[1].n_lines() == 1);
  check_hierarchy(mesh, levels);
  // A short line touching its middle joins it there; its other layers still pair along it.
  const auto branched = column_with_branch();
  const auto gb = autan::compute_geometry(branched);
  const auto with_branch = autan::solvers::agglomerate(branched, gb, options);
  REQUIRE(with_branch.size() >= 2);
  CHECK(2 * with_branch[1].n_nodes <= with_branch[0].n_nodes + 4);
  check_hierarchy(branched, with_branch);
  // Pairs along the line are taken wherever two consecutive nodes may share an agglomerate, not
  // at a fixed parity: here the middle pair.
  const auto patched = four_cell_line();
  const auto gp = autan::compute_geometry(patched);
  options.coarsest_nodes = 1;
  const auto four = autan::solvers::agglomerate(patched, gp, options);
  REQUIRE(four[0].n_lines() == 1);
  REQUIRE(four.size() >= 2);
  CHECK(four[1].n_nodes == 3);
  check_hierarchy(patched, four);
}

TEST_CASE("layers coarsen in the plane whichever way their lines run", "[multigrid]") {
  // Uniform cells of aspect ratio 16: every cell is a line seed, so the cell numbering decides
  // where lines start. Reversing the numbering along every other column makes neighbouring
  // lines run opposite ways; their layers must still be matched.
  const int nx = 8;
  const auto mesh = layered_box(nx, 1.0 / 128, Fill::hexa, false, 1.0 / 128);
  const auto nz = static_cast<int>(mesh.n_cells()) / (nx * nx);
  std::vector<index_t> new_to_old(static_cast<std::size_t>(mesh.n_cells()));
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < nx; ++j) {
      for (int i = 0; i < nx; ++i) {
        const int kk = (i + j) % 2 == 0 ? k : nz - 1 - k;
        new_to_old[static_cast<std::size_t>(i + nx * (j + nx * k))] =
            static_cast<index_t>(i + nx * (j + nx * kk));
      }
    }
  }
  const auto reversed = autan::renumber_cells(mesh, new_to_old);
  for (const Mesh* m : {&mesh, &reversed}) {
    const auto g = autan::compute_geometry(*m);
    const auto levels = autan::solvers::agglomerate(*m, g);
    REQUIRE(levels.size() >= 2);
    CHECK(2 * levels[1].n_nodes <= levels[0].n_nodes);
    check_hierarchy(*m, levels);
  }
  // Lines that start at different columns: their layers are matched with a shift, so the
  // coarsening reaches the coarsest size instead of stalling.
  const auto slab = staggered_slab(24);
  const auto g = autan::compute_geometry(slab);
  const autan::solvers::AgglomerationOptions options;
  const auto levels = autan::solvers::agglomerate(slab, g, options);
  REQUIRE(levels.size() >= 2);
  CHECK(2 * levels[1].n_nodes <= levels[0].n_nodes);
  CHECK(levels.back().n_nodes <= options.coarsest_nodes);
  check_hierarchy(slab, levels);
  // On the fine level, the wall-plane grouping holds: nearly every node of a long line shares
  // its agglomerate with a node of another line, and the nodes of long lines left alone are at
  // most the ends of runs (two per line).
  const Level& fine = levels[0];
  std::vector<std::size_t> line_of(fine.n_nodes), size_of(fine.n_nodes);
  std::size_t long_lines = 0;
  for (std::size_t l = 0; l < fine.n_lines(); ++l) {
    const auto b = static_cast<std::size_t>(fine.line_offsets[l]);
    const auto e = static_cast<std::size_t>(fine.line_offsets[l + 1]);
    if (e - b >= options.min_line) ++long_lines;
    for (std::size_t k = b; k < e; ++k) {
      line_of[static_cast<std::size_t>(fine.line_nodes[k])] = l;
      size_of[static_cast<std::size_t>(fine.line_nodes[k])] = e - b;
    }
  }
  std::vector<std::vector<std::size_t>> aggregates(levels[1].n_nodes);
  for (std::size_t c = 0; c < fine.n_nodes; ++c) {
    aggregates[static_cast<std::size_t>(fine.coarse_node[c])].push_back(c);
  }
  std::size_t in_long = 0, grouped = 0, alone = 0;
  for (const auto& a : aggregates) {
    if (a.size() == 1 && size_of[a[0]] >= options.min_line) ++alone;
    for (const std::size_t x : a) {
      if (size_of[x] < options.min_line) continue;
      ++in_long;
      if (std::any_of(a.begin(), a.end(), [&](std::size_t y) {
            return line_of[y] != line_of[x] && size_of[y] >= options.min_line;
          })) {
        ++grouped;
      }
    }
  }
  CHECK(10 * grouped >= 9 * in_long);
  CHECK(alone <= 2 * long_lines);
}

TEST_CASE("one line sweep solves a block-tridiagonal system exactly", "[multigrid]") {
  // One column of cells of aspect ratio 4 and more: a single line holding every cell.
  const auto mesh = layered_box(1, 1e-5, Fill::hexa, false, 0.25);
  const auto g = autan::compute_geometry(mesh);
  Multigrid mg(mesh, g);
  REQUIRE(mg.levels()[0].n_lines() == 1);
  const auto s = diffusion(mesh, g, 1e-3, 0.3);
  mg.set_operator(s.diag, s.upper, s.lower);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  std::vector<double> x_exact(5 * n);
  for (std::size_t k = 0; k < x_exact.size(); ++k)
    x_exact[k] = std::sin(0.37 * static_cast<double>(k));
  const auto b = apply(mesh, s, x_exact);
  std::vector<double> x(5 * n, 0.0);
  // The multigrid's smoother and the distributed line smoother on one part.
  const auto lines = autan::solvers::smoother_lines(mesh, g);
  REQUIRE(lines.offsets.size() == 2);
  const std::vector<std::int32_t> colours(n, 0);
  autan::solvers::LineSmoother ls(mesh, lines.offsets, lines.cells, colours, 1);
  ls.set_operator(s.diag, s.upper, s.lower);
  for (const bool distributed : {false, true}) {
    std::vector<double> x(5 * n, 0.0);
    if (distributed) {
      ls.relax(0, b, x);
    } else {
      mg.smooth(b, x, 1);
    }
    CHECK(std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); }));
    double err = 0.0, scale = 0.0;
    for (std::size_t k = 0; k < x.size(); ++k) {
      err = std::max(err, std::abs(x[k] - x_exact[k]));
      scale = std::max(scale, std::abs(x_exact[k]));
    }
    CHECK(err <= 1e3 * std::numeric_limits<double>::epsilon() * scale);
  }
}

TEST_CASE("the line smoother repeats its bits on 1, 2 and 4 parts", "[multigrid]") {
  const Fill fill = GENERATE(Fill::hexa, Fill::prism);
  const auto mesh = layered_box(6, 1e-4, fill);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto s = diffusion(mesh, g, 1e-3, 0.3, true);
  const auto lines = autan::solvers::smoother_lines(mesh, g);
  REQUIRE(lines.offsets.size() > 2);
  // Colours of the graph whose nodes are the lines and the cells outside them.
  std::vector<std::size_t> node(n);
  std::vector<std::int64_t> node_id;
  std::vector<std::uint8_t> in_line(n, 0);
  for (std::size_t l = 0; l + 1 < lines.offsets.size(); ++l) {
    std::int64_t least = std::numeric_limits<std::int64_t>::max();
    for (auto k = lines.offsets[l]; k < lines.offsets[l + 1]; ++k) {
      const auto c = static_cast<std::size_t>(lines.cells[static_cast<std::size_t>(k)]);
      node[c] = node_id.size();
      in_line[c] = 1;
      least = std::min(least, static_cast<std::int64_t>(c));
    }
    node_id.push_back(least);
  }
  for (std::size_t c = 0; c < n; ++c) {
    if (in_line[c] != 0) continue;
    node[c] = node_id.size();
    node_id.push_back(static_cast<std::int64_t>(c));
  }
  std::vector<std::set<index_t>> adjacency(node_id.size());
  for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_interior_faces()); ++f) {
    const auto a = node[static_cast<std::size_t>(mesh.owner()[f])];
    const auto c = node[static_cast<std::size_t>(mesh.neighbour()[f])];
    if (a == c) continue;
    adjacency[a].insert(static_cast<index_t>(c));
    adjacency[c].insert(static_cast<index_t>(a));
  }
  std::vector<autan::offset_t> offsets{0};
  std::vector<index_t> neighbours;
  for (const auto& adj : adjacency) {
    neighbours.insert(neighbours.end(), adj.begin(), adj.end());
    offsets.push_back(static_cast<autan::offset_t>(neighbours.size()));
  }
  // Greedy distance-1 colouring in increasing node id (partition's colour_greedy).
  std::vector<std::size_t> visit(node_id.size());
  std::iota(visit.begin(), visit.end(), std::size_t{0});
  std::sort(visit.begin(), visit.end(),
            [&](std::size_t x, std::size_t y) { return node_id[x] < node_id[y]; });
  std::vector<std::int32_t> node_colour(node_id.size(), -1);
  for (const std::size_t v : visit) {
    std::set<std::int32_t> taken;
    for (auto k = offsets[v]; k < offsets[v + 1]; ++k) {
      taken.insert(node_colour[static_cast<std::size_t>(neighbours[static_cast<std::size_t>(k)])]);
    }
    std::int32_t c = 0;
    while (taken.count(c) != 0) ++c;
    node_colour[v] = c;
  }
  const std::int32_t n_colours = 1 + *std::max_element(node_colour.begin(), node_colour.end());
  std::vector<double> b(5 * n);
  for (std::size_t k = 0; k < b.size(); ++k) b[k] = std::sin(0.37 * static_cast<double>(k));

  // Each part's share: the global blocks, lines and colours in local indices; lockstep sweeps,
  // each ghost then copied from its owner.
  auto solve = [&](std::int32_t n_parts) {
    // Slices in x, each line on the part of its first cell: any partition that keeps the lines
    // whole must give the same result.
    std::vector<std::int32_t> parts(n);
    for (std::size_t c = 0; c < n; ++c) {
      parts[c] = std::min(n_parts - 1, static_cast<std::int32_t>(g.cell_centroid_x[c] * n_parts));
    }
    for (std::size_t q = 0; q + 1 < lines.offsets.size(); ++q) {
      const auto first =
          static_cast<std::size_t>(lines.cells[static_cast<std::size_t>(lines.offsets[q])]);
      for (auto k = lines.offsets[q]; k < lines.offsets[q + 1]; ++k) {
        parts[static_cast<std::size_t>(lines.cells[static_cast<std::size_t>(k)])] = parts[first];
      }
    }
    std::vector<autan::LocalPartition> local;
    std::vector<System> sys;
    std::vector<std::vector<double>> bl, xl;
    std::vector<std::vector<std::int32_t>> colours;
    std::vector<std::vector<std::size_t>> local_of(static_cast<std::size_t>(n_parts),
                                                   std::vector<std::size_t>(n, n));
    std::vector<std::unique_ptr<autan::solvers::LineSmoother>> smoothers;
    for (std::int32_t p = 0; p < n_parts; ++p) {
      local.push_back(autan::extract_partition(mesh, g, parts, n_parts, p, 1));
      const auto& m = local.back().mesh;
      const auto nl = static_cast<std::size_t>(m.n_cells());
      const auto gid = m.cell_global_id();
      for (std::size_t c = 0; c < nl; ++c) {
        local_of[static_cast<std::size_t>(p)][static_cast<std::size_t>(gid[c])] = c;
      }
      System l;
      l.diag.resize(25 * nl);
      for (std::size_t c = 0; c < nl; ++c) {
        std::copy_n(&s.diag[25 * static_cast<std::size_t>(gid[c])], 25, &l.diag[25 * c]);
      }
      const auto ni = static_cast<std::size_t>(m.n_interior_faces());
      l.upper.resize(25 * ni);
      l.lower.resize(25 * ni);
      for (std::size_t f = 0; f < ni; ++f) {
        const auto gf = static_cast<std::size_t>(local.back().face_global_id[f]);
        const bool rev = local.back().face_reversed[f] != 0;
        std::copy_n(&(rev ? s.lower : s.upper)[25 * gf], 25, &l.upper[25 * f]);
        std::copy_n(&(rev ? s.upper : s.lower)[25 * gf], 25, &l.lower[25 * f]);
      }
      sys.push_back(std::move(l));
      std::vector<autan::offset_t> lo{0};
      std::vector<index_t> lc;
      for (std::size_t q = 0; q + 1 < lines.offsets.size(); ++q) {
        const auto first =
            static_cast<std::size_t>(lines.cells[static_cast<std::size_t>(lines.offsets[q])]);
        if (parts[first] != p) continue;
        for (auto k = lines.offsets[q]; k < lines.offsets[q + 1]; ++k) {
          const auto c = static_cast<std::size_t>(lines.cells[static_cast<std::size_t>(k)]);
          REQUIRE(parts[c] == p);
          lc.push_back(static_cast<index_t>(local_of[static_cast<std::size_t>(p)][c]));
        }
        lo.push_back(static_cast<autan::offset_t>(lc.size()));
      }
      std::vector<std::int32_t> col(nl);
      std::vector<double> bp(5 * nl);
      for (std::size_t c = 0; c < nl; ++c) {
        const auto gc = static_cast<std::size_t>(gid[c]);
        col[c] = node_colour[node[gc]];
        for (std::size_t q = 0; q < 5; ++q) bp[q * nl + c] = b[q * n + gc];
      }
      colours.push_back(col);
      bl.push_back(std::move(bp));
      xl.emplace_back(5 * nl, 0.0);
      smoothers.push_back(std::make_unique<autan::solvers::LineSmoother>(
          m, lo, lc, colours.back(), n_colours, local.back().face_global_id));
      smoothers.back()->set_operator(sys.back().diag, sys.back().upper, sys.back().lower);
    }
    auto exchange = [&] {
      for (std::int32_t p = 0; p < n_parts; ++p) {
        const auto& m = local[static_cast<std::size_t>(p)].mesh;
        const auto nl = static_cast<std::size_t>(m.n_cells());
        for (auto c = static_cast<std::size_t>(m.n_owned_cells()); c < nl; ++c) {
          const auto gc = static_cast<std::size_t>(m.cell_global_id()[c]);
          const auto q = static_cast<std::size_t>(parts[gc]);
          const auto& mq = local[q].mesh;
          const auto nq = static_cast<std::size_t>(mq.n_cells());
          for (std::size_t k = 0; k < 5; ++k) {
            xl[static_cast<std::size_t>(p)][k * nl + c] = xl[q][k * nq + local_of[q][gc]];
          }
        }
      }
    };
    for (int sweep = 0; sweep < 2; ++sweep) {
      for (std::int32_t c = 0; c < 2 * n_colours; ++c) {
        const std::int32_t colour = c < n_colours ? c : 2 * n_colours - 1 - c;
        for (std::int32_t p = 0; p < n_parts; ++p) {
          smoothers[static_cast<std::size_t>(p)]->relax(colour, bl[static_cast<std::size_t>(p)],
                                                        xl[static_cast<std::size_t>(p)]);
        }
        exchange();
      }
    }
    std::vector<double> x(5 * n);
    for (std::size_t gc = 0; gc < n; ++gc) {
      const auto p = static_cast<std::size_t>(parts[gc]);
      const auto nl = static_cast<std::size_t>(local[p].mesh.n_cells());
      for (std::size_t k = 0; k < 5; ++k) x[k * n + gc] = xl[p][k * nl + local_of[p][gc]];
    }
    return x;
  };
  const auto one = solve(1);
  // Sweeping reduces the residual: the smoother solves the system it was given.
  const auto r0 = apply(mesh, s, one);
  double res = 0.0, rhs = 0.0;
  for (std::size_t k = 0; k < b.size(); ++k) {
    res += (b[k] - r0[k]) * (b[k] - r0[k]);
    rhs += b[k] * b[k];
  }
  CHECK(res < 0.25 * rhs);
  for (const std::int32_t n_parts : {2, 4}) {
    CHECK(solve(n_parts) == one);
  }
}

TEST_CASE("a two-level cycle applies the Galerkin coarse correction P (P^T A P)^-1 P^T",
          "[multigrid]") {
  // Meshes in RCM order, so that coarse edges of both orientations occur.
  const char* file = GENERATE("hybrid/hybrid.ascii.msh", "poly/polyMesh");
  INFO(file);
  const auto read = autan::io::read_mesh(mesh_dir / file);
  const auto mesh = autan::renumber_cells(read, autan::rcm_order(read));
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  MultigridOptions opt;
  opt.pre_sweeps = 0;
  opt.post_sweeps = 0;
  opt.scaling = autan::solvers::Scaling::none;
  opt.agglomeration.coarsest_nodes = n - 1;
  Multigrid mg(mesh, g, opt);
  REQUIRE(mg.levels().size() == 2);
  const std::size_t nc = mg.levels()[1].n_nodes;
  REQUIRE(5 * nc <= 1000);
  const auto s = diffusion(mesh, g, 1.0, 0.3, true);
  mg.set_operator(s.diag, s.upper, s.lower);
  std::vector<double> b(5 * n);
  for (std::size_t k = 0; k < b.size(); ++k) b[k] = std::cos(0.11 * static_cast<double>(k));
  std::vector<double> x(5 * n, 0.0);
  mg.cycle(b, x);
  CHECK(std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); }));
  // The hierarchy has edges whose owner lies in the coarse neighbour: both block orders are
  // exercised.
  const auto& flipped = mg.levels()[0].coarse_flipped;
  CHECK(std::count(flipped.begin(), flipped.end(), std::uint8_t{1}) > 0);
  // Dense P^T A P from the columns A P e_j, then x = P (P^T A P)^-1 P^T b by elimination.
  const auto& parent = mg.levels()[0].coarse_node;
  const std::size_t m = 5 * nc;
  std::vector<double> ac(m * m, 0.0), bc(m, 0.0);
  for (std::size_t j = 0; j < m; ++j) {
    std::vector<double> pe(5 * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      if (static_cast<std::size_t>(parent[i]) == j % nc) pe[(j / nc) * n + i] = 1.0;
    }
    const auto ape = apply(mesh, s, pe);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t c = 0; c < 5; ++c) {
        ac[(c * nc + static_cast<std::size_t>(parent[i])) * m + j] += ape[c * n + i];
      }
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t c = 0; c < 5; ++c)
      bc[c * nc + static_cast<std::size_t>(parent[i])] += b[c * n + i];
  }
  for (std::size_t k = 0; k < m; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < m; ++i) {
      if (std::abs(ac[i * m + k]) > std::abs(ac[p * m + k])) p = i;
    }
    std::swap_ranges(&ac[k * m], &ac[k * m] + m, &ac[p * m]);
    std::swap(bc[k], bc[p]);
    for (std::size_t i = k + 1; i < m; ++i) {
      const double f = ac[i * m + k] / ac[k * m + k];
      for (std::size_t j = k; j < m; ++j) ac[i * m + j] -= f * ac[k * m + j];
      bc[i] -= f * bc[k];
    }
  }
  for (std::size_t i = m; i-- > 0;) {
    for (std::size_t j = i + 1; j < m; ++j) bc[i] -= ac[i * m + j] * bc[j];
    bc[i] /= ac[i * m + i];
  }
  double err = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t c = 0; c < 5; ++c) {
      const double want = bc[c * nc + static_cast<std::size_t>(parent[i])];
      err = std::max(err, std::abs(x[c * n + i] - want));
      scale = std::max(scale, std::abs(want));
    }
  }
  CHECK(err <= 1e6 * std::numeric_limits<double>::epsilon() * scale);
}

// Iterations of the multigrid-preconditioned solve to a 1e-6 residual drop.
int iterations(const Mesh& mesh, const autan::Geometry& g, const MultigridOptions& options) {
  const auto s = diffusion(mesh, g, 1e-3, 0.1);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  std::vector<double> b(5 * n), x(5 * n, 0.0);
  for (std::size_t k = 0; k < b.size(); ++k) b[k] = std::sin(0.7 * static_cast<double>(k));
  Multigrid mg(mesh, g, options);
  mg.set_operator(s.diag, s.upper, s.lower);
  const auto report = mg.solve(b, x, 60, 1e-6);
  CHECK(report.residual_drop <= 1e-6);
  const auto r = apply(mesh, s, x);
  double rr = 0.0;
  for (std::size_t k = 0; k < r.size(); ++k) rr += (b[k] - r[k]) * (b[k] - r[k]);
  CHECK(std::sqrt(rr) <= 1.01e-6 * norm(b));
  return report.cycles;
}

TEST_CASE("multigrid iterations do not grow with refinement", "[multigrid]") {
  // Diffusion, the stiff part of the viscous Jacobian, with a small pseudo-time shift on
  // isotropic cells; Gauss-Seidel alone needs O(n^2) sweeps for the same drop.
  const Fill fill = GENERATE(Fill::hexa, Fill::prism);
  std::vector<int> counts;
  for (const int nx : {8, 16, 32}) {
    const auto mesh = layered_box(nx, 1.0 / nx, fill);
    const auto g = autan::compute_geometry(mesh);
    counts.push_back(iterations(mesh, g, {}));
    INFO("nx " << nx << ": " << counts.back() << " iterations");
    CHECK(counts.back() <= 9);
  }
  INFO("fill " << static_cast<int>(fill) << ", iterations " << counts[0] << ", " << counts[1]
               << ", " << counts[2]);
  CHECK(counts[2] <= counts[0] + 2);
}

TEST_CASE("line smoothing keeps the iterations bounded on wall layers of aspect ratio 1e4",
          "[multigrid]") {
  const Fill fill = GENERATE(Fill::hexa, Fill::prism);
  for (const int nx : {8, 16}) {
    const auto mesh = layered_box(nx, 1e-5, fill);
    const auto g = autan::compute_geometry(mesh);
    const int with_lines = iterations(mesh, g, {});
    MultigridOptions point;
    point.agglomeration.line_ratio = std::numeric_limits<double>::infinity();
    const int without = iterations(mesh, g, point);
    INFO("nx " << nx << ": " << with_lines << " iterations with lines, " << without << " without");
    CHECK(with_lines <= 9);
    CHECK(without >= 2 * with_lines);
  }
}

TEST_CASE("GCR reports a preconditioned direction that is not finite as a failed solve",
          "[multigrid]") {
  // A = I, r = (1, 1): the first direction (1, 0) halves |r|^2, the second is (NaN, 0).
  autan::solvers::Gcr gcr;
  std::vector<double> r{1.0, 1.0}, x{0.0, 0.0};
  int calls = 0;
  const auto report =
      gcr.solve([](std::span<const double> v,
                   std::span<double> y) { std::copy(v.begin(), v.end(), y.begin()); },
                [&](std::span<const double>, std::span<double> z) {
                  z[0] = calls++ == 0 ? 1.0 : std::numeric_limits<double>::quiet_NaN();
                  z[1] = 0.0;
                },
                r, x, 5, 1e-12);
  CHECK(std::isnan(report.residual_drop));
}
