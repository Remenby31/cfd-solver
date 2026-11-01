#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/core/renumber.hpp"
#include "autan/partition/device_self.hpp"
#include "autan/partition/partition.hpp"
#include "fixtures.hpp"

#ifdef AUTAN_HAVE_CUDA
#include <cuda_runtime.h>
#endif

using autan::index_t;

namespace {

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

void require_error(autan::ErrorCode expected, const std::function<void()>& action) {
  try {
    action();
    FAIL("expected " << autan::error_code_string(expected));
  } catch (const autan::Error& e) {
    CHECK(e.code() == expected);
  }
}

autan::Mesh box_mesh(int n, autan::test::BoxFill fill) {
  return autan::assemble(autan::to_face_based(autan::test::box(n, fill)));
}

/// Parts as slabs of equal width along x, by cell centroid.
std::vector<std::int32_t> slabs(const autan::Geometry& g, std::int32_t n_parts, double x_max) {
  std::vector<std::int32_t> parts(g.cell_volume.size());
  for (std::size_t c = 0; c < parts.size(); ++c) {
    const auto p = static_cast<std::int32_t>(std::floor(g.cell_centroid_x[c] / x_max * n_parts));
    parts[c] = std::clamp(p, 0, n_parts - 1);
  }
  return parts;
}

bool same_bits(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

/// Checks every rank of `parts` against the global mesh: owned cells cover the
/// mesh once, faces and owned cells carry the global geometry bit for bit,
/// ghosts are NaN, and each send list matches the receiver's ghost list.
std::vector<autan::LocalPartition> check_split(const autan::Mesh& global,
                                               const std::vector<std::int32_t>& parts,
                                               std::int32_t n_parts, int layers = 2) {
  const autan::Geometry gg = autan::compute_geometry(global);
  std::vector<autan::LocalPartition> locals;
  std::vector<int> seen(at(global.n_cells()), 0);
  for (std::int32_t r = 0; r < n_parts; ++r) {
    locals.push_back(autan::extract_partition(global, gg, parts, n_parts, r, layers));
    const autan::LocalPartition& lp = locals.back();
    const autan::Mesh& m = lp.mesh;
    const auto gid = m.cell_global_id();
    for (index_t c = 0; c < m.n_owned_cells(); ++c) {
      const auto global_cell = at(gid[at(c)]);
      ++seen[global_cell];
      CHECK(parts[global_cell] == r);
      CHECK(same_bits(lp.geometry.cell_volume[at(c)], gg.cell_volume[global_cell]));
      CHECK(same_bits(lp.geometry.cell_centroid_x[at(c)], gg.cell_centroid_x[global_cell]));
    }
    for (index_t c = m.n_owned_cells(); c < m.n_cells(); ++c) {
      // A ghost of the own part is a periodic image, elsewhere.
      const auto& t = m.ghost_translation()[at(c - m.n_owned_cells())];
      CHECK((parts[at(gid[at(c)])] != r || t != std::array<double, 3>{}));
      CHECK(std::isnan(lp.geometry.cell_volume[at(c)]));
    }
    for (index_t f = 0; f < m.n_faces(); ++f) {
      const auto source = at(lp.face_global_id[at(f)]);
      // The local owner is the global owner's copy at the face's home iff it
      // is that cell and, moved by its own translation, puts the global
      // centroid where the local face is: an image of the same cell across
      // the pair moves it elsewhere.
      const index_t owner = m.owner()[at(f)];
      std::array<double, 3> t{};
      if (owner >= m.n_owned_cells()) t = m.ghost_translation()[at(owner - m.n_owned_cells())];
      const bool same_orientation =
          gid[at(owner)] == global.owner()[source] &&
          same_bits(lp.geometry.face_centroid_x[at(f)], gg.face_centroid_x[source] + t[0]) &&
          same_bits(lp.geometry.face_centroid_y[at(f)], gg.face_centroid_y[source] + t[1]) &&
          same_bits(lp.geometry.face_centroid_z[at(f)], gg.face_centroid_z[source] + t[2]);
      const double sign = same_orientation ? 1.0 : -1.0;
      CHECK(lp.face_reversed[at(f)] == (same_orientation ? 0 : 1));
      if (f >= m.n_interior_faces()) CHECK(lp.face_reversed[at(f)] == 0);
      CHECK(same_bits(lp.geometry.face_area_x[at(f)], sign * gg.face_area_x[source]));
      CHECK(same_bits(lp.geometry.face_area_z[at(f)], sign * gg.face_area_z[source]));
      if (global.periodic_pairs().empty())
        CHECK(same_bits(lp.geometry.face_centroid_y[at(f)], gg.face_centroid_y[source]));
    }
    const auto& h = lp.halo;
    REQUIRE(h.recv_offsets.back() == m.n_cells() - m.n_owned_cells());
    for (const index_t f : h.interface_faces) {
      CHECK(f < m.n_owned_interior_faces());
      CHECK(m.neighbour()[at(f)] >= m.n_owned_cells());
    }
  }
  CHECK(std::all_of(seen.begin(), seen.end(), [](int s) { return s == 1; }));

  // What r sends to q are, in order, the cells q holds as ghosts from r.
  for (std::int32_t r = 0; r < n_parts; ++r) {
    const auto& hr = locals[at(r)].halo;
    for (std::size_t i = 0; i < hr.neighbours.size(); ++i) {
      const std::int32_t q = hr.neighbours[i];
      const auto& lq = locals[at(q)];
      const auto& hq = lq.halo;
      const auto j =
          at(std::find(hq.neighbours.begin(), hq.neighbours.end(), r) - hq.neighbours.begin());
      REQUIRE(j < hq.neighbours.size());
      std::vector<std::int64_t> sent, ghosts;
      for (index_t s = hr.send_offsets[i]; s < hr.send_offsets[i + 1]; ++s) {
        sent.push_back(locals[at(r)].mesh.cell_global_id()[at(hr.send_cells[at(s)])]);
      }
      for (index_t g = hq.recv_offsets[j]; g < hq.recv_offsets[j + 1]; ++g) {
        ghosts.push_back(lq.mesh.cell_global_id()[at(lq.mesh.n_owned_cells() + g)]);
      }
      CHECK(sent == ghosts);
    }
  }
  return locals;
}

}  // namespace

TEST_CASE("Slab partitions of a hex box have the analytic halo", "[partition]") {
  constexpr int n = 8;
  const std::int32_t n_parts = GENERATE(1, 2, 4, 8);
  const autan::Mesh global = box_mesh(n, autan::test::BoxFill::hexa);
  const auto parts = slabs(autan::compute_geometry(global), n_parts, 1.0);

  const auto q = autan::partition_quality(global, parts, n_parts);
  const std::int64_t layer = n * n;
  CHECK(q.cut_faces == (n_parts - 1) * layer);
  CHECK(q.imbalance == 1.0);
  CHECK(q.max_neighbours == std::min(n_parts - 1, 2));
  CHECK(q.ghost_ratio * n * n * n == static_cast<double>(2 * (n_parts - 1) * layer));
  CHECK(q.max_ghost_ratio ==
        (n_parts == 1 ? 0.0 : (n_parts == 2 ? 1.0 : 2.0) * layer * n_parts / (n * n * n)));

  const auto locals = check_split(global, parts, n_parts, 1);
  for (std::int32_t r = 0; r < n_parts; ++r) {
    const auto& m = locals[at(r)].mesh;
    const bool end = r == 0 || r == n_parts - 1;
    const std::int64_t ghosts = n_parts == 1 ? 0 : (end ? layer : 2 * layer);
    CHECK(m.n_owned_cells() == n * n * n / n_parts);
    CHECK(m.n_cells() - m.n_owned_cells() == ghosts);
    CHECK(q.ghost_cells[at(r)] == ghosts);
    CHECK(std::ssize(locals[at(r)].halo.interface_faces) == ghosts);
  }
}

TEST_CASE("Every cell type splits into valid partitions", "[partition]") {
  using autan::test::BoxFill;
  const BoxFill fill =
      GENERATE(BoxFill::tetra, BoxFill::hexa, BoxFill::prism, BoxFill::pyramid, BoxFill::hybrid);
  const std::int32_t n_parts = GENERATE(2, 3, 8);
  const autan::Mesh global = box_mesh(6, fill);
  check_split(global, slabs(autan::compute_geometry(global), n_parts, 1.0), n_parts);
}

TEST_CASE("A 105-face polyhedron with hanging nodes splits from its neighbours", "[partition]") {
  const autan::Mesh global = autan::assemble(autan::to_face_based(autan::test::hanging_node_box()));
  const auto locals = check_split(global, slabs(autan::compute_geometry(global), 2, 2.0), 2);
  CHECK(locals[0].mesh.n_owned_cells() == 1);
  CHECK(locals[0].mesh.n_cells() == 101);
}

TEST_CASE("Cell cost weights count faces", "[partition]") {
  const autan::Mesh global = autan::assemble(autan::to_face_based(autan::test::hanging_node_box()));
  const auto w = autan::cell_cost_weights(global);
  CHECK(*std::max_element(w.begin(), w.end()) == 105);
  CHECK(std::all_of(w.begin(), w.end(), [](std::int32_t x) { return x == 6 || x == 105; }));
}

#if defined(AUTAN_HAVE_METIS) || defined(AUTAN_HAVE_SCOTCH)
TEST_CASE("Graph partitioners return balanced, reproducible parts", "[partition]") {
  std::vector<autan::PartitionMethod> methods;
#ifdef AUTAN_HAVE_METIS
  methods.push_back(autan::PartitionMethod::metis);
#endif
#ifdef AUTAN_HAVE_SCOTCH
  methods.push_back(autan::PartitionMethod::scotch);
#endif
  const auto method = GENERATE_COPY(Catch::Generators::from_range(methods));
  const std::int32_t n_parts = GENERATE(1, 2, 4, 8);
  CAPTURE(method, n_parts);
  const autan::Mesh global = box_mesh(12, autan::test::BoxFill::hybrid);
  autan::PartitionOptions options;
  options.method = method;
  options.imbalance_tolerance = 0.05;
  options.cell_weights = autan::cell_cost_weights(global);

  const auto parts = autan::partition_cells(global, n_parts, options);
  CHECK(parts == autan::partition_cells(global, n_parts, options));
  const auto q = autan::partition_quality(global, parts, n_parts, options.cell_weights);
  CHECK(q.imbalance <= 1.0 + options.imbalance_tolerance + 1e-12);
  CHECK(std::all_of(q.owned_cells.begin(), q.owned_cells.end(), [](auto c) { return c > 0; }));
  // A partitioner beats slabs, whose cut is (n_parts - 1) layers of the box.
  const auto slab_cut = autan::partition_quality(
                            global, slabs(autan::compute_geometry(global), n_parts, 1.0), n_parts)
                            .cut_faces;
  CHECK(q.cut_faces <= slab_cut);
  check_split(global, parts, n_parts);
}
#endif

TEST_CASE("Invalid partition requests are rejected with their documented code", "[partition]") {
  using autan::ErrorCode;
  const autan::Mesh global = box_mesh(2, autan::test::BoxFill::hexa);
  const autan::Geometry gg = autan::compute_geometry(global);
  const std::vector<std::int32_t> two(8, 0);
  require_error(ErrorCode::partition_invalid_request, [&] { autan::partition_cells(global, 0); });
  require_error(ErrorCode::partition_invalid_request, [&] { autan::partition_cells(global, 9); });
  autan::PartitionOptions short_weights;
  short_weights.cell_weights.assign(7, 1);
  require_error(ErrorCode::partition_invalid_request,
                [&] { autan::partition_cells(global, 1, short_weights); });
  autan::PartitionOptions zero_weights;
  zero_weights.cell_weights.assign(8, 0);
  require_error(ErrorCode::partition_invalid_request,
                [&] { autan::partition_cells(global, 1, zero_weights); });
  require_error(ErrorCode::partition_invalid_request,
                [&] { autan::extract_partition(global, gg, two, 2, 1); });
  require_error(ErrorCode::partition_invalid_request,
                [&] { autan::extract_partition(global, gg, two, 1, 1); });
  require_error(ErrorCode::partition_invalid_request,
                [&] { autan::partition_quality(global, std::vector<std::int32_t>(8, 3), 2); });

  const auto parts = slabs(gg, 2, 1.0);
  const auto half = autan::extract_partition(global, gg, parts, 2, 0);
  require_error(ErrorCode::partition_input_has_ghosts,
                [&] { autan::partition_cells(half.mesh, 1); });
  require_error(ErrorCode::partition_input_has_ghosts,
                [&] { autan::extract_partition(half.mesh, half.geometry, parts, 2, 0); });
#ifndef AUTAN_HAVE_SCOTCH
  autan::PartitionOptions scotch;
  scotch.method = autan::PartitionMethod::scotch;
  require_error(ErrorCode::partition_backend_failed,
                [&] { autan::partition_cells(global, 2, scotch); });
#endif
}

namespace {

// Global ids of the cells within two face steps of local cell c (c excluded).
std::set<std::int64_t> face2(const autan::Mesh& m, index_t c) {
  std::set<index_t> near{c};
  for (int step = 0; step < 2; ++step) {
    const std::set<index_t> from = near;
    for (const index_t u : from) {
      for (const index_t f : m.cell_faces(u)) {
        if (f >= m.n_interior_faces()) continue;
        near.insert(m.owner()[at(f)] == u ? m.neighbour()[at(f)] : m.owner()[at(f)]);
      }
    }
  }
  near.erase(c);
  std::set<std::int64_t> ids;
  for (const index_t u : near) ids.insert(m.cell_global_id()[at(u)]);
  return ids;
}

}  // namespace

TEST_CASE("Two ghost layers hold the face-of-face stencil of every owned cell", "[partition]") {
  using autan::test::BoxFill;
  const BoxFill fill = GENERATE(BoxFill::hexa, BoxFill::tetra, BoxFill::hybrid);
  const std::int32_t n_parts = GENERATE(2, 4);
  const autan::Mesh global = box_mesh(6, fill);
  const auto locals =
      check_split(global, slabs(autan::compute_geometry(global), n_parts, 1.0), n_parts);
  for (const auto& lp : locals) {
    const auto& m = lp.mesh;
    CHECK(m.n_owned_interior_faces() < m.n_interior_faces());
    for (index_t c = 0; c < m.n_owned_cells(); ++c) {
      CHECK(face2(m, c) == face2(global, static_cast<index_t>(m.cell_global_id()[at(c)])));
    }
  }
}

TEST_CASE("Image ghosts sit one cell across periodic faces; a z-linear field keeps its gradient",
          "[partition][periodic]") {
  // The unit box in 4^3 cubes of `fill`, periodic in x and y. f = 3 z + 2 is
  // the same in every image, so a least-squares gradient over the face
  // neighbours is (0, 0, 3) on every owned cell, at the periodic faces too,
  // iff each image sits at its translated centroid.
  using autan::test::BoxFill;
  const BoxFill fill = GENERATE(BoxFill::hexa, BoxFill::tetra);
  const std::int32_t n_parts = GENERATE(1, 2, 4);
  autan::CellwiseMesh cells = autan::test::box(4, fill, 0.2);
  for (int k = 0; k < 2; ++k) {
    cells.face_zones[at(2 * k)].kind = autan::BcKind::periodic;
    cells.face_zones[at(2 * k + 1)].kind = autan::BcKind::periodic;
    std::array<double, 3> t{};
    t[at(k)] = 1.0;
    cells.periodic.push_back({.zone_a = 2 * k, .zone_b = 2 * k + 1, .translation = t});
  }
  const autan::Mesh global = autan::assemble(autan::to_face_based(std::move(cells)));
  const autan::Geometry gg = autan::compute_geometry(global);
  const auto parts = slabs(gg, n_parts, 1.0);
  const auto locals = check_split(global, parts, n_parts);
  for (const auto& lp : locals) {
    const auto& m = lp.mesh;
    CHECK(m.n_boundary_faces() > 0);  // the z sides stay boundaries
    // Ghost centroids as the exchange fills them: the owner's plus the
    // image's translation.
    std::vector<std::array<double, 3>> x(at(m.n_cells()));
    for (index_t c = 0; c < m.n_cells(); ++c) {
      const auto source = at(m.cell_global_id()[at(c)]);
      x[at(c)] = {gg.cell_centroid_x[source], gg.cell_centroid_y[source],
                  gg.cell_centroid_z[source]};
      if (c >= m.n_owned_cells()) {
        const auto& t = m.ghost_translation()[at(c - m.n_owned_cells())];
        for (std::size_t d = 0; d < 3; ++d) x[at(c)][d] += t[d];
      }
    }
    // Every face of an owned cell joins it to a neighbour one cell away, on
    // the side its area vector points to: across a periodic face too.
    for (index_t face = 0; face < m.n_owned_interior_faces(); ++face) {
      const auto u = at(m.owner()[at(face)]), v = at(m.neighbour()[at(face)]);
      const std::array<double, 3> d{x[v][0] - x[u][0], x[v][1] - x[u][1], x[v][2] - x[u][2]};
      CHECK(std::hypot(d[0], d[1], d[2]) < 0.75);
      CHECK(d[0] * lp.geometry.face_area_x[at(face)] + d[1] * lp.geometry.face_area_y[at(face)] +
                d[2] * lp.geometry.face_area_z[at(face)] >
            0.0);
    }
    const auto f = [&](index_t c) {
      return 3.0 * gg.cell_centroid_z[at(m.cell_global_id()[at(c)])] + 2.0;
    };
    for (index_t c = 0; c < m.n_owned_cells(); ++c) {
      std::array<std::array<double, 3>, 3> a{};
      std::array<double, 3> b{};
      for (const index_t face : m.cell_faces(c)) {
        if (face >= m.n_interior_faces()) continue;
        const index_t n = m.owner()[at(face)] == c ? m.neighbour()[at(face)] : m.owner()[at(face)];
        std::array<double, 3> d{};
        for (std::size_t k = 0; k < 3; ++k) d[k] = x[at(n)][k] - x[at(c)][k];
        for (std::size_t i = 0; i < 3; ++i) {
          b[i] += d[i] * (f(n) - f(c));
          for (std::size_t j = 0; j < 3; ++j) a[i][j] += d[i] * d[j];
        }
      }
      // Cramer's rule on the 3x3 normal equations.
      const auto det = [](const std::array<std::array<double, 3>, 3>& m3) {
        return m3[0][0] * (m3[1][1] * m3[2][2] - m3[1][2] * m3[2][1]) -
               m3[0][1] * (m3[1][0] * m3[2][2] - m3[1][2] * m3[2][0]) +
               m3[0][2] * (m3[1][0] * m3[2][1] - m3[1][1] * m3[2][0]);
      };
      const double d0 = det(a);
      std::array<double, 3> grad{};
      for (std::size_t k = 0; k < 3; ++k) {
        auto ak = a;
        for (std::size_t i = 0; i < 3; ++i) ak[i][k] = b[i];
        grad[k] = det(ak) / d0;
      }
      CHECK_THAT(grad[0], Catch::Matchers::WithinAbs(0.0, 1e-12));
      CHECK_THAT(grad[1], Catch::Matchers::WithinAbs(0.0, 1e-12));
      CHECK_THAT(grad[2], Catch::Matchers::WithinAbs(3.0, 1e-12));
    }
  }
}

TEST_CASE("Periodic sides split into several pairs share their direction",
          "[partition][periodic]") {
  // The unit box in 2^3 hexahedra, periodic in x, y and z, with each x side cut
  // in two zones at y = 1/2: two pairs along x, of equal or opposite
  // translations, and one along y.
  const bool opposite = GENERATE(false, true);
  const std::int32_t n_parts = GENERATE(1, 2);
  autan::CellwiseMesh cells = autan::test::box(2, autan::test::BoxFill::hexa);
  for (const char* name : {"xmin_hi", "xmax_hi"}) {
    cells.face_zones.push_back(
        {.name = name, .bc_type = "", .kind = autan::BcKind::periodic, .provenance = {}});
  }
  for (std::size_t f = 0; f < cells.labelled_face_zone.size(); ++f) {
    auto& zone = cells.labelled_face_zone[f];
    if (zone > 1) continue;
    bool high = true;
    for (auto k = cells.labelled_face_node_offsets[f]; k < cells.labelled_face_node_offsets[f + 1];
         ++k)
      high = high && cells.y[at(cells.labelled_face_nodes[at(k)])] >= 0.5;
    if (high) zone += 6;
  }
  for (const int z : {0, 1, 2, 3}) cells.face_zones[at(z)].kind = autan::BcKind::periodic;
  cells.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
  cells.periodic.push_back(
      opposite ? autan::PeriodicPair{.zone_a = 7, .zone_b = 6, .translation = {-1.0, 0.0, 0.0}}
               : autan::PeriodicPair{.zone_a = 6, .zone_b = 7, .translation = {1.0, 0.0, 0.0}});
  cells.periodic.push_back({.zone_a = 2, .zone_b = 3, .translation = {0.0, 1.0, 0.0}});
  // A fourth pair, along z: four pairs, three directions.
  for (const int z : {4, 5}) cells.face_zones[at(z)].kind = autan::BcKind::periodic;
  cells.periodic.push_back({.zone_a = 4, .zone_b = 5, .translation = {0.0, 0.0, 1.0}});
  const autan::Mesh global = autan::assemble(autan::to_face_based(std::move(cells)));
  const autan::Geometry gg = autan::compute_geometry(global);
  for (const auto& lp : check_split(global, slabs(gg, n_parts, 1.0), n_parts)) {
    const auto& m = lp.mesh;
    // No physical cell twice: (id, translation) is unique, as assemble checked.
    // Each owned cell has its six face neighbours, one cell away.
    for (index_t c = 0; c < m.n_owned_cells(); ++c) CHECK(std::ssize(m.cell_faces(c)) == 6);
    for (index_t face = 0; face < m.n_owned_interior_faces(); ++face) {
      const auto u = m.cell_global_id()[at(m.owner()[at(face)])];
      const auto v = m.cell_global_id()[at(m.neighbour()[at(face)])];
      const auto shift = [&](index_t c) {
        return c < m.n_owned_cells() ? std::array<double, 3>{}
                                     : m.ghost_translation()[at(c - m.n_owned_cells())];
      };
      const auto tu = shift(m.owner()[at(face)]), tv = shift(m.neighbour()[at(face)]);
      const double dx = gg.cell_centroid_x[at(v)] + tv[0] - gg.cell_centroid_x[at(u)] - tu[0];
      const double dy = gg.cell_centroid_y[at(v)] + tv[1] - gg.cell_centroid_y[at(u)] - tu[1];
      CHECK(std::hypot(dx, dy) < 0.75);
    }
  }
}

TEST_CASE("A cell periodic with itself marks the faces it reverses", "[partition][periodic]") {
  // One hexahedron periodic in x: across x = 1 the cell meets its own image,
  // so the local face there runs from the image to the cell, opposite to its
  // global face, whatever the number of layers.
  const int layers = GENERATE(1, 2);
  autan::CellwiseMesh cells = autan::test::box(1, autan::test::BoxFill::hexa, 0.0);
  cells.face_zones[0].kind = autan::BcKind::periodic;
  cells.face_zones[1].kind = autan::BcKind::periodic;
  cells.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
  const autan::Mesh global = autan::assemble(autan::to_face_based(std::move(cells)));
  const auto locals = check_split(global, std::vector<std::int32_t>(1, 0), 1, layers);
  const auto& reversed = locals[0].face_reversed;
  CHECK(std::ranges::count(reversed, std::uint8_t{1}) > 0);
}

TEST_CASE("One part fills its periodic images alone", "[partition][periodic]") {
  // The unit box of 3^3 tetrahedra cubes, periodic in x and y, as one part:
  // every ghost is an image of an owned cell. forward_self copies the owner's
  // value; fill_ghost_geometry its volume and its centroid moved by the
  // image's translation, bit for bit.
  autan::CellwiseMesh cells = autan::test::box(3, autan::test::BoxFill::tetra, 0.2);
  for (int k = 0; k < 2; ++k) {
    cells.face_zones[at(2 * k)].kind = autan::BcKind::periodic;
    cells.face_zones[at(2 * k + 1)].kind = autan::BcKind::periodic;
    std::array<double, 3> t{};
    t[at(k)] = 1.0;
    cells.periodic.push_back({.zone_a = 2 * k, .zone_b = 2 * k + 1, .translation = t});
  }
  const autan::Mesh global = autan::assemble(autan::to_face_based(std::move(cells)));
  const autan::Geometry gg = autan::compute_geometry(global);
  autan::LocalPartition part = autan::extract_partition(
      global, gg, std::vector<std::int32_t>(at(global.n_cells()), 0), 1, 0);
  const auto& m = part.mesh;
  REQUIRE(m.n_cells() > m.n_owned_cells());

  std::vector<double> id(at(m.n_cells()), -1.0);
  for (index_t c = 0; c < m.n_owned_cells(); ++c)
    id[at(c)] = static_cast<double>(m.cell_global_id()[at(c)]);
  double* const component = id.data();
  autan::forward_self(part.halo, m.n_owned_cells(), std::span<double* const>(&component, 1));
  for (index_t c = 0; c < m.n_cells(); ++c)
    CHECK(id[at(c)] == static_cast<double>(m.cell_global_id()[at(c)]));

  // The same in single precision.
  std::vector<float> id32(at(m.n_cells()), -1.0F);
  for (index_t c = 0; c < m.n_owned_cells(); ++c)
    id32[at(c)] = static_cast<float>(m.cell_global_id()[at(c)]);
  float* const component32 = id32.data();
  autan::forward_self(part.halo, m.n_owned_cells(), std::span<float* const>(&component32, 1));
  for (index_t c = 0; c < m.n_cells(); ++c)
    CHECK(id32[at(c)] == static_cast<float>(m.cell_global_id()[at(c)]));

  autan::fill_ghost_geometry(part);
  for (index_t c = m.n_owned_cells(); c < m.n_cells(); ++c) {
    const auto source = at(m.cell_global_id()[at(c)]);
    const auto& t = m.ghost_translation()[at(c - m.n_owned_cells())];
    CHECK(same_bits(part.geometry.cell_volume[at(c)], gg.cell_volume[source]));
    CHECK(same_bits(part.geometry.cell_centroid_x[at(c)], gg.cell_centroid_x[source] + t[0]));
    CHECK(same_bits(part.geometry.cell_centroid_y[at(c)], gg.cell_centroid_y[source] + t[1]));
    CHECK(same_bits(part.geometry.cell_centroid_z[at(c)], gg.cell_centroid_z[source] + t[2]));
  }

  // On one of several parts the exchange fills the geometry, not this.
  autan::LocalPartition half = autan::extract_partition(global, gg, slabs(gg, 2, 1.0), 2, 0);
  require_error(autan::ErrorCode::partition_invalid_request,
                [&] { autan::fill_ghost_geometry(half); });
}

TEST_CASE("Periodic images fill in device memory as on the host", "[partition][periodic][gpu]") {
  // The one-part tetrahedra box of "One part fills its periodic images
  // alone": DeviceSelfForward on device arrays gives the host forward_self's
  // values; a build without CUDA refuses device memory (E0145).
  autan::CellwiseMesh cells = autan::test::box(3, autan::test::BoxFill::tetra, 0.2);
  for (int k = 0; k < 2; ++k) {
    cells.face_zones[at(2 * k)].kind = autan::BcKind::periodic;
    cells.face_zones[at(2 * k + 1)].kind = autan::BcKind::periodic;
    std::array<double, 3> t{};
    t[at(k)] = 1.0;
    cells.periodic.push_back({.zone_a = 2 * k, .zone_b = 2 * k + 1, .translation = t});
  }
  const autan::Mesh global = autan::assemble(autan::to_face_based(std::move(cells)));
  const autan::LocalPartition part =
      autan::extract_partition(global, autan::compute_geometry(global),
                               std::vector<std::int32_t>(at(global.n_cells()), 0), 1, 0);
  const auto& m = part.mesh;
#ifdef AUTAN_HAVE_CUDA
  const auto n = at(m.n_cells());
  autan::DeviceSelfForward images(part.halo, m.n_owned_cells());
  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  // Two components, owned values distinct, ghosts unset, in each precision.
  const auto check_precision = [&]<class T>(T) {
    std::vector<T> a(n, T{-1}), b(n, T{-1});
    for (std::size_t c = 0; c < at(m.n_owned_cells()); ++c) {
      a[c] = static_cast<T>(m.cell_global_id()[c]) + T{0.25};
      b[c] = -static_cast<T>(m.cell_global_id()[c]);
    }
    std::vector<T> host_a = a, host_b = b;
    const std::array<T*, 2> host{host_a.data(), host_b.data()};
    autan::forward_self(part.halo, m.n_owned_cells(), std::span<T* const>(host));
    std::array<T*, 2> device{};
    for (std::size_t k = 0; k < 2; ++k) {
      REQUIRE(cudaMalloc(&device[k], n * sizeof(T)) == cudaSuccess);
      REQUIRE(cudaMemcpy(device[k], (k == 0 ? a : b).data(), n * sizeof(T),
                         cudaMemcpyHostToDevice) == cudaSuccess);
    }
    images.forward(std::span<T* const>(device), stream);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(cudaMemcpy(a.data(), device[0], n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(b.data(), device[1], n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(a == host_a);
    CHECK(b == host_b);
    std::array<T*, 17> too_many{};
    too_many.fill(device[0]);
    require_error(autan::ErrorCode::partition_invalid_request,
                  [&] { images.forward(std::span<T* const>(too_many), stream); });
    for (T* d : device) cudaFree(d);
  };
  check_precision(float{});
  check_precision(double{});
  cudaStreamDestroy(stream);
#else
  require_error(autan::ErrorCode::halo_device_unsupported,
                [&] { autan::DeviceSelfForward images(part.halo, m.n_owned_cells()); });
#endif
}

namespace {

// The dual graph of a mesh as a GraphLevel: cells, interior faces, global ids.
struct DualLevel {
  std::vector<index_t> owner, neighbour;
  std::vector<std::int64_t> ids;
  autan::GraphLevel level;
  explicit DualLevel(const autan::Mesh& mesh) {
    owner.assign(mesh.owner().begin(), mesh.owner().begin() + mesh.n_interior_faces());
    neighbour.assign(mesh.neighbour().begin(), mesh.neighbour().end());
    ids.assign(mesh.cell_global_id().begin(), mesh.cell_global_id().end());
    level = {
        .n_nodes = at(mesh.n_cells()), .owner = owner, .neighbour = neighbour, .global_id = ids};
  }
};

}  // namespace

#if defined(AUTAN_HAVE_METIS) || defined(AUTAN_HAVE_SCOTCH)
TEST_CASE("Groups of cells stay in one part", "[partition][graph]") {
  // Pairs of cells across every third interior face that touch no earlier pair.
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(6, autan::test::BoxFill::hybrid, 0.2)));
  std::vector<autan::offset_t> offsets{0};
  std::vector<index_t> grouped;
  std::vector<char> used(at(mesh.n_cells()), 0);
  for (index_t f = 0; f < mesh.n_interior_faces(); f += 3) {
    const index_t a = mesh.owner()[at(f)], b = mesh.neighbour()[at(f)];
    if (used[at(a)] || used[at(b)]) continue;
    used[at(a)] = used[at(b)] = 1;
    grouped.insert(grouped.end(), {a, b});
    offsets.push_back(static_cast<autan::offset_t>(grouped.size()));
  }
  REQUIRE(offsets.size() > 10);
  std::vector<autan::PartitionMethod> methods;
#ifdef AUTAN_HAVE_METIS
  methods.push_back(autan::PartitionMethod::metis);
#endif
#ifdef AUTAN_HAVE_SCOTCH
  methods.push_back(autan::PartitionMethod::scotch);
#endif
  for (const auto method : methods) {
    for (const std::int32_t n : {2, 4}) {
      const auto parts = autan::partition_cells(
          mesh, n,
          {.method = method, .cell_weights = {}, .group_offsets = offsets, .group_cells = grouped});
      for (std::size_t g = 0; g + 1 < offsets.size(); ++g)
        CHECK(parts[at(grouped[at(offsets[g])])] == parts[at(grouped[at(offsets[g]) + 1])]);
      CHECK(parts == autan::partition_cells(mesh, n,
                                            {.method = method,
                                             .cell_weights = {},
                                             .group_offsets = offsets,
                                             .group_cells = grouped}));
    }
  }
  // Malformed groups, refused on one part as on two: a cell in two groups, an empty group
  // (no phantom node), cells without offsets.
  const auto refused = [&](std::vector<autan::offset_t> group_offsets,
                           std::vector<index_t> group_cells) {
    for (const std::int32_t n : {1, 2}) {
      require_error(autan::ErrorCode::partition_invalid_request, [&] {
        (void)autan::partition_cells(
            mesh, n,
            {.cell_weights = {}, .group_offsets = group_offsets, .group_cells = group_cells});
      });
    }
  };
  refused({0, 2, 4}, {0, 1, 1, 2});
  std::vector<index_t> all(at(mesh.n_cells()));
  std::iota(all.begin(), all.end(), index_t{0});
  refused({0, 0, static_cast<autan::offset_t>(all.size())}, all);
  refused({}, {0, 1});
}
#endif

TEST_CASE("A coarse node goes to the part of its least child", "[partition][graph]") {
  // Children (ids 5, 2, 7) of coarse 0 and (ids 4, 9) of coarse 1.
  const std::vector<std::int32_t> parts{0, 1, 0, 2, 1};
  const std::vector<index_t> coarse{0, 0, 0, 1, 1};
  const std::vector<std::int64_t> ids{5, 2, 7, 4, 9};
  const auto cp = autan::coarse_parts(parts, coarse, ids, 2);
  CHECK(cp == std::vector<std::int32_t>{1, 2});
  // Coarse 0 is rank 1's (node 1, id 2): its other children, nodes 0 and 2, live on rank 0.
  CHECK(autan::remote_children(1, parts, coarse, cp) == std::vector<index_t>{0, 2});
  CHECK(autan::remote_children(2, parts, coarse, cp) == std::vector<index_t>{4});
  CHECK(autan::remote_children(0, parts, coarse, cp).empty());
  require_error(autan::ErrorCode::partition_invalid_request,
                [&] { (void)autan::coarse_parts(parts, coarse, ids, 3); });  // coarse 2 empty
  // The largest id is an id like any other.
  const std::vector<std::int64_t> top{std::numeric_limits<std::int64_t>::max()};
  CHECK(autan::coarse_parts(std::vector<std::int32_t>{1}, std::vector<index_t>{0}, top, 1) ==
        std::vector<std::int32_t>{1});
}

TEST_CASE("Graph parts send each other exactly the ghosts they hold", "[partition][graph]") {
  // The dual graph of a box, cut in slabs, coarsened by groups of 4 consecutive cells (so
  // many coarse nodes have children in two slabs). For every pair of parts, what one receives
  // from the other (ghost nodes and ghost edges, in order) is what the other sends, by id.
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(4, autan::test::BoxFill::tetra, 0.2)));
  const DualLevel dual(mesh);
  const autan::Geometry g = autan::compute_geometry(mesh);
  const std::int32_t n_parts = GENERATE(1, 2, 4);
  CAPTURE(n_parts);
  const auto parts = slabs(g, n_parts, 1.0);
  std::vector<index_t> coarse(at(mesh.n_cells()));
  for (std::size_t c = 0; c < coarse.size(); ++c) coarse[c] = static_cast<index_t>(c / 4);
  const auto cp = autan::coarse_parts(parts, coarse, dual.ids, (coarse.size() + 3) / 4);
  std::vector<autan::LocalGraph> locals;
  for (std::int32_t r = 0; r < n_parts; ++r)
    locals.push_back(autan::extract_graph(dual.level, parts, n_parts, r, coarse, cp));

  std::vector<int> owned(at(mesh.n_cells()), 0);
  for (std::int32_t r = 0; r < n_parts; ++r) {
    const auto& l = locals[at(r)];
    for (std::size_t v = 0; v < l.n_owned; ++v) {
      ++owned[at(l.node_source[v])];
      CHECK(parts[at(l.node_source[v])] == r);
    }
    // Remote children are ghosts, with their edges.
    std::set<std::int64_t> held(l.node_global_id.begin(), l.node_global_id.end());
    std::set<std::int64_t> edges(l.edge_global_id.begin(), l.edge_global_id.end());
    for (const index_t v : autan::remote_children(r, parts, coarse, cp)) {
      CHECK(held.contains(dual.ids[at(v)]));
      for (index_t f = 0; f < mesh.n_interior_faces(); ++f) {
        if (dual.owner[at(f)] == v || dual.neighbour[at(f)] == v) CHECK(edges.contains(f));
      }
    }
    // Local edges: local owner < neighbour, the level's cells, reversed as flagged.
    for (std::size_t e = 0; e < l.owner.size(); ++e) {
      CHECK(l.owner[e] < l.neighbour[e]);
      const auto f = at(l.edge_global_id[e]);
      const bool reversed = l.edge_reversed[e] != 0;
      CHECK(l.node_source[at(reversed ? l.neighbour[e] : l.owner[e])] == dual.owner[f]);
      CHECK(l.node_source[at(reversed ? l.owner[e] : l.neighbour[e])] == dual.neighbour[f]);
      CHECK((e < l.n_owned_edges) == (parts[at(dual.owner[f])] == r));
    }
  }
  CHECK(std::all_of(owned.begin(), owned.end(), [](int k) { return k == 1; }));

  // The exchange, emulated: the ids received from q are the ids q sends, in order.
  const auto check = [&](auto halo_of, auto ids_of, auto n_owned_of) {
    for (std::int32_t r = 0; r < n_parts; ++r) {
      const autan::HaloPattern& h = halo_of(locals[at(r)]);
      for (std::size_t i = 0; i < h.neighbours.size(); ++i) {
        const std::int32_t q = h.neighbours[i];
        const autan::HaloPattern& hq = halo_of(locals[at(q)]);
        const auto j =
            at(std::find(hq.neighbours.begin(), hq.neighbours.end(), r) - hq.neighbours.begin());
        REQUIRE(j < hq.neighbours.size());
        std::vector<std::int64_t> received, sent;
        for (index_t k = h.recv_offsets[i]; k < h.recv_offsets[i + 1]; ++k)
          received.push_back(ids_of(locals[at(r)])[n_owned_of(locals[at(r)]) + at(k)]);
        for (index_t k = hq.send_offsets[j]; k < hq.send_offsets[j + 1]; ++k)
          sent.push_back(ids_of(locals[at(q)])[at(hq.send_cells[at(k)])]);
        CHECK(received == sent);
      }
      std::size_t total = 0;
      for (std::size_t i = 0; i + 1 < h.recv_offsets.size(); ++i)
        total += at(h.recv_offsets[i + 1] - h.recv_offsets[i]);
      CHECK(total == ids_of(locals[at(r)]).size() - n_owned_of(locals[at(r)]));
    }
  };
  check([](const autan::LocalGraph& l) -> const autan::HaloPattern& { return l.node_halo; },
        [](const autan::LocalGraph& l) -> const std::vector<std::int64_t>& {
          return l.node_global_id;
        },
        [](const autan::LocalGraph& l) { return l.n_owned; });
  check([](const autan::LocalGraph& l) -> const autan::HaloPattern& { return l.edge_halo; },
        [](const autan::LocalGraph& l) -> const std::vector<std::int64_t>& {
          return l.edge_global_id;
        },
        [](const autan::LocalGraph& l) { return l.n_owned_edges; });

  // A coarse node owned by no part is refused.
  for (const std::int32_t wrong : {-1, n_parts}) {
    auto bad = cp;
    bad.front() = wrong;
    require_error(autan::ErrorCode::partition_invalid_request,
                  [&] { (void)autan::extract_graph(dual.level, parts, n_parts, 0, coarse, bad); });
  }
}

TEST_CASE("Greedy colouring visits nodes by increasing global id", "[partition][colour]") {
  using autan::offset_t;
  // A path 0-1-2-3 with ids 3, 2, 1, 0: visited 3, 2, 1, 0, so colours alternate from node 3.
  const std::vector<offset_t> path_offsets{0, 1, 3, 5, 6};
  const std::vector<index_t> path{1, 0, 2, 1, 3, 2};
  CHECK(autan::colour_greedy(path_offsets, path, std::vector<std::int64_t>{3, 2, 1, 0}) ==
        std::vector<std::int32_t>{1, 0, 1, 0});
  // A star whose centre has the largest id: the leaves first, all 0, then the centre 1.
  const std::vector<offset_t> star_offsets{0, 3, 4, 5, 6};
  const std::vector<index_t> star{1, 2, 3, 0, 0, 0};
  CHECK(autan::colour_greedy(star_offsets, star, std::vector<std::int64_t>{9, 1, 2, 3}) ==
        std::vector<std::int32_t>{1, 0, 0, 0});
  // A triangle needs three.
  const std::vector<offset_t> triangle_offsets{0, 2, 4, 6};
  const std::vector<index_t> triangle{1, 2, 0, 2, 0, 1};
  CHECK(autan::colour_greedy(triangle_offsets, triangle, std::vector<std::int64_t>{0, 1, 2}) ==
        std::vector<std::int32_t>{0, 1, 2});
  require_error(autan::ErrorCode::partition_invalid_request, [&] {
    (void)autan::colour_greedy(path_offsets, path, std::vector<std::int64_t>{0, 1});
  });
  // Offsets that go back: their last entry matches, the middle one does not.
  require_error(autan::ErrorCode::partition_invalid_request, [&] {
    (void)autan::colour_greedy(std::vector<offset_t>{0, 3, 2}, std::vector<index_t>{1, 0},
                               std::vector<std::int64_t>{0, 1});
  });
  require_error(autan::ErrorCode::partition_invalid_request, [&] {
    (void)autan::colour_greedy(triangle_offsets, std::vector<index_t>{1, 2, 0, 2, 0, 7},
                               std::vector<std::int64_t>{0, 1, 2});
  });
}

TEST_CASE("Cells across a face or a periodic pair never share a colour, in any cell order",
          "[partition][colour]") {
  using autan::test::BoxFill;
  const BoxFill fill = GENERATE(BoxFill::hexa, BoxFill::tetra, BoxFill::hybrid);
  const bool periodic = GENERATE(false, true);
  CAPTURE(periodic);
  autan::CellwiseMesh cells = autan::test::box(4, fill, 0.2);
  if (periodic) {
    for (int k = 0; k < 2; ++k) {
      cells.face_zones[at(2 * k)].kind = autan::BcKind::periodic;
      cells.face_zones[at(2 * k + 1)].kind = autan::BcKind::periodic;
      std::array<double, 3> t{};
      t[at(k)] = 1.0;
      cells.periodic.push_back({.zone_a = 2 * k, .zone_b = 2 * k + 1, .translation = t});
    }
  }
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(std::move(cells)));
  const auto colour = autan::colour_cells(mesh);
  REQUIRE(colour.size() == at(mesh.n_cells()));
  for (index_t f = 0; f < mesh.n_interior_faces(); ++f)
    CHECK(colour[at(mesh.owner()[at(f)])] != colour[at(mesh.neighbour()[at(f)])]);
  std::size_t across = 0;
  const auto offsets = mesh.boundary_zone_offsets();
  for (std::size_t p = 0; p < mesh.periodic_pairs().size(); ++p) {
    const auto partners = mesh.periodic_partners(p);
    const index_t first = offsets[at(mesh.periodic_pairs()[p].zone_a)];
    for (std::size_t k = 0; k < partners.size(); ++k, ++across) {
      CHECK(colour[at(mesh.owner()[at(first) + k])] != colour[at(mesh.owner()[at(partners[k])])]);
    }
  }
  CHECK((across > 0) == periodic);
  // The same colour for each global id after renumbering the cells (RCM).
  const autan::Mesh renumbered = autan::renumber_cells(mesh, autan::rcm_order(mesh));
  const auto again = autan::colour_cells(renumbered);
  for (std::size_t c = 0; c < again.size(); ++c)
    CHECK(again[c] == colour[at(renumbered.cell_global_id()[c])]);
  // Groups of four consecutive cells, every other block: a group shares one colour, and two
  // cells of different nodes across a face or a periodic pair never do.
  std::vector<autan::offset_t> group_offsets{0};
  std::vector<index_t> group_cells;
  std::vector<std::int64_t> node(at(mesh.n_cells()));
  std::iota(node.begin(), node.end(), std::int64_t{mesh.n_cells()});
  for (index_t c = 0; c + 4 <= mesh.n_cells(); c += 8) {
    for (index_t k = c; k < c + 4; ++k) {
      group_cells.push_back(k);
      node[at(k)] = c;
    }
    group_offsets.push_back(static_cast<autan::offset_t>(group_cells.size()));
  }
  const auto grouped = autan::colour_cells(mesh, group_offsets, group_cells);
  REQUIRE(grouped.size() == at(mesh.n_cells()));
  const auto apart = [&](index_t a, index_t b) {
    if (node[at(a)] != node[at(b)]) CHECK(grouped[at(a)] != grouped[at(b)]);
  };
  for (std::size_t g = 0; g + 1 < group_offsets.size(); ++g) {
    for (auto k = group_offsets[g]; k < group_offsets[g + 1]; ++k)
      CHECK(grouped[at(group_cells[at(k)])] == grouped[at(group_cells[at(group_offsets[g])])]);
  }
  for (index_t f = 0; f < mesh.n_interior_faces(); ++f)
    apart(mesh.owner()[at(f)], mesh.neighbour()[at(f)]);
  for (std::size_t p = 0; p < mesh.periodic_pairs().size(); ++p) {
    const auto partners = mesh.periodic_partners(p);
    const index_t first = offsets[at(mesh.periodic_pairs()[p].zone_a)];
    for (std::size_t k = 0; k < partners.size(); ++k)
      apart(mesh.owner()[at(first) + k], mesh.owner()[at(partners[k])]);
  }
  // A group's id is its least cell_global_id, not its place: the cells renumbered (RCM) and
  // the groups listed backwards, each cell keeps its colour.
  std::vector<index_t> new_of(at(mesh.n_cells()));
  for (std::size_t c = 0; c < new_of.size(); ++c)
    new_of[at(renumbered.cell_global_id()[c])] = static_cast<index_t>(c);
  std::vector<autan::offset_t> backwards_offsets{0};
  std::vector<index_t> backwards_cells;
  for (std::size_t g = group_offsets.size() - 1; g > 0; --g) {
    for (auto k = group_offsets[g - 1]; k < group_offsets[g]; ++k)
      backwards_cells.push_back(new_of[at(group_cells[at(k)])]);
    backwards_offsets.push_back(static_cast<autan::offset_t>(backwards_cells.size()));
  }
  const auto regrouped = autan::colour_cells(renumbered, backwards_offsets, backwards_cells);
  for (std::size_t c = 0; c < regrouped.size(); ++c)
    CHECK(regrouped[c] == grouped[at(renumbered.cell_global_id()[c])]);
  require_error(autan::ErrorCode::partition_invalid_request, [&] {
    (void)autan::colour_cells(mesh, std::vector<autan::offset_t>{0, 2}, std::vector<index_t>{0, 0});
  });
  require_error(autan::ErrorCode::partition_input_has_ghosts, [&] {
    const autan::LocalPartition part =
        autan::extract_partition(mesh, autan::compute_geometry(mesh),
                                 std::vector<std::int32_t>(at(mesh.n_cells()), 0), 1, 0);
    if (part.mesh.n_cells() == part.mesh.n_owned_cells()) {
      throw autan::Error(autan::ErrorCode::partition_input_has_ghosts, "no ghosts to refuse");
    }
    (void)autan::colour_cells(part.mesh);
  });
}
