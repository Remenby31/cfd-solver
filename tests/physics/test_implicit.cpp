// The approximate Jacobian of the Euler residual and its block Gauss-Seidel solver.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "autan/core/geometry.hpp"
#include "autan/core/renumber.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/partition/partition.hpp"
#include "autan/physics/block_matrix.hpp"
#include "autan/physics/euler.hpp"

namespace {

namespace fs = std::filesystem;
using autan::physics::BlockMatrix;
using autan::physics::BoundaryKind;
using autan::physics::Euler;
using autan::physics::Flux;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
const std::array<const char*, 4> meshes = {"tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh",
                                           "poly/polyMesh", "hybrid/hybrid.ascii.msh"};

std::array<double, 5> conserved(double rho, std::array<double, 3> v, double p) {
  const double q2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  return {rho, rho * v[0], rho * v[1], rho * v[2], p / 0.4 + 0.5 * rho * q2};
}

}  // namespace

TEST_CASE("the assembled Jacobian of the first-order residual matches its central differences",
          "[implicit]") {
  // Without reconstruction the fused face Jacobians are exact: A v = dR/dU v.
  using P = autan::kernels::fp64;
  const char* file = GENERATE(from_range(meshes));
  const Flux flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  INFO(file << " flux " << static_cast<int>(flux));
  const auto mesh = autan::io::read_mesh(mesh_dir / file);
  const auto geometry = autan::compute_geometry(mesh);
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  const double a = std::sqrt(1.4 * 1e5 / 1.2);
  const std::array<double, 5> w = {1.2, 0.6 * a, 0.2 * a, -0.1 * a, 1e5};
  // Every condition, so that every boundary Jacobian is exercised.
  std::vector<BoundaryKind> kind(nb);
  std::vector<double> data(5 * nb);
  const std::array<BoundaryKind, 5> kinds = {BoundaryKind::farfield, BoundaryKind::slip,
                                             BoundaryKind::outlet, BoundaryKind::state,
                                             BoundaryKind::inlet};
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  for (std::size_t b = 0; b < nb; ++b) {
    kind[b] = kinds[b % kinds.size()];
    // An inlet where the flow enters: where it leaves, the inlet's normal velocity is clamped
    // at zero and the fluxes switch upwind side there, a kink central differences straddle.
    const auto f = ni + b;
    const double vn = w[1] * geometry.face_area_x[f] + w[2] * geometry.face_area_y[f] +
                      w[3] * geometry.face_area_z[f];
    if (kind[b] == BoundaryKind::inlet && vn >= 0.0) kind[b] = BoundaryKind::state;
    // Inlet data are total pressure and temperature, above the static ones of w.
    const std::array<double, 5> v =
        kind[b] == BoundaryKind::inlet ? std::array<double, 5>{1.3e5, 320.0, 0.0, 0.0, 0.0} : w;
    for (std::size_t j = 0; j < 5; ++j) data[j * nb + b] = v[j];
  }
  autan::physics::Scheme s;
  s.flux = flux;
  s.second_order = false;
  s.mach_ref = 1.0;  // AUSM+-up's scaling held fixed, away from its kink
  s.reference = conserved(1.1, {10.0, 5.0, 0.0}, 0.95e5);
  Euler<P> e(mesh, geometry, {1.4, 287.0}, s);
  e.set_boundary(kind, data);
  std::vector<double> u(5 * nc), v(5 * nc);
  const auto uc = conserved(w[0], {w[1], w[2], w[3]}, w[4]);
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t c = 0; c < nc; ++c) {
      const double x = geometry.cell_centroid_x[c], y = geometry.cell_centroid_y[c];
      u[k * nc + c] =
          uc[k] * (1.0 + 0.1 * std::sin(3.0 * x + static_cast<double>(k)) * std::cos(2.0 * y));
      v[k * nc + c] = uc[k] * std::cos(5.0 * x + 2.0 * y + static_cast<double>(k));
    }
  }
  std::vector<double> state(5 * nc), res(5 * nc), plain(5 * nc), av(5 * nc), rp(5 * nc), rm(5 * nc);
  e.state(u, state);
  e.residual(state, plain);
  e.residual_and_jacobian(state, res);
  // The fused loops compute the same residual, operation for operation.
  CHECK(res == plain);
  BlockMatrix m(mesh);
  e.assemble(std::vector<double>(nc, 0.0), m);
  m.multiply(v, av);
  const double h = 1e-6;
  std::vector<double> up(u), um(u);
  for (std::size_t i = 0; i < u.size(); ++i) {
    up[i] += h * v[i];
    um[i] -= h * v[i];
  }
  e.state(up, state);
  e.residual(state, rp);
  e.state(um, state);
  e.residual(state, rm);
  double diff = 0.0, norm = 0.0;
  for (std::size_t i = 0; i < u.size(); ++i) {
    REQUIRE(std::isfinite(av[i]));
    REQUIRE(std::isfinite(rp[i] - rm[i]));
    diff = std::max(diff, std::abs((rp[i] - rm[i]) / (2.0 * h) - av[i]));
    norm = std::max(norm, std::abs(av[i]));
  }
  // Central differences: O(h^2) truncation and O(eps / h) round-off, relative.
  CHECK(diff <= 1e-6 * norm);

  // The same blocks in fp32+acc64: FP32 Jacobians, so A v agrees with FP64's to FP32 rounding.
  using Q = autan::kernels::fp32_acc64;
  Euler<Q> e32(mesh, geometry, {1.4, 287.0}, s);
  e32.set_boundary(kind, data);
  std::vector<float> state32(5 * nc);
  std::vector<double> res32(5 * nc), plain32(5 * nc), av32(5 * nc);
  e32.state(u, state32);
  e32.residual(state32, plain32);
  e32.residual_and_jacobian(state32, res32);
  CHECK(res32 == plain32);
  BlockMatrix m32(mesh);
  e32.assemble(std::vector<double>(nc, 0.0), m32);
  m32.multiply(v, av32);
  double diff32 = 0.0;
  for (std::size_t i = 0; i < u.size(); ++i) {
    REQUIRE(std::isfinite(av32[i]));
    diff32 = std::max(diff32, std::abs(av32[i] - av[i]));
  }
  CHECK(diff32 <= 1024 * std::numeric_limits<float>::epsilon() * norm);
}

TEST_CASE("the assembled Jacobian matches its central differences across periodic image ghosts",
          "[implicit]") {
  // The periodic box as one part: each periodic face is an interior face to an image ghost
  // (partitioning.md), whose values the exchange copies from its owned original. A v must equal
  // dR/dU v on every owned row, the rows next to images included, so the blocks coupling a cell
  // to a ghost take the ghost's dW/dU.
  using P = autan::kernels::fp64;
  const auto global = autan::io::read_mesh(mesh_dir / "periodic-box" / "periodic-box.ascii.msh");
  const auto global_geometry = autan::compute_geometry(global);
  const std::vector<std::int32_t> parts(static_cast<std::size_t>(global.n_cells()), 0);
  auto part = autan::extract_partition(global, global_geometry, parts, 1, 0);
  autan::fill_ghost_geometry(part);
  const auto& mesh = part.mesh;
  const auto& geometry = part.geometry;
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  const auto owned = static_cast<std::size_t>(mesh.n_owned_cells());
  REQUIRE(owned < nc);
  struct Images final : autan::physics::GhostFill {
    const autan::LocalPartition& part;
    explicit Images(const autan::LocalPartition& p) : part(p) {}
    void forward(std::span<float* const> c) override {
      autan::forward_self(part.halo, part.mesh.n_owned_cells(), c);
    }
    void forward(std::span<double* const> c) override {
      autan::forward_self(part.halo, part.mesh.n_owned_cells(), c);
    }
  } images(part);
  auto fill = [&](std::vector<double>& x) {
    std::array<double*, 5> c{};
    for (std::size_t k = 0; k < 5; ++k) c[k] = x.data() + k * nc;
    images.forward(std::span<double* const>(c));
  };
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  const double a = std::sqrt(1.4 * 1e5 / 1.2);
  const std::array<double, 5> w = {1.2, 0.6 * a, 0.2 * a, -0.1 * a, 1e5};
  std::vector<BoundaryKind> kind(nb, BoundaryKind::farfield);
  std::vector<double> data(5 * nb);
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t j = 0; j < 5; ++j) data[j * nb + b] = w[j];
  }
  autan::physics::Scheme s;
  s.second_order = false;
  s.mach_ref = 1.0;
  s.reference = conserved(1.1, {10.0, 5.0, 0.0}, 0.95e5);
  Euler<P> e(mesh, geometry, {1.4, 287.0}, s);
  e.set_ghosts(&images);
  e.set_boundary(kind, data);
  // Periodic fields on the owned cells, copied into their images.
  std::vector<double> u(5 * nc), v(5 * nc);
  const auto uc = conserved(w[0], {w[1], w[2], w[3]}, w[4]);
  constexpr double two_pi = 6.283185307179586;
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t c = 0; c < owned; ++c) {
      const double x = geometry.cell_centroid_x[c], y = geometry.cell_centroid_y[c];
      u[k * nc + c] = uc[k] * (1.0 + 0.1 * std::sin(two_pi * x + static_cast<double>(k)) *
                                         std::cos(two_pi * y));
      v[k * nc + c] = uc[k] * std::cos(two_pi * (x + y) + static_cast<double>(k));
    }
  }
  fill(u);
  fill(v);
  std::vector<double> state(5 * nc), res(5 * nc), av(5 * nc), rp(5 * nc), rm(5 * nc);
  e.state(u, state);
  e.residual_and_jacobian(state, res);
  BlockMatrix m(mesh);
  e.assemble(std::vector<double>(nc, 0.0), m);
  m.multiply(v, av);
  const double h = 1e-6;
  std::vector<double> up(u), um(u);
  for (std::size_t i = 0; i < u.size(); ++i) {
    up[i] += h * v[i];
    um[i] -= h * v[i];
  }
  e.state(up, state);
  e.residual(state, rp);
  e.state(um, state);
  e.residual(state, rm);
  double diff = 0.0, norm = 0.0;
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t c = 0; c < owned; ++c) {
      const auto i = k * nc + c;
      REQUIRE(std::isfinite(av[i]));
      REQUIRE(std::isfinite(rp[i] - rm[i]));
      diff = std::max(diff, std::abs((rp[i] - rm[i]) / (2.0 * h) - av[i]));
      norm = std::max(norm, std::abs(av[i]));
    }
  }
  CHECK(diff <= 1e-6 * norm);
}

TEST_CASE("multicolour block Gauss-Seidel solves a block diagonally dominant system",
          "[implicit]") {
  const char* file = GENERATE(from_range(meshes));
  const auto mesh = autan::io::read_mesh(mesh_dir / file);
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  BlockMatrix m(mesh);
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> r(-1.0, 1.0);
  for (std::size_t f = 0; f < ni; ++f) {
    for (std::size_t q = 0; q < 25; ++q) {
      m.upper(f)[q] = r(rng);
      m.lower(f)[q] = r(rng);
    }
  }
  // Each block row sums at most 5 * (faces of the cell) in absolute value off the diagonal.
  std::vector<double> faces(nc, 0.0);
  for (std::size_t f = 0; f < ni; ++f) {
    faces[static_cast<std::size_t>(mesh.owner()[f])] += 1.0;
    faces[static_cast<std::size_t>(mesh.neighbour()[f])] += 1.0;
  }
  for (std::size_t c = 0; c < nc; ++c) {
    for (std::size_t q = 0; q < 25; ++q) m.diagonal(c)[q] = 0.1 * r(rng);
    for (std::size_t i = 0; i < 5; ++i) m.diagonal(c)[6 * i] = 10.0 * faces[c] + 10.0;
  }
  std::vector<double> x_exact(5 * nc), b(5 * nc), x(5 * nc, 0.0), ax(5 * nc);
  for (auto& v : x_exact) v = r(rng);
  m.multiply(x_exact, b);
  m.factor();
  m.gauss_seidel(b, x, 20);
  m.multiply(x, ax);
  double err = 0.0, ref = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    REQUIRE(std::isfinite(x[i]));
    err = std::max(err, std::abs(x[i] - x_exact[i]));
    ref = std::max(ref, std::abs(x_exact[i]));
  }
  CHECK(err <= 1e-12 * ref);
}

TEMPLATE_TEST_CASE("under global keys, renumbered cells have the residual and Jacobian bit for bit",
                   "[implicit]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // solver-input.md, reproducibility "ranks": what a cell sums must not depend on how the mesh is
  // numbered. Renumbering the cells (reverse Cuthill-McKee) re-sorts the faces and flips some,
  // as a partition does; with the global keys of the original mesh the second-order residual,
  // limiter on, and every block of the Jacobian must keep their bits.
  using P = TestType;
  const char* file = GENERATE(from_range(meshes));
  INFO(file);
  const auto a = autan::io::read_mesh(mesh_dir / file);
  const auto ga = autan::compute_geometry(a);
  const auto b = autan::renumber_cells(a, autan::rcm_order(a));
  const auto nc = static_cast<std::size_t>(a.n_cells());
  const auto nf = static_cast<std::size_t>(a.n_faces());
  const auto ni = static_cast<std::size_t>(a.n_interior_faces());
  const auto nb = nf - ni;
  // The face of `a` each face of `b` is, by its nodes, and whether it runs the other way.
  auto nodes = [](const autan::Mesh& m, std::size_t f) {
    const auto n = m.face_nodes(static_cast<autan::index_t>(f));
    std::vector<autan::index_t> v(n.begin(), n.end());
    std::ranges::sort(v);
    return v;
  };
  std::map<std::vector<autan::index_t>, std::size_t> face_of;
  for (std::size_t f = 0; f < nf; ++f) face_of[nodes(a, f)] = f;
  const auto gid = b.cell_global_id();
  auto cell = [&](std::size_t c) { return static_cast<std::size_t>(gid[c]); };
  std::vector<std::int64_t> face_a(nf), face_id(nf);
  std::vector<std::uint8_t> reversed(nf, 0), not_reversed(nf, 0);
  std::size_t n_reversed = 0;
  for (std::size_t f = 0; f < nf; ++f) {
    face_id[f] = static_cast<std::int64_t>(f);
    const auto fa = face_of.at(nodes(b, f));
    face_a[f] = static_cast<std::int64_t>(fa);
    reversed[f] = static_cast<std::uint8_t>(cell(static_cast<std::size_t>(b.owner()[f])) !=
                                            static_cast<std::size_t>(a.owner()[fa]));
    n_reversed += reversed[f];
  }
  REQUIRE(n_reversed > 0);
  // The geometry of `a`, carried over as a partition carries it: area vectors negated where the
  // face runs the other way.
  autan::Geometry gb = autan::compute_geometry(b);
  for (std::size_t f = 0; f < nf; ++f) {
    const auto fa = static_cast<std::size_t>(face_a[f]);
    const double sign = reversed[f] != 0 ? -1.0 : 1.0;
    gb.face_area_x[f] = sign * ga.face_area_x[fa];
    gb.face_area_y[f] = sign * ga.face_area_y[fa];
    gb.face_area_z[f] = sign * ga.face_area_z[fa];
    gb.face_area[f] = ga.face_area[fa];
    gb.face_centroid_x[f] = ga.face_centroid_x[fa];
    gb.face_centroid_y[f] = ga.face_centroid_y[fa];
    gb.face_centroid_z[f] = ga.face_centroid_z[fa];
  }
  for (std::size_t c = 0; c < nc; ++c) {
    gb.cell_volume[c] = ga.cell_volume[cell(c)];
    gb.cell_centroid_x[c] = ga.cell_centroid_x[cell(c)];
    gb.cell_centroid_y[c] = ga.cell_centroid_y[cell(c)];
    gb.cell_centroid_z[c] = ga.cell_centroid_z[cell(c)];
  }
  const autan::kernels::GlobalKeys keys_a{
      .cell = a.cell_global_id(), .face = face_id, .reversed = not_reversed, .shift = {}};
  const autan::kernels::GlobalKeys keys_b{
      .cell = gid, .face = face_a, .reversed = reversed, .shift = {}};

  const double sound = std::sqrt(1.4 * 1e5 / 1.2);
  const std::array<double, 5> w = {1.2, 0.6 * sound, 0.2 * sound, -0.1 * sound, 1e5};
  const std::array<BoundaryKind, 4> kinds = {BoundaryKind::farfield, BoundaryKind::slip,
                                             BoundaryKind::outlet, BoundaryKind::state};
  std::vector<BoundaryKind> kind_a(nb), kind_b(nb);
  std::vector<double> data_a(5 * nb), data_b(5 * nb);
  for (std::size_t k = 0; k < nb; ++k) {
    kind_a[k] = kinds[k % kinds.size()];
    for (std::size_t j = 0; j < 5; ++j) data_a[j * nb + k] = w[j];
  }
  for (std::size_t f = ni; f < nf; ++f) {
    const auto k = static_cast<std::size_t>(face_a[f]) - ni;
    kind_b[f - ni] = kind_a[k];
    for (std::size_t j = 0; j < 5; ++j) data_b[j * nb + f - ni] = data_a[j * nb + k];
  }
  autan::physics::Scheme s;
  s.flux = Flux::roe;
  s.second_order = true;
  s.limiter = true;
  s.reference = conserved(1.1, {10.0, 5.0, 0.0}, 0.95e5);
  Euler<P> ea(a, ga, {1.4, 287.0}, s, &keys_a);
  Euler<P> eb(b, gb, {1.4, 287.0}, s, &keys_b);
  ea.set_boundary(kind_a, data_a);
  eb.set_boundary(kind_b, data_b);
  const auto uc = conserved(w[0], {w[1], w[2], w[3]}, w[4]);
  std::vector<double> ua(5 * nc), ub(5 * nc), shift_a(nc), shift_b(nc);
  for (std::size_t c = 0; c < nc; ++c) {
    const double x = ga.cell_centroid_x[c], y = ga.cell_centroid_y[c], z = ga.cell_centroid_z[c];
    for (std::size_t k = 0; k < 5; ++k) {
      ua[k * nc + c] =
          uc[k] * (1.0 + 0.1 * std::sin(3.0 * x + static_cast<double>(k)) * std::cos(2.0 * y - z));
    }
    shift_a[c] = 1.0 + x;
  }
  for (std::size_t c = 0; c < nc; ++c) {
    for (std::size_t k = 0; k < 5; ++k) ub[k * nc + c] = ua[k * nc + cell(c)];
    shift_b[c] = shift_a[cell(c)];
  }
  std::vector<typename P::state> sa(5 * nc), sb(5 * nc);
  std::vector<typename P::residual> ra(5 * nc), rb(5 * nc);
  ea.state(ua, sa);
  eb.state(ub, sb);
  ea.residual_and_jacobian(sa, ra);
  eb.residual_and_jacobian(sb, rb);
  // Bit patterns, so that a signed zero or a NaN cannot pass for an equal value.
  auto bits_differ = [](auto x, auto y) {
    static_assert(sizeof(x) == sizeof(y));
    return std::memcmp(&x, &y, sizeof(x)) != 0;
  };
  REQUIRE(std::ranges::all_of(ra, [](auto x) { return std::isfinite(static_cast<double>(x)); }));
  std::size_t residual_differs = 0;
  for (std::size_t c = 0; c < nc; ++c) {
    for (std::size_t k = 0; k < 5; ++k) {
      residual_differs += bits_differ(rb[k * nc + c], ra[k * nc + cell(c)]);
    }
  }
  CHECK(residual_differs == 0);
  BlockMatrix ma(a), mb(b);
  ea.assemble(shift_a, ma);
  eb.assemble(shift_b, mb);
  REQUIRE(std::ranges::all_of(ma.diagonals(), [](double x) { return std::isfinite(x); }));
  REQUIRE(std::ranges::all_of(ma.uppers(), [](double x) { return std::isfinite(x); }));
  REQUIRE(std::ranges::all_of(ma.lowers(), [](double x) { return std::isfinite(x); }));
  auto same = [](const double* x, const double* y) {
    return std::memcmp(x, y, 25 * sizeof(double)) == 0;
  };
  std::size_t blocks_differ = 0;
  for (std::size_t c = 0; c < nc; ++c) {
    blocks_differ += !same(mb.diagonal(c), ma.diagonal(cell(c)));
  }
  // A(owner, neighbour) of `b` is upper or lower of `a`, as `a` orients the face.
  for (std::size_t f = 0; f < ni; ++f) {
    const auto fa = static_cast<std::size_t>(face_a[f]);
    blocks_differ += !same(mb.upper(f), reversed[f] != 0 ? ma.lower(fa) : ma.upper(fa));
    blocks_differ += !same(mb.lower(f), reversed[f] != 0 ? ma.upper(fa) : ma.lower(fa));
  }
  CHECK(blocks_differ == 0);
}
