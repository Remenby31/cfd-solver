#include "autan/physics/euler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
#include "euler_cpu.hpp"

namespace autan::physics {

namespace ek = autan::kernels::euler;

template <class P>
struct Euler<P>::Impl {
  using Gr = typename P::grad;

  Scheme scheme;
  kernels::Topology topo{};
  // Each geometry array has the element type its Data<P> member stores (the generator stores
  // geometry that bodies read only as values in P::geom_value).
  template <class Ptr>
  using stored = std::vector<std::remove_cvref_t<std::remove_pointer_t<Ptr>>>;
  using D = ek::Data<P>;
  stored<decltype(D::face_normal)> face_normal;
  stored<decltype(D::face_area_magnitude)> face_area_magnitude;
  stored<decltype(D::face_centroid)> face_centroid;
  stored<decltype(D::face_moment)> face_moment;
  stored<decltype(D::cell_centroid)> cell_centroid;
  stored<decltype(D::cell_volume)> cell_volume;
  stored<decltype(D::stencil_lsq_weight)> stencil_lsq_weight;
  kernels::CellStencil stencil;
  std::vector<double> bc_kind;
  std::vector<State> bc, prim_b, prim, wmin, wmax, phi, source, prim_tan, zero_state;
  std::vector<typename P::jac> face_jac, bnd_jac, dwdu;
  std::vector<Gr> grad, zero_grad;
  ek::Data<P> d{};
  GhostFill* ghosts = nullptr;
  // Under global keys: each cell's faces in global order, their twins and the faces that run
  // against their global orientation; face loops then gather (no scatter order).
  kernels::GlobalOrder order;
  std::vector<std::uint8_t> reversed;
  bool gather = false;

  // Fills the ghost values of `n` components of an array laid out c * n_cells + i.
  // The exchange takes at most 16 components a call.
  template <class T>
  void fill_ghosts(std::vector<T>& v, std::size_t n) const {
    if (ghosts == nullptr) return;
    const auto nc = static_cast<std::size_t>(topo.n_cells);
    for (std::size_t first = 0; first < n; first += 16) {
      const std::size_t count = std::min<std::size_t>(16, n - first);
      std::array<T*, 16> components{};
      for (std::size_t k = 0; k < count; ++k) components[k] = v.data() + (first + k) * nc;
      ghosts->forward(std::span<T* const>(components.data(), count));
    }
  }

  // The loops before the faces: primitives (and their Jacobian dW/dU, which the coupling blocks
  // of faces to ghost cells take), boundary states, gradients and limiter of the owned cells,
  // each filled into the ghosts before a later loop or the assembly reads it there.
  void cell_loops(bool jacobian) {
    if (jacobian) {
      ek::cpu::primitives_jacobian(d);
      fill_ghosts(dwdu, 25);
    } else {
      ek::cpu::primitives(d);
    }
    fill_ghosts(prim, 5);
    ek::cpu::boundary_states(d);
    if (scheme.second_order) {
      ek::cpu::gradients(d);
      fill_ghosts(grad, 15);
      if (scheme.limiter) {
        ek::cpu::limiter(d);
        fill_ghosts(phi, 5);
      }
    }
    ek::cpu::sources(d);
  }

  template <class T, class Source>
  static void fill(std::vector<T>& to, const Source& from) {
    to = kernels::converted<T>(from);
  }

  Impl(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme s,
       const kernels::GlobalKeys* keys)
      : scheme(s) {
    const auto packed = kernels::pack_geometry(mesh, geometry, keys);
    topo = kernels::topology(mesh.view());
    stencil =
        kernels::cell_stencil(mesh, packed,
                              s.gradient == GradientStencil::vertex ? kernels::StencilKind::vertex
                                                                    : kernels::StencilKind::face2,
                              keys);
    if (keys != nullptr) {
      order = kernels::global_order(mesh, *keys);
      reversed.assign(keys->reversed.begin(), keys->reversed.end());
      topo.cell_faces = order.cell_faces.data();
      topo.incidence_twin = order.twin.data();
      topo.face_reversed = reversed.data();
      topo.rank_independent = true;
      gather = true;
    }
    topo.n_stencil = static_cast<kernels::offset_t>(stencil.entries.size());
    topo.stencil_offsets = stencil.offsets.data();
    topo.stencil = stencil.entries.data();
    topo.stencil_twin = stencil.twin.data();
    fill(stencil_lsq_weight, stencil.lsq_weight);
    fill(face_normal, packed.face_normal);
    fill(face_area_magnitude, packed.face_area_magnitude);
    fill(face_centroid, packed.face_centroid);
    fill(face_moment, packed.face_moment);
    fill(cell_centroid, packed.cell_centroid);
    fill(cell_volume, packed.cell_volume);
    const auto nc = static_cast<std::size_t>(topo.n_cells);
    const auto nf = static_cast<std::size_t>(topo.n_faces);
    bc_kind.assign(nf, 0.0);
    bc.assign(5 * nf, State(0));
    prim_b.assign(5 * nf, State(0));
    prim.assign(5 * nc, State(0));
    wmin.assign(5 * nc, State(0));
    wmax.assign(5 * nc, State(0));
    phi.assign(5 * nc, State(scheme.second_order && !scheme.limiter ? 1 : 0));
    source.assign(5 * nc, State(0));
    prim_tan.assign(5 * nc, State(0));
    zero_state.assign(5 * nc, State(0));
    grad.assign(15 * nc, Gr(0));
    zero_grad.assign(15 * nc, Gr(0));

    d.topo = topo;
    d.prim = prim.data();
    d.grad = grad.data();
    d.wmin = wmin.data();
    d.wmax = wmax.data();
    d.phi = phi.data();
    d.source = source.data();
    d.prim_b = prim_b.data();
    d.bc_kind = bc_kind.data();
    d.bc = bc.data();
    d.face_normal = face_normal.data();
    d.face_area_magnitude = face_area_magnitude.data();
    d.face_centroid = face_centroid.data();
    d.face_moment = face_moment.data();
    d.cell_centroid = cell_centroid.data();
    d.cell_volume = cell_volume.data();
    d.stencil_lsq_weight = stencil_lsq_weight.data();
    d.gamma[0] = typename P::param(gas.gamma);
    d.gas_constant[0] = typename P::param(gas.gas_constant);
    if (!(scheme.mach_ref > 0.0)) {
      throw std::invalid_argument("AUSM+-up needs a positive cut-off Mach number");
    }
    d.mach_ref[0] = typename P::param(scheme.mach_ref);
    const auto& r = scheme.reference;
    if (!(r[0] > 0.0)) throw std::invalid_argument("reference density must be positive");
    for (std::size_t c = 0; c < 5; ++c) d.u_ref[c] = typename P::param(r[c]);
    d.p_ref[0] = typename P::param((gas.gamma - 1.0) *
                                   (r[4] - 0.5 * (r[1] * r[1] + r[2] * r[2] + r[3] * r[3]) / r[0]));
    // Venkatakrishnan (1995), eq. 5, threshold eps^2 = (K h)^3 made dimensional:
    // eps_k^2 = (K / L)^3 V q_k^2.
    const double k3 = std::pow(scheme.venkat_k / scheme.length, 3);
    for (std::size_t c = 0; c < 5; ++c) {
      d.limiter_scale[c] = typename P::param(k3 * scheme.scale[c] * scheme.scale[c]);
    }
  }

  // Least-squares weights over the stencil, 1/|d|^2 weighting as kernels::cell_stencil, but
  // with only the boundary faces whose condition gives the state there: elsewhere the boundary
  // value is the adjacent cell's, which sits at the cell centroid and not at the face, and would
  // make the gradient inconsistent next to slip walls and extrapolating boundaries. A cell whose
  // remaining entries do not determine a gradient keeps every boundary face.
  void compute_gradient_weights(std::span<const BoundaryKind> kind) {
    const auto nc = static_cast<std::size_t>(topo.n_cells);
    const auto nf = static_cast<std::size_t>(topo.n_faces);
    const auto ni = static_cast<std::size_t>(topo.n_interior_faces);
    const auto nst = stencil.entries.size();
    std::vector<double> w(3 * nst, 0.0);
    constexpr double coordinate_eps =
        std::numeric_limits<typename decltype(cell_centroid)::value_type>::epsilon();
    for (std::size_t c = 0; c + 1 < stencil.offsets.size(); ++c) {
      const auto begin = static_cast<std::size_t>(stencil.offsets[c]);
      const auto end = static_cast<std::size_t>(stencil.offsets[c + 1]);
      auto weights = [&](bool all_boundaries) {
        std::array<double, 9> a{};
        std::vector<std::array<double, 4>> offs;  // offset and weight per entry, 0 if excluded
        for (std::size_t k = begin; k < end; ++k) {
          const index_t e = stencil.entries[k];
          std::array<double, 3> x{};
          bool used = true;
          if (e >= 0) {
            for (std::size_t j = 0; j < 3; ++j) {
              x[j] = static_cast<double>(cell_centroid[j * nc + static_cast<std::size_t>(e)]);
            }
          } else {
            const auto f = static_cast<std::size_t>(-e - 1);
            used = all_boundaries || kind[f - ni] == BoundaryKind::state;
            for (std::size_t j = 0; j < 3; ++j) {
              x[j] = static_cast<double>(face_centroid[j * nf + f]);
            }
          }
          // Offsets within 64 ulps of the coordinates are round-off, not geometry: across a
          // one-cell-thick slab the centroids differ only by them, and an offset kept at 1e-17
          // would let the fit invert round-off into a gradient across the layer.
          std::array<double, 4> dk{};
          for (std::size_t j = 0; j < 3; ++j) {
            const double xc = static_cast<double>(cell_centroid[j * nc + c]);
            const double round_off = 64 * coordinate_eps * std::max(std::abs(x[j]), std::abs(xc));
            dk[j] = x[j] - xc;
            if (std::abs(dk[j]) <= round_off) dk[j] = 0.0;
          }
          dk[3] = used ? 1.0 / (dk[0] * dk[0] + dk[1] * dk[1] + dk[2] * dk[2]) : 0.0;
          for (std::size_t u = 0; u < 3; ++u) {
            for (std::size_t v = 0; v < 3; ++v) a[3 * u + v] += dk[3] * dk[u] * dk[v];
          }
          offs.push_back(dk);
        }
        const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                           a[2] * (a[3] * a[7] - a[4] * a[6]);
        const double scale = a[0] * a[4] * a[8];
        if (!(std::abs(det) > 1e-10 * std::abs(scale))) return false;
        const std::array<double, 9> inv = {
            (a[4] * a[8] - a[5] * a[7]) / det, (a[2] * a[7] - a[1] * a[8]) / det,
            (a[1] * a[5] - a[2] * a[4]) / det, (a[5] * a[6] - a[3] * a[8]) / det,
            (a[0] * a[8] - a[2] * a[6]) / det, (a[2] * a[3] - a[0] * a[5]) / det,
            (a[3] * a[7] - a[4] * a[6]) / det, (a[1] * a[6] - a[0] * a[7]) / det,
            (a[0] * a[4] - a[1] * a[3]) / det};
        for (std::size_t k = begin; k < end; ++k) {
          const auto& dk = offs[k - begin];
          for (std::size_t u = 0; u < 3; ++u) {
            w[u * nst + k] =
                dk[3] * (inv[3 * u] * dk[0] + inv[3 * u + 1] * dk[1] + inv[3 * u + 2] * dk[2]);
          }
        }
        return true;
      };
      if (!weights(false)) weights(true);
    }
    fill(stencil_lsq_weight, w);
    d.stencil_lsq_weight = stencil_lsq_weight.data();
  }

  // A face or boundary loop: its gather under global keys, else its scatter.
  template <class Loop>
  void run(Loop loop) const {
    if (gather) {
      loop(kernels::Strategy::store_gather);
    } else {
      loop();
    }
  }

  void faces() {
    switch (scheme.flux) {
      case Flux::roe:
        run([&](auto... strategy) { ek::cpu::roe_faces(d, strategy...); });
        run([&](auto... strategy) { ek::cpu::roe_boundaries(d, strategy...); });
        break;
      case Flux::hllc:
        run([&](auto... strategy) { ek::cpu::hllc_faces(d, strategy...); });
        run([&](auto... strategy) { ek::cpu::hllc_boundaries(d, strategy...); });
        break;
      case Flux::ausm_up:
        run([&](auto... strategy) { ek::cpu::ausm_faces(d, strategy...); });
        run([&](auto... strategy) { ek::cpu::ausm_boundaries(d, strategy...); });
        break;
    }
  }

  void faces_jacobian() {
    switch (scheme.flux) {
      case Flux::roe:
        run([&](auto... strategy) { ek::cpu::roe_faces_jacobian(d, strategy...); });
        run([&](auto... strategy) { ek::cpu::roe_boundaries_jacobian(d, strategy...); });
        break;
      case Flux::hllc:
        run([&](auto... strategy) { ek::cpu::hllc_faces_jacobian(d, strategy...); });
        run([&](auto... strategy) { ek::cpu::hllc_boundaries_jacobian(d, strategy...); });
        break;
      case Flux::ausm_up:
        run([&](auto... strategy) { ek::cpu::ausm_faces_jacobian(d, strategy...); });
        run([&](auto... strategy) { ek::cpu::ausm_boundaries_jacobian(d, strategy...); });
        break;
    }
  }

  void faces_tangent() {
    switch (scheme.flux) {
      case Flux::roe:
        run([&](auto... strategy) { ek::cpu::roe_faces_tangent(d, strategy...); });
        run([&](auto... strategy) { ek::cpu::roe_boundaries_tangent(d, strategy...); });
        break;
      case Flux::hllc:
        run([&](auto... strategy) { ek::cpu::hllc_faces_tangent(d, strategy...); });
        run([&](auto... strategy) { ek::cpu::hllc_boundaries_tangent(d, strategy...); });
        break;
      case Flux::ausm_up:
        run([&](auto... strategy) { ek::cpu::ausm_faces_tangent(d, strategy...); });
        run([&](auto... strategy) { ek::cpu::ausm_boundaries_tangent(d, strategy...); });
        break;
    }
  }
};

template <class P>
Euler<P>::Euler(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme scheme,
                const kernels::GlobalKeys* keys)
    : impl_(std::make_unique<Impl>(mesh, geometry, gas, scheme, keys)) {}

template <class P>
Euler<P>::~Euler() = default;

template <class P>
index_t Euler<P>::n_cells() const noexcept {
  return impl_->topo.n_cells;
}

template <class P>
index_t Euler<P>::n_boundary_faces() const noexcept {
  return impl_->topo.n_faces - impl_->topo.n_interior_faces;
}

template <class P>
void Euler<P>::set_boundary(std::span<const BoundaryKind> kind, std::span<const double> data) {
  auto& m = *impl_;
  const auto nb = static_cast<std::size_t>(n_boundary_faces());
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  const auto ni = static_cast<std::size_t>(m.topo.n_interior_faces);
  if (kind.size() != nb || data.size() != 5 * nb) {
    throw std::invalid_argument("boundary arrays do not match the boundary faces");
  }
  for (std::size_t b = 0; b < nb; ++b) {
    m.bc_kind[ni + b] = static_cast<double>(kind[b]);
    for (std::size_t c = 0; c < 5; ++c) {
      m.bc[c * nf + ni + b] = State(data[c * nb + b]);
    }
  }
  m.compute_gradient_weights(kind);
}

template <class P>
void Euler<P>::set_boundary_data(std::span<const double> data) {
  auto& m = *impl_;
  const auto nb = static_cast<std::size_t>(n_boundary_faces());
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  const auto ni = static_cast<std::size_t>(m.topo.n_interior_faces);
  if (data.size() != 5 * nb) {
    throw std::invalid_argument("boundary data does not match the boundary faces");
  }
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t c = 0; c < 5; ++c) m.bc[c * nf + ni + b] = State(data[c * nb + b]);
  }
}

template <class P>
std::vector<double> Euler<P>::gradient_weights() const {
  const auto& w = impl_->stencil_lsq_weight;
  return {w.begin(), w.end()};
}

template <class P>
void Euler<P>::set_second_order(bool on) {
  auto& m = *impl_;
  m.scheme.second_order = on;
  if (!on) {
    std::fill(m.grad.begin(), m.grad.end(), typename P::grad(0));
    std::fill(m.phi.begin(), m.phi.end(), State(0));
  } else if (!m.scheme.limiter) {
    std::fill(m.phi.begin(), m.phi.end(), State(1));
  }
}

template <class P>
void Euler<P>::set_source(std::span<const double> source) {
  auto& m = *impl_;
  if (source.size() != m.source.size()) {
    throw std::invalid_argument("source array does not match the cells");
  }
  for (std::size_t i = 0; i < source.size(); ++i) {
    m.source[i] = State(source[i]);
  }
}

template <class P>
void Euler<P>::state(std::span<const double> u, std::span<State> s) const {
  const auto n = static_cast<std::size_t>(n_cells());
  const auto& r = impl_->scheme.reference;
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t c = 0; c < n; ++c) s[k * n + c] = State(u[k * n + c] - r[k]);
  }
}

template <class P>
void Euler<P>::residual(std::span<const State> u, std::span<Res> res) {
  auto& m = *impl_;
  m.d.u = const_cast<State*>(u.data());
  m.d.res = res.data();
  m.cell_loops(false);
  m.faces();
}

template <class P>
void Euler<P>::residual_and_jacobian(std::span<const State> s, std::span<Res> res) {
  auto& m = *impl_;
  if (m.dwdu.empty()) {
    // Allocated on first use: residual-only users (explicit, unsteady) carry no Jacobian.
    const auto nc = static_cast<std::size_t>(m.topo.n_cells);
    const auto nf = static_cast<std::size_t>(m.topo.n_faces);
    m.face_jac.assign(50 * nf, typename P::jac(0));
    m.bnd_jac.assign(25 * nf, typename P::jac(0));
    m.dwdu.assign(25 * nc, typename P::jac(0));
    m.d.face_jac = m.face_jac.data();
    m.d.bnd_jac = m.bnd_jac.data();
    m.d.dwdu = m.dwdu.data();
  }
  m.d.u = const_cast<State*>(s.data());
  m.d.res = res.data();
  m.cell_loops(true);
  m.faces_jacobian();
}

template <class P>
void Euler<P>::assemble(std::span<const double> shift, BlockMatrix& a) const {
  const auto& m = *impl_;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  const auto ni = static_cast<std::size_t>(m.topo.n_interior_faces);
  // Block k of a structure-of-arrays Jac dataset over n entities: entry (i, j) of entity e at
  // (k * 25 + i * 5 + j) * n + e.
  auto block = [](const std::vector<typename P::jac>& v, std::size_t n, std::size_t e,
                  std::size_t k) {
    std::array<double, 25> b{};
    for (std::size_t q = 0; q < 25; ++q) b[q] = static_cast<double>(v[(25 * k + q) * n + e]);
    return b;
  };
  // out += sign * J D.
  auto add_product = [](double* out, const std::array<double, 25>& j,
                        const std::array<double, 25>& dm, double sign) {
    for (std::size_t r = 0; r < 5; ++r) {
      for (std::size_t c = 0; c < 5; ++c) {
        double t = 0.0;
        for (std::size_t k = 0; k < 5; ++k) t += j[5 * r + k] * dm[5 * k + c];
        out[5 * r + c] += sign * t;
      }
    }
  };
  // The face loops evaluate each face in its global orientation (Topology::face_reversed):
  // face_jac holds dF/dW of its global left (block 0) and right (block 1) cells, with
  // R_left += F, R_right -= F. The blocks are placed in the local orientation of BlockMatrix.
  auto left = [&](std::size_t f) {
    const bool rev = m.topo.face_reversed != nullptr && m.topo.face_reversed[f] != 0;
    return static_cast<std::size_t>(rev ? m.topo.neighbour[f] : m.topo.owner[f]);
  };
  a.zero();
  // Each diagonal block gathers its cell's faces in Topology::cell_faces order (global order
  // under global keys), as the device gather does, so that both sum in the same order.
  for (std::size_t c = 0; c < nc; ++c) {
    double* dg = a.diagonal(c);
    for (std::size_t i = 0; i < 5; ++i) dg[6 * i] += shift[c];
    const auto dc = block(m.dwdu, nc, c, 0);
    for (auto k = m.topo.cell_face_offsets[c]; k < m.topo.cell_face_offsets[c + 1]; ++k) {
      const auto f = static_cast<std::size_t>(m.topo.cell_faces[k]);
      if (f >= ni) {
        add_product(dg, block(m.bnd_jac, nf, f, 0), dc, 1.0);
      } else if (left(f) == c) {
        add_product(dg, block(m.face_jac, nf, f, 0), dc, 1.0);
      } else {
        add_product(dg, block(m.face_jac, nf, f, 1), dc, -1.0);
      }
    }
  }
  for (std::size_t f = 0; f < ni; ++f) {
    const auto o = static_cast<std::size_t>(m.topo.owner[f]);
    const auto n = static_cast<std::size_t>(m.topo.neighbour[f]);
    // A(left, right) = +J_right D_right and A(right, left) = -J_left D_left.
    const bool rev = left(f) != o;
    add_product(rev ? a.lower(f) : a.upper(f), block(m.face_jac, nf, f, 1),
                block(m.dwdu, nc, rev ? o : n, 0), 1.0);
    add_product(rev ? a.upper(f) : a.lower(f), block(m.face_jac, nf, f, 0),
                block(m.dwdu, nc, rev ? n : o, 0), -1.0);
  }
}

template <class P>
void Euler<P>::first_order_tangent(std::span<const State> u, std::span<const State> du,
                                   std::span<Res> dres) {
  auto& m = *impl_;
  auto d = m.d;
  d.u = const_cast<State*>(u.data());
  d.grad = m.zero_grad.data();
  d.phi = m.zero_state.data();
  ek::cpu::primitives(d);
  d.u_tan = const_cast<State*>(du.data());
  d.prim_tan = m.prim_tan.data();
  d.grad_tan = m.zero_grad.data();
  d.phi_tan = m.zero_state.data();
  d.source_tan = m.zero_state.data();
  d.res_tan = dres.data();
  ek::cpu::primitives_tangent(d);
  ek::cpu::sources_tangent(d);
  // The first-order fluxes use the primitives only: grad and phi are zero above.
  auto saved = m.d;
  m.d = d;
  m.faces_tangent();
  m.d = saved;
}

template <class P>
const kernels::euler::Data<P>& Euler<P>::data() const {
  return impl_->d;
}

template <class P>
const kernels::CellStencil& Euler<P>::stencil() const {
  return impl_->stencil;
}

template <class P>
const Scheme& Euler<P>::scheme() const {
  return impl_->scheme;
}

template <class P>
void Euler<P>::set_ghosts(GhostFill* ghosts) noexcept {
  impl_->ghosts = ghosts;
}

template <class P>
void Euler<P>::spectral_radius(std::span<State> lambda) {
  auto& m = *impl_;
  m.d.lam = lambda.data();
  ek::cpu::spectral_radius(m.d);
}

template <class P>
void Euler<P>::boundary_pressure(std::span<double> p, std::span<double> force) const {
  // The face state of the boundary flux (reconstruct in kernels/euler.py, with its fallback to
  // the cell value) and, for the force, the pressure correction of warped faces (wall_warp).
  const auto& m = *impl_;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  const auto ni = static_cast<std::size_t>(m.topo.n_interior_faces);
  const auto nb = nf - ni;
  // grad is 5 x 3 per cell, row-major: component (i, j) at (3 i + j) * n_cells + c.
  auto g = [&](std::size_t i, std::size_t j, std::size_t c) {
    return static_cast<double>(m.grad[(3 * i + j) * nc + c]);
  };
  for (std::size_t f = ni; f < nf; ++f) {
    const auto c = static_cast<std::size_t>(m.topo.owner[f]);
    double drho = 0.0, dp = 0.0;
    for (std::size_t j = 0; j < 3; ++j) {
      const double r = static_cast<double>(m.face_centroid[j * nf + f]) -
                       static_cast<double>(m.cell_centroid[j * nc + c]);
      drho += g(0, j, c) * r;
      dp += g(4, j, c) * r;
    }
    const double rho_c = static_cast<double>(m.prim[c]);
    // prim[4] is the gauge pressure p - p_ref.
    const double p_c = static_cast<double>(m.d.p_ref[0]) + static_cast<double>(m.prim[4 * nc + c]);
    const double rho_f = rho_c + static_cast<double>(m.phi[c]) * drho;
    double p_f = p_c + static_cast<double>(m.phi[4 * nc + c]) * dp;
    if (!(rho_f > 0.0 && p_f > 0.0)) p_f = p_c;
    p[f - ni] = p_f;
    if (force.empty()) continue;
    const double phi_p = static_cast<double>(m.phi[4 * nc + c]);
    for (std::size_t k = 0; k < 3; ++k) {
      // p S + phi_p M^T grad p, M row-major 3 x 3 per face: entry (j, k) at (3 j + k) * nf + f.
      double warp = 0.0;
      for (std::size_t j = 0; j < 3; ++j) {
        warp += static_cast<double>(m.face_moment[(3 * j + k) * nf + f]) * g(4, j, c);
      }
      const double area_k = static_cast<double>(m.face_normal[k * nf + f]) *
                            static_cast<double>(m.face_area_magnitude[f]);
      force[k * nb + (f - ni)] = p_f * area_k + phi_p * warp;
    }
  }
}

template class Euler<kernels::fp64>;
template class Euler<kernels::fp32_acc64>;

}  // namespace autan::physics
