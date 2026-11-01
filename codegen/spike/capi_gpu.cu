// C ABI of the GPU spike driver: the generated CUDA loops on device copies of a spike mesh, for
// the tests that compare them with the host loops and check determinism.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "advection_diffusion_gpu.cuh"
#include "advection_gpu.cuh"
#include "autan/kernels/colouring.hpp"
#include "autan/kernels/gpu.hpp"
#include "autan/kernels/norms.cuh"
#include "device.cuh"
#include "gradients_gpu.cuh"
#include "roe_gpu.cuh"
#include "spike.hpp"

namespace {

using autan::kernels::index_t;
using autan::kernels::offset_t;
using autan::spike::DeviceArray;
using autan::spike::DeviceMesh;
using autan::spike::guarded;
using autan::spike::SpikeMesh;
using autan::spike::strategy_of;
namespace gpu = autan::kernels::gpu;

template <class T>
std::vector<T> converted_in(const double* p, std::size_t n) {
  return std::vector<T>(p, p + n);
}

template <class T>
void copy_out(const std::vector<T>& v, double* p) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    p[i] = static_cast<double>(v[i]);
  }
}

void synchronise() { gpu::check(cudaDeviceSynchronize(), "kernel execution"); }

template <class P>
void advdiff_residual(const SpikeMesh& m, gpu::Strategy strategy, int block, const double* q,
                      const double* q_b, const double* source, const double* velocity, double nu,
                      double* res, double* grad, double* norm) {
  namespace ad = autan::kernels::advection_diffusion;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  DeviceMesh dm(m, block);
  DeviceArray<typename P::state> qs(converted_in<typename P::state>(q, nc));
  DeviceArray<typename P::state> qbs(converted_in<typename P::state>(q_b, nf));
  DeviceArray<typename P::state> ss(converted_in<typename P::state>(source, nc));
  DeviceArray<typename P::grad> gs(3 * nc);
  DeviceArray<typename P::residual> rs(nc);
  ad::Data<P> d{};
  d.topo = dm.topo;
  d.q = qs.get();
  d.q_b = qbs.get();
  dm.bind(d);
  d.grad = gs.get();
  d.source = ss.get();
  dm.bind(d);
  d.res = rs.get();
  dm.bind(d);
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = typename P::param(velocity[i]);
  }
  d.nu[0] = typename P::param(nu);
  if (m.topo.stencil != nullptr) {
    ad::gpu_loops::lsq_gradient_stencil(d, dm.schedule);
  } else {
    ad::gpu_loops::lsq_gradient(d, dm.schedule);
  }
  ad::gpu_loops::init_residual(d, dm.schedule);
  ad::gpu_loops::flux(d, strategy, dm.schedule);
  ad::gpu_loops::boundary_flux(d, strategy, dm.schedule);
  DeviceArray<double> partials(static_cast<std::size_t>(gpu::norm_partials(m.topo.n_owned_cells)));
  DeviceArray<double> sum(1);
  gpu::sum_of_squares(rs.get(), m.topo.n_owned_cells, partials.get(), sum.get(), nullptr);
  synchronise();
  copy_out(rs.download(), res);
  copy_out(gs.download(), grad);
  *norm = std::sqrt(sum.download()[0]);
}

template <class P>
void gradients(const SpikeMesh& m, std::string_view method, gpu::Strategy strategy, int block,
               const double* q, const double* q_b, int iterations, double* grad, double* qmin,
               double* qmax) {
  namespace gr = autan::kernels::gradients;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  DeviceMesh dm(m, block);
  DeviceArray<typename P::state> qs(converted_in<typename P::state>(q, nc));
  DeviceArray<typename P::state> qbs(converted_in<typename P::state>(q_b, nf));
  DeviceArray<typename P::grad> gs(converted_in<typename P::grad>(grad, 3 * nc));
  DeviceArray<typename P::state> mins(nc), maxs(nc);
  DeviceArray<typename P::residual> acc(3 * nc);
  gr::Data<P> d{};
  d.topo = dm.topo;
  d.q = qs.get();
  d.q_b = qbs.get();
  d.grad = gs.get();
  d.qmin = mins.get();
  d.qmax = maxs.get();
  d.gg_sum = acc.get();
  dm.bind(d);
  if (method == "lsq") {
    gr::gpu_loops::lsq_gradient_bounds(d, dm.schedule);
  } else {
    for (int it = 0; it < iterations; ++it) {
      gpu::check(cudaMemset(acc.get(), 0, 3 * nc * sizeof(typename P::residual)), "memset");
      gr::gpu_loops::green_gauss_interior(d, strategy, dm.schedule);
      gr::gpu_loops::green_gauss_boundary(d, strategy, dm.schedule);
      gr::gpu_loops::green_gauss_finalise(d, dm.schedule);
    }
  }
  synchronise();
  copy_out(gs.download(), grad);
  copy_out(mins.download(), qmin);
  copy_out(maxs.download(), qmax);
}

template <class P>
void advection(const SpikeMesh& m, gpu::Strategy strategy, int block, const double* q,
               const double* grad, const double* velocity, double* res) {
  namespace av = autan::kernels::advection;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  DeviceMesh dm(m, block);
  DeviceArray<typename P::state> qs(converted_in<typename P::state>(q, nc));
  DeviceArray<typename P::grad> gs(converted_in<typename P::grad>(grad, 3 * nc));
  DeviceArray<typename P::residual> rs(nc);
  av::Data<P> d{};
  d.topo = dm.topo;
  d.q = qs.get();
  d.grad = gs.get();
  d.res = rs.get();
  dm.bind(d);
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = typename P::param(velocity[i]);
  }
  av::gpu_loops::residual(d, strategy, dm.schedule);
  synchronise();
  copy_out(rs.download(), res);
}

template <class P>
void roe(const SpikeMesh& m, gpu::Strategy strategy, int block, int mode, const double* u,
         const double* seed, double* out, double* jac = nullptr) {
  namespace rf = autan::kernels::roe;
  using S = typename P::state;
  using R = typename P::residual;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  DeviceMesh dm(m, block);
  DeviceArray<S> us(converted_in<S>(u, 5 * nc));
  DeviceArray<S> u_tan(converted_in<S>(seed, 5 * nc));
  DeviceArray<R> r_adj(converted_in<R>(seed, 5 * nc));
  DeviceArray<R> rs(5 * nc), r_tan(5 * nc), u_adj(5 * nc);
  DeviceArray<typename P::jac> js(mode == 3 ? 50 * static_cast<std::size_t>(m.topo.n_faces) : 0);
  rf::Data<P> d{};
  d.topo = dm.topo;
  d.face_jac = js.get();
  d.u_cons = us.get();
  d.res = rs.get();
  dm.bind(d);
  d.u_cons_tan = u_tan.get();
  d.res_tan = r_tan.get();
  d.res_adj = r_adj.get();
  d.u_cons_adj = u_adj.get();
  if (mode == 0) {
    rf::gpu_loops::flux(d, strategy, dm.schedule);
  } else if (mode == 3) {
    rf::gpu_loops::flux_and_jacobian(d, strategy, dm.schedule);
  } else if (mode == 1) {
    rf::gpu_loops::flux_tangent(d, strategy, dm.schedule);
  } else {
    rf::gpu_loops::flux_adjoint(d, strategy, dm.schedule);
  }
  synchronise();
  copy_out((mode == 0 || mode == 3 ? rs : mode == 1 ? r_tan : u_adj).download(), out);
  if (jac != nullptr) {
    copy_out(js.download(), jac);
  }
}

/// Adjoint of the composed advection-diffusion residual on the device (seed R-bar -> q-bar).
template <class P>
void advdiff_adjoint(const SpikeMesh& m, gpu::Strategy strategy, int block, const double* q,
                     const double* q_b, const double* source, const double* velocity, double nu,
                     const double* seed, double* out) {
  namespace ad = autan::kernels::advection_diffusion;
  using S = typename P::state;
  using R = typename P::residual;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  DeviceMesh dm(m, block);
  DeviceArray<S> qs(converted_in<S>(q, nc)), qbs(converted_in<S>(q_b, nf)),
      ss(converted_in<S>(source, nc));
  DeviceArray<typename P::grad> gs(3 * nc);
  DeviceArray<R> rs(nc), r_adj(converted_in<R>(seed, nc)), q_adj(nc), g_adj(3 * nc), s_adj(nc);
  ad::Data<P> d{};
  d.topo = dm.topo;
  d.q = qs.get();
  d.q_b = qbs.get();
  dm.bind(d);
  d.grad = gs.get();
  d.source = ss.get();
  dm.bind(d);
  d.res = rs.get();
  dm.bind(d);
  d.res_adj = r_adj.get();
  d.q_adj = q_adj.get();
  d.grad_adj = g_adj.get();
  d.source_adj = s_adj.get();
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = typename P::param(velocity[i]);
  }
  d.nu[0] = typename P::param(nu);
  if (m.topo.stencil != nullptr) {
    ad::gpu_loops::lsq_gradient_stencil(d, dm.schedule);
  } else {
    ad::gpu_loops::lsq_gradient(d, dm.schedule);
  }
  ad::gpu_loops::boundary_flux_adjoint(d, strategy, dm.schedule);
  ad::gpu_loops::flux_adjoint(d, strategy, dm.schedule);
  ad::gpu_loops::init_residual_adjoint(d, dm.schedule);
  if (m.topo.stencil != nullptr) {
    ad::gpu_loops::lsq_gradient_stencil_adjoint(d, strategy, dm.schedule);
  } else {
    ad::gpu_loops::lsq_gradient_adjoint(d, strategy, dm.schedule);
  }
  synchronise();
  copy_out(q_adj.download(), out);
}

/// Adjoint of the LSQ gradient + bounds reduction on the device (seed 5 x n_cells -> q-bar):
/// Sum, Min and Max accumulators and the finalise body.
template <class P>
void lsq_bounds_adjoint(const SpikeMesh& m, gpu::Strategy strategy, int block, const double* q,
                        const double* q_b, const double* seed, double* out) {
  namespace gr = autan::kernels::gradients;
  using S = typename P::state;
  using R = typename P::residual;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  DeviceMesh dm(m, block);
  DeviceArray<S> qs(converted_in<S>(q, nc)), qbs(converted_in<S>(q_b, nf));
  DeviceArray<typename P::grad> gs(3 * nc);
  DeviceArray<S> lo(nc), hi(nc);
  DeviceArray<R> g_adj(converted_in<R>(seed, 3 * nc)), lo_adj(converted_in<R>(seed + 3 * nc, nc)),
      hi_adj(converted_in<R>(seed + 4 * nc, nc)), q_adj(nc);
  gr::Data<P> d{};
  d.topo = dm.topo;
  d.q = qs.get();
  d.q_b = qbs.get();
  d.grad = gs.get();
  d.qmin = lo.get();
  d.qmax = hi.get();
  dm.bind(d);
  d.grad_adj = g_adj.get();
  d.qmin_adj = lo_adj.get();
  d.qmax_adj = hi_adj.get();
  d.q_adj = q_adj.get();
  gr::gpu_loops::lsq_gradient_bounds_adjoint(d, strategy, dm.schedule);
  synchronise();
  copy_out(q_adj.download(), out);
}

}  // namespace

extern "C" {

int autan_spike_gpu_lsq_bounds_adjoint(const void* mesh, const char* policy, const char* strategy,
                                       int block, const double* q, const double* q_b,
                                       const double* seed, double* out, char* err,
                                       std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::with_policy(
        policy, [&](auto p) { lsq_bounds_adjoint<decltype(p)>(*m, s, block, q, q_b, seed, out); });
  });
}

/// Fixed-order FP64 sum of squares of x[0..n) computed on the device.
int autan_spike_gpu_sum_of_squares(const double* x, std::int64_t n, double* out, char* err,
                                   std::size_t err_len) {
  return guarded(err, err_len, [&] {
    DeviceArray<double> dx(x, static_cast<std::size_t>(n));
    DeviceArray<double> partials(static_cast<std::size_t>(gpu::norm_partials(n)));
    DeviceArray<double> sum(1);
    gpu::sum_of_squares(dx.get(), n, partials.get(), sum.get(), nullptr);
    synchronise();
    *out = sum.download()[0];
  });
}

/// Cells per slice of the stencil layout of the following device calls (0: CSR).
void autan_spike_set_stencil_slice(int slice) { autan::spike::stencil_slice() = slice; }

/// Cells per slice of the incidence layout of the following device calls (0: CSR).
void autan_spike_set_incidence_slice(int slice) { autan::spike::incidence_slice() = slice; }

/// Binds a Data-like struct holding built-in face_area (read-only) and a physics' own writable
/// cell_volume: out[0] = 1 if face_area was bound, out[1] = 1 if cell_volume was left alone.
int autan_spike_gpu_bind_check(const void* mesh, int* out, char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    struct Data {
      autan::kernels::Topology topo;
      const double* face_area = nullptr;
      double* cell_volume = nullptr;
    } d;
    double own = 0.0;
    d.cell_volume = &own;
    DeviceMesh dm(*m, 128);
    dm.bind(d);
    out[0] = d.face_area != nullptr ? 1 : 0;
    out[1] = d.cell_volume == &own ? 1 : 0;
  });
}

/// gpu::min of n doubles and of the same values as floats; gpu::first_index of n flags, given
/// as bytes and as doubles.
int autan_spike_gpu_reductions(const double* x, const std::uint8_t* flags, std::int64_t n,
                               double* min64, float* min32, std::int64_t* first,
                               std::int64_t* first64, char* err, std::size_t err_len) {
  return guarded(err, err_len, [&] {
    const auto count = static_cast<std::size_t>(n);
    std::vector<float> xf(x, x + count);
    DeviceArray<double> dx(x, count);
    DeviceArray<float> dxf(xf);
    DeviceArray<std::uint8_t> df(flags, count);
    DeviceArray<double> df64(std::vector<double>(flags, flags + count));
    *min64 = gpu::min(static_cast<index_t>(n), dx.get());
    *min32 = gpu::min(static_cast<index_t>(n), dxf.get());
    *first = gpu::first_index(static_cast<index_t>(n), df.get());
    *first64 = gpu::first_index(static_cast<index_t>(n), df64.get());
  });
}

int autan_spike_gpu_roe_jacobian(const void* mesh, const char* policy, const char* strategy,
                                 int block, const double* u, double* res, double* jac, char* err,
                                 std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::with_policy(policy,
                              [&](auto p) { roe<decltype(p)>(*m, s, block, 3, u, u, res, jac); });
  });
}

int autan_spike_gpu_roe(const void* mesh, const char* policy, const char* strategy, int block,
                        int mode, const double* u, const double* seed, double* out, char* err,
                        std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::with_policy(policy,
                              [&](auto p) { roe<decltype(p)>(*m, s, block, mode, u, seed, out); });
  });
}

int autan_spike_gpu_advdiff_adjoint(const void* mesh, const char* policy, const char* strategy,
                                    int block, const double* q, const double* q_b,
                                    const double* source, const double* velocity, double nu,
                                    const double* seed, double* out, char* err,
                                    std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::with_policy(policy, [&](auto p) {
      advdiff_adjoint<decltype(p)>(*m, s, block, q, q_b, source, velocity, nu, seed, out);
    });
  });
}

int autan_spike_gpu_advdiff_residual(const void* mesh, const char* policy, const char* strategy,
                                     int block, const double* q, const double* q_b,
                                     const double* source, const double* velocity, double nu,
                                     double* res, double* grad, double* norm, char* err,
                                     std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::with_policy(policy, [&](auto p) {
      advdiff_residual<decltype(p)>(*m, s, block, q, q_b, source, velocity, nu, res, grad, norm);
    });
  });
}

int autan_spike_gpu_gradients(const void* mesh, const char* policy, const char* method,
                              const char* strategy, int block, const double* q, const double* q_b,
                              int iterations, double* grad, double* qmin, double* qmax, char* err,
                              std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::with_policy(policy, [&](auto p) {
      gradients<decltype(p)>(*m, method, s, block, q, q_b, iterations, grad, qmin, qmax);
    });
  });
}

int autan_spike_gpu_advection(const void* mesh, const char* policy, const char* strategy, int block,
                              const double* q, const double* grad, const double* velocity,
                              double* res, char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::with_policy(
        policy, [&](auto p) { advection<decltype(p)>(*m, s, block, q, grad, velocity, res); });
  });
}

}  // extern "C"
