// Times the generated loops of the spike on one mesh and prints the raw JSON that autan-bench
// reads (one entry per loop, strategy and policy; bytes_per_rep is the loop's analytic minimum).
//
//   loop_bench MESH [--policy fp32|fp32+acc64|fp64] [--order input|rcm] [--block N] [--gate 1]
//   --gate 1: after the host setup, print `autan-bench: ready` on stderr and wait for a line on
//   stdin before any CUDA call (autan-bench run --ready takes the GPU lock only then).
//              [--warmup N] [--reps N] [--batch N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "advection_cpu.hpp"
#include "advection_gpu.cuh"
#include "autan/core/renumber.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "blocks_cpu.hpp"
#include "blocks_gpu.cuh"
#include "device.cuh"
#include "diagonal_only_gpu.cuh"
#include "gradients_cpu.hpp"
#include "gradients_gpu.cuh"
#include "revision.h"
#include "roe_gpu.cuh"
#include "spike.hpp"

namespace {

using autan::kernels::index_t;
using autan::kernels::offset_t;
using autan::spike::DeviceArray;
using autan::spike::DeviceMesh;
using autan::spike::GeomCopy;
using autan::spike::SpikeMesh;
namespace gpu = autan::kernels::gpu;

struct Options {
  std::string mesh;
  std::string policy = "fp32+acc64";
  std::string order = "rcm";
  int block = 256;
  bool gate = false;
  int warmup = 5;
  int reps = 30;
  int batch = 5;
};

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(2);
}

/// With --gate 1 (autan-bench run --ready): the host setup is done, so announce it on stderr and
/// wait for autan-bench to take the GPU lock and say go on stdin. No CUDA call comes before this,
/// so the waiting process holds no device context.
void wait_for_go(bool gate) {
  if (!gate) {
    return;
  }
  std::fprintf(stderr, "autan-bench: ready\n");
  std::fflush(stderr);
  std::string line;
  if (!std::getline(std::cin, line)) {
    fail("--gate: stdin closed before go");
  }
}

Options parse(int argc, char** argv) {
  Options o;
  if (argc < 2) fail("usage: loop_bench MESH [options]");
  o.mesh = argv[1];
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string_view f(argv[i]);
    const std::string v(argv[i + 1]);
    if (f == "--policy")
      o.policy = v;
    else if (f == "--order")
      o.order = v;
    else if (f == "--gate")
      o.gate = std::stoi(v) != 0;
    else if (f == "--block")
      o.block = std::stoi(v);
    else if (f == "--warmup")
      o.warmup = std::stoi(v);
    else if (f == "--reps")
      o.reps = std::stoi(v);
    else if (f == "--batch")
      o.batch = std::stoi(v);
    else
      fail("unknown flag " + std::string(f));
  }
  return o;
}

struct Timed {
  std::string name;
  double bytes;
  std::string strategy;
  std::vector<double> seconds;
};

template <class Launch>
std::vector<double> time_it(const Options& o, Launch launch) {
  cudaEvent_t start, stop;
  gpu::check(cudaEventCreate(&start), "event");
  gpu::check(cudaEventCreate(&stop), "event");
  std::vector<double> out;
  for (int r = 0; r < o.warmup + o.reps; ++r) {
    gpu::check(cudaEventRecord(start), "record");
    for (int i = 0; i < o.batch; ++i) launch();
    gpu::check(cudaEventRecord(stop), "record");
    gpu::check(cudaEventSynchronize(stop), "sync");
    float ms = 0.0f;
    gpu::check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    if (r >= o.warmup) out.push_back(ms * 1e-3 / o.batch);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return out;
}

double max_rel_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
      return std::numeric_limits<double>::infinity();
    }
    num = std::max(num, std::abs(a[i] - b[i]));
    den = std::max(den, std::abs(b[i]));
  }
  return den > 0.0 ? num / den : std::numeric_limits<double>::infinity();
}

template <class T>
std::vector<double> as_double(const std::vector<T>& v) {
  return std::vector<double>(v.begin(), v.end());
}

template <class P>
void run(const Options& o, const SpikeMesh& m, std::vector<Timed>& out, bool& verified,
         double& tolerance) {
  using S = typename P::state;
  using Gr = typename P::grad;
  using R = typename P::residual;
  const auto& t = m.topo;
  const auto nc = static_cast<std::size_t>(t.n_cells);
  const auto nf = static_cast<std::size_t>(t.n_faces);
  const double ni = t.n_interior_faces, nb = t.n_faces - t.n_interior_faces;
  const double cells = t.n_cells, inc = static_cast<double>(t.n_incidences);
  std::mt19937_64 rng(42);
  std::normal_distribution<double> normal;
  std::vector<S> q(nc), qb(nf);
  std::vector<Gr> grad(3 * nc);
  for (auto& v : q) v = S(normal(rng));
  for (auto& v : qb) v = S(normal(rng));
  for (auto& v : grad) v = Gr(normal(rng));

  wait_for_go(o.gate);
  const auto colour_start = std::chrono::steady_clock::now();
  DeviceMesh dm(m, o.block);
  const double colouring_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - colour_start).count();
  GeomCopy<P> hg(m.packed);
  DeviceArray<S> dq(q), dqb(qb), dmin(nc), dmax(nc);
  DeviceArray<Gr> dgrad(grad);
  DeviceArray<R> dres(nc), dacc(3 * nc);
  tolerance = std::is_same_v<typename P::compute, double> ? 1e-12 : 1e-4;

  namespace av = autan::kernels::advection;
  av::Data<P> a{};
  a.topo = dm.topo;
  a.q = dq.get();
  a.grad = dgrad.get();
  a.res = dres.get();
  dm.bind(a);
  a.velocity[0] = typename P::param(1.0);
  a.velocity[1] = typename P::param(0.7);
  a.velocity[2] = typename P::param(0.4);

  // Host reference of one residual evaluation.
  av::Data<P> ah = a;
  ah.topo = m.topo;
  std::vector<R> hres(nc, R(0));
  ah.q = q.data();
  ah.grad = grad.data();
  ah.res = hres.data();
  autan::spike::bind(ah.face_r_left, hg.face_r_left);
  autan::spike::bind(ah.face_r_right, hg.face_r_right);
  autan::spike::bind(ah.face_normal, hg.face_normal);
  autan::spike::bind(ah.face_area_magnitude, hg.face_area_magnitude);
  av::cpu::residual(ah);

  // Analytic minimum traffic: face geometry (r_left, r_right, n, |S|) and owner/neighbour once per
  // interior face; q and grad read once, res read and written once per cell.
  const double adv_bytes = ni * (10.0 * sizeof(*a.face_normal) + 2 * sizeof(index_t)) +
                           cells * (sizeof(S) + 3.0 * sizeof(Gr) + 2.0 * sizeof(R));
  for (auto s : {gpu::Strategy::atomic, gpu::Strategy::warp_aggregated, gpu::Strategy::coloured,
                 gpu::Strategy::store_gather, gpu::Strategy::gather_recompute}) {
    gpu::check(cudaMemset(dres.get(), 0, nc * sizeof(R)), "memset");
    av::gpu_loops::residual(a, s, dm.schedule);
    gpu::check(cudaDeviceSynchronize(), "advection");
    if (max_rel_diff(as_double(dres.download()), as_double(hres)) > tolerance) verified = false;
    std::string kernel = std::string("advection_") + gpu::name(s);
    std::replace(kernel.begin(), kernel.end(), '-', '_');
    out.push_back({kernel, adv_bytes, gpu::name(s),
                   time_it(o, [&] { av::gpu_loops::residual(a, s, dm.schedule); })});
  }

  namespace gr = autan::kernels::gradients;
  gr::Data<P> d{};
  d.topo = dm.topo;
  d.q = dq.get();
  d.q_b = dqb.get();
  d.grad = dgrad.get();
  d.qmin = dmin.get();
  d.qmax = dmax.get();
  d.gg_sum = dacc.get();
  dm.bind(d);
  gr::Data<P> dh = d;
  dh.topo = m.topo;
  std::vector<Gr> hgrad(3 * nc);
  std::vector<S> hmin(nc), hmax(nc);
  dh.q = q.data();
  dh.q_b = qb.data();
  dh.grad = hgrad.data();
  dh.qmin = hmin.data();
  dh.qmax = hmax.data();
  autan::spike::bind(dh.lsq_weight, hg.lsq_weight);
  gr::cpu::lsq_gradient_bounds(dh);
  gr::gpu_loops::lsq_gradient_bounds(d, dm.schedule);
  gpu::check(cudaDeviceSynchronize(), "lsq");
  if (max_rel_diff(as_double(dgrad.download()), as_double(hgrad)) > tolerance) verified = false;
  // Analytic minimum: per incidence the face index and LSQ weight; per cell its CSR offset, q,
  // and the gradient and two bounds written; owner/neighbour per interior face; q_b per boundary
  // face.
  const double lsq_bytes = inc * (sizeof(index_t) + 3.0 * sizeof(*d.lsq_weight)) +
                           cells * (sizeof(offset_t) + 3.0 * sizeof(S) + 3.0 * sizeof(Gr)) +
                           ni * 2 * sizeof(index_t) + nb * sizeof(S);
  out.push_back({"lsq_gradient_bounds", lsq_bytes, "cell",
                 time_it(o, [&] { gr::gpu_loops::lsq_gradient_bounds(d, dm.schedule); })});

  // Its adjoint: atomic scatter against the deterministic two-pass form. Bytes: the primal's
  // reads, the 5 seeds per cell read and zeroed, q_adj read and written.
  DeviceArray<R> g_adj(3 * nc), lo_adj(nc), hi_adj(nc), q_adj(nc);
  gr::Data<P> da = d;
  da.grad_adj = g_adj.get();
  da.qmin_adj = lo_adj.get();
  da.qmax_adj = hi_adj.get();
  da.q_adj = q_adj.get();
  const double lsq_adj_bytes = lsq_bytes - cells * (3.0 * sizeof(S) + 3.0 * sizeof(Gr)) +
                               cells * (sizeof(S) + 10.0 * sizeof(R) + 2.0 * sizeof(R));
  // Host reference with random seeds; the adjoint consumes (zeroes) its seeds, so each check
  // uploads them again. Timed repetitions after the first run on zeroed seeds: same work.
  std::vector<R> seeds(5 * nc);
  for (auto& v : seeds) v = R(normal(rng));
  std::vector<R> h_gadj(seeds.begin(), seeds.begin() + 3 * nc);
  std::vector<R> h_lo(seeds.begin() + 3 * nc, seeds.begin() + 4 * nc);
  std::vector<R> h_hi(seeds.begin() + 4 * nc, seeds.end());
  std::vector<R> h_qadj(nc, R(0));
  gr::Data<P> dha = dh;
  dha.grad_adj = h_gadj.data();
  dha.qmin_adj = h_lo.data();
  dha.qmax_adj = h_hi.data();
  dha.q_adj = h_qadj.data();
  gr::cpu::lsq_gradient_bounds(dh);
  gr::cpu::lsq_gradient_bounds_adjoint(dha);
  for (auto s : {gpu::Strategy::atomic, gpu::Strategy::store_gather}) {
    g_adj.upload(seeds.data());
    lo_adj.upload(seeds.data() + 3 * nc);
    hi_adj.upload(seeds.data() + 4 * nc);
    gpu::check(cudaMemset(q_adj.get(), 0, nc * sizeof(R)), "memset");
    gr::gpu_loops::lsq_gradient_bounds_adjoint(da, s, dm.schedule);
    gpu::check(cudaDeviceSynchronize(), "lsq adjoint");
    if (max_rel_diff(as_double(q_adj.download()), as_double(h_qadj)) > tolerance) verified = false;
    std::string tag = gpu::name(s);
    std::replace(tag.begin(), tag.end(), '-', '_');
    out.push_back(
        {"lsq_gradient_bounds_adjoint_" + tag, lsq_adj_bytes, gpu::name(s),
         time_it(o, [&] { gr::gpu_loops::lsq_gradient_bounds_adjoint(da, s, dm.schedule); })});
  }

  // LSQ gradient over the wider cell stencils that make the unlimited scheme stable on
  // tetrahedra. Analytic minimum: per entry its index and weight; per cell its offset, q, and
  // the gradient written.
  for (auto kind : {autan::kernels::StencilKind::vertex, autan::kernels::StencilKind::face2}) {
    const auto t_build = std::chrono::steady_clock::now();
    const auto st = autan::kernels::cell_stencil(m.mesh, m.packed, kind);
    const double build_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_build).count();
    const auto ns = st.entries.size();
    DeviceArray<offset_t> st_off(st.offsets), st_twin(st.twin);
    DeviceArray<index_t> st_entries(st.entries);
    const auto w_host = autan::kernels::converted<
        std::remove_cv_t<std::remove_pointer_t<decltype(d.stencil_lsq_weight)>>>(st.lsq_weight);
    DeviceArray<std::remove_cv_t<std::remove_pointer_t<decltype(d.stencil_lsq_weight)>>> w_dev(
        w_host);
    gr::Data<P> dst = d;
    dst.topo.n_stencil = static_cast<offset_t>(ns);
    dst.topo.stencil_offsets = st_off.get();
    dst.topo.stencil = st_entries.get();
    dst.topo.stencil_twin = st_twin.get();
    dst.stencil_lsq_weight = w_dev.get();
    gr::Data<P> hst = dh;
    hst.topo.n_stencil = static_cast<offset_t>(ns);
    hst.topo.stencil_offsets = st.offsets.data();
    hst.topo.stencil = st.entries.data();
    hst.topo.stencil_twin = st.twin.data();
    hst.stencil_lsq_weight = w_host.data();
    gr::cpu::lsq_gradient_stencil(hst);
    gr::gpu_loops::lsq_gradient_stencil(dst, dm.schedule);
    gpu::check(cudaDeviceSynchronize(), "lsq stencil");
    if (max_rel_diff(as_double(dgrad.download()), as_double(hgrad)) > tolerance) verified = false;
    const double st_bytes = static_cast<double>(ns) * (sizeof(index_t) + 3.0 * sizeof(w_host[0])) +
                            cells * (sizeof(offset_t) + sizeof(S) + 3.0 * sizeof(Gr)) +
                            nb * sizeof(S);
    const char* label = kind == autan::kernels::StencilKind::vertex ? "vertex" : "face2";
    std::fprintf(stderr, "%s stencil: %zu entries (%.1f per cell), built in %.3f s\n", label, ns,
                 static_cast<double>(ns) / cells, build_seconds);
    out.push_back({std::string("lsq_gradient_stencil_") + label, st_bytes, "cell",
                   time_it(o, [&] { gr::gpu_loops::lsq_gradient_stencil(dst, dm.schedule); })});
    // The same in the sliced layout (32 cells per slice): the CSR bytes (padding is skipped)
    // plus each cell's row length; the output is cleared first so the check sees this launch.
    gpu::DeviceMesh sm(m.mesh, m.packed, &st, {o.block, 0, 0, nullptr, 32});
    gr::Data<P> dsl = dst;
    sm.bind(dsl);
    gpu::check(cudaMemset(dgrad.get(), 0, 3 * nc * sizeof(Gr)), "clear gradient");
    gr::gpu_loops::lsq_gradient_stencil(dsl, sm.schedule());
    gpu::check(cudaDeviceSynchronize(), "lsq stencil sliced");
    if (max_rel_diff(as_double(dgrad.download()), as_double(hgrad)) > tolerance) verified = false;
    out.push_back({std::string("lsq_gradient_stencil_") + label + "_sliced",
                   st_bytes + cells * sizeof(index_t), "sliced32",
                   time_it(o, [&] { gr::gpu_loops::lsq_gradient_stencil(dsl, sm.schedule()); })});
  }

  // Roe flux, its tangent and its adjoint: the adjoint cost ratio of research section 8.
  namespace rf = autan::kernels::roe;
  std::vector<S> u(5 * nc);
  for (std::size_t c = 0; c < nc; ++c) {
    const double rho = 1.0 + 0.1 * std::abs(normal(rng));
    const double vx = 0.3 * normal(rng), vy = 0.3 * normal(rng), vz = 0.3 * normal(rng);
    u[c] = S(rho);
    u[nc + c] = S(rho * vx);
    u[2 * nc + c] = S(rho * vy);
    u[3 * nc + c] = S(rho * vz);
    u[4 * nc + c] = S(2.5 + 0.5 * rho * (vx * vx + vy * vy + vz * vz));
  }
  DeviceArray<S> du(u), du_tan(u);
  DeviceArray<R> dr(5 * nc), dr_tan(5 * nc), dr_adj(5 * nc), du_adj(5 * nc);
  rf::Data<P> rd{};
  rd.topo = dm.topo;
  rd.u_cons = du.get();
  rd.res = dr.get();
  dm.bind(rd);
  rd.u_cons_tan = du_tan.get();
  rd.res_tan = dr_tan.get();
  rd.res_adj = dr_adj.get();
  rd.u_cons_adj = du_adj.get();
  const double roe_bytes = ni * (4.0 * sizeof(*rd.face_normal) + 2 * sizeof(index_t)) +
                           cells * (5.0 * sizeof(S) + 10.0 * sizeof(R));
  for (auto s : {gpu::Strategy::atomic, gpu::Strategy::warp_aggregated, gpu::Strategy::store_gather,
                 gpu::Strategy::gather_recompute}) {
    std::string tag = gpu::name(s);
    std::replace(tag.begin(), tag.end(), '-', '_');
    out.push_back({"roe_" + tag, roe_bytes, gpu::name(s),
                   time_it(o, [&] { rf::gpu_loops::flux(rd, s, dm.schedule); })});
    out.push_back({"roe_tangent_" + tag, roe_bytes + cells * 5.0 * sizeof(S), gpu::name(s),
                   time_it(o, [&] { rf::gpu_loops::flux_tangent(rd, s, dm.schedule); })});
    out.push_back({"roe_adjoint_" + tag, roe_bytes, gpu::name(s),
                   time_it(o, [&] { rf::gpu_loops::flux_adjoint(rd, s, dm.schedule); })});
  }

  // Diagonal block inverses (kernels/blocks.py), in the accumulation type: diagonally dominant
  // random blocks, row-permuted so that pivoting swaps rows.
  using A = typename P::accum;
  std::vector<A> hdiag(25 * nc), hinv(25 * nc);
  for (std::size_t c = 0; c < nc; ++c) {
    const std::size_t shift = c % 5;
    for (std::size_t i = 0; i < 5; ++i) {
      for (std::size_t j = 0; j < 5; ++j) {
        const double v = normal(rng) + ((i + shift) % 5 == j ? 6.0 : 0.0);
        hdiag[(5 * i + j) * nc + c] = A(v);
      }
    }
  }
  namespace bk = autan::kernels::blocks;
  bk::Data<P> hb{};
  hb.topo = t;
  hb.diag = hdiag.data();
  hb.diag_inv = hinv.data();
  bk::cpu::invert_diagonal(hb);
  DeviceArray<A> ddiag(hdiag), dinv(25 * nc);
  bk::Data<P> db{};
  db.topo = t;
  db.diag = ddiag.get();
  db.diag_inv = dinv.get();
  bk::gpu_loops::invert_diagonal(db, dm.schedule);
  if (max_rel_diff(as_double(dinv.download()), as_double(hinv)) > tolerance) verified = false;
  out.push_back({"block_inverse", cells * 50.0 * sizeof(A), "cell",
                 time_it(o, [&] { bk::gpu_loops::invert_diagonal(db, dm.schedule); })});

  // The block matrix (kernels/blocks.py): `assemble` from random face and boundary blocks and a
  // shift, then y = A x, in the CSR and the sliced incidence layout (Topology::incidence_stride).
  using J = typename P::jac;
  std::vector<J> hjac(50 * nf), hbnd(25 * nf);
  for (auto& v : hjac) v = J(normal(rng));
  for (auto& v : hbnd) v = J(normal(rng));
  std::vector<A> hshift(nc), hdiag2(25 * nc);
  for (auto& v : hshift) v = A(6.0 + normal(rng));
  std::vector<J> hoff(25 * static_cast<std::size_t>(t.n_incidences));
  std::vector<R> hx(5 * nc), hy(5 * nc);
  for (auto& v : hx) v = R(normal(rng));
  bk::Data<P> hm = hb;
  hm.face_jac = hjac.data();
  hm.bnd_jac = hbnd.data();
  hm.shift = hshift.data();
  hm.diag = hdiag2.data();
  hm.off = hoff.data();
  hm.x = hx.data();
  hm.y = hy.data();
  autan::spike::bind(hm.incidence_side, hg.incidence_side);
  bk::cpu::assemble(hm);
  bk::cpu::multiply(hm);
  DeviceArray<J> djac(hjac), dbnd(hbnd);
  DeviceArray<A> dshift(hshift), ddiag2(25 * nc);
  DeviceArray<R> dx(hx), dy(5 * nc);
  const double nbf = static_cast<double>(t.n_faces) - ni;
  for (const int slice : {0, 32}) {
    autan::spike::incidence_slice() = slice;
    DeviceMesh dl(m, o.block);
    autan::spike::incidence_slice() = 0;
    bk::Data<P> dd{};
    dl.bind(dd);
    DeviceArray<J> doff(25 * static_cast<std::size_t>(dl.topo.n_incidences));
    dd.face_jac = djac.get();
    dd.bnd_jac = dbnd.get();
    dd.shift = dshift.get();
    dd.diag = ddiag2.get();
    dd.off = doff.get();
    dd.x = dx.get();
    dd.y = dy.get();
    bk::gpu_loops::assemble(dd, dl.schedule);
    bk::gpu_loops::multiply(dd, dl.schedule);
    if (max_rel_diff(as_double(ddiag2.download()), as_double(hdiag2)) > tolerance ||
        max_rel_diff(as_double(dy.download()), as_double(hy)) > tolerance) {
      verified = false;
    }
    // Analytic minima. Assembly: per cell its offset, shift and diagonal block; per incidence
    // its face, side and off-diagonal block; per interior face its two blocks and its owner and
    // neighbour; per boundary face its block. Product: per cell its offset, diagonal block, x
    // and y; per incidence its block and the cell across (the sliced column index, or the face
    // and its owner and neighbour in CSR).
    const double row = slice > 0 ? sizeof(offset_t) + sizeof(index_t) : sizeof(offset_t);
    const double asm_bytes =
        cells * (row + 26.0 * sizeof(A)) +
        inc * (sizeof(index_t) + sizeof(*dd.incidence_side) + 25.0 * sizeof(J)) +
        ni * (50.0 * sizeof(J) + 2 * sizeof(index_t)) + nbf * 25.0 * sizeof(J);
    const double mul_bytes = cells * (row + 25.0 * sizeof(A) + 10.0 * sizeof(R)) +
                             inc * (25.0 * sizeof(J) + sizeof(index_t)) +
                             (slice > 0 ? 0.0 : ni * 2 * sizeof(index_t));
    const std::string layout = slice > 0 ? "_sliced" : "_csr";
    out.push_back({"block_assemble" + layout, asm_bytes, "gather",
                   time_it(o, [&] { bk::gpu_loops::assemble(dd, dl.schedule); })});
    out.push_back({"block_multiply" + layout, mul_bytes, "gather",
                   time_it(o, [&] { bk::gpu_loops::multiply(dd, dl.schedule); })});
    // The same assembly without the off-diagonal blocks (codegen/tests/kernels): their cost.
    namespace dgo = autan::kernels::diagonal_only;
    DeviceArray<A> ddiag3(25 * nc);
    dgo::Data<P> dq{};
    dl.bind(dq);
    dq.face_jac = djac.get();
    dq.bnd_jac = dbnd.get();
    dq.shift = dshift.get();
    dq.diag = ddiag3.get();
    dgo::gpu_loops::assemble(dq, dl.schedule);
    if (max_rel_diff(as_double(ddiag3.download()), as_double(hdiag2)) > tolerance) verified = false;
    out.push_back({"block_diagonal" + layout, asm_bytes - inc * 25.0 * sizeof(J), "gather",
                   time_it(o, [&] { dgo::gpu_loops::assemble(dq, dl.schedule); })});
  }
  std::fprintf(stderr, "colouring %.3f s (%zu interior colours, %zu boundary colours)\n",
               colouring_seconds, dm.schedule.interior.offsets.size() - 1,
               dm.schedule.boundary.offsets.size() - 1);
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  const auto t0 = std::chrono::steady_clock::now();
  autan::Mesh mesh = autan::io::read_mesh(o.mesh);
  const auto t1 = std::chrono::steady_clock::now();
  double renumber_seconds = 0.0;
  if (o.order == "rcm") {
    mesh = autan::renumber_cells(mesh, autan::rcm_order(mesh));
    renumber_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  } else if (o.order != "input") {
    fail("--order is input or rcm");
  }
  SpikeMesh m{std::move(mesh), {}, {}, {}, {}, {}};
  m.geometry = autan::compute_geometry(m.mesh);
  m.packed = autan::kernels::pack_geometry(m.mesh, m.geometry);
  m.topo = autan::kernels::topology(m.mesh.view());
  m.twins = autan::kernels::incidence_twins(m.mesh);
  m.topo.incidence_twin = m.twins.data();
  const double read_seconds = std::chrono::duration<double>(t1 - t0).count();

  std::vector<Timed> timed;
  bool verified = true;
  double tolerance = 0.0;
  autan::spike::with_policy(o.policy,
                            [&](auto p) { run<decltype(p)>(o, m, timed, verified, tolerance); });

  int device = 0;
  cudaDeviceProp prop{};
  gpu::check(cudaGetDevice(&device), "device");
  gpu::check(cudaGetDeviceProperties(&prop, device), "properties");
  int runtime = 0, driver = 0;
  cudaRuntimeGetVersion(&runtime);
  cudaDriverGetVersion(&driver);
  char uuid[37];
  const auto* u = reinterpret_cast<const unsigned char*>(prop.uuid.bytes);
  std::snprintf(uuid, sizeof uuid,
                "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1],
                u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14],
                u[15]);
  std::printf(
      "{\"device\":{\"name\":\"%s\",\"uuid\":\"GPU-%s\",\"compute_capability\":\"%d.%d\","
      "\"sm_count\":%d,\"l2_bytes\":%d,\"memory_bytes\":%zu},",
      prop.name, uuid, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize,
      prop.totalGlobalMem);
  std::printf("\"revision\":{\"commit\":\"%s\",\"dirty\":%s},", AUTAN_BENCH_COMMIT,
              AUTAN_BENCH_DIRTY);
  std::printf("\"backend\":\"cuda\",\"runtime_version\":%d,\"driver_version\":%d,", runtime,
              driver);
  // `config` identifies a run by content (mesh hash, not path) and holds no measurement, so that
  // autan-bench compare can match runs; one-off host costs go to `setup_seconds`.
  const std::string sha = autan::io::mesh_hash(o.mesh);
  std::printf(
      "\"config\":{\"mesh_file\":\"%s\",\"mesh_sha256\":\"%s\",\"policy\":\"%s\","
      "\"order\":\"%s\",\"layout\":\"soa\",\"n_cells\":%d,\"n_faces\":%d,"
      "\"n_interior_faces\":%d,\"n_incidences\":%lld,\"block\":%d,\"warmup\":%d,\"reps\":%d,"
      "\"batch\":%d,\"tolerance\":%.1e},",
      std::filesystem::path(o.mesh).filename().string().c_str(), sha.c_str(), o.policy.c_str(),
      o.order.c_str(), m.topo.n_cells, m.topo.n_faces, m.topo.n_interior_faces,
      static_cast<long long>(m.topo.n_incidences), o.block, o.warmup, o.reps, o.batch, tolerance);
  std::printf("\"setup_seconds\":{\"read\":%.3f,\"renumber\":%.3f},", read_seconds,
              renumber_seconds);
  std::printf("\"verified\":%s,\"kernels\":[", verified ? "true" : "false");
  for (std::size_t k = 0; k < timed.size(); ++k) {
    const auto& t = timed[k];
    std::printf(
        "%s{\"name\":\"%s\",\"bytes_per_rep\":%.0f,\"cells\":%d,\"params\":{\"strategy\":"
        "\"%s\",\"policy\":\"%s\",\"order\":\"%s\"},\"seconds\":[",
        k ? "," : "", t.name.c_str(), t.bytes, m.topo.n_cells, t.strategy.c_str(), o.policy.c_str(),
        o.order.c_str());
    for (std::size_t r = 0; r < t.seconds.size(); ++r)
      std::printf("%s%.9e", r ? "," : "", t.seconds[r]);
    std::printf("]}");
  }
  std::printf("]}\n");
  return verified ? 0 : 1;
}
