"""Matrix blocks (Jac, JacAccum) read by generated loops: passive inputs of a linear operator."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import autan_spike as s
import meshes
import numpy as np
import pytest
from autan_codegen.errors import Code, CodegenError
from autan_codegen.generate import generate
from conftest import PRECISIONS, eps

ROOT = Path(__file__).resolve().parents[2]

MODULE = """
from autan_codegen.dsl import Inc, Jac, JacAccum, Residual, State, Write, cell_loop, cells, faces
from autan_codegen.dsl import face_loop, pointwise

x = cells.field("x", State.Vec5)
diag = cells.field("diag", JacAccum.Mat5)
shift = cells.field("shift", Jac.Mat5)
off = faces.field("off", Jac.Mat5)
y = cells.field("y", Residual.Vec5)


@pointwise
def apply(a: JacAccum.Mat5, s: Jac.Mat5, v: State.Vec5) -> Residual.Vec5:
    return a @ v + s @ v


@pointwise
def couple(b: Jac.Mat5, vl: State.Vec5, vr: State.Vec5) -> Residual.Vec5:
    return b @ (vr - vl)


diagonal = cell_loop(apply, a=diag, s=shift, v=x, out=Write(y))
coupling = face_loop(couple, b=off, vl=x.left, vr=x.right, out=Inc(y, left=+1, right=-1))
"""

# Three cells in a chain, faces 0 (0-1) and 1 (1-2), no boundary face: y_c = D_c x_c, then
# face f adds B_f (x_r - x_l) to its left cell and subtracts it from its right one. A shared
# library loaded by ctypes, as the spike drivers (no executable to launch).
DRIVER = r"""
#include <algorithm>
#include <vector>
#include "blocks_cpu.hpp"
using namespace autan::kernels;
extern "C" void run(const double* in, double* out) {
  constexpr index_t nc = 3, nf = 2;
  std::vector<double> x(in, in + 5 * nc), diag(in + 5 * nc, in + 30 * nc),
      shift(in + 30 * nc, in + 55 * nc), off(in + 55 * nc, in + 55 * nc + 25 * nf),
      y(5 * nc, 0.0);
  std::vector<index_t> owner{0, 1}, neighbour{1, 2}, cell_faces{0, 0, 1, 1};
  std::vector<offset_t> offsets{0, 1, 3, 4};
  Topology t{};
  t.n_cells = t.n_owned_cells = nc;
  t.n_faces = t.n_interior_faces = t.n_owned_interior_faces = t.n_owned_faces = nf;
  t.n_incidences = 4;
  t.owner = owner.data();
  t.neighbour = neighbour.data();
  t.cell_face_offsets = offsets.data();
  t.cell_faces = cell_faces.data();
  blocks::Data<fp64> d{};
  d.topo = t;
  d.x = x.data();
  d.diag = diag.data();
  d.shift = shift.data();
  d.off = off.data();
  d.y = y.data();
  blocks::cpu::diagonal(d);
  blocks::cpu::coupling(d);
  for (std::size_t i = 0; i < y.size(); ++i) out[i] = y[i];
  // Again as a gather, on the incidences sliced by 2 cells: cells 0 and 1 share a slice of
  // two rows (entries 0, 1 | 2 padding, 3), cell 2 its own (entry 4 | 5 padding).
  std::vector<offset_t> starts{0, 1, 4, 6};
  std::vector<index_t> length{1, 2, 1}, sliced_faces{0, 0, 0, 1, 1, 0}, cell{1, 0, -1, 2, 1, -1};
  t.n_incidences = 6;
  t.cell_face_offsets = starts.data();
  t.cell_faces = sliced_faces.data();
  t.incidence_stride = 2;
  t.incidence_length = length.data();
  t.incidence_cell = cell.data();
  d.topo = t;
  std::fill(y.begin(), y.end(), 0.0);
  blocks::cpu::diagonal(d);
  blocks::cpu::coupling(d, Strategy::gather_recompute);
  for (std::size_t i = 0; i < y.size(); ++i) out[y.size() + i] = y[i];
}
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
def test_blocks_are_passive_inputs_of_a_block_operator(tmp_path: Path) -> None:
    src = tmp_path / "blocks.py"
    src.write_text(MODULE)
    out = tmp_path / "gen"
    generate(src, out, ["cpu"])
    header = (out / "blocks.hpp").read_text()
    # Blocks are never differentiated, cell blocks included: no tangent or adjoint of them.
    for name in ("diag_tan", "diag_adj", "shift_tan", "shift_adj", "off_tan", "off_adj"):
        assert name not in header, name
    assert "typename P::accum* diag;" in header  # JacAccum: accumulation precision
    assert "typename P::jac* off;" in header
    (tmp_path / "driver.cpp").write_text(DRIVER)
    lib = tmp_path / "libblocks.so"
    subprocess.run(
        ["c++", "-std=c++20", "-O1", "-shared", "-fPIC", f"-I{ROOT / 'codegen' / 'include'}",
         f"-I{ROOT / 'core' / 'include'}", f"-I{out}", str(tmp_path / "driver.cpp"), "-o", str(lib)],
        check=True, capture_output=True,
    )  # fmt: skip
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 5))
    diag, shift = rng.standard_normal((3, 5, 5)), rng.standard_normal((3, 5, 5))
    off = rng.standard_normal((2, 5, 5))
    # Generated datasets are component-major: component k of entity i at k * n + i.
    inputs = np.concatenate(
        [
            x.T.ravel(),
            diag.reshape(3, 25).T.ravel(),
            shift.reshape(3, 25).T.ravel(),
            off.reshape(2, 25).T.ravel(),
        ]
    )
    y = np.zeros(30)  # the scatter loop, then the gather on sliced incidences
    ctypes.CDLL(str(lib)).run(
        inputs.ctypes.data_as(ctypes.c_void_p), y.ctypes.data_as(ctypes.c_void_p)
    )
    expect = np.einsum("cij,cj->ci", diag, x) + np.einsum("cij,cj->ci", shift, x)
    for f, (left, right) in enumerate(((0, 1), (1, 2))):
        flow = off[f] @ (x[right] - x[left])
        expect[left] += flow
        expect[right] -= flow
    for run in (y[:15], y[15:]):
        np.testing.assert_allclose(run.reshape(5, 3).T, expect, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("kind", ["tet", "hex", "poly", "hybrid"])
def test_incidence_side_is_plus_one_on_the_left_cell_and_on_boundary_faces(kind: str) -> None:
    m = s.mesh(meshes.build(kind, 4, 0.15, seed=7))
    for c in range(m.n_cells):
        for k in range(m.cell_face_offsets[c], m.cell_face_offsets[c + 1]):
            f = m.cell_faces[k]
            left = f >= m.n_interior or m.owner[f] == c
            assert m.incidence_side[k] == (1.0 if left else -1.0), (c, f)


# One driver for the host and the device: kernels/blocks.py inverts n blocks given as doubles
# (component-major), stored in the policy's accumulation type.
INVERSE_DRIVER = r"""
#include <vector>
#include "blocks.hpp"
#ifdef __CUDACC__
#include "blocks_gpu.cuh"
#else
#include "blocks_cpu.hpp"
#endif
using namespace autan::kernels;
using P = POLICY;
using A = P::accum;
extern "C" void run(int n, const double* in, double* out) {
  std::vector<A> a(in, in + 25 * n), inv(25 * n);
  Topology t{};
  t.n_cells = t.n_owned_cells = n;
  blocks::Data<P> d{};
  d.topo = t;
#ifdef __CUDACC__
  const std::size_t bytes = sizeof(A) * a.size();
  cudaMalloc(&d.diag, bytes);
  cudaMalloc(&d.diag_inv, bytes);
  cudaMemcpy(d.diag, a.data(), bytes, cudaMemcpyHostToDevice);
  blocks::gpu_loops::invert_diagonal(d, gpu::Schedule{});
  cudaMemcpy(inv.data(), d.diag_inv, bytes, cudaMemcpyDeviceToHost);
  cudaFree(d.diag);
  cudaFree(d.diag_inv);
#else
  d.diag = a.data();
  d.diag_inv = inv.data();
  blocks::cpu::invert_diagonal(d);
#endif
  for (std::size_t i = 0; i < inv.size(); ++i) out[i] = double(inv[i]);
}
"""

POLICY_TYPES = {"fp64": "fp64", "fp32+acc64": "fp32_acc64", "fp32": "fp32"}
N_REGULAR = 48


def invert5(a: np.ndarray) -> np.ndarray:
    """Gauss-Jordan of the host BlockMatrix (physics/src/block_matrix.cpp), in Python floats."""
    m = [[float(a[i, j]) for j in range(5)] + [float(i == j) for j in range(5)] for i in range(5)]
    for k in range(5):
        p = k
        for i in range(k + 1, 5):
            if abs(m[i][k]) > abs(m[p][k]):
                p = i
        if m[p][k] == 0.0 or not np.isfinite(m[p][k]):
            return np.full((5, 5), np.nan)
        m[k], m[p] = m[p], m[k]
        d = 1.0 / m[k][k]
        m[k] = [x * d for x in m[k]]
        for i in range(5):
            if i != k:
                f = m[i][k]
                m[i] = [x - f * y for x, y in zip(m[i], m[k], strict=True)]
    return np.array([row[5:] for row in m])


def _test_blocks() -> np.ndarray:
    """Regular blocks (diagonally dominant, row-permuted so that pivoting swaps, row-scaled over
    six decades), then exactly singular ones (zero, a zero column, an infinite or a NaN entry)."""
    rng = np.random.default_rng(5)
    regular = rng.standard_normal((N_REGULAR, 5, 5)) + 6.0 * np.eye(5)
    for b in regular[16:32]:
        b[:] = b[rng.permutation(5)]
    regular[32:] *= 10.0 ** rng.uniform(-3, 3, (16, 5, 1))
    singular = np.repeat(rng.standard_normal((1, 5, 5)) + 6.0 * np.eye(5), 4, axis=0)
    singular[0] = 0.0
    singular[1][:, 2] = 0.0
    singular[2][3, 1] = np.inf
    singular[3][0, 4] = np.nan
    return np.concatenate([regular, singular])


def _build(tmp_path: Path, backend: str, policy: str) -> ctypes.CDLL:
    out = tmp_path / f"gen_{backend}"
    generate(ROOT / "kernels" / "blocks.py", out, [backend], policies=[policy])
    source = tmp_path / ("driver.cu" if backend == "gpu" else "driver.cpp")
    source.write_text(INVERSE_DRIVER)
    lib = tmp_path / f"libinverse_{backend}.so"
    flags = [f"-DPOLICY={POLICY_TYPES[policy]}", f"-I{ROOT / 'codegen' / 'include'}",
             f"-I{ROOT / 'core' / 'include'}", f"-I{out}", "-o", str(lib)]  # fmt: skip
    if backend == "gpu":
        nvcc = shutil.which("nvcc")
        assert nvcc is not None, "the GPU backend tests need nvcc on PATH"
        cmd = [nvcc, "-std=c++20", "-O2", "-shared", "-Xcompiler", "-fPIC", *flags,
               str(source), str(out / "blocks_gpu.cu")]  # fmt: skip
    else:
        cmd = ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-shared", "-fPIC", *flags,
               str(source)]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True)
    return ctypes.CDLL(str(lib))


def _invert(lib: ctypes.CDLL, blocks: np.ndarray) -> np.ndarray:
    n = len(blocks)
    inputs = np.ascontiguousarray(blocks.reshape(n, 25).T)  # component-major
    inv = np.zeros(25 * n)
    lib.run(
        ctypes.c_int(n), inputs.ctypes.data_as(ctypes.c_void_p), inv.ctypes.data_as(ctypes.c_void_p)
    )
    return inv.reshape(25, n).T.reshape(n, 5, 5)


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
@pytest.mark.parametrize("policy", PRECISIONS)
def test_diagonal_blocks_invert_as_the_host_block_matrix(
    tmp_path: Path, backend: str, policy: str
) -> None:
    if backend == "metal":
        with pytest.raises(CodegenError) as err:
            generate(ROOT / "kernels" / "blocks.py", tmp_path, ["metal"])
        assert err.value.code == Code.POLICY_UNSUPPORTED_BY_BACKEND
        return
    blocks = _test_blocks()
    a = blocks[:N_REGULAR]
    host = _invert(_build(tmp_path, "cpu", policy), blocks)
    # The inverse runs in the accumulation type: FP64 under fp32+acc64, as the host matrix.
    tol = 64 * float(np.finfo(np.float32 if policy == "fp32" else np.float64).eps)
    bound = (
        tol
        * np.linalg.cond(a)[:, None, None]
        * np.abs(np.linalg.inv(a)).max(axis=(1, 2))[:, None, None]
    )
    assert (np.abs(host[:N_REGULAR] - np.linalg.inv(a)) <= bound).all()
    if policy != "fp32":
        np.testing.assert_array_equal(host, np.array([invert5(b) for b in blocks]))
    # A zero or non-finite pivot, where the host matrix throws, makes the whole inverse NaN.
    assert np.isnan(host[N_REGULAR:]).all()
    if backend == "gpu":
        lib = _build(tmp_path, "gpu", policy)
        dev = _invert(lib, blocks)
        assert (np.abs(dev[:N_REGULAR] - host[:N_REGULAR]) <= bound).all()
        assert np.isnan(dev[N_REGULAR:]).all()
        np.testing.assert_array_equal(_invert(lib, blocks), dev)


# kernels/blocks.py `assemble` then `multiply` on a mesh's topology, CSR or sliced, every input
# given as doubles and stored in the type the policy gives its dataset (component-major).
MATRIX_DRIVER = r"""
#include <cstdint>
#include <type_traits>
#include <vector>
#include "blocks.hpp"
#ifdef __CUDACC__
#include "blocks_gpu.cuh"
#else
#include "blocks_cpu.hpp"
#endif
using namespace autan::kernels;
using P = POLICY;
using D = blocks::Data<P>;
template <class T> using Of = std::remove_cvref_t<T>;
#ifdef __CUDACC__
template <class T> T* place(const std::vector<T>& v) {
  T* p = nullptr;
  cudaMalloc(&p, sizeof(T) * v.size());
  cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice);
  return p;
}
template <class T> void fetch(std::vector<T>& v, const T* p) {
  cudaMemcpy(v.data(), p, sizeof(T) * v.size(), cudaMemcpyDeviceToHost);
}
#else
template <class T> T* place(std::vector<T>& v) { return v.data(); }
template <class T> void fetch(std::vector<T>&, const T*) {}
#endif
template <class T> std::vector<T> as(const double* p, std::size_t n) { return {p, p + n}; }
template <class T> void give(const std::vector<T>& v, double* out) {
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = double(v[i]);
}
// stride 1: CSR (length and cell unused); otherwise the sliced layout of offsets, length and cell.
extern "C" void run(int nc, int nf, int ni, long long nk, long long stride, const index_t* owner,
                    const index_t* neighbour, const offset_t* offsets, const index_t* length,
                    const index_t* cell_faces, const index_t* cell, const double* side,
                    const std::uint8_t* reversed, const double* face_jac, const double* bnd_jac, const double* shift,
                    const double* x, double* diag, double* off, double* y) {
  D d{};
  auto o = std::vector<index_t>(owner, owner + nf);
  auto n = std::vector<index_t>(neighbour, neighbour + ni);
  auto of = std::vector<offset_t>(offsets, offsets + nc + 1);
  auto ln = std::vector<index_t>(length, length + (stride == 1 ? 0 : nc));
  auto cf = std::vector<index_t>(cell_faces, cell_faces + nk);
  auto cl = std::vector<index_t>(cell, cell + (stride == 1 ? 0 : nk));
  auto rv = std::vector<std::uint8_t>(reversed, reversed + nf);
  auto sd = as<Of<decltype(*d.incidence_side)>>(side, nk);
  auto fj = as<Of<decltype(*d.face_jac)>>(face_jac, 50 * nf);
  auto bj = as<Of<decltype(*d.bnd_jac)>>(bnd_jac, 25 * nf);
  auto sh = as<Of<decltype(*d.shift)>>(shift, nc);
  auto xv = as<Of<decltype(*d.x)>>(x, 5 * nc);
  std::vector<Of<decltype(*d.diag)>> dg(25 * nc);
  std::vector<Of<decltype(*d.off)>> ov(25 * nk);
  std::vector<Of<decltype(*d.y)>> yv(5 * nc);
  Topology t{};
  t.n_cells = t.n_owned_cells = nc;
  t.n_faces = t.n_owned_faces = nf;
  t.n_interior_faces = t.n_owned_interior_faces = ni;
  t.n_incidences = nk;
  t.owner = place(o);
  t.neighbour = place(n);
  t.cell_face_offsets = place(of);
  t.cell_faces = place(cf);
  t.face_reversed = place(rv);
  if (stride != 1) {
    t.incidence_stride = stride;
    t.incidence_length = place(ln);
    t.incidence_cell = place(cl);
  }
  d.topo = t;
  d.incidence_side = place(sd);
  d.face_jac = place(fj);
  d.bnd_jac = place(bj);
  d.shift = place(sh);
  d.x = place(xv);
  d.diag = place(dg);
  d.off = place(ov);
  d.y = place(yv);
#ifdef __CUDACC__
  blocks::gpu_loops::assemble(d, gpu::Schedule{});
  blocks::gpu_loops::multiply(d, gpu::Schedule{});
#else
  blocks::cpu::assemble(d);
  blocks::cpu::multiply(d);
#endif
  fetch(dg, d.diag);
  fetch(ov, d.off);
  fetch(yv, d.y);
  give(dg, diag);
  give(ov, off);
  give(yv, y);
}
"""


def _build_matrix(tmp_path: Path, backend: str, policy: str) -> ctypes.CDLL:
    out = tmp_path / f"gen_{backend}"
    generate(ROOT / "kernels" / "blocks.py", out, [backend], policies=[policy])
    source = tmp_path / ("matrix.cu" if backend == "gpu" else "matrix.cpp")
    source.write_text(MATRIX_DRIVER)
    lib = tmp_path / f"libmatrix_{backend}.so"
    flags = [f"-DPOLICY={POLICY_TYPES[policy]}", f"-I{ROOT / 'codegen' / 'include'}",
             f"-I{ROOT / 'core' / 'include'}", f"-I{out}", "-o", str(lib)]  # fmt: skip
    if backend == "gpu":
        nvcc = shutil.which("nvcc")
        assert nvcc is not None, "the GPU backend tests need nvcc on PATH"
        cmd = [nvcc, "-std=c++20", "-O2", "-shared", "-Xcompiler", "-fPIC", *flags, str(source)]
    else:
        cmd = ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-shared", "-fPIC", *flags,
               str(source)]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True)
    return ctypes.CDLL(str(lib))


def _sliced(m: s.Mesh, slice_: int) -> dict[str, np.ndarray | int]:
    """The incidences of m in slices of `slice_` cells: entry j of lane l of a slice at the
    slice's start + j * slice_ + l, a slice as long as its longest row (Topology docs)."""
    off = m.cell_face_offsets
    length = np.diff(off).astype(np.int32)
    starts, position, total = np.zeros(m.n_cells + 1, np.int64), np.zeros(off[-1], np.int64), 0
    for s0 in range(0, m.n_cells, slice_):
        for c in range(s0, min(s0 + slice_, m.n_cells)):
            starts[c] = total + c - s0
            position[off[c] : off[c + 1]] = starts[c] + slice_ * np.arange(length[c])
        total += slice_ * int(length[s0 : s0 + slice_].max())
    starts[-1] = total
    cell = np.full(total, -1, np.int32)
    faces = np.zeros(total, np.int32)
    faces[position] = m.cell_faces
    for c in range(m.n_cells):
        for k in range(off[c], off[c + 1]):
            f = m.cell_faces[k]
            if f < m.n_interior:
                cell[position[k]] = m.neighbour[f] if m.owner[f] == c else m.owner[f]
    return {"offsets": starts, "length": length, "cell": cell, "cell_faces": faces,
            "position": position, "total": total, "stride": slice_}  # fmt: skip


def _matrix(
    lib: ctypes.CDLL, m: s.Mesh, layout: dict | None, reversed_, side, face_jac, bnd_jac, shift, x
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """diag (cells, 5, 5), off (CSR incidences, 5, 5) and y (cells, 5) from the generated loops,
    `reversed_` flagging the faces whose global orientation is the reverse of the local one."""

    def p(a: np.ndarray) -> ctypes.c_void_p:
        return a.ctypes.data_as(ctypes.c_void_p)

    nc, nk = m.n_cells, m.n_incidences
    csr_side = side
    if layout is None:
        offsets, length, faces, cell, total, stride = (
            m.cell_face_offsets, np.zeros(1, np.int32), m.cell_faces, np.zeros(1, np.int32), nk, 1,
        )  # fmt: skip
    else:
        offsets, length, faces, cell = (
            layout[k] for k in ("offsets", "length", "cell_faces", "cell")
        )
        total, stride = layout["total"], layout["stride"]
        side = np.zeros(total)
        side[layout["position"]] = csr_side
    soa = lambda a: np.ascontiguousarray(a.reshape(len(a), -1).T)
    diag, off, y = np.zeros(25 * nc), np.zeros(25 * total), np.zeros(5 * nc)
    lib.run(
        ctypes.c_int(nc), ctypes.c_int(m.n_faces), ctypes.c_int(m.n_interior),
        ctypes.c_longlong(total), ctypes.c_longlong(stride), p(m.owner), p(m.neighbour),
        p(offsets), p(length), p(faces), p(cell), p(np.ascontiguousarray(side)),
        p(reversed_.astype(np.uint8)), p(soa(face_jac)), p(soa(bnd_jac)), p(shift), p(soa(x)), p(diag), p(off), p(y),
    )  # fmt: skip
    off = off.reshape(25, total).T
    if layout is not None:
        off = off[layout["position"]]
    return diag.reshape(25, nc).T.reshape(nc, 5, 5), off.reshape(nk, 5, 5), y.reshape(5, nc).T


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
@pytest.mark.parametrize("policy", PRECISIONS)
@pytest.mark.parametrize("kind", ["tet", "hex", "poly", "hybrid"])
def test_block_matrix_assembles_and_multiplies_as_the_host_block_matrix(
    tmp_path: Path, backend: str, policy: str, kind: str
) -> None:
    if backend == "metal":
        pytest.skip("kernels/blocks.py is not generated for Metal (E1013, see the inverse test)")
    m = s.mesh(meshes.build(kind, 4, 0.15, seed=9))
    nc, ni, nf = m.n_cells, m.n_interior, m.n_faces
    rng = np.random.default_rng(9)
    face_jac, bnd_jac = rng.standard_normal((nf, 2, 5, 5)), rng.standard_normal((nf, 5, 5))
    if policy != "fp64":  # Jac is stored in FP32
        face_jac = face_jac.astype(np.float32).astype(np.float64)
        bnd_jac = bnd_jac.astype(np.float32).astype(np.float64)
    # Blocks a face does not have are never written by the physics: NaN must not reach a result.
    face_jac[ni:] = np.nan
    bnd_jac[:ni] = np.nan
    shift, x = rng.uniform(1.0, 2.0, nc), rng.standard_normal((nc, 5))
    lib = _build_matrix(tmp_path, "cpu", policy)
    dev = _build_matrix(tmp_path, "gpu", policy) if backend == "gpu" else None
    # Local orientation, then random interior faces reversed (a partition's faces under the
    # global order, runtime.md): the matrix follows each face's global left and right cells.
    flips = np.zeros(nf, bool)
    flips[:ni] = rng.random(ni) < 0.5
    for reversed_ in (np.zeros(nf, bool), flips):
        # The host assembly (physics Euler::assemble, BlockMatrix): per cell, in the order of
        # its faces, left +face_jac[0], right -face_jac[1], boundary bnd_jac, then the shift;
        # off-diagonal left +face_jac[1], right -face_jac[0], zero on a boundary face.
        diag, off = np.zeros((nc, 5, 5)), np.zeros((m.n_incidences, 5, 5))
        size = np.zeros((nc, 5, 5))  # sum of the magnitudes of the diagonal's terms
        side = np.ones(m.n_incidences)
        a = np.zeros((nc, 5, nc, 5))
        for c in range(nc):
            for k in range(m.cell_face_offsets[c], m.cell_face_offsets[c + 1]):
                f = m.cell_faces[k]
                if f >= ni:
                    diag[c] = diag[c] + bnd_jac[f]
                    size[c] += np.abs(bnd_jac[f])
                    continue
                other = m.neighbour[f] if m.owner[f] == c else m.owner[f]
                if (m.owner[f] == c) != reversed_[f]:
                    diag[c] = diag[c] + face_jac[f, 0]
                    size[c] += np.abs(face_jac[f, 0])
                    off[k] = face_jac[f, 1]
                else:
                    side[k] = -1.0
                    diag[c] = diag[c] - face_jac[f, 1]
                    size[c] += np.abs(face_jac[f, 1])
                    off[k] = -face_jac[f, 0]
                a[c, :, other, :] += off[k]
            diag[c] = diag[c] + shift[c] * np.eye(5)
            size[c] += shift[c] * np.eye(5)
            a[c, :, c, :] += diag[c]
        a = a.reshape(5 * nc, 5 * nc)
        expect = (a @ x.ravel()).reshape(nc, 5)
        bound = 64 * eps(policy) * (np.abs(a) @ np.abs(x.ravel())).reshape(nc, 5)
        inputs = (reversed_, side, face_jac, bnd_jac, shift, x)
        runs = [_matrix(lib, m, None, *inputs), _matrix(lib, m, _sliced(m, 4), *inputs)]
        if dev is not None:
            runs += [_matrix(dev, m, None, *inputs), _matrix(dev, m, _sliced(m, 32), *inputs)]
        for got_diag, got_off, y in runs:
            np.testing.assert_array_equal(got_off, off)
            if policy == "fp32":  # the diagonal sums in FP32
                assert (np.abs(got_diag - diag) <= 64 * eps(policy) * size).all()
            else:
                np.testing.assert_array_equal(got_diag, diag)
            assert (np.abs(y - expect) <= bound).all()
        # The sliced layout visits each cell's faces in the same order: the same bits as CSR.
        for sliced, csr in zip(runs[1::2], runs[::2], strict=True):
            for got, ref in zip(sliced, csr, strict=True):
                np.testing.assert_array_equal(got, ref)


ACCUM_REDUCTION = """
from autan_codegen.dsl import JacAccum, Sum, cell_reduce, cells, incidences, pointwise

w = incidences.field("w", JacAccum)
total = cells.field("total", JacAccum)


@pointwise(compute="accum")
def doubled(v: JacAccum) -> JacAccum:
    return v + v


summed = cell_reduce(doubled, v=w, acc=Sum(total))
"""

# The chain of DRIVER (incidences 0 | 1 2 | 3) under fp32+acc64: per-incidence values that FP32
# would round, summed by a body computing in the accumulation type.
ACCUM_DRIVER = r"""
#include <vector>
#include "accum_reduction_cpu.hpp"
using namespace autan::kernels;
extern "C" void run(const double* in, double* out) {
  std::vector<double> w(in, in + 4), total(3, 0.0);
  std::vector<index_t> owner{0, 1}, neighbour{1, 2}, cell_faces{0, 0, 1, 1};
  std::vector<offset_t> offsets{0, 1, 3, 4};
  Topology t{};
  t.n_cells = t.n_owned_cells = 3;
  t.n_faces = t.n_interior_faces = t.n_owned_interior_faces = t.n_owned_faces = 2;
  t.n_incidences = 4;
  t.owner = owner.data();
  t.neighbour = neighbour.data();
  t.cell_face_offsets = offsets.data();
  t.cell_faces = cell_faces.data();
  accum_reduction::Data<fp32_acc64> d{};
  d.topo = t;
  d.w = w.data();
  d.total = total.data();
  accum_reduction::cpu::summed(d);
  for (int c = 0; c < 3; ++c) out[c] = total[c];
}
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
def test_reduction_bodies_compute_in_the_accumulation_type(tmp_path: Path) -> None:
    src = tmp_path / "accum_reduction.py"
    src.write_text(ACCUM_REDUCTION)
    out = tmp_path / "gen"
    generate(src, out, ["cpu"])
    (tmp_path / "driver.cpp").write_text(ACCUM_DRIVER)
    lib = tmp_path / "libaccum.so"
    subprocess.run(
        ["c++", "-std=c++20", "-O1", "-shared", "-fPIC", f"-I{ROOT / 'codegen' / 'include'}",
         f"-I{ROOT / 'core' / 'include'}", f"-I{out}", str(tmp_path / "driver.cpp"), "-o", str(lib)],
        check=True, capture_output=True,
    )  # fmt: skip
    w = 1.0 + 2.0 ** -np.arange(30, 34)  # 1 in FP32
    total = np.zeros(3)
    ctypes.CDLL(str(lib)).run(
        w.ctypes.data_as(ctypes.c_void_p), total.ctypes.data_as(ctypes.c_void_p)
    )
    np.testing.assert_array_equal(total, [2 * w[0], 2 * w[1] + 2 * w[2], 2 * w[3]])
