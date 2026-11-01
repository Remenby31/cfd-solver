"""ctypes bindings to the spike library (codegen/spike/capi.cpp): meshes through the core and the
generated host loops. The library path comes from AUTAN_SPIKE_LIB."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from meshes import FaceArrays

F64 = npt.NDArray[np.float64]


class SpikeError(Exception):
    def __init__(self, message: str) -> None:
        self.code = message.split(":", 1)[0]
        super().__init__(message)


@cache
def lib() -> ctypes.CDLL:
    return ctypes.CDLL(os.environ["AUTAN_SPIKE_LIB"])


def _p(a: np.ndarray) -> ctypes.c_void_p:  # type: ignore[type-arg]
    assert a.flags.c_contiguous
    return ctypes.c_void_p(a.ctypes.data)


def _check(rc: int, err: ctypes.Array[ctypes.c_char]) -> None:
    if rc != 0:
        raise SpikeError(err.value.decode())


@dataclass
class Mesh:
    handle: ctypes.c_void_p
    n_cells: int
    n_owned: int
    n_faces: int
    n_interior: int
    n_incidences: int
    owner: npt.NDArray[np.int32]
    neighbour: npt.NDArray[np.int32]
    cell_face_offsets: npt.NDArray[np.int64]
    cell_faces: npt.NDArray[np.int32]
    face_area: F64  # (n_faces, 3)
    face_centroid: F64
    face_moment: F64  # (n_faces, 3, 3)
    cell_centroid: F64
    cell_volume: F64
    lsq_weight: F64
    incidence_side: F64  # (n_incidences,)

    def __del__(self) -> None:
        if self.handle:
            lib().autan_spike_mesh_free(self.handle)

    @property
    def h(self) -> float:
        return float((self.cell_volume.sum() / self.n_cells) ** (1.0 / 3.0))


def read(path: str) -> Mesh:
    """Mesh file through the core readers (autan::io::read_mesh)."""
    err = ctypes.create_string_buffer(512)
    h = ctypes.c_void_p()
    _check(
        lib().autan_spike_mesh_read(path.encode(), ctypes.byref(h), err, ctypes.c_size_t(len(err))),
        err,
    )
    return _wrap(h)


def mesh(fa: FaceArrays, n_ghost_cells: int = 0) -> Mesh:
    err = ctypes.create_string_buffer(512)
    h = ctypes.c_void_p()
    xyz = np.ascontiguousarray(fa.points, dtype=np.float64)
    rc = lib().autan_spike_mesh_from_faces(
        ctypes.c_int64(len(xyz)), _p(xyz), ctypes.c_int64(len(fa.cell0)), _p(fa.face_offsets),
        _p(fa.face_nodes), _p(fa.cell0), _p(fa.cell1), ctypes.c_int32(fa.n_cells),
        ctypes.c_int32(n_ghost_cells), ctypes.byref(h), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return _wrap(h)


def partition(m: Mesh, parts: npt.NDArray[np.int32], n_parts: int, rank: int, layers: int) -> Mesh:
    """Part `rank` of `parts` (one per cell of the unpartitioned `m`) with `layers` ghost layers,
    ghost geometry filled from `m` (autan::extract_partition)."""
    err = ctypes.create_string_buffer(512)
    h = ctypes.c_void_p()
    parts = np.ascontiguousarray(parts, dtype=np.int32)
    rc = lib().autan_spike_mesh_partition(
        m.handle, _p(parts), ctypes.c_int64(len(parts)), ctypes.c_int32(n_parts), ctypes.c_int32(rank), ctypes.c_int32(layers),
        ctypes.byref(h), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return _wrap(h)


def set_global_order(m: Mesh, stencil: str | None) -> None:
    """Order `m` by its global numbering, with a cell stencil of `stencil` ordered the same way
    (autan_spike_set_global_order): results then do not depend on the partition."""
    err = ctypes.create_string_buffer(512)
    kind = {"vertex": 0, "face2": 1, None: -1}[stencil]
    _check(
        lib().autan_spike_set_global_order(
            m.handle, ctypes.c_int(kind), err, ctypes.c_size_t(len(err))
        ),
        err,
    )


def global_ids(m: Mesh) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Global cell of each local cell and global face of each local face."""
    cells, faces = np.zeros(m.n_cells, np.int64), np.zeros(m.n_faces, np.int64)
    lib().autan_spike_global_ids(m.handle, _p(cells), _p(faces))
    return cells, faces


def incidences(m: Mesh) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], F64]:
    """Cell -> face offsets, faces and incidence sides as `m` holds them now (after
    set_global_order: in the global order, sides for the faces' global orientation)."""
    a = _arrays(m.handle)
    return a["cell_face_offsets"], a["cell_faces"], a["incidence_side"]


def _arrays(h: ctypes.c_void_p) -> dict[str, np.ndarray]:
    counts = np.zeros(5, dtype=np.int64)
    lib().autan_spike_mesh_counts(h, _p(counts))
    nc, _, nf, ni, nk = (int(c) for c in counts)
    a = {
        "owner": np.zeros(nf, np.int32),
        "neighbour": np.zeros(ni, np.int32),
        "cell_face_offsets": np.zeros(nc + 1, np.int64),
        "cell_faces": np.zeros(nk, np.int32),
        "face_area": np.zeros(3 * nf),
        "face_centroid": np.zeros(3 * nf),
        "face_moment": np.zeros(9 * nf),
        "cell_centroid": np.zeros(3 * nc),
        "cell_volume": np.zeros(nc),
        "lsq_weight": np.zeros(3 * nk),
        "incidence_side": np.zeros(nk),
    }
    lib().autan_spike_mesh_arrays(h, *(_p(v) for v in a.values()))
    return a


def _wrap(h: ctypes.c_void_p) -> Mesh:
    counts = np.zeros(5, dtype=np.int64)
    lib().autan_spike_mesh_counts(h, _p(counts))
    nc, no, nf, ni, nk = (int(c) for c in counts)
    a = _arrays(h)
    soa = {k: a[k].reshape(3, n).T.copy() for k, n in
           (("face_area", nf), ("face_centroid", nf), ("cell_centroid", nc), ("lsq_weight", nk))}  # fmt: skip
    moment = a["face_moment"].reshape(9, nf).T.reshape(nf, 3, 3).copy()
    return Mesh(h, nc, no, nf, ni, nk, a["owner"], a["neighbour"], a["cell_face_offsets"], a["cell_faces"],
                soa["face_area"], soa["face_centroid"], moment, soa["cell_centroid"], a["cell_volume"],
                soa["lsq_weight"], a["incidence_side"])  # fmt: skip


def soa(v: F64) -> F64:
    """(n, k) -> SoA flat array (component-major), the layout of generated datasets."""
    return np.ascontiguousarray(v.T).ravel()


def advdiff_residual(
    m: Mesh,
    policy: str,
    q: F64,
    q_b: F64,
    source: F64,
    velocity: tuple[float, float, float],
    nu: float,
) -> tuple[F64, F64]:
    err = ctypes.create_string_buffer(512)
    res = np.zeros(m.n_cells)
    grad = np.zeros(3 * m.n_cells)
    vel = np.array(velocity, dtype=np.float64)
    q, q_b, source = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b, source))
    rc = lib().autan_spike_advdiff_residual(
        m.handle, policy.encode(), _p(q), _p(q_b), _p(source), _p(vel), ctypes.c_double(nu),
        _p(res), _p(grad), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return res, grad.reshape(3, -1).T


def gradients(
    m: Mesh,
    policy: str,
    method: str,
    q: F64,
    q_b: F64,
    iterations: int = 1,
    grad0: F64 | None = None,
) -> tuple[F64, F64, F64]:
    """LSQ gradient and bounds, or `iterations` of corrected Green-Gauss starting from `grad0`."""
    err = ctypes.create_string_buffer(512)
    grad = np.zeros(3 * m.n_cells) if grad0 is None else soa(grad0)
    qmin = np.zeros(m.n_cells)
    qmax = np.zeros(m.n_cells)
    q, q_b = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b))
    rc = lib().autan_spike_gradients(
        m.handle, policy.encode(), method.encode(), _p(q), _p(q_b), ctypes.c_int(iterations),
        _p(grad), _p(qmin), _p(qmax), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return grad.reshape(3, -1).T, qmin, qmax


def advection(
    m: Mesh,
    policy: str,
    q: F64,
    grad: F64,
    velocity: tuple[float, float, float],
    strategy: str = "scatter",
) -> F64:
    """Residual of the section 3.3 upwind face loop on the host: the face loop ("scatter") or a
    per-cell gather ("gather-recompute", "store-gather")."""
    err = ctypes.create_string_buffer(512)
    res = np.zeros(m.n_cells)
    vel = np.array(velocity, dtype=np.float64)
    q, g = np.ascontiguousarray(q, dtype=np.float64), soa(grad)
    rc = lib().autan_spike_advection(
        m.handle,
        policy.encode(),
        strategy.encode(),
        _p(q),
        _p(g),
        _p(vel),
        _p(res),
        err,
        ctypes.c_size_t(len(err)),
    )
    _check(rc, err)
    return res


STRATEGIES = ("atomic", "warp-aggregated", "coloured", "store-gather", "gather-recompute")
DETERMINISTIC = ("coloured", "store-gather", "gather-recompute")


def gpu_advdiff_residual(
    m: Mesh,
    policy: str,
    strategy: str,
    block: int,
    q: F64,
    q_b: F64,
    source: F64,
    velocity: tuple[float, float, float],
    nu: float,
) -> tuple[F64, F64, float]:
    """Residual, gradient and the FP64 norm of the owned residuals, computed on the device."""
    err = ctypes.create_string_buffer(512)
    res = np.zeros(m.n_cells)
    grad = np.zeros(3 * m.n_cells)
    norm = ctypes.c_double()
    vel = np.array(velocity, dtype=np.float64)
    q, q_b, source = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b, source))
    rc = lib().autan_spike_gpu_advdiff_residual(
        m.handle, policy.encode(), strategy.encode(), ctypes.c_int(block), _p(q), _p(q_b), _p(source),
        _p(vel), ctypes.c_double(nu), _p(res), _p(grad), ctypes.byref(norm), err,
        ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return res, grad.reshape(3, -1).T, norm.value


def gpu_gradients(
    m: Mesh,
    policy: str,
    method: str,
    strategy: str,
    block: int,
    q: F64,
    q_b: F64,
    iterations: int = 1,
    grad0: F64 | None = None,
) -> tuple[F64, F64, F64]:
    err = ctypes.create_string_buffer(512)
    grad = np.zeros(3 * m.n_cells) if grad0 is None else soa(grad0)
    qmin = np.zeros(m.n_cells)
    qmax = np.zeros(m.n_cells)
    q, q_b = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b))
    rc = lib().autan_spike_gpu_gradients(
        m.handle, policy.encode(), method.encode(), strategy.encode(), ctypes.c_int(block), _p(q), _p(q_b),
        ctypes.c_int(iterations), _p(grad), _p(qmin), _p(qmax), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return grad.reshape(3, -1).T, qmin, qmax


def gpu_advection(
    m: Mesh,
    policy: str,
    strategy: str,
    block: int,
    q: F64,
    grad: F64,
    velocity: tuple[float, float, float],
) -> F64:
    err = ctypes.create_string_buffer(512)
    res = np.zeros(m.n_cells)
    vel = np.array(velocity, dtype=np.float64)
    q, g = np.ascontiguousarray(q, dtype=np.float64), soa(grad)
    rc = lib().autan_spike_gpu_advection(
        m.handle, policy.encode(), strategy.encode(), ctypes.c_int(block), _p(q), _p(g), _p(vel), _p(res),
        err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return res


def advdiff_derivative(
    m: Mesh,
    policy: str,
    mode: str,
    q: F64,
    q_b: F64,
    source: F64,
    velocity: tuple[float, float, float],
    nu: float,
    seed: F64,
) -> F64:
    """mode "tangent": dR for dq = seed; "adjoint": q-bar for R-bar = seed (dR/dq only)."""
    err = ctypes.create_string_buffer(512)
    out = np.zeros(m.n_cells)
    vel = np.array(velocity, dtype=np.float64)
    q, q_b, source, seed = (
        np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b, source, seed)
    )
    rc = lib().autan_spike_advdiff_derivative(
        m.handle, policy.encode(), ctypes.c_int(0 if mode == "tangent" else 1), _p(q), _p(q_b), _p(source),
        _p(vel), ctypes.c_double(nu), _p(seed), _p(out), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return out


def advdiff_complex_step(
    m: Mesh,
    q: F64,
    q_b: F64,
    source: F64,
    velocity: tuple[float, float, float],
    nu: float,
    dq: F64,
    h: float = 1e-30,
) -> F64:
    err = ctypes.create_string_buffer(512)
    out = np.zeros(m.n_cells)
    vel = np.array(velocity, dtype=np.float64)
    q, q_b, source, dq = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b, source, dq))
    rc = lib().autan_spike_advdiff_complex_step(
        m.handle, _p(q), _p(q_b), _p(source), _p(vel), ctypes.c_double(nu), _p(dq), ctypes.c_double(h),
        _p(out), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return out


def roe(m: Mesh, policy: str, mode: str, u: F64, seed: F64 | None = None) -> F64:
    """Roe face loop on U (n, 5): residual, tangent (seed dU) or adjoint (seed R-bar); (n, 5)."""
    err = ctypes.create_string_buffer(512)
    out = np.zeros(5 * m.n_cells)
    us = soa(u)
    sd = soa(seed) if seed is not None else np.zeros(5 * m.n_cells)
    code = {"residual": 0, "tangent": 1, "adjoint": 2}[mode]
    rc = lib().autan_spike_roe(
        m.handle,
        policy.encode(),
        ctypes.c_int(code),
        _p(us),
        _p(sd),
        _p(out),
        err,
        ctypes.c_size_t(len(err)),
    )
    _check(rc, err)
    return out.reshape(5, -1).T


def roe_jacobian(
    m: Mesh, policy: str, u: F64, strategy: str | None = None, block: int = 128
) -> tuple[F64, F64]:
    """Residual (n, 5) and face Jacobians (n_faces, 2, 5, 5) of the fused Roe loop, on the host
    or, with a strategy, on the device."""
    err = ctypes.create_string_buffer(512)
    res = np.zeros(5 * m.n_cells)
    jac = np.zeros(50 * m.n_faces)
    us = soa(u)
    if strategy is None:
        rc = lib().autan_spike_roe_jacobian(
            m.handle, policy.encode(), _p(us), _p(res), _p(jac), err, ctypes.c_size_t(len(err))
        )
    else:
        rc = lib().autan_spike_gpu_roe_jacobian(
            m.handle, policy.encode(), strategy.encode(), ctypes.c_int(block), _p(us), _p(res), _p(jac),
            err, ctypes.c_size_t(len(err)),
        )  # fmt: skip
    _check(rc, err)
    return res.reshape(5, -1).T, jac.reshape(50, -1).T.reshape(-1, 2, 5, 5)


def roe_complex_step(m: Mesh, u: F64, du: F64, h: float = 1e-30) -> F64:
    err = ctypes.create_string_buffer(512)
    out = np.zeros(5 * m.n_cells)
    us, ds = soa(u), soa(du)
    rc = lib().autan_spike_roe_complex_step(
        m.handle, _p(us), _p(ds), ctypes.c_double(h), _p(out), err, ctypes.c_size_t(len(err))
    )
    _check(rc, err)
    return out.reshape(5, -1).T


def lsq_bounds_derivative(m: Mesh, policy: str, mode: str, q: F64, q_b: F64, seed: F64) -> F64:
    """Tangent: seed dq (n,) -> (n, 5) = (dgrad, dqmin, dqmax). Adjoint: seed (n, 5) -> q-bar (n,)."""
    err = ctypes.create_string_buffer(512)
    q, q_b = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b))
    if mode == "tangent":
        sd, out = np.ascontiguousarray(seed, dtype=np.float64), np.zeros(5 * m.n_cells)
    else:
        sd, out = soa(seed), np.zeros(m.n_cells)
    rc = lib().autan_spike_lsq_bounds_derivative(
        m.handle, policy.encode(), ctypes.c_int(1 if mode == "tangent" else 2), _p(q), _p(q_b), _p(sd), _p(out),
        err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return out.reshape(5, -1).T if mode == "tangent" else out


def gpu_roe(
    m: Mesh, policy: str, strategy: str, block: int, mode: str, u: F64, seed: F64 | None = None
) -> F64:
    err = ctypes.create_string_buffer(512)
    out = np.zeros(5 * m.n_cells)
    us = soa(u)
    sd = soa(seed) if seed is not None else np.zeros(5 * m.n_cells)
    code = {"residual": 0, "tangent": 1, "adjoint": 2}[mode]
    rc = lib().autan_spike_gpu_roe(
        m.handle, policy.encode(), strategy.encode(), ctypes.c_int(block), ctypes.c_int(code), _p(us), _p(sd),
        _p(out), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return out.reshape(5, -1).T


def gpu_advdiff_adjoint(
    m: Mesh,
    policy: str,
    strategy: str,
    block: int,
    q: F64,
    q_b: F64,
    source: F64,
    velocity: tuple[float, float, float],
    nu: float,
    seed: F64,
) -> F64:
    err = ctypes.create_string_buffer(512)
    out = np.zeros(m.n_cells)
    vel = np.array(velocity, dtype=np.float64)
    q, q_b, source, seed = (
        np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b, source, seed)
    )
    rc = lib().autan_spike_gpu_advdiff_adjoint(
        m.handle, policy.encode(), strategy.encode(), ctypes.c_int(block), _p(q), _p(q_b), _p(source), _p(vel),
        ctypes.c_double(nu), _p(seed), _p(out), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return out


def gpu_lsq_bounds_adjoint(
    m: Mesh, policy: str, strategy: str, block: int, q: F64, q_b: F64, seed: F64
) -> F64:
    """Adjoint of the LSQ gradient + bounds reduction on the device: seed (n, 5) -> q-bar (n,)."""
    err = ctypes.create_string_buffer(512)
    out = np.zeros(m.n_cells)
    sd = soa(seed)
    q, q_b = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b))
    rc = lib().autan_spike_gpu_lsq_bounds_adjoint(
        m.handle, policy.encode(), strategy.encode(), ctypes.c_int(block), _p(q), _p(q_b), _p(sd), _p(out),
        err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return out


def set_stencil_slice(slice_: int) -> None:
    """Cells per slice of the device stencil layout of the following calls (0: CSR)."""
    lib().autan_spike_set_stencil_slice(ctypes.c_int(slice_))


def set_incidence_slice(slice_: int) -> None:
    """Cells per slice of the device incidence layout of the following calls (0: CSR)."""
    lib().autan_spike_set_incidence_slice(ctypes.c_int(slice_))


def gpu_bind_check(m: Mesh) -> tuple[bool, bool]:
    """(built-in face_area bound, a physics' own writable cell_volume left alone) by DeviceMesh."""
    err = ctypes.create_string_buffer(512)
    out = np.zeros(2, dtype=np.int32)
    _check(lib().autan_spike_gpu_bind_check(m.handle, _p(out), err, ctypes.c_size_t(len(err))), err)
    return bool(out[0]), bool(out[1])


def gpu_reductions(x: F64, flags: npt.NDArray[np.bool_]) -> tuple[float, float, int, int]:
    """Device min of x in FP64 and in FP32, and the first index where flags is set, from flags
    stored as bytes and as doubles."""
    err = ctypes.create_string_buffer(512)
    m64, m32, first, first64 = (
        ctypes.c_double(),
        ctypes.c_float(),
        ctypes.c_int64(),
        ctypes.c_int64(),
    )
    xs = np.ascontiguousarray(x, dtype=np.float64)
    fs = np.ascontiguousarray(flags, dtype=np.uint8)
    rc = lib().autan_spike_gpu_reductions(
        _p(xs), _p(fs), ctypes.c_int64(xs.size), ctypes.byref(m64), ctypes.byref(m32),
        ctypes.byref(first), ctypes.byref(first64), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return m64.value, m32.value, first.value, first64.value


def gpu_sum_of_squares(x: F64) -> float:
    """Fixed-order FP64 sum of squares computed on the device."""
    err = ctypes.create_string_buffer(512)
    out = ctypes.c_double()
    xs = np.ascontiguousarray(x, dtype=np.float64)
    rc = lib().autan_spike_gpu_sum_of_squares(
        _p(xs), ctypes.c_int64(xs.size), ctypes.byref(out), err, ctypes.c_size_t(len(err))
    )
    _check(rc, err)
    return out.value


def set_stencil(m: Mesh, kind: str | None) -> int:
    """Give the advection-diffusion residual a stencil LSQ gradient ("vertex" or "face2"), or
    back to face neighbours (None). Returns the number of stencil entries."""
    err = ctypes.create_string_buffer(512)
    size = ctypes.c_int64()
    code = {None: -1, "vertex": 0, "face2": 1}[kind]
    rc = lib().autan_spike_set_stencil(
        m.handle, ctypes.c_int(code), ctypes.byref(size), err, ctypes.c_size_t(len(err))
    )
    _check(rc, err)
    return size.value


def fill_ghost_geometry(m: Mesh, centroid: F64, volume: F64) -> None:
    """Give ghost cells the centroid (n, 3) and volume (n,) a partitioner would copy over."""
    err = ctypes.create_string_buffer(512)
    cs, vs = soa(centroid), np.ascontiguousarray(volume, dtype=np.float64)
    _check(
        lib().autan_spike_fill_ghost_geometry(
            m.handle, _p(cs), _p(vs), err, ctypes.c_size_t(len(err))
        ),
        err,
    )


def stencil_arrays(m: Mesh, size: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """Offsets and entries of the mesh's cell stencil (cell index, or -(f + 1) for face f)."""
    offsets = np.zeros(m.n_cells + 1, np.int64)
    entries = np.zeros(size, np.int32)
    lib().autan_spike_stencil_arrays(m.handle, _p(offsets), _p(entries))
    return offsets, entries


def face_weighted(
    m: Mesh, policy: str, mode: str, q: F64, q_b: F64, seed: F64 | None = None
) -> F64:
    """Test kernel codegen/tests/kernels/face_weighted.py: value, tangent or adjoint."""
    err = ctypes.create_string_buffer(512)
    out = np.zeros(m.n_cells)
    sd = np.zeros(m.n_cells) if seed is None else np.ascontiguousarray(seed, dtype=np.float64)
    q, q_b = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b))
    code = {"value": 0, "tangent": 1, "adjoint": 2}[mode]
    rc = lib().autan_spike_face_weighted(
        m.handle, policy.encode(), ctypes.c_int(code), _p(q), _p(q_b), _p(sd), _p(out), err,
        ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return out
