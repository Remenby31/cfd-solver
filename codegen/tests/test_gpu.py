"""Generated device loops: every race strategy equals the host loop to rounding, and the
deterministic strategies are bitwise reproducible across runs and launch shapes. A policy or
strategy the backend lacks is refused with its code.

Needs `--backend gpu` and AUTAN_SPIKE_LIB pointing to libautan_spike_gpu (-DAUTAN_ENABLE_CUDA=ON),
or `--backend metal` and libautan_spike_metal (-DAUTAN_ENABLE_METAL=ON).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import autan_spike as s
import meshes
import mms
import numpy as np
import pytest
from conftest import KINDS, PRECISIONS, eps, refused

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.gpu
# The most accurate policy each backend runs.
TOP = {"gpu": "fp64", "metal": "fp32+acc64"}


def is_refused(backend: str, policy: str, strategy: str, call: Callable[[], object]) -> bool:
    """True if the backend lacks `policy` or `strategy` and `call` raises its code."""
    code = refused(backend, policy, strategy)
    if code is None:
        return False
    with pytest.raises(s.SpikeError) as err:
        call()
    assert err.value.code == code
    return True


@pytest.fixture(scope="module", params=KINDS)
def case(request: pytest.FixtureRequest) -> tuple[s.Mesh, dict[str, np.ndarray]]:
    m = s.mesh(meshes.build(request.param, 8, 0.15, seed=11))
    rng = np.random.default_rng(3)
    data = {
        "q": rng.standard_normal(m.n_cells),
        "q_b": rng.standard_normal(m.n_faces),
        "src": rng.standard_normal(m.n_cells),
        "grad": rng.standard_normal((m.n_cells, 3)),
    }
    return m, data


@pytest.mark.parametrize("strategy", s.STRATEGIES)
@pytest.mark.parametrize("policy", PRECISIONS)
def test_gpu_residual_equals_host_residual(
    case: tuple[s.Mesh, dict[str, np.ndarray]], backend: str, policy: str, strategy: str
) -> None:
    m, d = case
    call = lambda: s.gpu_advection(m, policy, strategy, 128, d["q"], d["grad"], mms.VELOCITY)
    if is_refused(backend, policy, strategy, call):
        return
    host, hg = s.advdiff_residual(m, policy, d["q"], d["q_b"], d["src"], mms.VELOCITY, mms.NU)
    dev, dg, norm = s.gpu_advdiff_residual(
        m, policy, strategy, 128, d["q"], d["q_b"], d["src"], mms.VELOCITY, mms.NU
    )
    scale = np.abs(host).max()
    assert np.abs(dev - host).max() <= 200 * eps(policy) * scale
    owned = dev[: m.n_cells]
    assert abs(norm - np.sqrt(np.sum(owned * owned))) <= 1e-13 * norm
    assert np.abs(dg - hg).max() <= 200 * eps(policy) * np.abs(hg).max()
    adv_host = s.advection(m, policy, d["q"], d["grad"], mms.VELOCITY)
    adv_dev = s.gpu_advection(m, policy, strategy, 128, d["q"], d["grad"], mms.VELOCITY)
    assert np.abs(adv_dev - adv_host).max() <= 200 * eps(policy) * np.abs(adv_host).max()


@pytest.mark.parametrize("strategy", s.STRATEGIES)
def test_gpu_green_gauss_equals_host(
    case: tuple[s.Mesh, dict[str, np.ndarray]], backend: str, strategy: str
) -> None:
    m, d = case
    pol = TOP[backend]
    call = lambda: s.gpu_gradients(m, pol, "gg", strategy, 128, d["q"], d["q_b"], 3, d["grad"])
    if is_refused(backend, pol, strategy, call):
        return
    host, _, _ = s.gradients(m, pol, "gg", d["q"], d["q_b"], 3, d["grad"])
    dev, _, _ = call()
    assert np.abs(dev - host).max() <= 200 * eps(pol) * np.abs(host).max()
    lsq_h, lo_h, hi_h = s.gradients(m, pol, "lsq", d["q"], d["q_b"])
    lsq_d, lo_d, hi_d = s.gpu_gradients(m, pol, "lsq", strategy, 128, d["q"], d["q_b"])
    np.testing.assert_allclose(lsq_d, lsq_h, rtol=0, atol=200 * eps(pol) * np.abs(lsq_h).max())
    np.testing.assert_array_equal(lo_d, lo_h)
    np.testing.assert_array_equal(hi_d, hi_h)


@pytest.mark.parametrize("strategy", s.DETERMINISTIC)
@pytest.mark.parametrize("policy", PRECISIONS)
def test_deterministic_strategies_are_bitwise_reproducible(
    case: tuple[s.Mesh, dict[str, np.ndarray]], backend: str, policy: str, strategy: str
) -> None:
    # Residual, gradients, the residual norm computed on the device, and the adjoint of the
    # composed residual (whose LSQ-gradient reduction adjoint scatters to neighbours): bitwise
    # equal over 10 runs, two launch shapes and, on CUDA and HIP, both incidence layouts (CSR
    # and sliced, which visits each cell's faces in the same order) (research section 8).
    m, d = case
    call = lambda: s.gpu_advection(m, policy, strategy, 128, d["q"], d["grad"], mms.VELOCITY)
    if is_refused(backend, policy, strategy, call):
        return
    ybar = np.random.default_rng(6).standard_normal(m.n_cells)
    runs = []
    try:
        for k in range(10):
            block = 64 if k % 2 else 256
            if backend != "metal":  # DeviceMesh (and its sliced layouts) is CUDA/HIP
                s.set_incidence_slice(32 if k % 3 == 2 else 0)
            res, grad, norm = s.gpu_advdiff_residual(
                m, policy, strategy, block, d["q"], d["q_b"], d["src"], mms.VELOCITY, mms.NU
            )
            gg, _, _ = s.gpu_gradients(
                m, policy, "gg", strategy, block, d["q"], d["q_b"], 2, d["grad"]
            )
            qbar = s.gpu_advdiff_adjoint(
                m, policy, strategy, block, d["q"], d["q_b"], d["src"], mms.VELOCITY, mms.NU, ybar
            )
            runs.append((res.tobytes(), grad.tobytes(), gg.tobytes(), norm, qbar.tobytes()))
    finally:
        if backend != "metal":
            s.set_incidence_slice(0)
    assert len(set(runs)) == 1


@pytest.mark.parametrize("strategy", s.STRATEGIES)
@pytest.mark.parametrize("policy", PRECISIONS)
def test_gpu_tangent_and_adjoint_equal_host(
    case: tuple[s.Mesh, dict[str, np.ndarray]], backend: str, policy: str, strategy: str
) -> None:
    m, d = case
    call = lambda: s.gpu_advection(m, policy, strategy, 128, d["q"], d["grad"], mms.VELOCITY)
    if is_refused(backend, policy, strategy, call):
        return
    rng = np.random.default_rng(4)
    rho = 1.0 + 0.2 * rng.random(m.n_cells)
    vel = 0.3 * rng.standard_normal((m.n_cells, 3))
    p = 1.0 + 0.2 * rng.random(m.n_cells)
    u = np.column_stack([rho, rho[:, None] * vel, p / 0.4 + 0.5 * rho * np.sum(vel * vel, axis=1)])
    x = rng.standard_normal(u.shape)
    for mode in ("residual", "tangent", "adjoint"):
        host = s.roe(m, policy, mode, u, x)
        dev = s.gpu_roe(m, policy, strategy, 128, mode, u, x)
        assert np.abs(dev - host).max() <= 500 * eps(policy) * np.abs(host).max()
    y = rng.standard_normal(m.n_cells)
    host = s.advdiff_derivative(
        m, policy, "adjoint", d["q"], d["q_b"], d["src"], mms.VELOCITY, mms.NU, y
    )
    dev = s.gpu_advdiff_adjoint(
        m, policy, strategy, 128, d["q"], d["q_b"], d["src"], mms.VELOCITY, mms.NU, y
    )
    assert np.abs(dev - host).max() <= 500 * eps(policy) * np.abs(host).max()


@pytest.mark.parametrize("strategy", s.DETERMINISTIC)
def test_face_loop_adjoints_are_bitwise_reproducible(
    case: tuple[s.Mesh, dict[str, np.ndarray]], strategy: str
) -> None:  # fp32+acc64 runs on every backend
    # The adjoint of a face loop is a face loop: it runs in the same race strategies. (The adjoint
    # of a cell reduction scatters with atomics in this spike and is not covered.)
    m, _ = case
    rng = np.random.default_rng(5)
    rho = 1.0 + 0.2 * rng.random(m.n_cells)
    vel = 0.3 * rng.standard_normal((m.n_cells, 3))
    u = np.column_stack([rho, rho[:, None] * vel, 2.5 + 0.5 * rho * np.sum(vel * vel, axis=1)])
    y = rng.standard_normal(u.shape)
    runs = {
        s.gpu_roe(m, "fp32+acc64", strategy, 64 if k % 2 else 256, "adjoint", u, y).tobytes()
        for k in range(10)
    }
    assert len(runs) == 1


@pytest.mark.parametrize("strategy", s.STRATEGIES)
def test_ghost_cells_receive_their_partial_sums_in_every_strategy(
    backend: str, strategy: str
) -> None:
    # Faces between owned and ghost cells increment the ghost's residual (mesh contract,
    # invariant 6): a reverse halo exchange sends those partial sums to the owner.
    fa, n_ghost = meshes.with_ghost_layer(meshes.build("hex", 6), 6)
    m = s.mesh(fa, n_ghost)
    rng = np.random.default_rng(14)
    u = np.column_stack([1.0 + 0.1 * rng.random(m.n_cells), 0.2 * rng.standard_normal((m.n_cells, 3)),
                         2.5 + 0.1 * rng.random(m.n_cells)])  # fmt: skip
    pol = TOP[backend]
    call = lambda: s.gpu_roe(m, pol, strategy, 64, "residual", u)
    if is_refused(backend, pol, strategy, call):
        return
    host = s.roe(m, pol, "residual", u)
    dev = call()
    ghosts = slice(m.n_cells - n_ghost, m.n_cells)
    assert np.abs(host[ghosts]).max() > 0.0
    assert np.abs(dev - host).max() <= 500 * eps(pol) * np.abs(host).max()


@pytest.mark.parametrize("strategy", s.STRATEGIES)
@pytest.mark.parametrize("policy", PRECISIONS)
def test_reduction_adjoints_with_min_max_and_finalise_equal_host(
    case: tuple[s.Mesh, dict[str, np.ndarray]], backend: str, policy: str, strategy: str
) -> None:
    # Values rounded to one decimal, so Min/Max tie often; both GPU forms (atomic scatter and the
    # deterministic two passes) must send each adjoint to the incidence the host selects.
    m, _ = case
    rng = np.random.default_rng(15)
    q = np.round(rng.standard_normal(m.n_cells), 1)
    q_b = np.round(rng.standard_normal(m.n_faces), 1)
    seed = rng.standard_normal((m.n_cells, 5))
    call = lambda: s.gpu_lsq_bounds_adjoint(m, policy, strategy, 64, q, q_b, seed)
    if is_refused(backend, policy, strategy, call):
        return
    host = s.lsq_bounds_derivative(m, policy, "adjoint", q, q_b, seed)
    dev = call()
    assert np.abs(dev - host).max() <= 500 * eps(policy) * np.abs(host).max()


def test_device_mesh_binds_built_in_geometry_and_leaves_physics_datasets(backend: str) -> None:
    if backend == "metal":
        pytest.skip("DeviceMesh is CUDA/HIP (autan/kernels/device.cuh)")
    m = s.mesh(meshes.build("hex", 3))
    assert s.gpu_bind_check(m) == (True, True)


@pytest.mark.parametrize("n", [0, 1, 255, 256, 257, 65_537, 1_000_003])
def test_device_min_and_first_index_are_exact(backend: str, n: int) -> None:
    # Global time step and first diverged cell: exact, NaN never the minimum, -0 before +0.
    if backend == "metal":
        pytest.skip("gpu::min and gpu::first_index are CUDA/HIP (autan/kernels/device.cuh)")
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    flags = rng.random(n) < 3.0 / max(n, 1)
    if n > 2:
        x[n // 3] = np.nan
    m64, m32, first, first64 = s.gpu_reductions(x, flags)
    if n == 0:
        assert (m64, m32, first, first64) == (np.inf, np.float32(np.inf), 0, 0)
        return
    assert m64 == np.nanmin(x)
    assert m32 == np.nanmin(x.astype(np.float32))
    assert first == first64 == (np.flatnonzero(flags)[0] if flags.any() else n)
    zeros = np.where(np.arange(n) % 2 == 0, 0.0, -0.0) if n > 1 else np.zeros(n)
    z64, z32, _, _ = s.gpu_reductions(zeros, flags)
    assert np.signbit(z64) == np.signbit(z32) == (n > 1)


@pytest.mark.parametrize("n", [0, 1, 4095, 4096, 4097, 250_001])
def test_device_sum_of_squares_is_exact_to_rounding_and_order_fixed(backend: str, n: int) -> None:
    x = np.random.default_rng(n).standard_normal(n)
    if is_refused(backend, "fp64", "", lambda: s.gpu_sum_of_squares(x)):
        return
    # Reference: correctly rounded sum (math.fsum). Pairwise trees err by O(log n) ulps.
    got = s.gpu_sum_of_squares(x)
    ref = math.fsum(x * x)
    assert abs(got - ref) <= 64 * np.finfo(np.float64).eps * ref
    assert {s.gpu_sum_of_squares(x) for _ in range(5)} == {got}


@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("policy", PRECISIONS)
def test_mms_solution_on_the_device_converges_at_second_order(
    backend: str, kind: str, policy: str
) -> None:
    # The steady manufactured solution solved with the device residual (deterministic strategy).
    m = s.mesh(meshes.build(kind, 4, 0.15, seed=4))
    call = lambda: mms.solve(m, policy, 1e-6, "gather-recompute")
    if is_refused(backend, policy, "gather-recompute", call):
        return
    h, e = [], []
    for n in (8, 16, 24) if kind == "poly" else (4, 8, 16):
        m = s.mesh(meshes.build(kind, n, 0.15, seed=n))
        sol = mms.solve(m, policy, 1e-12 if policy == "fp64" else 1e-6, "gather-recompute")
        h.append(m.h)
        e.append(sol.error_l2)
    fit, _ = mms.observed_order(h, e)
    assert fit >= 1.9


@pytest.mark.parametrize("scale", [1e-30, 1e-20, 1.0, 1e20])
@pytest.mark.parametrize("policy", PRECISIONS)
def test_device_residual_norm_holds_at_extreme_amplitudes(
    case: tuple[s.Mesh, dict[str, np.ndarray]], backend: str, policy: str, scale: float
) -> None:
    # Squares of 1e-20 and 1e20 leave FP32's exponent range: the norm must still be the norm of
    # the residual the device returns (computed here in FP64 from those values).
    m, d = case
    src = scale * d["src"]
    zero_q, zero_b = np.zeros(m.n_cells), np.zeros(m.n_faces)
    call = lambda: s.gpu_advdiff_residual(
        m, policy, "gather-recompute", 128, zero_q, zero_b, src, mms.VELOCITY, mms.NU
    )
    if is_refused(backend, policy, "gather-recompute", call):
        return
    res, _, norm = call()
    expected = float(np.sqrt(np.sum(res * res)))
    assert expected > 0.0
    assert abs(norm - expected) <= 1e3 * eps("fp64") * expected


@pytest.mark.parametrize("stencil", ["vertex", "face2"])
@pytest.mark.parametrize("strategy", s.STRATEGIES)
def test_stencil_gradient_residual_and_adjoint_equal_host(
    backend: str, stencil: str, strategy: str
) -> None:
    pol = TOP[backend]
    m = s.mesh(meshes.build("tet", 5, 0.15, seed=12))
    s.set_stencil(m, stencil)
    rng = np.random.default_rng(16)
    q, q_b, src = (
        rng.standard_normal(m.n_cells),
        rng.standard_normal(m.n_faces),
        rng.standard_normal(m.n_cells),
    )
    ybar = rng.standard_normal(m.n_cells)
    call = lambda: s.gpu_advdiff_adjoint(
        m, pol, strategy, 64, q, q_b, src, mms.VELOCITY, mms.NU, ybar
    )
    if is_refused(backend, pol, strategy, call):
        return
    host, hg = s.advdiff_residual(m, pol, q, q_b, src, mms.VELOCITY, mms.NU)
    hbar = s.advdiff_derivative(m, pol, "adjoint", q, q_b, src, mms.VELOCITY, mms.NU, ybar)
    runs = set()
    for block in (64, 256, 64, 256):
        dev, dg, _ = s.gpu_advdiff_residual(
            m, pol, strategy, block, q, q_b, src, mms.VELOCITY, mms.NU
        )
        dbar = s.gpu_advdiff_adjoint(
            m, pol, strategy, block, q, q_b, src, mms.VELOCITY, mms.NU, ybar
        )
        assert np.abs(dev - host).max() <= 500 * eps(pol) * np.abs(host).max()
        assert np.abs(dg - hg).max() <= 500 * eps(pol) * np.abs(hg).max()
        assert np.abs(dbar - hbar).max() <= 500 * eps(pol) * np.abs(hbar).max()
        runs.add((dev.tobytes(), dbar.tobytes()))
    if strategy in s.DETERMINISTIC:
        assert len(runs) == 1
    # The sliced stencil layout visits each cell's entries in the same order: same bits.
    if backend == "metal":
        return  # DeviceMesh (and its sliced layout) is CUDA/HIP
    s.set_stencil_slice(32)
    try:
        dev_s, _, _ = s.gpu_advdiff_residual(
            m, pol, strategy, 64, q, q_b, src, mms.VELOCITY, mms.NU
        )
        dbar_s = s.gpu_advdiff_adjoint(
            m, pol, strategy, 64, q, q_b, src, mms.VELOCITY, mms.NU, ybar
        )
    finally:
        s.set_stencil_slice(0)
    if strategy in s.DETERMINISTIC:
        assert (dev_s.tobytes(), dbar_s.tobytes()) in runs
    else:
        assert np.abs(dev_s - host).max() <= 500 * eps(pol) * np.abs(host).max()
        assert np.abs(dbar_s - hbar).max() <= 500 * eps(pol) * np.abs(hbar).max()


@pytest.mark.parametrize("stencil", ["vertex", "face2"])
@pytest.mark.parametrize("strategy", s.STRATEGIES)
def test_stencil_adjoint_sends_ghost_cells_their_partial_sums(
    backend: str, stencil: str, strategy: str
) -> None:
    pol = TOP[backend]
    # Owned cells next to the ghost layer send adjoint contributions to ghost cells through the
    # stencil; every strategy must deliver them (the host scatters them directly).
    full = s.mesh(meshes.build("hex", 6))
    fa, n_ghost = meshes.with_ghost_layer(meshes.build("hex", 6), 6)
    m = s.mesh(fa, n_ghost)
    s.fill_ghost_geometry(m, full.cell_centroid, full.cell_volume)
    s.set_stencil(m, stencil)
    rng = np.random.default_rng(17)
    q, q_b, src = (
        rng.standard_normal(m.n_cells),
        rng.standard_normal(m.n_faces),
        rng.standard_normal(m.n_cells),
    )
    ybar = rng.standard_normal(m.n_cells)
    call = lambda: s.gpu_advdiff_adjoint(
        m, pol, strategy, 64, q, q_b, src, mms.VELOCITY, mms.NU, ybar
    )
    if is_refused(backend, pol, strategy, call):
        return
    host = s.advdiff_derivative(m, pol, "adjoint", q, q_b, src, mms.VELOCITY, mms.NU, ybar)
    dev = s.gpu_advdiff_adjoint(m, pol, strategy, 64, q, q_b, src, mms.VELOCITY, mms.NU, ybar)
    ghosts = slice(m.n_cells - n_ghost, m.n_cells)
    assert np.abs(host[ghosts]).max() > 0.0
    assert np.abs(dev - host).max() <= 500 * eps(pol) * np.abs(host).max()
