"""Generated loops on a partition with two ghost layers (mesh contract, invariant 6)."""

from __future__ import annotations

import autan_spike as s
import meshes
import mms
import numpy as np
import pytest
from conftest import KINDS, eps, refused

DEVICE_POLICY = {"cpu": "fp64", "gpu": "fp64", "metal": "fp32+acc64"}
WHERE = ["host", *(pytest.param(x, marks=pytest.mark.gpu) for x in s.STRATEGIES)]


@pytest.mark.parametrize("where", WHERE)
@pytest.mark.parametrize("kind", KINDS)
def test_face_loops_skip_halo_faces_of_a_two_layer_partition(
    backend: str, kind: str, where: str
) -> None:
    # A one-layer partition has no halo faces: its residual on owned cells and its partial sums
    # on ghost cells (the faces they share with owned cells) are what the two-layer partition
    # must give. Faces between two ghosts and boundary faces of ghosts add nothing.
    pol = DEVICE_POLICY[backend] if where != "host" else "fp64"
    if where != "host" and refused(backend, pol, where):
        pytest.skip(f"{backend} refuses {pol} with {where}")
    full = s.mesh(meshes.build(kind, 6, 0.15, seed=21))
    x = full.cell_centroid[:, 0]
    parts = (x > np.median(x)).astype(np.int32)
    rng = np.random.default_rng(22)
    q, src = rng.standard_normal(full.n_cells), rng.standard_normal(full.n_cells)
    q_b = rng.standard_normal(full.n_faces)

    def residual(m: s.Mesh) -> dict[int, float]:
        cells, faces = s.global_ids(m)
        args = (q[cells], q_b[faces], src[cells], mms.VELOCITY, mms.NU)
        if where == "host":
            res, _ = s.advdiff_residual(m, pol, *args)
        else:
            res, _, _ = s.gpu_advdiff_residual(m, pol, where, 64, *args)
        return dict(zip(cells.tolist(), res.tolist(), strict=True))

    for rank in (0, 1):
        two = s.partition(full, parts, 2, rank, layers=2)
        one = s.partition(full, parts, 2, rank, layers=1)
        assert two.n_faces > one.n_faces  # the halo faces this test is about
        r2, r1 = residual(two), residual(one)
        scale = max(abs(v) for v in r1.values())
        for g, v in r2.items():
            assert abs(v - r1.get(g, 0.0)) <= 500 * eps(pol) * scale, g


# Gathers sum each cell in its incidence order; the coloured order depends on the numbering.
GATHERS = ("store-gather", "gather-recompute")
WHERE_DETERMINISTIC = ["host", *(pytest.param(x, marks=pytest.mark.gpu) for x in GATHERS)]


@pytest.mark.parametrize("where", WHERE_DETERMINISTIC)
@pytest.mark.parametrize("kind", KINDS)
def test_results_are_bit_identical_across_rank_counts(backend: str, kind: str, where: str) -> None:
    # Ordered by the global numbering (runtime.md, Ranks), a cell sums its faces by global face
    # id and a face is evaluated in its global orientation, so every owned cell of a 2-rank
    # partition holds the bits of the same cell of a 1-rank partition. Compared: the advection
    # face loop, the face2 stencil gradient, the incidence LSQ gradient with its bounds, and the
    # sides of the cell's faces (geometry.incidence_side). The
    # advection-diffusion residual is not: it reads ghost gradients, which a solver fills by a
    # halo exchange between the two loops and this test does not.
    if where != "host" and backend == "metal":
        pytest.skip("the global order is a DeviceMesh option (CUDA/HIP)")
    full = s.mesh(meshes.build(kind, 6, 0.15, seed=31))
    x = full.cell_centroid[:, 0]
    parts = (x > np.median(x)).astype(np.int32)
    rng = np.random.default_rng(32)
    q, src = rng.standard_normal(full.n_cells), rng.standard_normal(full.n_cells)
    grad, q_b = rng.standard_normal((full.n_cells, 3)), rng.standard_normal(full.n_faces)

    def owned(m: s.Mesh) -> dict[int, bytes]:
        cells, faces = s.global_ids(m)
        args = (q[cells], q_b[faces], src[cells], mms.VELOCITY, mms.NU)
        if where == "host":
            adv = s.advection(m, "fp64", q[cells], grad[cells], mms.VELOCITY, "gather-recompute")
            _, stencil_grad = s.advdiff_residual(m, "fp64", *args)
            lsq, lo, hi = s.gradients(m, "fp64", "lsq", q[cells], q_b[faces])
        else:
            adv = s.gpu_advection(m, "fp64", where, 64, q[cells], grad[cells], mms.VELOCITY)
            _, stencil_grad, _ = s.gpu_advdiff_residual(m, "fp64", where, 64, *args)
            lsq, lo, hi = s.gpu_gradients(m, "fp64", "lsq", where, 64, q[cells], q_b[faces])
        # Each cell's faces by global id with their sides (+1 for the global left cell).
        off, cell_faces, side = s.incidences(m)
        sides = [
            faces[cell_faces[off[c] : off[c + 1]]] * side[off[c] : off[c + 1]]
            for c in range(m.n_owned)
        ]
        return {
            int(cells[c]): b"".join(a[c].tobytes() for a in (adv, stencil_grad, lsq, lo, hi))
            + sides[c].tobytes()
            for c in range(m.n_owned)
        }

    one = s.partition(full, np.zeros(full.n_cells, np.int32), 1, 0, layers=2)
    s.set_global_order(one, "face2")
    reference = owned(one)
    for rank in (0, 1):
        two = s.partition(full, parts, 2, rank, layers=2)
        s.set_global_order(two, "face2")
        got = owned(two)
        assert got.keys() <= reference.keys()
        assert all(got[g] == reference[g] for g in got), kind


NUMBERED = [
    "scatter",
    *(pytest.param(x, marks=pytest.mark.gpu) for x in ("atomic", "warp-aggregated", "coloured")),
]


@pytest.mark.parametrize("strategy", NUMBERED)
def test_orders_that_follow_the_numbering_are_refused_when_rank_independent(
    backend: str, strategy: str
) -> None:
    # Scatter, atomic, warp-aggregated and coloured sum in an order set by the local numbering
    # (a coloured run on 2 ranks differs from 1 rank): a rank-independent topology refuses them.
    if strategy != "scatter" and backend == "metal":
        pytest.skip("the global order is a DeviceMesh option (CUDA/HIP)")
    full = s.mesh(meshes.build("hex", 4))
    m = s.partition(full, np.zeros(full.n_cells, np.int32), 1, 0, layers=2)
    s.set_global_order(m, None)
    q, grad = np.zeros(m.n_cells), np.zeros((m.n_cells, 3))
    run = (
        (lambda: s.advection(m, "fp64", q, grad, mms.VELOCITY, "scatter"))
        if strategy == "scatter"
        else (lambda: s.gpu_advection(m, "fp64", strategy, 64, q, grad, mms.VELOCITY))
    )
    with pytest.raises(s.SpikeError) as err:
        run()
    assert err.value.code == "E1109"
