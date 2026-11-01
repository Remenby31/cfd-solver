"""Nested mesh families of the periodic square [-10, 10]^2 of the convected isentropic vortex
(Spiegel, Huynh and DeBonis, AIAA 2015-2444), one cell thick in z.

`vortex-hex`: n x n quadrilaterals. `vortex-prism`: triangles from an 8 x 8 lattice whose nodes are
moved by a seeded jitter (along the boundary only tangentially, the same on both sides of a
periodic pair) and whose squares are split along a seeded diagonal; every finer level splits each
triangle into four at its edge midpoints, so a level is the next finer one with every other line
removed; `vortex-hex` takes any n, and its levels are nested when they double, as these are. Both
are extruded over a thickness of 1 at every level, as the order study of a 2D oracle requires (one
layer, one domain); its in-plane size h = sqrt(area / N) is 20/n for quadrilaterals and 20/(sqrt(2) n)
for triangles, two per lattice square. The zones xmin
and xmax, ymin and ymax are periodic pairs, translated by 20 along x and y; zmin and zmax bound
the layer. A case on these meshes sets zmin and zmax to symmetry and names no periodic zone in its
boundary rules (E0424).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from autan_vv.errors import VVError

if TYPE_CHECKING:
    from autan_vv.meshes.facemesh import FloatArray, IntArray

KINDS = ("vortex-hex", "vortex-prism")
SIDE = 20.0
LOWER = -10.0
COARSE = 8
# (zone a, zone b, translation): x_b = x_a + translation (mesh.md, Periodic pairs).
PERIODIC = (("xmin", "xmax", (SIDE, 0.0, 0.0)), ("ymin", "ymax", (0.0, SIDE, 0.0)))
JITTER = 0.25
THICKNESS = 1.0


def _lattice(seed: int) -> tuple[FloatArray, IntArray]:
    """The coarse triangulation: (COARSE + 1)^2 jittered nodes in the plane and its triangles,
    counter-clockwise."""
    m = COARSE
    h = SIDE / m
    rng = np.random.default_rng(np.random.SeedSequence([seed, m, 1]))
    # One jitter per node of the periodic lattice, so that the copies of a node on both sides of a
    # pair move alike; nodes on x = const lines move only along y, and conversely.
    jitter = rng.uniform(-JITTER * h, JITTER * h, size=(m, m, 2))
    i, j = np.meshgrid(np.arange(m + 1), np.arange(m + 1), indexing="ij")
    xy = np.stack([LOWER + h * i, LOWER + h * j], axis=-1).astype(float)
    d = jitter[i % m, j % m]
    d[..., 0] = np.where(i % m == 0, 0.0, d[..., 0])
    d[..., 1] = np.where(j % m == 0, 0.0, d[..., 1])
    xy += d
    node = (i * (m + 1) + j)[:-1, :-1].ravel()
    a, b, c, e = node, node + (m + 1), node + (m + 2), node + 1
    flip = rng.integers(0, 2, size=a.size).astype(bool)
    first = np.where(flip[:, None], np.stack([a, b, e], 1), np.stack([a, b, c], 1))
    second = np.where(flip[:, None], np.stack([b, c, e], 1), np.stack([a, c, e], 1))
    return xy.reshape(-1, 2), np.concatenate([first, second])


def _refine(xy: FloatArray, tri: IntArray) -> tuple[FloatArray, IntArray]:
    """Each triangle split into four at its edge midpoints, counter-clockwise kept."""
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    key = np.sort(edges, axis=1)
    unique, inverse = np.unique(key, axis=0, return_inverse=True)
    mid = len(xy) + inverse.reshape(3, -1)  # midpoint of edge 01, 12, 20 of each triangle
    xy = np.concatenate([xy, 0.5 * (xy[unique[:, 0]] + xy[unique[:, 1]])])
    m01, m12, m20 = mid
    a, b, c = tri.T
    tri = np.concatenate(
        [
            np.stack([a, m01, m20], 1),
            np.stack([m01, b, m12], 1),
            np.stack([m20, m12, c], 1),
            np.stack([m01, m12, m20], 1),
        ]
    )
    return xy, tri


def grid(kind: str, n: int, seed: int = 0) -> tuple[FloatArray, dict[str, IntArray]]:
    """Points and cells (Gmsh node order) of level `n` of family `kind`."""
    dz = THICKNESS
    if kind == "vortex-hex":
        axis = LOWER + SIDE * np.arange(n + 1) / n
        gx, gy = np.meshgrid(axis, axis, indexing="ij")
        xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        p = (i * (n + 1) + j).ravel()
        quads = np.stack([p, p + (n + 1), p + (n + 2), p + 1], axis=1)
        top = len(xy)
        points = np.concatenate([np.c_[xy, np.zeros(top)], np.c_[xy, np.full(top, dz)]])
        return points, {"hex": np.concatenate([quads, quads + top], axis=1)}
    refinements = 0
    while COARSE * 2**refinements < n:
        refinements += 1
    if kind != "vortex-prism" or COARSE * 2**refinements != n:
        msg = f"{kind} has levels n = {COARSE} x 2^k (vortex-hex: any n), got n = {n}"
        raise VVError(msg, code="V0309")
    xy, tri = _lattice(seed)
    for _ in range(refinements):
        xy, tri = _refine(xy, tri)
    top = len(xy)
    points = np.concatenate([np.c_[xy, np.zeros(top)], np.c_[xy, np.full(top, dz)]])
    return points, {"prism": np.concatenate([tri, tri + top], axis=1)}
