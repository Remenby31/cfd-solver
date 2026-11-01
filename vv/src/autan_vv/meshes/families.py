"""Seeded mesh families of the unit cube for observed-order studies.

Kinds: `tet` (unstructured Delaunay), `hex` (structured), `prism` (unstructured triangles extruded
in z), `hybrid` (hex slab, tet slab with pyramids at the quad interface, prism slab) and `poly`
(median-dual polyhedra of the tet mesh) and `agglomerated` (the median dual with the cells of every
other box of a lattice of edge about 4/n merged, giving non-convex polyhedra with about a hundred faces or more
among ordinary ones). A positive `perturbation` moves every node by a seeded
random vector of length up to `perturbation` times its shortest incident edge, tangentially on the
boundary, to make the "dirty" variant of a family.

Gmsh (GPL-2.0-or-later) is a development-time dependency of this package only; it is imported
lazily so that the rest of `autan_vv` works without it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from autan_vv.errors import VVError
from autan_vv.meshes import ringleb, vortex
from autan_vv.meshes.dual import median_dual
from autan_vv.meshes.facemesh import FaceMesh, FloatArray, IntArray, from_faces, merge_cells
from autan_vv.meshes.fluent import write_fluent

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

KINDS = ("tet", "hex", "prism", "hybrid", "poly", "agglomerated")
BOX_ZONES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
# The y = 0 face is the wall of the RANS-SA manufactured solution (wall distance d = y).
BC_TYPES = {name: ("wall" if name == "ymin" else "pressure-far-field") for name in BOX_ZONES}
CELL_ZONE = "fluid"
MANIFEST_SCHEMA = "autan.mesh-family/1"

# Local faces of Gmsh linear elements, ordered so that the right-hand normal points out of a
# positively oriented element (Gmsh reference elements, "Node ordering" in the Gmsh manual).
_LOCAL_FACES: dict[str, tuple[tuple[int, ...], ...]] = {
    "tet": ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
    "hex": ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
    "prism": ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
    "pyramid": ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
}
_GMSH_TYPES = {4: "tet", 5: "hex", 6: "prism", 7: "pyramid"}
_NODES = {"tet": 4, "hex": 8, "prism": 6, "pyramid": 5}
# Gmsh writes elements grouped by type number (tet 4, hex 5, prism 6, pyramid 7): cells follow the
# same order so that cell i of the FaceMesh is element i of the file.
_CELL_ORDER = ("tet", "hex", "prism", "pyramid")


def _gmsh() -> ModuleType:
    try:
        import gmsh  # noqa: PLC0415  (GPL tool, imported only when generating meshes)
    except ImportError as exc:
        msg = "gmsh is not installed: run `uv sync` with the 'mesh' group on macOS or x86-64"
        raise VVError(msg, code="V0303") from exc
    module: ModuleType = gmsh
    return module


@dataclass(frozen=True)
class Level:
    """One generated mesh of a family, as recorded in the family manifest."""

    n: int
    file: str
    file_sha256: str
    mesh_sha256: str
    n_cells: int
    n_faces: int
    n_points: int
    cell_counts: dict[str, int]
    min_subvolume_ratio: float
    volume: float
    h: float


@dataclass(frozen=True)
class Family:
    """A mesh family and the provenance needed to regenerate it bit for bit."""

    kind: str
    perturbation: float
    seed: int
    gmsh_version: str
    levels: tuple[Level, ...]

    def manifest(self) -> dict[str, Any]:
        return {"schema": MANIFEST_SCHEMA, **asdict(self)}


def level_record(n: int, path: Path, mesh: FaceMesh) -> Level:
    """The manifest entry of level `n`, written to `path` from `mesh`."""
    volume = float(mesh.geometry.cell_volume.sum())
    counts: dict[str, int] = {}
    for k in mesh.cell_kinds or ("polyhedron",) * mesh.n_cells:
        counts[k] = counts.get(k, 0) + 1
    return Level(
        n=n,
        file=path.name,
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        mesh_sha256=mesh.sha256(),
        n_cells=mesh.n_cells,
        n_faces=mesh.n_faces,
        n_points=int(mesh.points.shape[0]),
        cell_counts=counts,
        min_subvolume_ratio=mesh.min_subvolume_ratio(),
        volume=volume,
        h=(volume / mesh.n_cells) ** (1.0 / 3.0),
    )


def write_manifest(family: Family, out_dir: Path) -> None:
    """Write `manifest.json` (schema autan.mesh-family/1) of `family` into `out_dir`."""
    manifest = json.dumps(family.manifest(), indent=2, allow_nan=False)
    (out_dir / "manifest.json").write_text(manifest + "\n")


def _build_geometry(gmsh: ModuleType, kind: str, n: int) -> None:
    geo = gmsh.model.geo
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    pts = [geo.addPoint(*c) for c in corners]
    lines = [geo.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    base = geo.addPlaneSurface([geo.addCurveLoop(lines)])
    structured_base = kind == "hex"
    if structured_base:
        for c in lines:
            geo.mesh.setTransfiniteCurve(c, n + 1)
        geo.mesh.setTransfiniteSurface(base)
        geo.mesh.setRecombine(2, base)
    if kind == "hex":
        geo.extrude([(2, base)], 0, 0, 1, numElements=[n], recombine=True)
    elif kind in ("tet", "poly"):
        geo.extrude([(2, base)], 0, 0, 1)
    elif kind == "prism":
        geo.extrude([(2, base)], 0, 0, 1, numElements=[n], recombine=True)
    else:
        msg = f"unknown mesh kind {kind!r}; expected one of {KINDS}"
        raise VVError(msg, code="V0304")
    geo.synchronize()


def _nodes(gmsh: ModuleType) -> tuple[IntArray, FloatArray]:
    tags, coords, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    return np.asarray(tags)[order].astype(np.int64), np.asarray(coords).reshape(-1, 3)[order]


def _cells(gmsh: ModuleType, index: dict[int, int] | IntArray) -> dict[str, IntArray]:
    out: dict[str, IntArray] = {}
    types, _, node_tags = gmsh.model.mesh.getElements(3)
    for t, nodes in zip(types, node_tags, strict=True):
        kind = _GMSH_TYPES.get(int(t))
        if kind is None:
            msg = f"unexpected Gmsh element type {t}"
            raise VVError(msg, code="V0305")
        out[kind] = np.asarray(index)[np.asarray(nodes, dtype=np.int64)].reshape(-1, _NODES[kind])
    return out


def _min_edge(points: FloatArray, cells: dict[str, IntArray]) -> FloatArray:
    """Shortest distance from each node to another node of a cell containing it (a lower bound on edges)."""
    h = np.full(points.shape[0], np.inf)
    for conn in cells.values():
        k = conn.shape[1]
        for i in range(k):
            for j in range(i + 1, k):
                d = np.linalg.norm(points[conn[:, i]] - points[conn[:, j]], axis=1)
                np.minimum.at(h, conn[:, i], d)
                np.minimum.at(h, conn[:, j], d)
    return h


def perturb(
    points: FloatArray,
    cells: dict[str, IntArray],
    boundary: dict[str, IntArray],
    amplitude: float,
    seed: int,
) -> FloatArray:
    """Move each node uniformly in a ball of radius `amplitude` x its shortest incident edge.

    Coordinates on the faces of the unit cube stay on them, so the domain is unchanged. Nodes of a
    cell that would become inverted or non-star-shaped have their displacement halved until every
    cell is valid, so the result is deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=points.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = amplitude * _min_edge(points, cells) * rng.uniform(size=points.shape[0]) ** (1.0 / 3.0)
    step = direction * radius[:, None]
    on_face = (np.abs(points) < 1e-12) | (np.abs(points - 1.0) < 1e-12)
    step[on_face] = 0.0
    width = max(c.shape[1] for c in cells.values())
    cell_nodes = np.concatenate(
        [
            np.pad(cells[k], ((0, 0), (0, width - cells[k].shape[1])), constant_values=-1)
            for k in _CELL_ORDER
            if k in cells
        ]
    )
    for _ in range(60):
        moved: FloatArray = points + step
        bad = _to_facemesh(moved, cells, boundary).geometry.min_subvolume <= 0.0
        if not bad.any():
            return moved
        nodes = cell_nodes[bad]
        step[nodes[nodes >= 0]] *= 0.5
    msg = "could not perturb the mesh without inverting cells"
    raise VVError(msg, code="V0308")


def _to_facemesh(
    points: FloatArray, cells: dict[str, IntArray], boundary: dict[str, IntArray]
) -> FaceMesh:
    """Vectorised element-to-face conversion: match faces by sorted nodes; orientation from the tables."""
    polys, owners = [], []
    cell_kinds: list[str] = []
    first = 0
    for kind in _CELL_ORDER:
        conn = cells.get(kind)
        if conn is None:
            continue
        for local in _LOCAL_FACES[kind]:
            f = conn[:, list(local)]
            polys.append(np.pad(f, ((0, 0), (0, 4 - f.shape[1])), constant_values=-1))
            owners.append(first + np.arange(conn.shape[0]))
        cell_kinds += [kind] * conn.shape[0]
        first += conn.shape[0]
    poly = np.concatenate(polys)
    owner = np.concatenate(owners)
    key = np.sort(np.where(poly < 0, np.iinfo(np.int64).max, poly), axis=1)
    order = np.lexsort(key.T[::-1])
    key_sorted = key[order]
    same_as_next = np.all(key_sorted[1:] == key_sorted[:-1], axis=1)
    first_of_pair = np.flatnonzero(same_as_next)
    if np.any(same_as_next[1:] & same_as_next[:-1]):
        msg = "a face is shared by more than two cells"
        raise VVError(msg, code="V0306")
    a, b = order[first_of_pair], order[first_of_pair + 1]
    lo = np.where(owner[a] < owner[b], a, b)
    hi = np.where(owner[a] < owner[b], b, a)
    paired = np.zeros(poly.shape[0], dtype=bool)
    paired[a] = paired[b] = True
    single = np.flatnonzero(~paired)
    single_key = {tuple(key[i]): i for i in single}
    zones: dict[str, tuple[str, IntArray, IntArray]] = {}
    used = np.zeros(poly.shape[0], dtype=bool)
    for name, faces in boundary.items():
        bkey = np.sort(np.where(faces < 0, np.iinfo(np.int64).max, faces), axis=1)
        idx = np.array([single_key.get(tuple(k), -1) for k in bkey], dtype=np.int64)
        if np.any(idx < 0):
            msg = f"zone {name} has a face that is not a cell boundary face"
            raise VVError(msg, code="V0302")
        used[idx] = True
        zones[name] = (BC_TYPES.get(name, "pressure-far-field"), poly[idx], owner[idx])
    if not np.all(used[single]):
        msg = "boundary faces without a zone"
        raise VVError(msg, code="V0302")
    return from_faces(points, first, (poly[lo], owner[lo], owner[hi]), zones, tuple(cell_kinds))


def _structured_hybrid(n: int) -> tuple[FloatArray, dict[str, IntArray]]:
    """Hex, pyramid, tet and prism slabs on an n^3 grid, conforming through min-max face diagonals.

    From z = 0 up: hexahedra; one transition layer where each hexahedron becomes a pyramid on its
    bottom quad plus ten tetrahedra on its triangulated other faces, all with apex at its centre;
    tetrahedra from the Kuhn split along the (0,0,0)-(1,1,1) diagonal; prisms from splitting each
    hexahedron along its horizontal (0,0)-(1,1) diagonal. Every split quad uses the diagonal joining
    its lowest and highest corners, so neighbouring cells always share whole faces.
    """
    n_hex = max(1, n // 3)
    n_prism = max(1, n // 3)
    grid = np.linspace(0.0, 1.0, n + 1)
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def node(i: IntArray, j: IntArray, k: IntArray) -> IntArray:
        return (i * (n + 1) + j) * (n + 1) + k

    def corners(k: int) -> IntArray:
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        i, j = ii.ravel(), jj.ravel()
        kk = np.full(i.size, k)
        bits = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ]
        return np.stack([node(i + a, j + b, kk + c) for a, b, c in bits], axis=1)

    cells: dict[str, list[IntArray]] = {"hex": [], "pyramid": [], "tet": [], "prism": []}
    kuhn = [(0, 1, 2, 6), (0, 1, 5, 6), (0, 4, 5, 6), (0, 4, 7, 6), (0, 3, 7, 6), (0, 3, 2, 6)]
    # Other faces of the transition hexahedron as triangle pairs split on their min-max diagonal.
    side_tris = [
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (3, 2, 6),
        (3, 6, 7),
        (0, 3, 7),
        (0, 7, 4),
        (4, 5, 6),
        (4, 6, 7),
    ]
    centres: list[FloatArray] = []
    for k in range(n):
        c = corners(k)
        if k < n_hex:
            cells["hex"].append(c)
        elif k == n_hex:
            centre = points.shape[0] + sum(x.shape[0] for x in centres) + np.arange(c.shape[0])
            centres.append(points[c].mean(axis=1))
            cells["pyramid"].append(np.column_stack([c[:, [0, 1, 2, 3]], centre]))
            cells["tet"] += [np.column_stack([c[:, list(t)], centre]) for t in side_tris]
        elif k < n - n_prism:
            cells["tet"] += [c[:, list(t)] for t in kuhn]
        else:
            cells["prism"] += [c[:, [0, 1, 2, 4, 5, 6]], c[:, [0, 2, 3, 4, 6, 7]]]
    points = np.concatenate([points, *centres])
    out = {k: np.concatenate(v) for k, v in cells.items()}
    # Orient tetrahedra positively (Gmsh convention) by swapping two nodes where needed.
    t = out["tet"]
    a, b, c_, d = (points[t[:, i]] for i in range(4))
    negative = np.einsum("ij,ij->i", np.cross(b - a, c_ - a), d - a) < 0
    t[negative] = t[negative][:, [0, 2, 1, 3]]
    return points, out


def _box_boundary(points: FloatArray, cells: dict[str, IntArray]) -> dict[str, IntArray]:
    """Faces used by one cell only, grouped by the face of the bounding box they lie on."""
    polys = []
    for kind in _CELL_ORDER:
        if kind in cells:
            for local in _LOCAL_FACES[kind]:
                f = cells[kind][:, list(local)]
                polys.append(np.pad(f, ((0, 0), (0, 4 - f.shape[1])), constant_values=-1))
    poly = np.concatenate(polys)
    key = np.sort(np.where(poly < 0, np.iinfo(np.int64).max, poly), axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    single = poly[counts[inverse.ravel()] == 1]
    out = {}
    zone_of = np.full(single.shape[0], -1)
    for k, name in enumerate(BOX_ZONES):
        axis = "xyz".index(name[0])
        value = points[:, axis].min() if name.endswith("min") else points[:, axis].max()
        tol = 1e-12 * float(np.ptp(points[:, axis]))
        coord = np.where(single >= 0, points[np.maximum(single, 0), axis], value)
        on = np.all(np.abs(coord - value) <= tol, axis=1)
        if np.any(zone_of[on] >= 0):
            msg = "a boundary face lies on two faces of the box: the box is degenerate"
            raise VVError(msg, code="V0302")
        zone_of[on] = k
        out[name] = single[on]
    if np.any(zone_of < 0):
        msg = "a boundary face lies on no face of the box"
        raise VVError(msg, code="V0302")
    return out


def cartesian(
    cells: tuple[int, int, int],
    lower: tuple[float, float, float] = (0.0, 0.0, 0.0),
    upper: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> FaceMesh:
    """Uniform hexahedral box mesh with `cells` = (nx, ny, nz) and zones `xmin` ... `zmax`.

    With one cell along a direction it is the one-layer mesh of a 1D or 2D problem. Needs no Gmsh.
    """
    nx, ny, nz = cells
    axes = [np.linspace(lo, hi, n + 1) for lo, hi, n in zip(lower, upper, cells, strict=True)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    i, j, k = (
        a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    )

    def node(a: IntArray, b: IntArray, c: IntArray) -> IntArray:
        return (a * (ny + 1) + b) * (nz + 1) + c

    bits = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    conn = {"hex": np.stack([node(i + a, j + b, k + c) for a, b, c in bits], axis=1)}
    mesh = _to_facemesh(points, conn, _box_boundary(points, conn))
    mesh.check()
    return mesh


def _gmsh_mesh(
    gmsh: ModuleType, kind: str, n: int, seed: int
) -> tuple[FloatArray, dict[str, IntArray]]:
    gmsh.option.setNumber("Mesh.RandomSeed", seed)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 1.0 / n)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0 / n)
    gmsh.model.add(f"{kind}-{n}")
    _build_geometry(gmsh, kind, n)
    gmsh.model.mesh.generate(3)
    tags, points = _nodes(gmsh)
    index = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    index[tags] = np.arange(tags.size)
    cells = _cells(gmsh, index)
    gmsh.model.remove()
    return points, cells


def _write_msh(
    gmsh: ModuleType,
    points: FloatArray,
    cells: dict[str, IntArray],
    boundary: dict[str, IntArray],
    path: Path,
    periodic: tuple[tuple[str, str, tuple[float, float, float]], ...] = (),
) -> None:
    """Write arrays as MSH 4.1 ASCII through a discrete Gmsh model, so the file is exactly the arrays.
    Each (zone a, zone b, translation) of `periodic` becomes a `$Periodic` surface link, b the
    translate of a (mesh.md, Periodic pairs: the reader pairs the faces by geometry)."""
    gmsh.model.add("export")
    volume = gmsh.model.addDiscreteEntity(3)
    gmsh.model.mesh.addNodes(
        3, volume, list(range(1, points.shape[0] + 1)), points.ravel().tolist()
    )
    tag = 1
    for kind in _CELL_ORDER:
        if kind in cells:
            conn = cells[kind]
            gmsh_type = {v: k for k, v in _GMSH_TYPES.items()}[kind]
            gmsh.model.mesh.addElementsByType(
                volume,
                gmsh_type,
                list(range(tag, tag + conn.shape[0])),
                (conn + 1).ravel().tolist(),
            )
            tag += conn.shape[0]
    surfaces: dict[str, int] = {}
    for name, faces in boundary.items():
        surface = gmsh.model.addDiscreteEntity(2)
        surfaces[name] = surface
        for k, gmsh_type in ((3, 2), (4, 3)):
            rows = faces[(faces >= 0).sum(axis=1) == k][:, :k]
            if rows.size:
                gmsh.model.mesh.addElementsByType(
                    surface,
                    gmsh_type,
                    list(range(tag, tag + rows.shape[0])),
                    (rows + 1).ravel().tolist(),
                )
                tag += rows.shape[0]
        gmsh.model.addPhysicalGroup(2, [surface], name=name)
    gmsh.model.addPhysicalGroup(3, [volume], name=CELL_ZONE)
    for a, b, (tx, ty, tz) in periodic:
        affine = [1.0, 0.0, 0.0, tx, 0.0, 1.0, 0.0, ty, 0.0, 0.0, 1.0, tz, 0.0, 0.0, 0.0, 1.0]
        gmsh.model.mesh.setPeriodic(2, [surfaces[b]], [surfaces[a]], affine)
    gmsh.write(str(path))
    gmsh.model.remove()


def _agglomerate(mesh: FaceMesh, primal_points: FloatArray, n: int) -> FaceMesh:
    """Merge the dual cells whose primal vertex lies in the same box of a lattice of edge about 4/n,
    in a checkerboard (at least two boxes per side)."""
    m = max(n // 4, 2)
    box = np.minimum(np.floor(primal_points * m).astype(np.int64), m - 1)
    box_id = (box[:, 0] * m + box[:, 1]) * m + box[:, 2]
    merged = box.sum(axis=1) % 2 == 0
    key = np.where(merged, box_id, -1 - np.arange(mesh.n_cells))
    _, cell_map = np.unique(key, return_inverse=True)
    return merge_cells(mesh, cell_map.astype(np.int64), int(cell_map.max()) + 1)


def generate_level(
    kind: str, n: int, *, perturbation: float = 0.0, seed: int = 0, path: Path | None = None
) -> tuple[FaceMesh, str]:
    """Generate one level with about n cells per edge; optionally write it to `path`.

    Returns the mesh and the Gmsh version. `poly` is written as Fluent ASCII, the others as MSH 4.1
    ASCII. The file holds exactly the in-memory mesh: coordinates are rounded to the 16 significant
    digits Gmsh writes.
    """
    if kind not in (*KINDS, "ringleb", *vortex.KINDS):
        msg = f"unknown mesh kind {kind!r}; expected one of {KINDS}, 'ringleb' or {vortex.KINDS}"
        raise VVError(msg, code="V0304")
    if kind == "ringleb" and perturbation > 0.0:
        msg = "the Ringleb mesh follows the flow's streamlines and is not perturbed"
        raise VVError(msg, code="V0309")
    if kind in vortex.KINDS and perturbation > 0.0:
        msg = f"{kind} is nested and periodic: it carries its own seeded jitter, not a perturbation"
        raise VVError(msg, code="V0309")
    if not (0.0 <= perturbation < 1.0) or n < 1:
        msg = f"need 0 <= perturbation < 1 and n >= 1, got {perturbation} and {n}"
        raise VVError(msg, code="V0309")
    gmsh = _gmsh()
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        version = str(gmsh.__version__)
        periodic: tuple[tuple[str, str, tuple[float, float, float]], ...] = ()
        if kind == "ringleb":
            points, cells, boundary = ringleb.grid(n)
        elif kind in vortex.KINDS:
            points, cells = vortex.grid(kind, n, seed)
            boundary = _box_boundary(points, cells)
            periodic = vortex.PERIODIC
        else:
            if kind == "hybrid":
                points, cells = _structured_hybrid(n)
            else:
                points, cells = _gmsh_mesh(
                    gmsh, "tet" if kind in ("poly", "agglomerated") else kind, n, seed
                )
            boundary = _box_boundary(points, cells)
        if perturbation > 0.0:
            ss = np.random.SeedSequence([seed, n, KINDS.index(kind)])
            points = perturb(points, cells, boundary, perturbation, int(ss.generate_state(1)[0]))
        points = np.array([float(f"{v:.16g}") for v in points.ravel()]).reshape(-1, 3)
        if kind in ("poly", "agglomerated"):
            mesh = median_dual(points, cells["tet"], boundary, BC_TYPES)
            if kind == "agglomerated":
                mesh = _agglomerate(mesh, points, n)
            if path is not None:
                write_fluent(mesh, path, CELL_ZONE)
        else:
            mesh = _to_facemesh(points, cells, boundary)
            if path is not None:
                _write_msh(gmsh, points, cells, boundary, path, periodic)
    finally:
        gmsh.finalize()
    mesh.check()
    return mesh, version


def generate_family(
    kind: str, levels: tuple[int, ...], out_dir: Path, *, perturbation: float = 0.0, seed: int = 0
) -> tuple[Family, list[FaceMesh]]:
    """Generate all levels of a family into `out_dir`, write its `manifest.json`, return both."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    meshes = []
    version = ""
    for n in levels:
        name = f"{kind}{'-dirty' if perturbation > 0 else ''}-n{n:03d}.msh"
        path = out_dir / name
        mesh, version = generate_level(kind, n, perturbation=perturbation, seed=seed, path=path)
        meshes.append(mesh)
        if kind == "ringleb":
            ringleb.write_face_states(mesh, path, path.with_suffix(".states.json"))
        records.append(level_record(n, path, mesh))
    family = Family(kind, perturbation, seed, version, tuple(records))
    write_manifest(family, out_dir)
    return family, meshes
