"""Tetrahedral meshes of the ONERA M6 wing in a half sphere, for the M2 inviscid validation case.

Geometry, from its public definition (Schmitt and Charpin, AGARD AR-138, 1979, chapter B1, as
restated by Destarac and Dumont for the NASA Turbulence Modeling Resource, 2016, section 2.2):

- planform: root leading edge A = (0, 0), root trailing edge B = (0.8059 m, 0), apex of the
  prolonged trapezoid C = (1.5787 m, 2.7345 m), cut at the semispan b = 1.1963 m (leading-edge
  sweep 30 deg, trailing-edge sweep 15.8 deg);
- every section is the root section scaled about the apex (conical generation), so the wing is the
  ruled surface between its root and tip sections;
- the root section is the 72-point table B1-1 (upper half, symmetric section);
- the tip is a half body of revolution: the tip section revolved about its chord line.

The original trailing edge has a base 0.141 % of the chord thick. As the TMR model does (0.55 % of
the local chord), the trailing edge is made sharp by extending the last segment of each surface to
the chord line, here 0.57 %. Chords and x/c keep the original definition.

Axes: x downstream, y along the span (the symmetry plane is y = 0), z up, so that the angle of
attack of case-schema.md rotates the flow in the x-z plane. The far field is the half sphere y >= 0
of radius FARFIELD_RADIUS centred near the root mid-chord. Zones: `wing`, `symmetry`, `farfield`.

Level n has a surface size about c(y) / n on the wing (c the local chord), a third of it along the
leading and trailing edges and the tip, growing linearly away from the wing; every size scales as
1/n, so n -> 2n multiplies the cell count by about 8.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from autan_vv.errors import VVError
from autan_vv.meshes import families

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from autan_vv.meshes.facemesh import FaceMesh, FloatArray, IntArray

ROOT_CHORD = 0.8059
APEX = (1.5787, 2.7345)
SEMISPAN = 1.1963
REFERENCE_AREA = 0.7532
MEAN_AERODYNAMIC_CHORD = 0.64607
FARFIELD_RADIUS = 20.0
CENTRE = (0.6, 0.0, 0.0)
ZONES = ("wing", "symmetry", "farfield")
# Size growth away from the wing per unit distance (at 20 Gmsh 4.15 creates invalid tetrahedra), and
# size along the edges relative to the wing's.
GROWTH = 6.0
EDGE_FACTOR = 1.0 / 3.0

# AGARD AR-138 table B1-1 (Destarac and Dumont 2016, table 1): root section, upper half, x/c, z/c.
SECTION = np.array(
    [
        (0.0000000, 0.0000000), (0.0000165, 0.0006914), (0.0000696, 0.0014416),
        (0.0001675, 0.0022554), (0.0003232, 0.0031382), (0.0005508, 0.0040959),
        (0.0008657, 0.0051343), (0.0012868, 0.0062598), (0.0018364, 0.0074784),
        (0.0025441, 0.0087958), (0.0034428, 0.0102163), (0.0045704, 0.0117419),
        (0.0059751, 0.0133708), (0.0077112, 0.0150951), (0.0098413, 0.0168984),
        (0.0124479, 0.0187537), (0.0156171, 0.0206220), (0.0194609, 0.0224545),
        (0.0241067, 0.0242004), (0.0297008, 0.0258245), (0.0364261, 0.0273317),
        (0.0444852, 0.0287912), (0.0541248, 0.0303278), (0.0656303, 0.0320138),
        (0.0793366, 0.0338372), (0.0956354, 0.0357742), (0.1149796, 0.0377923),
        (0.1378963, 0.0398522), (0.1649976, 0.0419089), (0.1919327, 0.0436214),
        (0.2187096, 0.0450507), (0.2453310, 0.0462358), (0.2717978, 0.0471987),
        (0.2981113, 0.0479494), (0.3242726, 0.0484902), (0.3502830, 0.0488183),
        (0.3761446, 0.0489296), (0.4018567, 0.0488202), (0.4274223, 0.0484833),
        (0.4528441, 0.0479351), (0.4781197, 0.0471661), (0.5032514, 0.0461903),
        (0.5282426, 0.0450209), (0.5530937, 0.0436741), (0.5778043, 0.0421684),
        (0.6023757, 0.0405241), (0.6268104, 0.0387613), (0.6511093, 0.0368990),
        (0.6752726, 0.0349542), (0.6993027, 0.0329402), (0.7231995, 0.0308662),
        (0.7469658, 0.0287365), (0.7705998, 0.0265505), (0.7941055, 0.0243027),
        (0.8174828, 0.0219842), (0.8407324, 0.0195838), (0.8638564, 0.0170915),
        (0.8868235, 0.0145051), (0.9061905, 0.0122389), (0.9225336, 0.0102727),
        (0.9363346, 0.0085827), (0.9479946, 0.0071423), (0.9578511, 0.0059224),
        (0.9661860, 0.0048907), (0.9732361, 0.0040180), (0.9792020, 0.0032796),
        (0.9842508, 0.0026547), (0.9885252, 0.0021257), (0.9921438, 0.0016778),
        (0.9952080, 0.0012985), (0.9978030, 0.0009773), (1.0000000, 0.0007052),
    ]
)  # fmt: skip


def te_slope() -> float:
    """dz/dx of the last segment of the upper surface."""
    (x0, z0), (x1, z1) = SECTION[-2], SECTION[-1]
    return float((z1 - z0) / (x1 - x0))


def sharp_section() -> FloatArray:
    """Upper half of the root section with the sharp trailing edge appended (x/c, z/c)."""
    extension = -SECTION[-1, 1] / te_slope()
    return np.vstack([SECTION, [1.0 + extension, 0.0]])


def scale(y: float | FloatArray) -> float | FloatArray:
    """Homothety ratio of the section at span y about the apex: local chord / root chord."""
    return (APEX[1] - y) / APEX[1]


def leading_edge(y: float | FloatArray) -> float | FloatArray:
    """x of the leading edge at span y (y <= b)."""
    return APEX[0] * (1.0 - scale(y))


def chord(y: float | FloatArray) -> float | FloatArray:
    """Local chord of the original (blunt) wing at span y."""
    return ROOT_CHORD * scale(y)


def _upper_curve(gmsh: ModuleType, s: float, y: float) -> tuple[int, FloatArray]:
    occ = gmsh.model.occ
    section = sharp_section()
    x = APEX[0] + s * (section[:, 0] * ROOT_CHORD - APEX[0])
    z = s * section[:, 1] * ROOT_CHORD
    points = [occ.addPoint(a, y, c) for a, c in zip(x[::-1], z[::-1], strict=True)]
    # From the trailing edge to the leading edge: tangent along the extended last segment at the
    # trailing edge, vertical at the leading edge, so the mirrored lower half joins it smoothly.
    te = np.array([-1.0, 0.0, -te_slope()])
    te /= np.linalg.norm(te)
    return occ.addSpline(points, tangents=[*te.tolist(), 0.0, 0.0, -1.0]), x


def build_geometry(gmsh: ModuleType) -> dict[str, list[int]]:
    """Build the fluid volume in the current Gmsh model; return the surface tags of each zone.

    The wing is assembled from its faces (ruled upper surface, its mirror image, the tip surface of
    revolution and the root section), not from Boolean operations on solids, which damage the
    surface of revolution; one cut removes it from the half ball.
    """
    occ = gmsh.model.occ
    s_tip = float(scale(SEMISPAN))
    root_upper, _ = _upper_curve(gmsh, 1.0, 0.0)
    tip_upper, _ = _upper_curve(gmsh, s_tip, SEMISPAN)
    wires = [occ.addWire([root_upper]), occ.addWire([tip_upper])]
    upper = [e for e in occ.addThruSections(wires, makeSolid=False, makeRuled=True) if e[0] == 2]
    lower = occ.copy(upper)
    occ.mirror(lower, 0.0, 0.0, 1.0, 0.0)
    tip = [
        e
        for e in occ.revolve([(1, tip_upper)], 0.0, SEMISPAN, 0.0, 1.0, 0.0, 0.0, -math.pi)
        if e[0] == 2
    ]
    root_lower = occ.copy([(1, root_upper)])
    occ.mirror(root_lower, 0.0, 0.0, 1.0, 0.0)
    root = occ.addPlaneSurface([occ.addCurveLoop([root_upper, root_lower[0][1]])])
    shell = occ.addSurfaceLoop([t for _, t in upper + lower + tip] + [root], sewing=True)
    wing = occ.addVolume([shell])
    r = FARFIELD_RADIUS
    ball = occ.addSphere(*CENTRE, r)
    half_space = occ.addBox(CENTRE[0] - r - 1, 0.0, -r - 1, 2 * r + 2, r + 1, 2 * r + 2)
    half, _ = occ.intersect([(3, ball)], [(3, half_space)])
    fluid, _ = occ.cut(half, [(3, wing)])
    occ.synchronize()
    if len(fluid) != 1:
        msg = f"the ONERA M6 domain has {len(fluid)} volumes instead of 1"
        raise VVError(msg, code="V0310")
    zones: dict[str, list[int]] = {name: [] for name in ZONES}
    for _, tag in gmsh.model.getBoundary(fluid, oriented=False):
        lo = gmsh.model.getBoundingBox(2, tag)
        if lo[3] - lo[0] > FARFIELD_RADIUS:
            zones["symmetry" if gmsh.model.getType(2, tag) == "Plane" else "farfield"].append(tag)
        else:
            zones["wing"].append(tag)
    if [len(zones[z]) for z in ZONES] != [3, 1, 1]:
        msg = f"unexpected ONERA M6 boundary surfaces: {zones}"
        raise VVError(msg, code="V0310")
    return zones


def _size_field(gmsh: ModuleType, n: int, wing: list[int]) -> None:
    """Background size: c(y)/n + GROWTH/n * d(wing), a third of it near the wing's edges."""
    fields = gmsh.model.mesh.field
    edges = sorted(
        {abs(c) for t in wing for _, c in gmsh.model.getBoundary([(2, t)], oriented=False)}
    )
    to_wing = fields.add("Distance")
    fields.setNumbers(to_wing, "SurfacesList", wing)
    fields.setNumber(to_wing, "Sampling", 200)
    to_edges = fields.add("Distance")
    fields.setNumbers(to_edges, "CurvesList", edges)
    fields.setNumber(to_edges, "Sampling", 400)
    k = ROOT_CHORD / APEX[1]
    local = f"({ROOT_CHORD}-{k}*Min(Max(y,0),{SEMISPAN}))"
    near = f"({local}+{GROWTH}*F{to_wing})/{n}"
    edge = f"{EDGE_FACTOR}*({local}+{GROWTH / EDGE_FACTOR}*F{to_edges})/{n}"
    cap = f"{GROWTH * FARFIELD_RADIUS / 8.0}/{n}"
    size = fields.add("MathEval")
    fields.setString(size, "F", f"Min(Min({near},{edge}),{cap})")
    fields.setAsBackgroundMesh(size)
    for option in (
        "Mesh.MeshSizeExtendFromBoundary",
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
    ):
        gmsh.option.setNumber(option, 0)


def _mesh(gmsh: ModuleType, n: int) -> tuple[FloatArray, dict[str, IntArray], dict[str, IntArray]]:
    gmsh.model.add(f"onera-m6-{n}")
    zones = build_geometry(gmsh)
    _size_field(gmsh, n, zones["wing"])
    for tag in zones["wing"]:
        # Frontal-Delaunay cannot mesh the surface of revolution with its two poles.
        if gmsh.model.getType(2, tag) == "Surface of Revolution":
            gmsh.model.mesh.setAlgorithm(2, tag, 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    # Netgen's optimiser removes the flat tetrahedra Delaunay leaves (max non-orthogonality 70 -> 53
    # deg at n = 24).
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.model.mesh.generate(3)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    tags = np.asarray(tags)[order].astype(np.int64)
    points = np.asarray(coords).reshape(-1, 3)[order]
    index = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    index[tags] = np.arange(tags.size)
    types, _, nodes = gmsh.model.mesh.getElements(3)
    if [int(t) for t in types] != [4]:
        msg = f"unexpected volume element types {list(types)}"
        raise VVError(msg, code="V0305")
    cells = {"tet": index[np.asarray(nodes[0], dtype=np.int64)].reshape(-1, 4)}
    boundary: dict[str, IntArray] = {}
    for name in ZONES:
        tris = []
        for tag in zones[name]:
            t2, _, n2 = gmsh.model.mesh.getElements(2, tag)
            if [int(t) for t in t2] != [2]:
                msg = f"unexpected surface element types {list(t2)} on {name}"
                raise VVError(msg, code="V0305")
            tris.append(index[np.asarray(n2[0], dtype=np.int64)].reshape(-1, 3))
        boundary[name] = np.concatenate(tris)
    gmsh.model.remove()
    # Gmsh keeps nodes of geometric points that no element uses (section control points).
    used = np.zeros(points.shape[0], dtype=bool)
    used[cells["tet"].ravel()] = True
    renumber = np.cumsum(used) - 1
    cells = {"tet": renumber[cells["tet"]]}
    # Boundary faces are rows of four node indices padded with -1, as families stores them.
    boundary = {
        k: np.pad(renumber[v], ((0, 0), (0, 1)), constant_values=-1) for k, v in boundary.items()
    }
    points = points[used]
    # The symmetry plane is y = 0 exactly; Gmsh leaves round-off on the nodes it places there.
    points[np.unique(boundary["symmetry"][:, :3]), 1] = 0.0
    return points, cells, boundary


def generate_level(n: int, path: Path | None = None, *, seed: int = 0) -> tuple[FaceMesh, str]:
    """Level n as a FaceMesh, optionally written to `path` as MSH 4.1 ASCII; returns Gmsh's version.

    Coordinates are rounded to the 16 significant digits Gmsh writes, so file and memory agree.
    """
    if n < 1:
        msg = f"need n >= 1, got {n}"
        raise VVError(msg, code="V0309")
    gmsh = families._gmsh()
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", seed)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        version = str(gmsh.__version__)
        points, cells, boundary = _mesh(gmsh, n)
        points = np.array([float(f"{v:.16g}") for v in points.ravel()]).reshape(-1, 3)
        mesh = families._to_facemesh(points, cells, boundary)
        mesh.check()
        if path is not None:
            families._write_msh(gmsh, points, cells, boundary, path)
    finally:
        gmsh.finalize()
    return mesh, version


def generate_family(
    levels: tuple[int, ...], out_dir: Path, *, seed: int = 0
) -> tuple[families.Family, list[FaceMesh]]:
    """Write levels `onera-m6-nNNN.msh` and `manifest.json` into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records, meshes, version = [], [], ""
    for n in levels:
        path = out_dir / f"onera-m6-n{n:03d}.msh"
        mesh, version = generate_level(n, path, seed=seed)
        records.append(families.level_record(n, path, mesh))
        meshes.append(mesh)
    family = families.Family("onera-m6", 0.0, seed, version, tuple(records))
    families.write_manifest(family, out_dir)
    return family, meshes
