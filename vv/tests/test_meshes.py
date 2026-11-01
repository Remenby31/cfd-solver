"""Mesh families: validity, reproducibility, and files read back by independent consumers."""

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pytest
from autan_vv.errors import VVError
from autan_vv.meshes.facemesh import FaceMesh, from_faces
from autan_vv.meshes.families import BOX_ZONES, KINDS, generate_family, generate_level

gmsh = pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")

EXPECTED_KINDS = {
    "tet": {"tet"},
    "hex": {"hex"},
    "prism": {"prism"},
    "hybrid": {"hex", "prism", "pyramid", "tet"},
    "poly": {"polyhedron"},
    "agglomerated": {"polyhedron"},
}


def _box_normal(name: str) -> np.ndarray:
    axis = "xyz".index(name[0])
    n = np.zeros(3)
    n[axis] = -1.0 if name.endswith("min") else 1.0
    return n


@pytest.mark.parametrize("perturbation", [0.0, 0.3])
@pytest.mark.parametrize("kind", KINDS)
def test_family_meshes_tile_the_unit_cube(kind: str, perturbation: float) -> None:
    mesh, _ = generate_level(kind, 6, perturbation=perturbation, seed=3)
    g = mesh.geometry
    assert set(mesh.cell_kinds) == EXPECTED_KINDS[kind]
    assert g.cell_volume.sum() == pytest.approx(1.0, abs=1e-13)
    assert g.cell_volume.min() > 0.0
    assert [z.name for z in mesh.zones] == list(BOX_ZONES)
    for z in mesh.zones:
        area = g.face_area[z.start : z.start + z.count]
        # Each box face has unit area and outward normals.
        np.testing.assert_allclose(area.sum(axis=0), _box_normal(z.name), atol=1e-13)
        centroid = g.face_centroid[z.start : z.start + z.count]
        axis = "xyz".index(z.name[0])
        np.testing.assert_allclose(
            centroid[:, axis], 0.0 if z.name.endswith("min") else 1.0, atol=1e-14
        )
    # Interior normals point from owner to neighbour (centroids of non-convex agglomerates may not).
    if kind == "agglomerated":
        return
    d = g.cell_centroid[mesh.neighbour] - g.cell_centroid[mesh.owner[: mesh.n_interior]]
    assert np.all(np.einsum("ij,ij->i", d, g.face_area[: mesh.n_interior]) > 0.0)


def test_perturbation_moves_interior_nodes_only_tangentially_on_the_boundary() -> None:
    clean, _ = generate_level("hex", 6)
    dirty, _ = generate_level("hex", 6, perturbation=0.3, seed=3)
    moved = np.linalg.norm(dirty.points - clean.points, axis=1)
    assert np.count_nonzero(moved) > 0.9 * clean.points.shape[0] - 6 * 7 * 7 * 2
    on_face = (np.abs(clean.points) < 1e-12) | (np.abs(clean.points - 1.0) < 1e-12)
    np.testing.assert_array_equal(dirty.points[on_face], clean.points[on_face])
    assert np.max(moved) < 0.3 / 6 + 1e-12


@pytest.mark.parametrize("kind", ["tet", "poly", "agglomerated"])
def test_families_regenerate_bit_for_bit(kind: str, tmp_path: Path) -> None:
    first, _ = generate_family(kind, (4, 6), tmp_path / "a", perturbation=0.2, seed=11)
    second, _ = generate_family(kind, (4, 6), tmp_path / "b", perturbation=0.2, seed=11)
    assert first.levels == second.levels
    other_seed, _ = generate_family(kind, (4,), tmp_path / "c", perturbation=0.2, seed=12)
    assert other_seed.levels[0].file_sha256 != first.levels[0].file_sha256
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert manifest["schema"] == "autan.mesh-family/1"
    assert [lv["n"] for lv in manifest["levels"]] == [4, 6]


def test_gmsh_file_reads_back_through_gmsh(tmp_path: Path) -> None:
    path = tmp_path / "hybrid.msh"
    mesh, _ = generate_level("hybrid", 6, perturbation=0.2, seed=5, path=path)
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        assert path.read_text().startswith("$MeshFormat\n4.1 0 8")
        names = {gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()}
        assert names == {*BOX_ZONES, "fluid"}
        types, tags, nodes = gmsh.model.mesh.getElements(3)
        assert sum(len(t) for t in tags) == mesh.n_cells
        assert {int(t) for t in types} == {4, 5, 6, 7}
        # Element i of the file is cell i of the mesh: same type, same vertices.
        npe = {4: 4, 5: 8, 6: 6, 7: 5}
        kinds = {4: "tet", 5: "hex", 6: "prism", 7: "pyramid"}
        file_cells = [
            (kinds[int(t)], set((c - 1).tolist()))
            for t, conn in zip(types, nodes, strict=True)
            for c in np.asarray(conn, dtype=np.int64).reshape(-1, npe[int(t)])
        ]
        own = np.r_[mesh.owner, mesh.neighbour]
        face_ids = np.r_[np.arange(mesh.n_faces), np.arange(mesh.n_interior)]
        for i in (0, mesh.n_cells // 2, mesh.n_cells - 1):
            vertices = set(np.concatenate([mesh.face(int(f)) for f in face_ids[own == i]]).tolist())
            assert file_cells[i] == (mesh.cell_kinds[i], vertices)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(coords).reshape(-1, 3)[np.argsort(node_tags)]
        np.testing.assert_array_equal(xyz, mesh.points)
    finally:
        gmsh.finalize()


def _read_fluent(path: Path) -> FaceMesh:
    """Independent reader for the Fluent subset we write: nodes, mixed faces, zone names."""
    text = path.read_text()
    nodes = re.search(r"\(10 \(1 1 ([0-9a-f]+) 1 3\)\(\n(.*?)\n\)\)", text, re.DOTALL)
    assert nodes is not None
    points = np.array([[float(v) for v in ln.split()] for ln in nodes.group(2).splitlines()])
    assert points.shape[0] == int(nodes.group(1), 16)
    n_cells = int(re.search(r"\(12 \(0 1 ([0-9a-f]+) 0\)\)", text).group(1), 16)  # type: ignore[union-attr]
    names = {
        int(i, 16): (kind, name)
        for i, kind, name in re.findall(r"\(45 \(([0-9a-f]+) (\S+) (\S+)\)\(\)\)", text)
    }
    interior: list[list[int]] = [[], [], []]
    zones: dict[str, tuple[str, list[list[int]], list[int]]] = {}
    for zid, body in re.findall(
        r"\(13 \(([0-9a-f]+) [0-9a-f]+ [0-9a-f]+ [0-9a-f]+ 0\)\(\n(.*?)\n\)\)", text, re.DOTALL
    ):
        kind, name = names[int(zid, 16)]
        for line in body.splitlines():
            v = [int(t, 16) for t in line.split()]
            assert v[0] == len(v) - 3
            poly = [p - 1 for p in v[1:-2]][
                ::-1
            ]  # Fluent normal points to c0; ours away from owner.
            c0, c1 = v[-2] - 1, v[-1] - 1
            if kind == "interior":
                interior[0].append(poly)  # type: ignore[arg-type]
                interior[1].append(c0)
                interior[2].append(c1)
            else:
                assert c1 == -1
                zones.setdefault(name, (kind, [], []))
                zones[name][1].append(poly)
                zones[name][2].append(c0)

    def pad(rows: list[list[int]]) -> np.ndarray:
        out = np.full((len(rows), max(map(len, rows))), -1)
        for i, r in enumerate(rows):
            out[i, : len(r)] = r
        return out

    return from_faces(
        points,
        n_cells,
        (pad(interior[0]), np.array(interior[1]), np.array(interior[2])),  # type: ignore[arg-type]
        {k: (t, pad(p), np.array(o)) for k, (t, p, o) in zones.items()},
    )


def test_fluent_polyhedral_file_reads_back_closed_with_zones(tmp_path: Path) -> None:
    path = tmp_path / "poly.msh"
    mesh, _ = generate_level("poly", 5, perturbation=0.2, seed=2, path=path)
    back = _read_fluent(path)
    back.check()
    assert back.n_cells == mesh.n_cells
    assert back.n_faces == mesh.n_faces
    assert back.geometry.cell_volume.sum() == pytest.approx(1.0, abs=1e-13)
    assert {z.name: (z.bc_type, z.count) for z in back.zones} == {
        z.name: (z.bc_type, z.count) for z in mesh.zones
    }
    assert dict(next(iter(back.zones)).__dict__)["bc_type"] in {"wall", "pressure-far-field"}
    np.testing.assert_allclose(back.geometry.cell_volume, mesh.geometry.cell_volume, rtol=1e-13)
    # Every edge of every cell is used exactly twice, in opposite directions (core reader rule).
    for cell in range(back.n_cells):
        edges: dict[tuple[int, int], int] = {}
        faces = np.flatnonzero(
            (back.owner == cell)
            | np.r_[back.neighbour == cell, np.zeros(back.n_faces - back.n_interior, bool)]
        )
        for f in faces.tolist():
            nodes = back.face(f) if back.owner[f] == cell else back.face(f)[::-1]
            for a, b in zip(nodes, np.roll(nodes, -1), strict=True):
                edges[(int(a), int(b))] = edges.get((int(a), int(b)), 0) + 1
        assert all(c == 1 and edges.get((b, a)) == 1 for (a, b), c in edges.items())


@pytest.mark.parametrize(
    ("kind", "perturbation", "code"),
    [
        ("octree", 0.0, "V0304"),
        ("hex", float("inf"), "V0309"),
        ("hex", float("nan"), "V0309"),
        ("hex", 1.0, "V0309"),
    ],
)
def test_invalid_requests_are_coded_errors(kind: str, perturbation: float, code: str) -> None:
    with pytest.raises(VVError) as err:
        generate_level(kind, 4, perturbation=perturbation)
    assert err.value.code == code


def test_check_rejects_non_finite_geometry() -> None:
    mesh, _ = generate_level("hex", 2)
    points = mesh.points.copy()
    points[5] = np.nan
    broken = FaceMesh(
        points,
        mesh.face_offsets,
        mesh.face_nodes,
        mesh.owner,
        mesh.neighbour,
        mesh.n_cells,
        mesh.zones,
    )
    with pytest.raises(VVError) as err:
        broken.check()
    assert err.value.code == "V0301"


def test_ringleb_family_carries_the_exact_states_of_its_boundary(tmp_path: Path) -> None:
    from autan import face_states  # noqa: PLC0415
    from autan_vv.oracles.inviscid import RINGLEB_SPEED_MIN, ringleb_fields  # noqa: PLC0415

    family, meshes = generate_family("ringleb", (4, 8), tmp_path)
    for level, mesh in zip(family.levels, meshes, strict=True):
        g = mesh.geometry
        assert g.cell_volume.min() > 0.0
        # The cells fill the C1.2 domain: the strict oracle accepts every cell centroid.
        ringleb_fields(g.cell_centroid)
        # The SDK reads the file as `autan plan` does: valid, and for this mesh file.
        msh = tmp_path / level.file
        doc = face_states.parse(msh.with_suffix(".states.json").read_bytes())
        assert not isinstance(doc, str), doc
        assert doc.mesh_sha256 == level.file_sha256
        assert set(doc.zones) == {"k-min", "k-max", "inflow", "outflow"}
        zones = {z.name: z for z in mesh.zones}
        for name, states in doc.zones.items():
            z = zones[name]
            centroid = g.face_centroid[z.start : z.start + z.count]
            np.testing.assert_array_equal(np.array(states.centroids), centroid)
            # Nodes of a side lie on its curve of the hodograph: a streamline k = 0.7 or 1.5,
            # or the iso-speed V = 0.5; the face states are the oracle there.
            nodes = np.unique(
                np.concatenate([mesh.face(f) for f in range(z.start, z.start + z.count)])
            )
            exact = ringleb_fields(mesh.points[nodes], extended=True)
            speed = np.hypot(exact["VelocityX"], exact["VelocityY"])
            if name in ("inflow", "outflow"):
                np.testing.assert_allclose(speed, RINGLEB_SPEED_MIN, rtol=1e-12)
            values = np.array(states.values)
            rho, vel, p = values[:, 0], values[:, 1:4], values[:, 4]
            # Independent of the oracle's inversion: isentropic flow in stagnation units
            # (p = rho^gamma / gamma), and the speed or direction along the side's curve.
            np.testing.assert_allclose(p, rho**1.4 / 1.4, rtol=1e-12)
            np.testing.assert_array_equal(vel[:, 2], 0.0)
            speed = np.linalg.norm(vel, axis=1)
            normal = g.face_area[z.start : z.start + z.count]
            normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
            h = 1.0 / level.n
            if name in ("inflow", "outflow"):
                # Faces on the iso-speed V = 0.5, to the chord's O(h^2).
                np.testing.assert_allclose(speed, RINGLEB_SPEED_MIN, atol=2.0 * h**2)
                # The flow enters at y > 0 and leaves at y < 0.
                flux = np.einsum("ij,ij->i", vel, normal)
                assert np.all(flux < 0.0) if name == "inflow" else np.all(flux > 0.0)
            else:
                # Streamlines: the flow runs along the side, to the chord's O(h).
                tangential = np.abs(np.einsum("ij,ij->i", vel, normal)) / speed
                assert tangential.max() < 4.0 * h


def _zone_faces(mesh: FaceMesh, name: str, axis: int, shift: float) -> set[tuple[float, ...]]:
    """The faces of zone `name`, each its sorted nodes moved by `shift` along `axis`."""
    zone = next(z for z in mesh.zones if z.name == name)
    out = set()
    for f in range(zone.start, zone.start + zone.count):
        p = mesh.points[mesh.face(f)].copy()
        p[:, axis] += shift
        out.add(tuple(sorted(map(tuple, p.round(12).tolist()))))
    return out


def _zone_polygons(mesh: FaceMesh, name: str) -> list[np.ndarray]:
    """The node coordinates of each face of zone `name`."""
    zone = next(z for z in mesh.zones if z.name == name)
    return [mesh.points[mesh.face(f)] for f in range(zone.start, zone.start + zone.count)]


def _polygon_key(corners: np.ndarray) -> frozenset[tuple[float, ...]]:
    """A polygon as the set of its corners, rounded off the 16 digits the files keep."""
    return frozenset(map(tuple, corners.round(9).tolist()))


@pytest.mark.parametrize("kind", ["vortex-hex", "vortex-prism"])
def test_vortex_families_are_periodic_and_nested(kind: str, tmp_path: Path) -> None:
    # The square [-10, 10]^2 of the isentropic vortex, one layer of thickness 20/n: each face of
    # xmax (ymax) is a face of xmin (ymin) moved by 20 along x (y), the file declares both links,
    # and each level holds the nodes of the coarser one in the plane, with four times its cells.
    family, meshes = generate_family(kind, (8, 16, 32), tmp_path, seed=5)
    for level, mesh in zip(family.levels, meshes, strict=True):
        g = mesh.geometry
        side = 20.0
        assert g.cell_volume.min() > 0.0
        assert g.cell_volume.sum() == pytest.approx(side * side, rel=1e-13)
        # One layer of thickness 1 at every level; n^2 squares, two triangles each for prisms.
        np.testing.assert_array_equal(np.unique(mesh.points[:, 2]), [0.0, 1.0])
        assert mesh.n_cells == level.n**2 * (1 if kind == "vortex-hex" else 2)
        for a, b, axis in (("xmin", "xmax", 0), ("ymin", "ymax", 1)):
            moved = _zone_faces(mesh, a, axis, side)
            assert moved
            assert moved == _zone_faces(mesh, b, axis, 0.0)
        text = (tmp_path / level.file).read_text()
        links = re.search(r"\$Periodic\n(\d+)\n", text)
        assert links is not None
        assert int(links.group(1)) == 2
    # Nested: the faces of zmin at the finer level are exactly the four children of each coarser
    # face, at its edge midpoints (and, for a quadrilateral, its centre).
    for coarse, fine in itertools.pairwise(meshes):
        children = set()
        for corners in _zone_polygons(coarse, "zmin"):
            k = len(corners)
            mid = [0.5 * (corners[i] + corners[(i + 1) % k]) for i in range(k)]
            if k == 3:
                parts = [
                    (corners[0], mid[0], mid[2]),
                    (mid[0], corners[1], mid[1]),
                    (mid[2], mid[1], corners[2]),
                    (mid[0], mid[1], mid[2]),
                ]
            else:
                centre = corners.mean(axis=0)
                parts = [(corners[i], mid[i], centre, mid[(i - 1) % 4]) for i in range(4)]
            children |= {_polygon_key(np.array(q)) for q in parts}
        assert {_polygon_key(q) for q in _zone_polygons(fine, "zmin")} == children
    again, _ = generate_family(kind, (8,), tmp_path / "again", seed=5)
    assert again.levels[0].file_sha256 == family.levels[0].file_sha256
    # Read back through Gmsh: the nodes are the arrays bit for bit, and each link makes the max
    # zone the translate of the min zone.
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(tmp_path / family.levels[0].file))
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(coords).reshape(-1, 3)[np.argsort(node_tags)]
        np.testing.assert_array_equal(xyz, meshes[0].points)
        entity = {
            gmsh.model.getPhysicalName(d, t): gmsh.model.getEntitiesForPhysicalGroup(d, t)[0]
            for d, t in gmsh.model.getPhysicalGroups(2)
        }
        for a, b, shift in (("xmin", "xmax", (20.0, 0.0, 0.0)), ("ymin", "ymax", (0.0, 20.0, 0.0))):
            master, _, _, affine = gmsh.model.mesh.getPeriodicNodes(2, entity[b])
            assert master == entity[a]
            np.testing.assert_array_equal(np.asarray(affine).reshape(4, 4)[:3, 3], shift)
    finally:
        gmsh.finalize()
