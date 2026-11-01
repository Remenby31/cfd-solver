"""Run directories in the `autan.results/1` format, written from a FaceMesh and its cell fields.

These runs are fixtures for the verification harness: they let it be exercised on any mesh family
with fields of known error before the solver exists. They are written through `autan`'s own
`RunWriter` and CGNS node writer and read back with `autan.Results.open`, which checks them against
docs/contracts/results.md. They always carry backend `fake`: they are not solver results and
can never be evidence.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import autan
import numpy as np
import numpy.typing as npt
from autan import cgns
from autan.fake import bc_type, node_names
from autan.results import (
    CaseIdentity,
    Devices,
    MeshIdentity,
    RunInfo,
    RunWriter,
    Solver,
    field_names,
    identity,
    iso,
    residual_names,
    run_metadata,
    utc_now,
)
from autan.version import COMMIT, __version__

from autan_vv.errors import VVError

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.meshes.facemesh import FaceMesh, FloatArray

KINDS = {
    "tet": "tetra",
    "hex": "hexa",
    "prism": "prism",
    "pyramid": "pyramid",
    "polyhedron": "polyhedron",
}
NORMALIZED = {"wall": "wall", "pressure-far-field": "farfield"}


def mesh_info(mesh: FaceMesh, path: Path, mesh_format: str) -> autan.MeshInfo:
    """The MeshInfo the core reader reports for `mesh` stored at `path` (docs/contracts/mesh-info.md).

    Gmsh files carry no boundary types (`type: null`); Fluent files carry theirs.
    """
    kinds = mesh.cell_kinds or ("polyhedron",) * mesh.n_cells
    by_type = dict.fromkeys(KINDS.values(), 0)
    for k in kinds:
        by_type[KINDS[k]] += 1
    g = mesh.geometry
    ni = mesh.n_interior
    d = g.cell_centroid[mesh.neighbour] - g.cell_centroid[mesh.owner[:ni]]
    db = g.face_centroid[ni:] - g.cell_centroid[mesh.owner[ni:]]
    normal = g.face_area / np.linalg.norm(g.face_area, axis=1, keepdims=True)
    cos = np.einsum("ij,ij->i", np.concatenate([d, db]), normal) / np.linalg.norm(
        np.concatenate([d, db]), axis=1
    )
    faces_per_cell = np.bincount(
        np.concatenate([mesh.owner, mesh.neighbour]), minlength=mesh.n_cells
    )
    typed = mesh_format == "fluent"
    document = {
        "schema": "autan.mesh-info/1",
        "source": {
            "path": str(path),
            "format": mesh_format,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "groups": [],
        "groups_source": None,
        "bounds": {
            "min": [float(x) for x in mesh.points.min(axis=0)],
            "max": [float(x) for x in mesh.points.max(axis=0)],
        },
        "counts": {
            "cells": mesh.n_cells,
            "faces": mesh.n_faces,
            "nodes": int(mesh.points.shape[0]),
            "cells_by_type": by_type,
        },
        "boundary_zones": [
            {
                "name": z.name,
                "type": NORMALIZED[z.bc_type] if typed else None,
                "native_type": z.bc_type if typed else None,
                "faces": z.count,
            }
            for z in mesh.zones
        ],
        "cell_zones": [{"name": "fluid", "type": None, "native_type": None, "cells": mesh.n_cells}],
        "interfaces": [],
        "periodic": [],
        "quality": {
            "max_non_orthogonality_deg": float(
                np.degrees(np.arccos(np.clip(cos.min(), -1.0, 1.0)))
            ),
            "max_faces_per_cell": int(faces_per_cell.max()),
        },
    }
    return autan.MeshInfo.from_document(document, autan.Location(file=str(path)))


def _point_list(parent: Any, ids: npt.ArrayLike) -> None:
    cgns.node(parent, "PointList", "IndexArray_t", np.asarray(ids, dtype=np.int64).reshape(1, -1))


def _elements(
    zone: Any, name: str, kind: int, first: int, offsets: npt.ArrayLike, connectivity: npt.ArrayLike
) -> None:
    elements = cgns.node(zone, name, "Elements_t", np.array([kind, 0], dtype=np.int32))
    count = np.asarray(offsets).size - 1
    cgns.node(
        elements,
        "ElementRange",
        "IndexRange_t",
        np.array([first, first + count - 1], dtype=np.int64),
    )
    cgns.node(elements, "ElementStartOffset", "DataArray_t", np.asarray(offsets, dtype=np.int64))
    cgns.node(
        elements, "ElementConnectivity", "DataArray_t", np.asarray(connectivity, dtype=np.int64)
    )


def _write_solution(
    path: Path, mesh: FaceMesh, plan: autan.Plan, info: RunInfo, fields: dict[str, FloatArray]
) -> None:
    boundary_names = [b.zone for b in plan.boundaries]
    boundary_nodes, cell_nodes = node_names(boundary_names, ["fluid"])
    # A plan that is ok gives every zone with faces a condition; V&V meshes have no empty zone.
    conditions = {b.zone: b.condition for b in plan.boundaries if b.condition is not None}
    zone_of = {z.name: z for z in mesh.zones}
    n_faces = mesh.n_faces
    # NFACE: each cell lists its faces, positive when the face normal points out of the cell.
    cell_faces: list[list[int]] = [[] for _ in range(mesh.n_cells)]
    for f, c in enumerate(mesh.owner.tolist()):
        cell_faces[c].append(f + 1)
    for f, c in enumerate(mesh.neighbour.tolist()):
        cell_faces[c].append(-(f + 1))
    with cgns.create(path) as handle:
        base = cgns.node(handle, "Base", "CGNSBase_t", np.array([3, 3], dtype=np.int32))
        cgns.node(base, "DataClass", "DataClass_t", "Dimensional")
        units = "".join(u.ljust(32) for u in ("Kilogram", "Meter", "Second", "Kelvin", "Radian"))
        cgns.node(
            base,
            "DimensionalUnits",
            "DimensionalUnits_t",
            np.frombuffer(units.encode(), dtype=np.int8).reshape(5, 32).T,
        )
        metadata = run_metadata(info)
        cgns.node(base, "RunMetadata", "Descriptor_t", json.dumps(metadata, sort_keys=True))
        cgns.node(
            base,
            "ZoneNames",
            "Descriptor_t",
            json.dumps(
                {
                    "boundary": dict(zip(boundary_nodes, boundary_names, strict=True)),
                    "cell": dict(zip(cell_nodes, ["fluid"], strict=True)),
                },
                sort_keys=True,
            ),
        )
        for zone_name, node_name in zip(boundary_names, boundary_nodes, strict=True):
            family = cgns.node(base, node_name, "Family_t")
            cgns.node(family, "FamilyBC", "FamilyBC_t", bc_type(conditions[zone_name]))
        cgns.node(base, cell_nodes[0], "Family_t")
        zone = cgns.node(
            base,
            "Zone",
            "Zone_t",
            np.array([[mesh.points.shape[0], mesh.n_cells, 0]], dtype=np.int64),
        )
        cgns.node(zone, "ZoneType", "ZoneType_t", "Unstructured")
        coordinates = cgns.node(zone, "GridCoordinates", "GridCoordinates_t")
        for axis in range(3):
            cgns.data_array(coordinates, f"Coordinate{'XYZ'[axis]}", mesh.points[:, axis].copy())
        _elements(zone, "NGonElements", 22, 1, mesh.face_offsets, mesh.face_nodes + 1)
        offsets = np.concatenate([[0], np.cumsum([len(c) for c in cell_faces])])
        _elements(
            zone, "NFaceElements", 23, n_faces + 1, offsets, [f for c in cell_faces for f in c]
        )
        zone_bc = cgns.node(zone, "ZoneBC", "ZoneBC_t")
        walls = []
        for zone_name, node_name in zip(boundary_names, boundary_nodes, strict=True):
            z = zone_of[zone_name]
            condition = conditions[zone_name]
            bc = cgns.node(zone_bc, node_name, "BC_t", bc_type(condition))
            _point_list(bc, np.arange(z.start, z.start + z.count) + 1)
            cgns.node(bc, "GridLocation", "GridLocation_t", "FaceCenter")
            cgns.node(bc, "FamilyName", "FamilyName_t", node_name)
            if condition["kind"] in ("wall", "slip"):
                walls.append(np.arange(z.start, z.start + z.count))
        region = cgns.node(zone, cell_nodes[0], "ZoneSubRegion_t", np.array([3], dtype=np.int32))
        _point_list(region, np.arange(mesh.n_cells) + n_faces + 1)
        cgns.node(region, "GridLocation", "GridLocation_t", "CellCenter")
        cgns.node(region, "FamilyName", "FamilyName_t", cell_nodes[0])
        solution = cgns.node(zone, "FlowSolution", "FlowSolution_t")
        cgns.node(solution, "GridLocation", "GridLocation_t", "CellCenter")
        for name, values in fields.items():
            cgns.data_array(solution, name, np.asarray(values, dtype=np.float64))
        if walls:
            ids = np.sort(np.concatenate(walls))
            wall = cgns.node(zone, "WallSolution", "ZoneSubRegion_t", np.array([2], np.int32))
            _point_list(wall, ids + 1)
            cgns.node(wall, "GridLocation", "GridLocation_t", "FaceCenter")
            for name in ("Pressure", "WallShearStressX", "WallShearStressY", "WallShearStressZ"):
                cgns.data_array(wall, name, np.zeros(ids.size))


def write_run(
    runs: Path,
    case: autan.Case,
    mesh: FaceMesh,
    info: autan.MeshInfo,
    fields: dict[str, FloatArray],
    *,
    time: float | None = None,
    status: str = "converged",
) -> autan.Results:
    """Write one finished run of `case` on `mesh` with the given cell fields; return it opened.

    `fields` must hold exactly the fields results.md requires for the case's physics, one value per
    cell in mesh order. The history holds one line with unit residuals. Cases with monitors or time
    averaging are refused (V0731); runs are marked non-deterministic, since their fields are given.
    """
    spec = case.spec
    if spec.monitors or getattr(spec.time, "average_from", None) is not None:
        msg = "fixture runs carry no monitors and no time averaging"
        raise VVError(msg, code="V0731")
    plan = case.plan(info)
    if not plan.ok:
        raise autan.CaseError(d for d in plan.diagnostics if d.severity == "error")
    missing = set(field_names(spec)) ^ set(fields)
    if missing:
        msg = f"fields must be exactly {field_names(spec)}, differing in {sorted(missing)}"
        raise VVError(msg, code="V0731")
    malformed = [k for k, v in fields.items() if np.shape(v) != (mesh.n_cells,)]
    if malformed:
        msg = f"fields {malformed} must hold one value per cell ({mesh.n_cells})"
        raise VVError(msg, code="V0731")
    started = utc_now()
    # Always the fake backend: fixture fields are given, never computed, so they are never evidence.
    solver = Solver(version=__version__, commit=COMMIT, backend="fake")
    devices = Devices(kind="cpu", count=1)
    groups = info.groups_source.sha256 if info.groups_source else None
    run = RunInfo(
        run_id=RunWriter.new_run_id(started),
        identity=identity(
            spec.hash(), info.source.sha256, groups, solver, spec.physics.precision, devices
        ),
        status="running",
        case=CaseIdentity(name=spec.name, sha256=spec.hash()),
        mesh=MeshIdentity(
            path=str(case.mesh_path), sha256=info.source.sha256, groups_sha256=groups
        ),
        solver=solver,
        precision=spec.physics.precision,
        # The fields come from the caller, not from the identified configuration.
        deterministic=False,
        devices=devices,
        started=iso(started),
        finished=None,
        iterations=0,
        time=None,
        averaging=None,
        diagnostics=(),
    )
    # oracle-state conditions: the face-state files, archived as results.md#run-directory says.
    states = {
        rule.condition.sha256: case.resolve(rule.condition.path).read_bytes()
        for rule in spec.boundaries
        if rule.condition.kind == "oracle-state"
    }
    writer = RunWriter(runs, run, spec, info, states=states)
    writer.append(
        {
            "iteration": 1,
            "time": time,
            "residuals": dict.fromkeys(residual_names(spec), 1.0),
            "monitors": {},
        }
    )
    final = writer.info.model_copy(update={"status": status, "iterations": 1, "time": time})
    staging = writer.path / ".solution.cgns.tmp"
    _write_solution(staging, mesh, plan, final, {k: fields[k] for k in field_names(spec)})
    staging.rename(writer.path / "solution.cgns")
    writer.finish(status=status, iterations=1, time=time)
    return autan.Results.open(writer.path)
