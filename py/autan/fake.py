"""``autan.FakeBackend``: writes synthetic runs that conform to ``autan.results/1``.

The geometry is a unit cube of hexahedra standing in for the real mesh; zone names,
conditions and media come from the resolved case. Fields hold the freestream state,
histories are smooth synthetic curves. Nothing here is a computation: results of
this backend must never be used as reference data.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time as clock
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import TypeAdapter

from autan import cgns
from autan.errors import CaseError, diagnostic
from autan.plan import read_regular
from autan.results import (
    Averaging,
    CaseIdentity,
    Devices,
    FinalMonitor,
    MeshIdentity,
    Results,
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
from autan.schema import (
    CaseSpec,
    Fluid,
    Medium,
    OracleState,
    SolidZone,
    Steady,
    Unsteady,
)
from autan.solver_input import (
    SolidInitial,
    SolidMedium,
    given_state,
    initial_state,
    medium_input,
    turbulence_input,
)
from autan.solver_input import freestream_state as closed_freestream
from autan.units import si
from autan.version import COMMIT, __version__

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from autan.case import Case
    from autan.meshinfo import MeshInfo, PeriodicPair
    from autan.plan import Plan
    from autan.restart import Start

MEDIUM: TypeAdapter[Fluid | SolidZone] = TypeAdapter(Medium)

BC_TYPES = {
    "slip": "BCWallInviscid",
    "symmetry": "BCSymmetryPlane",
    "farfield": "BCFarfield",
    "inlet": "BCInflowSubsonic",
    "velocity-inlet": "BCInflowSubsonic",
    "outlet": "BCOutflowSubsonic",
    "manufactured": "UserDefined",
    "oracle-state": "BCDirichlet",
}
BASE_CHILDREN = frozenset({"Zone", "ZoneNames", "RunMetadata", "DataClass", "DimensionalUnits"})
ZONE_CHILDREN = frozenset(
    {
        "ZoneType",
        "GridCoordinates",
        "NGonElements",
        "NFaceElements",
        "ZoneBC",
        "ZoneGridConnectivity",
        "FlowSolution",
        "FlowSolutionMean",
        "WallSolution",
        "RestartState",
    }
)
GENERATED = re.compile(r"(boundary|cell-zone)-\d{4}")
CONVERGED_AT = 100


def bc_type(condition: dict[str, Any]) -> str:
    if condition["kind"] == "wall":
        return "BCWallViscousIsothermal" if "temperature" in condition else "BCWallViscousHeatFlux"
    return BC_TYPES[condition["kind"]]


def node_names(boundary: list[str], cells: list[str]) -> tuple[list[str], list[str]]:
    """CGNS node names of the zones, by the rule of results.md#solutioncgns."""
    everything = boundary + cells

    def kept(name: str, reserved: frozenset[str]) -> bool:
        return (
            0 < len(name) <= 32
            and name.isascii()
            and name.isprintable()
            and "/" not in name
            and name not in (".", "..")
            and everything.count(name) == 1
            and name not in reserved
            and not GENERATED.fullmatch(name)
        )

    return (
        [n if kept(n, BASE_CHILDREN) else f"boundary-{i:04d}" for i, n in enumerate(boundary)],
        [
            n if kept(n, BASE_CHILDREN | ZONE_CHILDREN) else f"cell-zone-{i:04d}"
            for i, n in enumerate(cells)
        ],
    )


@dataclass(frozen=True)
class Box:
    """A cube of n³ hexahedra written as polyhedra."""

    n: int

    def nodes(self) -> NDArray[np.float64]:
        axis = np.linspace(0.0, 1.0, self.n + 1)
        z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
        return np.stack([x.ravel(), y.ravel(), z.ravel()])

    def _node(self, i: int, j: int, k: int) -> int:
        return i + (self.n + 1) * (j + (self.n + 1) * k)

    def faces(self) -> tuple[list[list[int]], list[int], dict[tuple[str, int, int, int], int]]:
        """Faces (0-based node ids, normal along +axis), boundary face ids, face index."""
        n, at = self.n, self._node
        faces: list[list[int]] = []
        boundary: list[int] = []
        index: dict[tuple[str, int, int, int], int] = {}
        for axis in "xyz":
            for k in range(n + (axis == "z")):
                for j in range(n + (axis == "y")):
                    for i in range(n + (axis == "x")):
                        if axis == "x":
                            quad = [
                                at(i, j, k),
                                at(i, j + 1, k),
                                at(i, j + 1, k + 1),
                                at(i, j, k + 1),
                            ]
                            edge = i in (0, n)
                        elif axis == "y":
                            quad = [
                                at(i, j, k),
                                at(i, j, k + 1),
                                at(i + 1, j, k + 1),
                                at(i + 1, j, k),
                            ]
                            edge = j in (0, n)
                        else:
                            quad = [
                                at(i, j, k),
                                at(i + 1, j, k),
                                at(i + 1, j + 1, k),
                                at(i, j + 1, k),
                            ]
                            edge = k in (0, n)
                        index[axis, i, j, k] = len(faces)
                        if edge:
                            boundary.append(len(faces))
                        faces.append(quad)
        return faces, boundary, index

    def cells(self, index: dict[tuple[str, int, int, int], int]) -> list[list[int]]:
        """Signed 1-based face ids per cell; positive when the normal points out of the cell."""
        n = self.n
        return [
            [
                -(index["x", i, j, k] + 1),
                index["x", i + 1, j, k] + 1,
                -(index["y", i, j, k] + 1),
                index["y", i, j + 1, k] + 1,
                -(index["z", i, j, k] + 1),
                index["z", i, j, k + 1] + 1,
            ]
            for k in range(n)
            for j in range(n)
            for i in range(n)
        ]


def freestream_state(spec: CaseSpec) -> dict[str, Any]:
    """The closed freestream state of the solver input, as the fake's working values."""
    state = closed_freestream(spec)
    return {
        "density": state.density,
        "pressure": state.pressure,
        "temperature": state.temperature,
        "speed": state.speed,
        "viscosity": state.viscosity,
        "direction": np.array(state.velocity) / state.speed,
    }


@dataclass(frozen=True)
class FakeBackend:
    """Writes a synthetic run directory that conforms to ``autan.results/1``.

    The synthetic trajectory depends only on the case and the mesh: a steady run
    converges at iteration 100 unless its budget ends first. ``iterations`` stops the
    run earlier (status ``stopped``); ``pace`` sleeps that many seconds per iteration to
    imitate a live run; ``diverge_at`` makes the run diverge there with ``D0107``.
    """

    iterations: int = 200
    pace: float = 0.0
    diverge_at: int | None = None

    def run(
        self, case: Case, mesh_info: MeshInfo, runs: Path, start: Start | None = None
    ) -> Results:
        result = case.plan(mesh_info)
        if not result.ok:
            raise CaseError(d for d in result.diagnostics if d.severity == "error")
        spec = case.spec
        started = utc_now()
        solver = Solver(version=__version__, commit=COMMIT, backend="fake")
        devices = Devices(kind="cpu", count=1)
        info = RunInfo(
            run_id=RunWriter.new_run_id(started),
            identity=identity(
                spec.hash(),
                mesh_info.source.sha256,
                solver,
                spec.physics.precision,
                devices,
                "input",
                start.info if start is not None else None,
            ),
            status="running",
            case=CaseIdentity(name=spec.name, sha256=spec.hash()),
            mesh=MeshIdentity(path=str(case.mesh_path), sha256=mesh_info.source.sha256),
            solver=solver,
            precision=spec.physics.precision,
            devices=devices,
            cell_order="input",
            started=iso(started),
            finished=None,
            iterations=0,
            time=None,
            averaging=None,
            start=start.info if start is not None else None,
            diagnostics=(),
        )
        manufactured = (
            case.resolve(spec.verification.manufactured.path).read_bytes()
            if spec.verification is not None
            else None
        )
        states = {
            rule.condition.sha256: _archived(
                case.resolve(rule.condition.path), rule.condition.sha256, "E0420"
            )
            for rule in spec.boundaries
            if isinstance(rule.condition, OracleState)
        }
        if spec.initial_state is not None:
            states[spec.initial_state.sha256] = _archived(
                case.resolve(spec.initial_state.path), spec.initial_state.sha256, "E0508"
            )
        writer = RunWriter(runs, info, spec, mesh_info, manufactured, source="fake", states=states)
        for warning in result.diagnostics:
            writer.diagnose(warning)
        history = _History(spec, freestream_state(spec))
        steady = isinstance(spec.time, Steady)
        budget = spec.time.max_iterations if isinstance(spec.time, Steady) else spec.time.steps
        end = min(budget, CONVERGED_AT) if steady else budget
        # A started run begins from the state of the previous one, which the synthetic
        # trajectory leaves unchanged.
        inherited = (
            {name: Results.open(start.run).field(name) for name in field_names(spec)}
            if start is not None
            else None
        )
        # A continued run goes on counting; its trajectory is that of an uninterrupted run.
        offset = start.info.iterations if start and start.info.mode == "continue" else 0
        cap = offset + self.iterations
        diverges = self.diverge_at is not None and self.diverge_at <= min(end, cap)
        last = min(end, cap, (self.diverge_at or end + 1) - 1)
        done, status = offset, "running"
        try:
            for iteration in range(offset + 1, last + 1):
                writer.append(history.record(iteration))
                done = iteration
                if iteration == CONVERGED_AT // 2:
                    writer.log(
                        "info",
                        "synthetic residuals halfway to convergence",
                        source="fake",
                        iteration=iteration,
                    )
                if self.pace:
                    clock.sleep(self.pace)
        except KeyboardInterrupt:
            status = "stopped"
            raise
        except Exception as error:
            status = "failed"
            failure = diagnostic("E0704", f"{type(error).__name__}: {error}")
            writer.finish(
                status=status, iterations=done, time=history.time(done), diagnostics=(failure,)
            )
            raise
        finally:
            if status == "stopped":
                self._finish(writer, spec, result, history, done, "stopped", mesh_info, inherited)
        if diverges:
            status = "diverged"
        elif done < end:
            status = "stopped"
        elif steady:
            status = "converged" if done == CONVERGED_AT else "completed"
        else:
            status = "completed"
        self._finish(writer, spec, result, history, done, status, mesh_info, inherited)
        return Results.open(writer.path)

    def _finish(
        self,
        writer: RunWriter,
        spec: CaseSpec,
        plan: Plan,
        history: _History,
        done: int,
        status: str,
        mesh_info: MeshInfo,
        inherited: dict[str, NDArray[np.float64]] | None = None,
    ) -> None:
        box = Box(
            max(
                8,
                len(mesh_info.cell_zones),
                math.ceil(math.sqrt(len(mesh_info.boundary_zones) / 6)),
            )
        )
        layout = cell_layout(box, plan)
        changes: dict[str, Any] = {"status": status, "iterations": done, "time": history.time(done)}
        if status in ("converged", "completed", "stopped"):
            final = {m.name: history.forces(m, done) for m in spec.monitors}
            if all(math.isfinite(v) for record in final.values() for v in record.values()):
                changes["monitors"] = {
                    n: FinalMonitor.model_validate({"iteration": done, **r})
                    for n, r in final.items()
                }
            else:
                message = "a final monitor value is not finite"
                status = "failed"
                changes.update(status=status, diagnostics=(diagnostic("E0704", message),))
        if status == "diverged":
            assert self.diverge_at is not None
            cell = box.n**3 // 2
            changes["diagnostics"] = (
                diagnostic(
                    "D0107",
                    f"divergence at iteration {self.diverge_at} near cell {cell} "
                    f"(zone {layout[cell]!r})",
                    iteration=self.diverge_at,
                    cell=cell,
                    zone=layout[cell],
                ),
            )
        if isinstance(spec.time, Unsteady) and spec.time.average_from is not None:
            start = spec.time.average_from
            if done > start:
                changes["averaging"] = Averaging(
                    from_step=start + 1,
                    to_step=done,
                    from_time=float(history.time(start) or 0.0),
                    to_time=float(history.time(done) or 0.0),
                )
        info = writer.info.model_copy(update=changes)
        staging = writer.path / ".solution.cgns.tmp"
        _write_solution(staging, spec, plan, mesh_info.periodic, box, layout, info, inherited)
        staging.rename(writer.path / "solution.cgns")
        writer.finish(**changes)


def cell_layout(box: Box, plan: Plan) -> list[str]:
    """Cell zone of every cell: slabs along x, in MeshInfo order."""
    names = [z.zone for z in plan.cell_zones]
    return [names[(c % box.n) * len(names) // box.n] for c in range(box.n**3)]


class _History:
    """Synthetic residuals reaching ``residual_orders`` at CONVERGED_AT, forces approaching a
    limit; both depend only on the case."""

    def __init__(self, spec: CaseSpec, state: dict[str, Any]) -> None:
        self.spec, self.state = spec, state
        self.equations = residual_names(spec)
        self.orders = spec.time.residual_orders if isinstance(spec.time, Steady) else 3.0

    def time(self, step: int) -> float | None:
        if isinstance(self.spec.time, Steady):
            return None
        dt = self.spec.time.dt
        return step * (si(dt, "s") if dt is not None else 1e-3)

    def record(self, iteration: int) -> dict[str, Any]:
        fraction = (iteration - 1) / (CONVERGED_AT - 1)
        residuals = {
            name: 10.0 ** (-self.orders * fraction) * (1 + index)
            for index, name in enumerate(self.equations)
        }
        monitors = {
            monitor.name: self.forces(monitor, iteration)
            for monitor in self.spec.monitors
            if iteration % monitor.every == 0
        }
        return {
            "iteration": iteration,
            "time": self.time(iteration),
            "residuals": residuals,
            "monitors": monitors,
        }

    def forces(self, monitor: Any, iteration: int) -> dict[str, float]:
        fs, state = self.spec.freestream, self.state
        alpha, beta = si(fs.aoa, "rad"), si(fs.sideslip, "rad")
        q = 0.5 * state["density"] * state["speed"] ** 2
        area = si(monitor.ref_area, "m**2") if monitor.ref_area is not None else 1.0
        approach = 1.0 - math.exp(-iteration / (0.2 * CONVERGED_AT))
        cl = 2 * math.pi * alpha * approach
        cd = 0.02 + cl**2 / (math.pi * 10.0)
        e_d = np.array(
            [math.cos(alpha) * math.cos(beta), -math.sin(beta), math.sin(alpha) * math.cos(beta)]
        )
        e_l = np.array([-math.sin(alpha), 0.0, math.cos(alpha)])
        e_s = np.cross(e_l, e_d)
        force = q * area * (cd * e_d + cl * e_l)
        length = si(fs.ref_length, "m") if fs.ref_length is not None else 1.0
        center = np.array([si(c, "m") for c in monitor.moment_center])
        with np.errstate(over="ignore", invalid="ignore"):  # non-finite: the run fails
            moment = np.cross(np.array([0.25 * length, 0.0, 0.0]) - center, force)
        values = {key: float(v) for key, v in zip(("fx", "fy", "fz"), force, strict=True)}
        values.update({key: float(v) for key, v in zip(("mx", "my", "mz"), moment, strict=True)})
        if monitor.ref_area is not None:
            values.update(
                cd=float(force @ e_d / (q * area)),
                cs=float(force @ e_s / (q * area)),
                cl=float(force @ e_l / (q * area)),
            )
            if fs.ref_length is not None:
                values.update(
                    {
                        f"c{key}": float(v / (q * area * length))
                        for key, v in zip(("mx", "my", "mz"), moment, strict=True)
                    }
                )
        return values


def _write_solution(
    path: Path,
    spec: CaseSpec,
    plan: Plan,
    periodic: tuple[PeriodicPair, ...],
    box: Box,
    layout: list[str],
    info: RunInfo,
    inherited: dict[str, NDArray[np.float64]] | None = None,
) -> None:
    """Write solution.cgns for the final ``info``."""
    faces, boundary_faces, index = box.faces()
    cells = box.cells(index)
    boundary_zones = [b for b in plan.boundaries if b.faces > 0]
    boundary_names = [b.zone for b in plan.boundaries]
    cell_names = [z.zone for z in plan.cell_zones]
    boundary_nodes, cell_nodes = node_names(boundary_names, cell_names)
    pairs = [p.zones for p in periodic]
    sizes = _sizes(len(boundary_faces), [b.zone for b in boundary_zones], pairs)
    bounds = np.cumsum([0, *(sizes[b.zone] for b in boundary_zones)])
    owned = {
        b.zone: np.array(boundary_faces[bounds[i] : bounds[i + 1]])
        for i, b in enumerate(boundary_zones)
    }
    n_faces = len(faces)

    with cgns.create(path) as handle:
        base = cgns.node(handle, "Base", "CGNSBase_t", np.array([3, 3], dtype=np.int32))
        cgns.node(base, "DataClass", "DataClass_t", "Dimensional")
        cgns.node(
            base,
            "DimensionalUnits",
            "DimensionalUnits_t",
            np.frombuffer(
                "".join(
                    u.ljust(32) for u in ("Kilogram", "Meter", "Second", "Kelvin", "Radian")
                ).encode(),
                dtype=np.int8,
            )
            .reshape(5, 32)
            .T,
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
                    "cell": dict(zip(cell_nodes, cell_names, strict=True)),
                },
                sort_keys=True,
            ),
        )
        conditions = {b.zone: b.condition for b in plan.boundaries}
        for zone, name in zip(boundary_names, boundary_nodes, strict=True):
            family = cgns.node(base, name, "Family_t")
            condition = conditions[zone]
            assert condition is not None or zone not in owned
            if condition is not None and condition["kind"] != "periodic":
                cgns.node(family, "FamilyBC", "FamilyBC_t", bc_type(condition))
        for name in cell_nodes:
            cgns.node(base, name, "Family_t")

        zone = cgns.node(
            base,
            "Zone",
            "Zone_t",
            np.array([[box.nodes().shape[1], len(cells), 0]], dtype=np.int64),
        )
        cgns.node(zone, "ZoneType", "ZoneType_t", "Unstructured")
        coordinates = cgns.node(zone, "GridCoordinates", "GridCoordinates_t")
        for axis, values in zip("XYZ", box.nodes(), strict=True):
            cgns.data_array(coordinates, f"Coordinate{axis}", values)
        _elements(zone, "NGonElements", 22, 1, [[v + 1 for v in f] for f in faces])
        _elements(zone, "NFaceElements", 23, n_faces + 1, cells)

        node_of = dict(zip(boundary_names, boundary_nodes, strict=True))
        if any(
            c is not None and c["kind"] != "periodic" for z, c in conditions.items() if z in owned
        ):
            zone_bc = cgns.node(zone, "ZoneBC", "ZoneBC_t")
        for zone_name, name in zip(boundary_names, boundary_nodes, strict=True):
            condition = conditions[zone_name]
            if zone_name not in owned or (condition and condition["kind"] == "periodic"):
                continue
            assert condition is not None
            bc = cgns.node(zone_bc, name, "BC_t", bc_type(condition))
            _point_list(bc, owned[zone_name] + 1)
            cgns.node(bc, "GridLocation", "GridLocation_t", "FaceCenter")
            cgns.node(bc, "FamilyName", "FamilyName_t", name)

        if periodic:
            _write_periodic(zone, periodic, owned, node_of)

        for zone_name, name in zip(cell_names, cell_nodes, strict=True):
            members = [c for c, owner in enumerate(layout) if owner == zone_name]
            region = cgns.node(zone, name, "ZoneSubRegion_t", np.array([3], dtype=np.int32))
            _point_list(region, np.array(members) + n_faces + 1)
            cgns.node(region, "GridLocation", "GridLocation_t", "CellCenter")
            cgns.node(region, "FamilyName", "FamilyName_t", name)

        fields = inherited if inherited is not None else _fields(spec, plan, layout)
        for solution_name in ("FlowSolution",) + (
            ("FlowSolutionMean",) if info.averaging is not None else ()
        ):
            solution = cgns.node(zone, solution_name, "FlowSolution_t")
            cgns.node(solution, "GridLocation", "GridLocation_t", "CellCenter")
            for name, values in fields.items():
                cgns.data_array(solution, name, values)

        walls = np.sort(
            np.concatenate(
                [
                    owned[b.zone]
                    for b in boundary_zones
                    if b.condition and b.condition["kind"] in ("wall", "slip")
                ]
                or [np.array([], dtype=np.int64)]
            )
        )
        if walls.size:
            region = cgns.node(zone, "WallSolution", "ZoneSubRegion_t", np.array([2], np.int32))
            _point_list(region, walls + 1)
            cgns.node(region, "GridLocation", "GridLocation_t", "FaceCenter")
            owner = {abs(face) - 1: cell for cell, faces in enumerate(cells) for face in faces}
            cells_of_walls = np.array([owner[int(face)] for face in walls])
            cgns.data_array(region, "Pressure", fields["Pressure"][cells_of_walls])
            for axis in "XYZ":
                cgns.data_array(region, f"WallShearStress{axis}", np.zeros(walls.size))


def _sizes(total: int, zones: list[str], pairs: list[tuple[str, str]]) -> dict[str, int]:
    """Boundary faces per zone of the synthetic box, equal on both sides of a periodic pair
    (the box has 6 n^2 boundary faces, an even number)."""
    paired = {zone for pair in pairs for zone in pair}
    share = total // len(zones)
    sizes = dict.fromkeys(paired, share)
    rest = total - share * len(paired)
    others = [zone for zone in zones if zone not in paired]
    if others:
        sizes |= {
            z: len(c)
            for z, c in zip(others, np.array_split(np.arange(rest), len(others)), strict=True)
        }
    else:
        for a, b in pairs[: rest // 2]:
            sizes[a] += 1
            sizes[b] += 1
    return sizes


def _write_periodic(
    zone: Any,
    periodic: tuple[PeriodicPair, ...],
    owned: dict[str, NDArray[Any]],
    node_of: dict[str, str],
) -> None:
    """Each periodic pair as two reciprocal GridConnectivity_t (results.md#solutioncgns); the
    synthetic box pairs the faces of the two zones in order."""
    connectivity = cgns.node(zone, "ZoneGridConnectivity", "ZoneGridConnectivity_t")
    for pair in periodic:
        a, b = pair.zones
        vector = np.array(pair.transform.vector)
        for side, other, shift in ((a, b, vector), (b, a, -vector)):
            gc = cgns.node(connectivity, node_of[side], "GridConnectivity_t", "Zone")
            cgns.node(gc, "GridConnectivityType", "GridConnectivityType_t", "Abutting1to1")
            cgns.node(gc, "GridLocation", "GridLocation_t", "FaceCenter")
            _point_list(gc, owned[side] + 1)
            cgns.node(
                gc,
                "PointListDonor",
                "IndexArray_t",
                (owned[other] + 1).astype(np.int64).reshape(1, -1),
            )
            properties = cgns.node(gc, "GridConnectivityProperty", "GridConnectivityProperty_t")
            periodic = cgns.node(properties, "Periodic", "Periodic_t")
            zero = np.zeros(3, dtype=np.float32)
            cgns.data_array(periodic, "RotationCenter", zero)
            cgns.data_array(periodic, "RotationAngle", zero)
            cgns.data_array(periodic, "Translation", shift.astype(np.float32))
            cgns.node(
                gc, "PeriodicTranslation", "Descriptor_t", json.dumps([float(v) for v in shift])
            )


def _point_list(parent: Any, ids: NDArray[Any]) -> None:
    cgns.node(parent, "PointList", "IndexArray_t", np.asarray(ids, dtype=np.int64).reshape(1, -1))


def _elements(zone: Any, name: str, kind: int, first: int, items: list[list[int]]) -> None:
    elements = cgns.node(zone, name, "Elements_t", np.array([kind, 0], dtype=np.int32))
    cgns.node(
        elements,
        "ElementRange",
        "IndexRange_t",
        np.array([first, first + len(items) - 1], dtype=np.int64),
    )
    offsets = np.concatenate([[0], np.cumsum([len(i) for i in items])]).astype(np.int64)
    cgns.node(elements, "ElementStartOffset", "DataArray_t", offsets)
    cgns.node(
        elements,
        "ElementConnectivity",
        "DataArray_t",
        np.array([v for i in items for v in i], dtype=np.int64),
    )


def _fields(spec: CaseSpec, plan: Plan, layout: list[str]) -> dict[str, NDArray[np.float64]]:
    """Every cell holds the initial state of its zone (solver-input.md#initial-state)."""
    freestream, turbulence = closed_freestream(spec), turbulence_input(spec)
    initial = {
        z.zone: initial_state(
            medium_input(MEDIUM.validate_python(z.medium)),
            freestream,
            turbulence,
            given_state(MEDIUM.validate_python(z.medium)),
        )
        for z in plan.cell_zones
    }
    densities = {}
    for z in plan.cell_zones:
        medium = medium_input(MEDIUM.validate_python(z.medium))
        if isinstance(medium, SolidMedium):
            densities[z.zone] = medium.density
    names = field_names(spec)
    fields = {name: np.zeros(len(layout)) for name in names}
    for cell, zone in enumerate(layout):
        zone_state = initial[zone]
        if isinstance(zone_state, SolidInitial):
            fields["Density"][cell] = densities[zone]
            if "Temperature" in fields:
                fields["Temperature"][cell] = zone_state.temperature
            continue
        values = {
            "Density": zone_state.density,
            "Pressure": zone_state.pressure,
            "VelocityX": zone_state.velocity[0],
            "VelocityY": zone_state.velocity[1],
            "VelocityZ": zone_state.velocity[2],
            "Temperature": zone_state.temperature,
            "TurbulentSANuTilde": zone_state.nu_tilde,
            "TurbulentEnergyKinetic": zone_state.k,
            "TurbulentDissipationRate": zone_state.omega,
        }
        for name in names:
            fields[name][cell] = values[name]
    return fields


def _archived(path: Path, sha256: str, code: str) -> bytes:
    """The bytes of a file the plan checked, read the same way, which must not have changed."""
    try:
        data = read_regular(path)
    except OSError as error:
        message = f"{path} cannot be read: {error}"
        raise CaseError([diagnostic(code, message, path=str(path))]) from None
    if hashlib.sha256(data).hexdigest() != sha256:
        message = f"{path} changed after it was checked: its SHA-256 differs from the case's"
        raise CaseError([diagnostic(code, message, path=str(path))])
    return data
