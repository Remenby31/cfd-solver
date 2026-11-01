"""Verify solver runs (`autan.results/1`) against an oracle: errors, observed order, gate verdict."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import autan
import numpy as np
from autan import cgns, initial_state
from autan.results import field_names
from autan.schema import (
    DDES,
    SA,
    SST,
    Compressible,
    ConstantDensityFluid,
    ConstantViscosity,
    Fluid,
    IdealGas,
    Incompressible,
    Inviscid,
    Laminar,
    SolidZone,
)
from autan.units import si

from autan_vv.errors import VVError
from autan_vv.gates import CONSERVED, FLOOR_FACTOR, GATES, MIN_LEVELS, PRESERVED, Gate
from autan_vv.meshes.facemesh import FaceMesh, FloatArray, Zone
from autan_vv.oracles import Oracle, conserved
from autan_vv.oracles.registry import get_oracle
from autan_vv.order import fit_order

if TYPE_CHECKING:
    from pathlib import Path

VERDICT_SCHEMA = "autan.vv-verdict/1"
UNIVERSAL_GAS_CONSTANT = 8.314462618  # J/(mol K), CODATA 2018 exact
PRECISION_EPS = {
    "fp64": float(np.finfo(np.float64).eps),
    "fp32": float(np.finfo(np.float32).eps),
    "fp32+acc64": float(np.finfo(np.float32).eps),
}
FINISHED = ("converged", "completed")


def read_geometry(results: autan.Results) -> FaceMesh:
    """Rebuild the face-based mesh of a run from solution.cgns (NGON_n faces, NFACE_n cells).

    A positive face id in a cell's NFACE list means the face normal points out of that cell
    (results.md); that cell becomes the face's owner. Cell i is NFACE element F + 1 + i.
    """
    try:
        return _read_geometry(results)
    except (KeyError, IndexError, ValueError) as exc:
        msg = f"cannot rebuild the geometry of {results.path.name}: {exc}"
        raise VVError(msg, code="V0721") from exc


def _read_geometry(results: autan.Results) -> FaceMesh:
    zone = cgns.read(results.path / "solution.cgns").child("Base").child("Zone")
    coords = zone.child("GridCoordinates")
    points = np.column_stack(
        [np.asarray(coords.child(f"Coordinate{a}").data, dtype=np.float64) for a in "XYZ"]
    )
    ngon, nface = zone.child("NGonElements"), zone.child("NFaceElements")
    face_offsets = np.asarray(ngon.child("ElementStartOffset").data, dtype=np.int64)
    face_nodes = np.asarray(ngon.child("ElementConnectivity").data, dtype=np.int64) - 1
    cell_offsets = np.asarray(nface.child("ElementStartOffset").data, dtype=np.int64)
    signed = np.asarray(nface.child("ElementConnectivity").data, dtype=np.int64)
    n_faces, n_cells = face_offsets.size - 1, cell_offsets.size - 1
    cell_of = np.repeat(np.arange(n_cells), np.diff(cell_offsets))
    face = np.abs(signed) - 1
    owner = np.full(n_faces, -1, dtype=np.int64)
    neighbour = np.full(n_faces, -1, dtype=np.int64)
    outward = signed > 0
    owner[face[outward]] = cell_of[outward]
    neighbour[face[~outward]] = cell_of[~outward]
    flip = owner < 0  # a face referenced only inward: reverse it so that its only cell owns it
    if np.any(face_nodes < 0) or np.any(face_nodes >= points.shape[0]) or np.any(face >= n_faces):
        msg = "a face or node index is out of range"
        raise VVError(msg, code="V0721")
    if np.any(owner < 0) and np.any(neighbour[flip] < 0):
        msg = "a face belongs to no cell"
        raise VVError(msg, code="V0721")
    owner[flip], neighbour[flip] = neighbour[flip], -1
    polys = [face_nodes[face_offsets[f] : face_offsets[f + 1]] for f in range(n_faces)]
    for f in np.flatnonzero(flip):
        polys[f] = polys[f][::-1]
    interior = np.flatnonzero(neighbour >= 0)
    boundary = np.flatnonzero(neighbour < 0)
    order = np.concatenate([interior, boundary])
    sizes = np.array([polys[f].size for f in order])
    mesh = FaceMesh(
        points=points,
        face_offsets=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        face_nodes=np.concatenate([polys[f] for f in order]).astype(np.int64),
        owner=owner[order],
        neighbour=neighbour[interior],
        n_cells=n_cells,
        zones=(Zone("boundary", "boundary", interior.size, boundary.size),),
    )
    mesh.check()
    return mesh


def _viscosity(gas: IdealGas, temperature: float) -> float:
    """Dynamic viscosity of the gas at `temperature` (constant, or Sutherland's law)."""
    law = gas.viscosity
    if isinstance(law, ConstantViscosity):
        return float(si(law.mu, "Pa * s"))
    t_ref, s = si(law.t_ref, "K"), si(law.s, "K")
    return float(
        si(law.mu_ref, "Pa * s") * (temperature / t_ref) ** 1.5 * (t_ref + s) / (temperature + s)
    )


def freestream_oracle(spec: autan.CaseSpec) -> Oracle:
    """The uniform freestream state of a compressible case (case-schema.md, Freestream; T defaults to 288.15 K).

    With `pressure`: rho = p/(R T). With `re`: rho = Re mu(T)/(V L) and p = rho R T. V = M sqrt(gamma R T),
    direction (cos a cos b, -sin b, sin a cos b). Turbulence (case-schema.md, Physics): SA nu~ =
    ratio mu/rho; SST k = 1.5 (I V)^2, omega = rho k/(ratio mu).
    """
    physics = spec.physics
    if not isinstance(physics, Compressible) or not isinstance(physics.gas, IdealGas):
        msg = "the freestream oracle needs a compressible ideal-gas case"
        raise VVError(msg, code="V0722")
    gas = physics.gas
    fs = spec.freestream
    # A validated compressible case has mach, and either pressure or re with ref_length.
    if fs.mach is None or (fs.pressure is None and (fs.re is None or fs.ref_length is None)):
        msg = "the case's freestream lacks mach, or both pressure and re with ref_length"
        raise VVError(msg, code="V0722")
    r = UNIVERSAL_GAS_CONSTANT / si(gas.molar_mass, "kg / mol")
    temperature = si(fs.temperature, "K") if fs.temperature is not None else 288.15
    speed = fs.mach * math.sqrt(gas.gamma * r * temperature)
    if fs.pressure is not None:
        pressure = si(fs.pressure, "Pa")
        density = pressure / (r * temperature)
    else:
        mu = _viscosity(gas, temperature)
        assert fs.re is not None
        assert fs.ref_length is not None
        density = fs.re * mu / (speed * si(fs.ref_length, "m"))
        pressure = density * r * temperature
    a, b = si(fs.aoa, "rad"), si(fs.sideslip, "rad")
    velocity = speed * np.array(
        [math.cos(a) * math.cos(b), -math.sin(b), math.sin(a) * math.cos(b)]
    )

    mu = _viscosity(gas, temperature)
    turbulence = physics.turbulence
    base = turbulence.base if isinstance(turbulence, DDES) else turbulence
    extra: dict[str, float] = {}
    if isinstance(base, SA):
        extra["TurbulentSANuTilde"] = base.nu_tilde_ratio * mu / density
    elif isinstance(base, SST):
        k = 1.5 * (base.intensity * speed) ** 2
        extra["TurbulentEnergyKinetic"] = k
        extra["TurbulentDissipationRate"] = density * k / (base.viscosity_ratio * mu)

    def evaluate(points: FloatArray, _t: float | None) -> dict[str, FloatArray]:
        one = np.ones(points.shape[0])
        return {
            "Density": density * one,
            "VelocityX": velocity[0] * one,
            "VelocityY": velocity[1] * one,
            "VelocityZ": velocity[2] * one,
            "Pressure": pressure * one,
            "Temperature": temperature * one,
        } | {name: value * one for name, value in extra.items()}

    kind = (
        "euler"
        if isinstance(turbulence, Inviscid)
        else "navier-stokes"
        if isinstance(turbulence, Laminar)
        else "any"
    )
    return Oracle(
        "freestream",
        "Uniform freestream state of the case.",
        kind,
        (0, 1, 2),
        False,
        gas.gamma,
        r,
        evaluate,
    )


def _check_compatible(results: autan.Results, oracle: Oracle) -> list[str]:
    """Reasons the run cannot be compared with the oracle (empty if it can)."""
    spec = results.case
    reasons = []
    if results.run.status not in FINISHED:
        reasons.append(f"status {results.run.status}, not converged or completed")
    physics = spec.physics
    expected = {"euler": Inviscid, "navier-stokes": Laminar, "rans-sa": SA}
    if any(isinstance(rule.medium, SolidZone) for rule in spec.cell_zones):
        reasons.append("the case has solid zones; oracles describe one fluid")
    if oracle.name == "freestream" and any(
        isinstance(rule.medium, Fluid) and rule.medium.material != getattr(physics, "gas", None)
        for rule in spec.cell_zones
    ):
        reasons.append(
            "a cell zone overrides the gas: the freestream is defined for the default gas only"
        )
    if oracle.physics in expected:
        if not isinstance(physics, Compressible) or not isinstance(
            physics.turbulence, expected[oracle.physics]
        ):
            reasons.append(
                f"physics {physics.kind}/{physics.turbulence.model}, oracle needs {oracle.physics}"
            )
        else:
            # Every fluid material the case uses: the default gas and each cell-zone override.
            gases = [
                physics.gas,
                *(r.medium.material for r in spec.cell_zones if isinstance(r.medium, Fluid)),
            ]
            for gas in gases:
                if not isinstance(gas, IdealGas):
                    reasons.append(f"material {gas.name} is not an ideal gas")
                    continue
                r = UNIVERSAL_GAS_CONSTANT / si(gas.molar_mass, "kg / mol")
                if not math.isclose(gas.gamma, oracle.gamma, rel_tol=1e-12):
                    reasons.append(f"{gas.name}: gamma {gas.gamma}, oracle uses {oracle.gamma}")
                if not math.isclose(r, oracle.gas_constant, rel_tol=1e-9):
                    reasons.append(
                        f"{gas.name}: gas constant {r:.9g} J/(kg K), oracle uses {oracle.gas_constant:.9g}"
                    )
                if oracle.physics == "euler":
                    continue
                viscosity = oracle.parameters.get("viscosity")
                law = gas.viscosity
                if viscosity is not None and not (
                    isinstance(law, ConstantViscosity)
                    and math.isclose(si(law.mu, "Pa * s"), viscosity, rel_tol=1e-12)
                ):
                    reasons.append(
                        f"{gas.name}: oracle assumes a constant viscosity {viscosity} Pa s"
                    )
                prandtl = oracle.parameters.get("prandtl")
                if prandtl is not None and not math.isclose(gas.prandtl, prandtl, rel_tol=1e-12):
                    reasons.append(
                        f"{gas.name}: Prandtl number {gas.prandtl}, oracle uses {prandtl}"
                    )
    if oracle.physics == "incompressible-navier-stokes":
        if not isinstance(physics, Incompressible) or not isinstance(physics.turbulence, Laminar):
            reasons.append("oracle needs a laminar incompressible case")
        else:
            fluids = [
                physics.fluid,
                *(r.medium.material for r in spec.cell_zones if isinstance(r.medium, Fluid)),
            ]
            for fluid in fluids:
                if not isinstance(fluid, ConstantDensityFluid):
                    reasons.append(f"material {fluid.name} is not a constant-density fluid")
                    continue
                for key, unit, value in (
                    ("density", "kg / m**3", fluid.density),
                    ("viscosity", "Pa * s", fluid.viscosity),
                ):
                    if not math.isclose(si(value, unit), oracle.parameters[key], rel_tol=1e-12):
                        reasons.append(
                            f"{fluid.name}: {key} {si(value, unit)}, oracle uses {oracle.parameters[key]}"
                        )
    if oracle.unsteady and results.run.time is None:
        reasons.append("unsteady oracle but the run has no physical time")
    return reasons


@dataclass(frozen=True)
class RunErrors:
    """Errors of one run against the oracle, with the run's provenance and mesh extent."""

    provenance: dict[str, Any]
    n_cells: int
    volume: float
    h: float
    bounds: tuple[tuple[float, float], ...]
    errors: dict[str, dict[str, float]]


def _h(mesh: FaceMesh, axes: tuple[int, ...]) -> tuple[float, float]:
    """Total volume and mesh size along the oracle's axes. Along every other direction each cell must
    span the whole mesh thickness, one layer (V0723); h = (V / (their extent) / N)^(1/len(axes))."""
    g = mesh.geometry
    volume = float(g.cell_volume.sum())
    span = 1.0
    cell_of_node = np.repeat(mesh.owner, np.diff(mesh.face_offsets))
    nb = np.repeat(mesh.neighbour, np.diff(mesh.face_offsets)[: mesh.n_interior])
    for axis in sorted(set(range(3)) - set(axes)):
        coord = mesh.points[:, axis]
        lo, hi = float(coord.min()), float(coord.max())
        extent = hi - lo
        values = coord[mesh.face_nodes]
        cell_min = np.full(mesh.n_cells, np.inf)
        cell_max = np.full(mesh.n_cells, -np.inf)
        for cells, vals in ((cell_of_node, values), (nb, values[: nb.size])):
            np.minimum.at(cell_min, cells, vals)
            np.maximum.at(cell_max, cells, vals)
        one_layer = (
            extent > 0
            and np.all(cell_min - lo <= 1e-9 * extent)
            and np.all(hi - cell_max <= 1e-9 * extent)
        )
        if not one_layer:
            msg = f"the oracle varies along {[('xyz')[a] for a in axes]}: the mesh must be one cell thick along {'xyz'[axis]}"
            raise VVError(msg, code="V0723")
        span *= extent
    return volume, (volume / span / mesh.n_cells) ** (1.0 / len(axes))


def run_errors(results: autan.Results, oracle: Oracle) -> RunErrors:
    """L1, L2 (volume-weighted) and L-infinity errors of every field the oracle and run share.

    The run's cell values are compared with the oracle at the cell centroids (the difference from a
    cell average is O(h^2), which leaves second-order observed orders unchanged).
    """
    reasons = _check_compatible(results, oracle)
    if reasons:
        msg = f"run {results.run.run_id} cannot be verified against {oracle.name}: {'; '.join(reasons)}"
        raise VVError(msg, code="V0724")
    names = field_names(results.case)
    run = {
        name: results.field(name) for name in names
    }  # the Results API checks solution.cgns first
    mesh = read_geometry(results)
    malformed = [n for n, v in run.items() if np.shape(v) != (mesh.n_cells,)]
    if malformed:
        msg = f"fields {malformed} do not hold one value per cell ({mesh.n_cells})"
        raise VVError(msg, code="V0721")
    g = mesh.geometry
    exact = oracle.fields(g.cell_centroid, results.run.time)
    run = {name: value for name, value in run.items() if name in exact}
    if math.isfinite(oracle.gamma) and all(n in run for n in ("Density", "VelocityX", "Pressure")):
        exact |= conserved(exact, oracle.gamma)
        run |= conserved(run, oracle.gamma)
        exact["Entropy"] = exact["Pressure"] / exact["Density"] ** oracle.gamma
        # A non-physical run (rho <= 0) gives a NaN entropy: reported as a failed measurement.
        with np.errstate(invalid="ignore", divide="ignore"):
            run["Entropy"] = run["Pressure"] / run["Density"] ** oracle.gamma
    weight = g.cell_volume / g.cell_volume.sum()
    errors = {}
    for name, value in run.items():
        diff = value - exact[name]
        errors[name] = {
            "L1": float(np.sum(weight * np.abs(diff))),
            "L2": float(np.sqrt(np.sum(weight * diff**2))),
            "Linf": float(np.max(np.abs(diff))),
            "scale": float(np.max(np.abs(exact[name]))),
        }
    volume, h = _h(mesh, oracle.axes)
    run_info = results.run
    provenance = {
        "path": str(results.path),
        "run_id": run_info.run_id,
        "identity": run_info.identity,
        "status": run_info.status,
        "backend": run_info.solver.backend,
        "solver_commit": run_info.solver.commit,
        "precision": run_info.precision,
        "case_sha256": run_info.case.sha256,
        "mesh_sha256": run_info.mesh.sha256,
        "time": run_info.time,
    }
    bounds = tuple(
        (float(mesh.points[:, a].min()), float(mesh.points[:, a].max())) for a in range(3)
    )
    return RunErrors(provenance, mesh.n_cells, volume, h, bounds, errors)


GROUPS = (("rhou", "rhov", "rhow"), ("VelocityX", "VelocityY", "VelocityZ"))


def _scale(run: RunErrors, name: str) -> float:
    """Magnitude a field's error is measured against: the largest of its vector's components (so a
    component the oracle makes zero is judged against the whole vector), and 1 for a zero field."""
    group = next((g for g in GROUPS if name in g), (name,))
    scale = max(run.errors[n]["scale"] for n in group if n in run.errors)
    return scale if scale > 0.0 else 1.0


# The initial-state file's variables (initial-state.md), as the oracle names its fields.
_INITIAL_FIELDS = ("Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure")


def _initial_state_problem(results: autan.Results, oracle: Oracle) -> str | None:
    """Why the run's initial state, if it has one, is not `oracle` at its start: the archived file
    must name the oracle and hold its values at t = 0 (steady: its only state) at its centroids,
    to the file's 17 significant digits."""
    spec = results.case.initial_state
    if spec is None:
        return None
    path = results.path / "states" / f"{spec.sha256}.json"
    document = initial_state.parse(path.read_bytes())
    run = results.run.run_id
    if isinstance(document, str):
        return f"run {run}: its initial state is not a valid document ({document})"
    if document.oracle != oracle.name:
        return f"run {run} starts from {document.oracle!r}, not from {oracle.name!r}"
    exact = oracle.fields(np.array(document.centroids), 0.0 if oracle.unsteady else None)
    values = np.array(document.values)
    for k, name in enumerate(_INITIAL_FIELDS):
        scale = max(float(np.abs(exact[name]).max()), 1e-300)
        if float(np.abs(values[:, k] - exact[name]).max()) > 1e-12 * scale:
            return f"run {run} does not start from {oracle.name} at t = 0 ({name})"
    return None


def _family_problem(opened: list[autan.Results]) -> list[str]:
    """Reasons runs cannot form one refinement family: one problem, one solver build, one time."""
    reasons = []

    def problem(results: autan.Results) -> dict[str, Any]:
        doc: dict[str, Any] = results.case.model_dump(mode="json")
        doc.pop("mesh", None)
        doc.pop("name", None)
        # Face states and initial states are exported per mesh: each level names its own file
        # (whether there is one stays part of the case; its content is checked in verify).
        if doc.get("initial_state") is not None:
            doc["initial_state"] = "per mesh"
        for rule in doc.get("boundaries", []):
            condition = rule.get("condition") or {}
            if condition.get("kind") == "oracle-state":
                condition.pop("path", None)
                condition.pop("sha256", None)
        return doc

    first = opened[0]
    for results in opened[1:]:
        if problem(results) != problem(first):
            reasons.append(f"{results.run.run_id} solves a different case than {first.run.run_id}")
        if (results.run.solver.commit, results.run.solver.backend) != (
            first.run.solver.commit,
            first.run.solver.backend,
        ):
            reasons.append(f"{results.run.run_id} comes from another solver build or backend")
        t0, t1 = first.run.time, results.run.time
        if (t0 is None) != (t1 is None) or (
            t0 is not None and t1 is not None and not math.isclose(t0, t1, rel_tol=1e-12)
        ):
            reasons.append(f"{results.run.run_id} stops at another physical time")
    return reasons


def _fits(runs: list[RunErrors], eps: float) -> dict[str, dict[str, Any]]:
    """Order of every norm of every field, over the levels whose relative error is at least 100 times
    the rounding floor of the precision; fewer than two such levels give no fit."""
    fits: dict[str, dict[str, Any]] = {}
    for name in runs[0].errors:
        fits[name] = {}
        for norm in ("L1", "L2", "Linf"):
            values = [r.errors[name][norm] for r in runs]
            if not all(math.isfinite(v) for v in values):
                fits[name][norm] = {"levels": 0, "excluded": len(runs), "non_finite": True}
                continue
            eligible = [
                r for r in runs if r.errors[name][norm] >= FLOOR_FACTOR * eps * _scale(r, name)
            ]
            entry: dict[str, Any] = {"levels": len(eligible), "excluded": len(runs) - len(eligible)}
            if len(eligible) >= 2:
                fit = fit_order([r.h for r in eligible], [r.errors[name][norm] for r in eligible])
                entry |= {
                    "order": fit.order,
                    "pairwise_orders": list(fit.pairwise_orders),
                    "r_squared": fit.r_squared,
                }
            fits[name][norm] = entry
    return fits


def _gate_checks(
    gate: Gate, runs: list[RunErrors], fits: dict[str, dict[str, Any]], eps: float
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if gate.time is not None:
        times = [r.provenance["time"] for r in runs]
        ok = all(t is not None and math.isclose(t, gate.time, rel_tol=1e-9) for t in times)
        checks.append(
            {"prerequisite": "time", "required": gate.time, "measured": times, "pass": ok}
        )
    if gate.domain is not None:
        ok = all(
            math.isclose(r.bounds[a][0], lo, abs_tol=1e-9 * (hi - lo))
            and math.isclose(r.bounds[a][1], hi, abs_tol=1e-9 * (hi - lo))
            for r in runs
            for a, (lo, hi) in enumerate(gate.domain)
        )
        checks.append(
            {
                "prerequisite": "domain",
                "required": [list(d) for d in gate.domain],
                "measured": [list(b) for b in runs[0].bounds[: len(gate.domain)]],
                "pass": ok,
            }
        )
    derived = (*CONSERVED, "Entropy")
    names = (
        gate.fields
        if gate.fields is not None
        else tuple(n for n in runs[0].errors if n not in derived)
    )
    for name in names:
        if any(name not in r.errors for r in runs):
            checks.append({"field": name, "pass": False, "reason": "not written by the runs"})
            continue
        if not all(math.isfinite(r.errors[name][gate.norm]) for r in runs):
            checks.append(
                {
                    "field": name,
                    "pass": False,
                    "reason": "non-finite error (NaN or infinity) on a run",
                }
            )
            continue
        worst = max(r.errors[name][gate.norm] / _scale(r, name) / eps for r in runs)
        if gate.measure == "max-error":
            checks.append(
                {
                    "field": name,
                    "max_relative_error_in_eps": worst,
                    "upper": gate.upper,
                    "pass": worst <= gate.upper,
                }
            )
            continue
        if all(r.errors[name]["scale"] == 0.0 for r in runs):
            checks.append(
                {
                    "field": name,
                    "identically_zero": True,
                    "max_relative_error_in_eps": worst,
                    "upper": PRESERVED,
                    "pass": worst <= PRESERVED,
                }
            )
            continue
        fit = fits[name][gate.norm]
        if fit["levels"] < MIN_LEVELS:
            checks.append(
                {
                    "field": name,
                    "pass": False,
                    "levels": fit["levels"],
                    "reason": f"{fit['levels']} levels above {FLOOR_FACTOR:g}x the rounding floor, {MIN_LEVELS} needed",
                }
            )
            continue
        fitted, finest = fit["order"], fit["pairwise_orders"][-1]
        orders = (fitted, finest)
        above = max(orders) > gate.upper
        check: dict[str, Any] = {
            "field": name,
            "fitted_order": fitted,
            "finest_pair_order": finest,
            "levels": fit["levels"],
            "lower": gate.lower,
            "upper": gate.upper,
            "pass": all(math.isfinite(v) for v in orders)
            and min(orders) >= gate.lower
            and (not above or not gate.upper_blocks),
        }
        if above and not gate.upper_blocks:
            check["warning"] = f"order above {gate.upper:g}"
            if gate.investigate_above is not None and max(orders) > gate.investigate_above:
                check["investigate"] = True
        if gate.monotone:
            # Every adjacent pair of levels; only two levels both under the rounding floor may
            # stall, so that a level under the floor cannot hide a coarser error rising again.
            levels = [
                (
                    r.errors[name][gate.norm],
                    r.errors[name][gate.norm] < FLOOR_FACTOR * eps * _scale(r, name),
                )
                for r in runs
            ]
            decreasing = all(
                b < a or (a_floor and b_floor)
                for (a, a_floor), (b, b_floor) in itertools.pairwise(levels)
            )
            check["errors_decrease"] = decreasing
            check["pass"] = check["pass"] and decreasing
        checks.append(check)
    return checks


def _json_ready(value: Any) -> Any:
    """Non-finite numbers become null (a failed measurement), so the report is strict JSON."""
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def verify(run_dirs: list[Path], against: str, gate_id: str | None = None) -> dict[str, Any]:
    """Compare runs with an oracle and, with `gate_id`, give a verdict (schema autan.vv-verdict/1).

    `against` is an oracle name (`autan-vv oracles`) or `freestream`, built from each run's case.
    Several runs form a refinement family: one case apart from its mesh, one solver build and
    backend, one precision, one physical time, distinct meshes (V0726, V0727). Orders are fitted over
    the levels whose error is at least 100 times the rounding floor. `evidence` is false when a run
    comes from the `fake` backend or the oracle is a numerical reference, whatever the gate says.
    """
    if not run_dirs:
        msg = "no run to verify"
        raise VVError(msg, code="V0725")
    gate = None
    if gate_id is not None:
        gate = GATES.get(gate_id)
        if gate is None:
            msg = f"unknown gate {gate_id!r}; known: {', '.join(sorted(GATES))}"
            raise VVError(msg, code="V0728")
        if against not in gate.oracles:
            msg = f"gate {gate_id} is checked against {', '.join(gate.oracles)}, not {against}"
            raise VVError(msg, code="V0728")
    opened = [autan.Results.open(p) for p in run_dirs]
    precisions = {r.run.precision for r in opened}
    if len(precisions) > 1:
        msg = f"runs of one family must share a precision, got {sorted(precisions)}"
        raise VVError(msg, code="V0726")
    reasons = _family_problem(opened)
    if reasons:
        raise VVError("; ".join(reasons), code="V0727")
    oracle = freestream_oracle(opened[0].case) if against == "freestream" else get_oracle(against)
    for results in opened:
        problem = _initial_state_problem(results, oracle)
        if problem:
            raise VVError(problem, code="V0727")
    runs = [run_errors(results, oracle) for results in opened]
    hs = [r.h for r in runs]
    for r in runs[1:]:
        for a in range(3):
            if oracle.curved and a in oracle.axes:
                continue
            extent = max(runs[0].bounds[a][1] - runs[0].bounds[a][0], 1e-300)
            if any(abs(r.bounds[a][k] - runs[0].bounds[a][k]) > 1e-9 * extent for k in (0, 1)):
                msg = f"run {r.provenance['run_id']} covers another domain along {'xyz'[a]}: a family refines one domain"
                raise VVError(msg, code="V0727")
    if len(set(hs)) != len(hs) or len({r.provenance["mesh_sha256"] for r in runs}) != len(runs):
        msg = "a refinement family needs distinct meshes"
        raise VVError(msg, code="V0727")
    eps = PRECISION_EPS[precisions.pop()]
    runs.sort(key=lambda r: -r.h)
    fits = _fits(runs, eps) if len(runs) >= 2 else {}
    report: dict[str, Any] = {
        "schema": VERDICT_SCHEMA,
        "oracle": {
            "name": oracle.name,
            "physics": oracle.physics,
            "reference_only": oracle.reference_only,
            "parameters": dict(oracle.parameters),
        },
        "runs": [
            {
                **r.provenance,
                "n_cells": r.n_cells,
                "volume": r.volume,
                "h": r.h,
                "bounds": r.bounds,
                "errors": r.errors,
            }
            for r in runs
        ],
        "fits": fits,
        "verdict": None,
    }
    if gate is not None:
        if gate.measure == "order" and len(runs) < MIN_LEVELS:
            msg = f"an order gate needs at least {MIN_LEVELS} runs"
            raise VVError(msg, code="V0729")
        checks = _gate_checks(gate, runs, fits, eps)
        not_evidence = []
        if any(r.provenance["backend"] == "fake" for r in runs):
            not_evidence.append("fake backend: synthetic results are never evidence")
        if oracle.reference_only:
            not_evidence.append("the oracle is a numerical reference, not an exact solution")
        report["verdict"] = {
            "gate": gate.id,
            "milestone": gate.milestone,
            "checks": checks,
            "pass": all(c["pass"] for c in checks),
            "warnings": [f"{c['field']}: {c['warning']}" for c in checks if "warning" in c],
            "investigate": any(c.get("investigate", False) for c in checks),
            "evidence": not not_evidence,
            "not_evidence_because": not_evidence,
        }
    return _json_ready(report)  # type: ignore[no-any-return]
