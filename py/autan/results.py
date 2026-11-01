"""Run directories ``autan.results/1``: writing them and reading them back (docs/contracts/results.md)."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from autan import cgns
from autan.errors import CaseError, Diagnostic, Location, diagnostic
from autan.meshinfo import MeshInfo
from autan.schema import DDES, SA, SST, CaseSpec, Compressible, Steady

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from autan.schema import Forces as ForcesSpec

SCHEMA_ID = "autan.results/1"
Status = Literal["running", "converged", "completed", "stopped", "diverged", "failed"]
SOLUTION_REQUIRED: frozenset[str] = frozenset({"converged", "completed", "stopped"})
Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CaseIdentity(_Strict):
    name: str
    sha256: Hex


class MeshIdentity(_Strict):
    path: str
    sha256: Hex


class Solver(_Strict):
    version: str
    commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    backend: Literal["cuda", "hip", "cpu", "fake"]


class Devices(_Strict):
    kind: Literal["cuda", "hip", "cpu"]
    count: Annotated[int, Field(ge=1)]


class Averaging(_Strict):
    from_step: int
    to_step: int
    from_time: float
    to_time: float


EQUATIONS = (
    "density",
    "pressure",
    "momentum-x",
    "momentum-y",
    "momentum-z",
    "energy",
    "nu-tilde",
    "k",
    "omega",
)
# What a JSON Schema can say of ``woken``; membership and order are checked by the readers.
WOKEN_SCHEMA = {"type": "array", "items": {"enum": list(EQUATIONS)}, "uniqueItems": True}


def woken_problem(reference: dict[str, float] | None, woken: list[str] | None) -> str | None:
    """Why ``woken`` does not go with ``residual_reference`` (results.md#runjson, ``start``)."""
    if (woken is None) != (reference is None):
        return "start.woken is given exactly with start.residual_reference"
    if woken is not None and (
        woken != sorted(set(woken)) or not set(woken) <= set(reference or {})
    ):
        return "start.woken lists sorted, distinct equations of residual_reference"
    return None


class StartInfo(_Strict):
    """Where a restarted run started from (results.md#runjson, ``start``)."""

    run_id: Annotated[str, Field(pattern=r"^run-\d{8}T\d{6}Z-[0-9a-f]{8}$")]
    identity: Hex
    mode: Literal["continue", "warm"]
    iterations: Annotated[int, Field(strict=True, ge=1)]
    time: float | None
    solution_sha256: Hex
    residual_reference: dict[str, float] | None
    woken: Annotated[
        list[str] | None, Field(json_schema_extra={"anyOf": [WOKEN_SCHEMA, {"type": "null"}]})
    ]

    @model_validator(mode="after")
    def _woken(self) -> Self:
        problem = woken_problem(self.residual_reference, self.woken)
        if problem:
            raise ValueError(problem)
        return self


class Convergence(_Strict):
    """The round-off floor of each equation of an fp32+acc64 steady run (case-schema.md#time)."""

    floor: dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]]


class FinalMonitor(BaseModel):
    """A monitor's record on the final state: its History quantities and the iteration."""

    model_config = ConfigDict(extra="allow", frozen=True)
    iteration: Annotated[int, Field(strict=True, ge=0)]
    __pydantic_extra__: dict[str, Annotated[float, Field(strict=True, allow_inf_nan=False)]]


class RunInfo(_Strict):
    """The content of ``run.json``."""

    schema_: Literal["autan.results/1"] = Field(default="autan.results/1", alias="schema")
    run_id: Annotated[str, Field(pattern=r"^run-\d{8}T\d{6}Z-[0-9a-f]{8}$")]
    identity: Hex
    status: Status
    case: CaseIdentity
    mesh: MeshIdentity
    solver: Solver
    precision: Literal["fp32+acc64", "fp64"]
    devices: Devices
    cell_order: Literal["rcm", "input"]
    started: str
    finished: str | None
    iterations: Annotated[int, Field(ge=0)]
    time: float | None
    averaging: Averaging | None
    start: StartInfo | None = None
    convergence: Convergence | None = None
    monitors: dict[str, FinalMonitor] | None = None
    diagnostics: tuple[Diagnostic, ...]

    model_config = ConfigDict(populate_by_name=True)

    def document(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def identity(
    case: str,
    mesh: str,
    solver: Solver,
    precision: str,
    devices: Devices,
    cell_order: str,
    start: StartInfo | None = None,
) -> str:
    """The reproducibility fingerprint of a run (results.md, ``identity``)."""
    parts = [case, mesh, solver.commit, precision, solver.backend, devices.kind, str(devices.count)]
    parts.append(cell_order)
    if start is not None:
        parts += [start.identity, start.mode, str(start.iterations), start.solution_sha256]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


class LogLine(_Strict):
    """One line of ``log.jsonl`` (results.md#log)."""

    time: Annotated[str, Field(pattern=r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")]
    iteration: Annotated[int, Field(strict=True, ge=1)] | None
    level: Literal["info", "warning", "error"]
    source: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]*$")]
    message: str
    code: str | None
    data: dict[str, Any]


def log_line_problems(entry: LogLine) -> list[str]:
    """Rules of results.md#log beyond the shape: registered code, level from the code's
    letter, no data without a code, the line's iteration equal to the data's."""
    from autan.errors import REGISTRY  # noqa: PLC0415 -- errors imports nothing from here

    found = []
    try:
        datetime.strptime(entry.time, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        found.append(f"time {entry.time!r} is not a valid UTC timestamp")
    if entry.code is None:
        return found
    if entry.code not in REGISTRY:
        return [*found, f"{entry.code!r} is not a registered code"]
    if entry.level != _LEVEL[REGISTRY[entry.code].severity]:
        found.append(f"{entry.code} is logged at level {_LEVEL[REGISTRY[entry.code].severity]}")
    iteration = entry.data.get("iteration")
    if isinstance(iteration, int) and entry.iteration != iteration:
        found.append("iteration differs from the diagnostic's")
    return found


def now_ms() -> str:
    moment = datetime.now(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


LogLevel = Literal["info", "warning", "error"]
_LEVEL: dict[str, LogLevel] = {"error": "error", "divergence": "error", "warning": "warning"}


class RunWriter:
    """Creates a run directory and keeps ``run.json`` and ``history.jsonl`` up to date.

    The directory is prepared under a hidden name and renamed once ``run.json``,
    ``case.toml`` and ``mesh-info.json`` are written, so a reader listing ``runs``
    never sees a run without them.
    """

    def __init__(
        self,
        runs: Path,
        info: RunInfo,
        spec: CaseSpec,
        mesh_info: MeshInfo,
        manufactured: bytes | None = None,
        source: str = "solver",
        states: dict[str, bytes] | None = None,
    ) -> None:
        self.source = source
        runs.mkdir(parents=True, exist_ok=True)
        staging = runs / f".{info.run_id}.tmp"
        staging.mkdir()
        self.path = staging
        self.info = info
        self.spec = spec
        (staging / "case.toml").write_text(_toml(spec))
        (staging / "mesh-info.json").write_text(json.dumps(mesh_info.document(), indent=2))
        if manufactured is not None:
            (staging / "manufactured.json").write_bytes(manufactured)
        if states:
            (staging / "states").mkdir()
            for digest, content in states.items():
                (staging / "states" / f"{digest}.json").write_bytes(content)
        # The log is written before the metadata that refers to it (results.md#log).
        self.log("info", f"run {info.run_id} started", source=self.source)
        for found in info.diagnostics:
            self.diagnose(found)
        self._write_info()
        final = runs / info.run_id
        if final.exists():
            message = f"run directory {final} already exists"
            raise FileExistsError(message)
        staging.rename(final)
        self.path = final

    @staticmethod
    def new_run_id(started: datetime) -> str:
        return f"run-{started.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(4)}"

    def _write_info(self) -> None:
        temporary = self.path / "run.json.tmp"
        temporary.write_text(json.dumps(self.info.document(), indent=2) + "\n")
        temporary.replace(self.path / "run.json")

    def append(self, record: dict[str, Any]) -> None:
        with (self.path / "history.jsonl").open("a") as history:
            history.write(json.dumps(record, allow_nan=False) + "\n")
            history.flush()

    def log(
        self,
        level: LogLevel,
        message: str,
        *,
        source: str,
        iteration: int | None = None,
        code: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Append one line to log.jsonl, written in one piece and flushed."""
        line = LogLine(
            time=now_ms(),
            iteration=iteration,
            level=level,
            source=source,
            message=message,
            code=code,
            data=data or {},
        )
        problems = log_line_problems(line)
        if problems:
            raise ValueError("; ".join(problems))
        with (self.path / "log.jsonl").open("a") as log:
            log.write(json.dumps(line.model_dump(mode="json"), allow_nan=False) + "\n")
            log.flush()

    def diagnose(self, found: Diagnostic) -> None:
        """Log a diagnostic with its code and data, as run.json will carry it; its line's
        iteration is the diagnostic's ``iteration`` datum, if any."""
        iteration = found.data.get("iteration")
        self.log(
            _LEVEL[found.severity],
            found.message,
            source=self.source,
            iteration=iteration if isinstance(iteration, int) else None,
            code=found.code,
            data=found.data,
        )

    def finish(self, **changes: Any) -> None:
        if changes.get("status") in FINAL and "monitors" not in changes:
            if self.spec.monitors:
                message = "a finished run of a case with monitors records their final values"
                raise ValueError(message)
            changes["monitors"] = {}
        self.info = self.info.model_copy(update={**changes, "finished": iso(utc_now())})
        for found in self.info.diagnostics:
            self.diagnose(found)
        self.log(
            "info",
            f"run {self.info.status} after {self.info.iterations} iterations",
            source=self.source,
            iteration=self.info.iterations or None,
        )
        self._write_info()


def _toml(spec: CaseSpec) -> str:
    import tomli_w  # noqa: PLC0415 -- only writers need it

    return tomli_w.dumps(spec.document())


class History:
    """Residual and monitor time series of a run, as NumPy arrays."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self.iterations: NDArray[np.int64] = np.array(
            [r["iteration"] for r in records], dtype=np.int64
        )

    def residual(self, equation: str) -> NDArray[np.float64]:
        return np.array([r["residuals"][equation] for r in self._records], dtype=np.float64)

    @property
    def equations(self) -> list[str]:
        return list(self._records[0]["residuals"]) if self._records else []

    def monitor(self, name: str) -> dict[str, NDArray[Any]]:
        """Every quantity of monitor ``name``, with the iterations it was recorded at."""
        rows = [
            (r["iteration"], r["monitors"][name]) for r in self._records if name in r["monitors"]
        ]
        if not rows:
            raise CaseError([diagnostic("E0702", f"no monitor named {name!r} in the history")])
        keys = rows[0][1].keys()
        series: dict[str, NDArray[Any]] = {
            "iteration": np.array([i for i, _ in rows], dtype=np.int64)
        }
        for key in keys:
            series[key] = np.array([values[key] for _, values in rows], dtype=np.float64)
        return series

    @property
    def monitors(self) -> list[str]:
        names: dict[str, None] = {}
        for record in self._records:
            names.update(dict.fromkeys(record["monitors"]))
        return list(names)


class Forces:
    """A forces monitor: ``fx`` … ``mz`` in SI and, when available, coefficients."""

    def __init__(self, series: dict[str, NDArray[Any]]) -> None:
        self._series = series

    def __getattr__(self, name: str) -> NDArray[Any]:
        try:
            return self._series[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def quantities(self) -> list[str]:
        return list(self._series)


class Results:
    """A run directory read back and checked against ``autan.results/1``."""

    def __init__(
        self,
        path: Path,
        run: RunInfo,
        case: CaseSpec,
        mesh_info: MeshInfo,
        history: History,
        log: tuple[LogLine, ...] = (),
    ) -> None:
        self.log = log
        self.path, self.run, self.case, self.mesh_info, self.history = (
            path,
            run,
            case,
            mesh_info,
            history,
        )
        self._solution: cgns.Node | None = None

    @classmethod
    def open(cls, path: str | Path) -> Results:
        """Read the run directory at ``path``; any departure from the contract is ``E0701``."""
        path = Path(path)

        def broken(message: str, file: str | None = None) -> NoReturn:
            where = Location(file=str(path / file) if file else str(path))
            raise CaseError([diagnostic("E0701", message, location=where)])

        try:
            document = json.loads((path / "run.json").read_text())
        except (OSError, ValueError) as error:
            broken(f"cannot read run.json: {error}", "run.json")
        if not isinstance(document, dict) or document.get("schema") != SCHEMA_ID:
            broken(f"run.json schema is not {SCHEMA_ID!r}", "run.json")
        for key in ("start", "convergence", "monitors"):
            if key not in document:
                broken(f"run.json has no {key} key (null when it does not apply)", "run.json")
        try:
            run = RunInfo.model_validate(document)
        except ValidationError as error:
            first = error.errors(include_url=False)[0]
            broken(f"run.json: {first['msg']} at {first['loc']}", "run.json")
        try:
            case_document = tomllib.loads((path / "case.toml").read_text())
            mesh_document = json.loads((path / "mesh-info.json").read_text())
        except (OSError, ValueError) as error:
            broken(str(error))
        try:
            case = CaseSpec.from_document(case_document, Location(file=str(path / "case.toml")))
            mesh_info = MeshInfo.from_document(
                mesh_document, Location(file=str(path / "mesh-info.json"))
            )
        except CaseError as error:
            broken(f"stored case or MeshInfo is invalid: {error.codes}")
        if run.convergence is not None and (
            run.precision != "fp32+acc64"
            or not isinstance(case.time, Steady)
            or set(run.convergence.floor) != set(residual_names(case))
        ):
            broken(
                "convergence floors exist only for a steady fp32+acc64 run, one per equation",
                "run.json",
            )
        digests = [
            r.condition.sha256 for r in case.boundaries if r.condition.kind == "oracle-state"
        ]
        if case.initial_state is not None:
            digests.append(case.initial_state.sha256)
        for digest in digests:
            archived = path / "states" / f"{digest}.json"
            if (
                not archived.is_file()
                or hashlib.sha256(archived.read_bytes()).hexdigest() != digest
            ):
                broken(f"{archived.name} is missing or differs from the case", "states")
        if case.verification is not None:
            archived = path / "manufactured.json"
            if (
                not archived.is_file()
                or hashlib.sha256(archived.read_bytes()).hexdigest()
                != case.verification.manufactured.sha256
            ):
                broken("manufactured.json is missing or differs from the case", "manufactured.json")
        for problem in run_problems(path, run, case, mesh_info):
            broken(problem, "run.json")
        records = _read_history(path / "history.jsonl", run.status == "running", broken)
        if not (path / "log.jsonl").is_file():
            broken("log.jsonl is missing", "log.jsonl")
        lines = _read_history(path / "log.jsonl", run.status == "running", broken)
        for problem in log_problems(lines, run):
            broken(problem, "log.jsonl")
        for problem in history_problems(records, run, case):
            broken(problem, "history.jsonl")
        for problem in final_monitor_problems(records, run, case):
            broken(problem, "run.json")
        results = cls(
            path,
            run,
            case,
            mesh_info,
            History(records),
            tuple(LogLine.model_validate(line) for line in lines),
        )
        if run.status in SOLUTION_REQUIRED and not (path / "solution.cgns").exists():
            broken(f"a {run.status} run must have solution.cgns", "solution.cgns")
        return results

    @property
    def forces(self) -> Forces:
        """The forces monitor of the case, when it has exactly one."""
        names = [m.name for m in self.case.monitors if m.kind == "forces"]
        if len(names) != 1:
            raise CaseError(
                [
                    diagnostic(
                        "E0702",
                        f"the case has {len(names)} forces monitors ({', '.join(names)}); "
                        "use results.history.monitor(name)",
                    )
                ]
            )
        return Forces(self.history.monitor(names[0]))

    def _tree(self) -> cgns.Node:
        """The solution, read and checked against the contract on first use."""
        if self._solution is not None:
            return self._solution
        path = self.path / "solution.cgns"
        where = Location(file=str(path))
        if self.run.status == "running" or not path.exists():
            reason = "is still running" if self.run.status == "running" else "has no solution"
            message = f"run {self.run.run_id} ({self.run.status}) {reason}"
            raise CaseError([diagnostic("E0703", message, location=where)])
        try:
            tree = cgns.read(path)
        except (OSError, ValueError, KeyError) as error:
            message = f"cannot read solution.cgns: {error}"
            raise CaseError([diagnostic("E0701", message, location=where)]) from None
        problems = solution_problems(tree, self.run, self.case, self.mesh_info)
        if problems:
            raise CaseError(diagnostic("E0701", p, location=where) for p in problems)
        self._solution = tree
        return tree

    def _zone(self) -> cgns.Node:
        return self._tree().child("Base").child("Zone")

    def field(self, name: str, *, mean: bool = False) -> NDArray[np.float64]:
        """Cell-centred field ``name`` (SIDS name, SI units), in mesh cell order."""
        name_of_solution = "FlowSolutionMean" if mean else "FlowSolution"
        zone = self._zone()
        if name_of_solution not in zone.children:
            message = f"run {self.run.run_id} has no time average"
            raise CaseError([diagnostic("E0703", message)])
        solution = zone.child(name_of_solution)
        if name not in solution.children:
            raise CaseError(
                [
                    diagnostic(
                        "E0703",
                        f"no field {name!r}; fields: "
                        f"{', '.join(k for k in solution.children if k != 'GridLocation')}",
                    )
                ]
            )
        return np.asarray(solution.child(name).data, dtype=np.float64)

    def wall(self, name: str) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Wall field ``name`` with the 0-based indices of the faces it is given on."""
        region = self._zone().child("WallSolution")
        faces = np.asarray(region.child("PointList").data, dtype=np.int64).ravel() - 1
        return faces, np.asarray(region.child(name).data, dtype=np.float64)

    def cell_zone(self, zone: str) -> NDArray[np.int64]:
        """0-based indices of the cells of cell zone ``zone``."""
        names = json.loads(self._tree().child("Base").child("ZoneNames").text())["cell"]
        node_name = next(k for k, v in names.items() if v == zone)
        zone_node = self._zone()
        offset = int(zone_node.child("NFaceElements").child("ElementRange").data[0])
        points = np.asarray(zone_node.child(node_name).child("PointList").data).ravel()
        return points.astype(np.int64) - offset


def residual_names(spec: CaseSpec) -> list[str]:
    """The equations of the history, in order (results.md#history)."""
    names = (
        ["density", "momentum-x", "momentum-y", "momentum-z", "energy"]
        if isinstance(spec.physics, Compressible)
        else ["pressure", "momentum-x", "momentum-y", "momentum-z"]
    )
    base = _base(spec)
    if isinstance(base, SA):
        return [*names, "nu-tilde"]
    return [*names, "k", "omega"] if isinstance(base, SST) else names


def field_names(spec: CaseSpec) -> list[str]:
    """The FlowSolution fields a run of ``spec`` writes (results.md#solutioncgns)."""
    names = ["Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"]
    if isinstance(spec.physics, Compressible):
        names.append("Temperature")
    base = _base(spec)
    if isinstance(base, SA):
        return [*names, "TurbulentSANuTilde"]
    if isinstance(base, SST):
        return [*names, "TurbulentEnergyKinetic", "TurbulentDissipationRate"]
    return names


def force_keys(monitor: ForcesSpec, spec: CaseSpec) -> set[str]:
    keys = {"fx", "fy", "fz", "mx", "my", "mz"}
    if monitor.ref_area is not None:
        keys |= {"cd", "cs", "cl"}
        if spec.freestream.ref_length is not None:
            keys |= {"cmx", "cmy", "cmz"}
    return keys


def _base(spec: CaseSpec) -> object:
    turbulence = spec.physics.turbulence
    return turbulence.base if isinstance(turbulence, DDES) else turbulence


_TIMESTAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")


def run_problems(path: Path, run: RunInfo, case: CaseSpec, mesh_info: MeshInfo) -> list[str]:
    """Invariants of run.json against its directory, case and MeshInfo."""
    found = []
    if path.name != run.run_id:
        found.append(f"directory {path.name!r} is not named after run_id {run.run_id!r}")
    if not _TIMESTAMP.fullmatch(run.started):
        found.append(f"started {run.started!r} is not a UTC ISO 8601 timestamp")
    elif run.run_id[4:20] != run.started.replace("-", "").replace(":", ""):
        found.append("run_id does not carry the started timestamp")
    if (run.finished is None) != (run.status == "running"):
        found.append("finished must be null exactly while running")
    if run.finished is not None and not _TIMESTAMP.fullmatch(run.finished):
        found.append(f"finished {run.finished!r} is not a UTC ISO 8601 timestamp")
    expected = identity(
        case.hash(),
        mesh_info.source.sha256,
        run.solver,
        run.precision,
        run.devices,
        run.cell_order,
        run.start,
    )
    if run.identity != expected:
        found.append("identity does not match its definition")
    if case.hash() != run.case.sha256 or case.name != run.case.name:
        found.append("case.toml does not match run.json case")
    if mesh_info.source.sha256 != run.mesh.sha256:
        found.append("mesh-info.json does not match run.json mesh.sha256")
    if not Path(run.mesh.path).is_absolute():
        found.append("mesh.path must be absolute")
    if run.precision != case.physics.precision:
        found.append("precision differs from the case's")
    steady = isinstance(case.time, Steady)
    if steady != (run.time is None):
        found.append("time must be null exactly for steady runs")
    if run.averaging is not None and steady:
        found.append("averaging is only for unsteady runs")
    if run.status == "diverged" and not any(d.code.startswith("D") for d in run.diagnostics):
        found.append("a diverged run must carry a D diagnostic")
    return found


def _read_history(path: Path, live: bool, broken: Any) -> list[dict[str, Any]]:
    """Parse a JSON Lines file of the run directory. Lines are framed on LF bytes; while the
    run is live the unterminated tail may still be being written and is skipped."""
    if not path.exists():
        return []
    data = path.read_bytes()
    complete, _, partial = data.rpartition(b"\n")
    if partial and not live:
        broken("the last line is not terminated", path.name)
    records = []
    for number, raw in enumerate(complete.split(b"\n") if complete else [], start=1):
        try:
            record = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        except ValueError as error:
            broken(f"line {number}: {error}", path.name)
        if not isinstance(record, dict):
            broken(f"line {number}: not an object", path.name)
        records.append(record)
    return records


def _reject_constant(name: str) -> object:
    message = f"{name} is not JSON"
    raise ValueError(message)


def log_problems(lines: list[dict[str, Any]], run: RunInfo) -> list[str]:
    """Every line a LogLine (log_line_problems); the first one before any iteration; every
    run.json diagnostic logged."""
    found: list[str] = []
    logged = set()
    for number, line in enumerate(lines, start=1):
        try:
            entry = LogLine.model_validate(line)
        except ValidationError as error:
            found.append(f"line {number}: {error.errors(include_url=False)[0]['msg']}")
            continue
        found.extend(f"line {number}: {problem}" for problem in log_line_problems(entry))
        logged.add((entry.code, json.dumps(entry.data, sort_keys=True)))
    if not lines:
        found.append("log.jsonl has no line; the run's start is logged first")
    elif lines[0].get("iteration") is not None:
        found.append("the first line logs the start of the run, before any iteration")
    found.extend(
        f"diagnostic {d.code} of run.json is not in log.jsonl"
        for d in run.diagnostics
        if (d.code, json.dumps(d.data, sort_keys=True)) not in logged
    )
    return found


def history_problems(records: list[dict[str, Any]], run: RunInfo, case: CaseSpec) -> list[str]:
    """Every line complete and finite, iterations increasing, the last one the run's."""
    found: list[str] = []
    equations = residual_names(case)
    monitors = {m.name: m for m in case.monitors}
    steady = isinstance(case.time, Steady)
    previous = 0
    for record in records:
        where = f"iteration {record.get('iteration')!r}"
        if set(record) != {"iteration", "time", "residuals", "monitors"}:
            return [f"{where}: keys must be iteration, time, residuals, monitors"]
        iteration = record["iteration"]
        if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration <= previous:
            return [f"{where}: iterations must be increasing integers"]
        previous = iteration
        if steady != (record["time"] is None) or not (steady or _finite(record["time"])):
            found.append(f"{where}: time must be null for steady runs, a number otherwise")
        residuals = record["residuals"]
        if not isinstance(residuals, dict) or list(residuals) != equations:
            found.append(f"{where}: residuals must be {equations}")
        elif not all(_finite(v) and v >= 0 for v in residuals.values()):
            found.append(f"{where}: residuals must be finite and non-negative")
        recorded = record["monitors"]
        due = {name for name, m in monitors.items() if iteration % m.every == 0}
        if not isinstance(recorded, dict) or set(recorded) != due:
            found.append(f"{where}: monitors must be exactly {sorted(due)}")
            continue
        for name, values in recorded.items():
            if not isinstance(values, dict) or set(values) != force_keys(monitors[name], case):
                found.append(f"{where}: monitor {name!r} has the wrong quantities")
            elif not all(_finite(v) for v in values.values()):
                found.append(f"{where}: monitor {name!r} has non-finite values")
    continued = run.start is not None and run.start.mode == "continue"
    base = run.start.iterations if run.start is not None and continued else 0
    if records and records[0].get("iteration") != base + 1 and continued:
        found.append(f"a continued run's history starts at iteration {base + 1}")
    if run.status != "running":
        last = records[-1]["iteration"] if records else base
        if last != run.iterations:
            found.append(f"history ends at iteration {last}, run.json says {run.iterations}")
    return found


FINAL = frozenset({"converged", "completed", "stopped"})
# run.json keys solution.cgns RunMetadata leaves out (results.md#solutioncgns): they change
# after the solution is written, or name its final state.
NOT_IN_METADATA = ("status", "finished", "iterations", "monitors", "diagnostics")


def run_metadata(info: RunInfo) -> dict[str, Any]:
    """The RunMetadata descriptor of solution.cgns for ``info``."""
    return {k: v for k, v in info.document().items() if k not in NOT_IN_METADATA}


def final_monitor_problems(
    records: list[dict[str, Any]], run: RunInfo, case: CaseSpec
) -> list[str]:
    """run.json ``monitors``: each monitor on the final state of a finished run, equal to the
    last history line when that line carries it (results.md#runjson)."""
    final = run.monitors
    if (final is None) != (run.status not in FINAL):
        return ["monitors must be given exactly for converged, completed and stopped runs"]
    if final is None:
        return []
    monitors = {m.name: m for m in case.monitors}
    if set(final) != set(monitors):
        return [f"monitors must be exactly {sorted(monitors)}"]
    found = []
    last = records[-1] if records and records[-1]["iteration"] == run.iterations else None
    for name, monitor in final.items():
        values = dict(monitor.model_extra or {})
        iteration = monitor.iteration
        if iteration != run.iterations:
            found.append(f"monitor {name!r}: iteration must be {run.iterations}")
        elif set(values) != force_keys(monitors[name], case) or not all(
            _finite(v) for v in values.values()
        ):
            found.append(f"monitor {name!r} has the wrong or non-finite quantities")
        elif last is not None and name in last["monitors"] and last["monitors"][name] != values:
            found.append(f"monitor {name!r} differs from the last history line")
    return found


def _finite(value: object) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an integer beyond the double range
        return False


UNITS = ("Kilogram", "Meter", "Second", "Kelvin", "Radian")
# BCType_t values of the CGNS SIDS (section 9.7), as written in files.
SIDS_BC_TYPES = frozenset(
    {
        "Null",
        "UserDefined",
        "BCAxisymmetricWedge",
        "BCDegenerateLine",
        "BCDegeneratePoint",
        "BCDirichlet",
        "BCExtrapolate",
        "BCFarfield",
        "BCGeneral",
        "BCInflow",
        "BCInflowSubsonic",
        "BCInflowSupersonic",
        "BCNeumann",
        "BCOutflow",
        "BCOutflowSubsonic",
        "BCOutflowSupersonic",
        "BCSymmetryPlane",
        "BCSymmetryPolar",
        "BCTunnelInflow",
        "BCTunnelOutflow",
        "BCWall",
        "BCWallInviscid",
        "BCWallViscous",
        "BCWallViscousHeatFlux",
        "BCWallViscousIsothermal",
        "FamilySpecified",
    }
)


def solution_problems(
    tree: cgns.Node, run: RunInfo, case: CaseSpec, mesh_info: MeshInfo
) -> list[str]:
    """The solution.cgns layout of results.md: labels, sizes, connectivity, fields, names."""
    try:
        return _solution_problems(tree, run, case, mesh_info)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        return [f"missing or malformed node: {type(error).__name__}: {error}"]


def _solution_problems(
    tree: cgns.Node, run: RunInfo, case: CaseSpec, mesh_info: MeshInfo
) -> list[str]:
    found: list[str] = []
    base = tree.child("Base")
    zone = base.child("Zone")
    for node, label in ((base, "CGNSBase_t"), (zone, "Zone_t")):
        if node.label != label:
            found.append(f"{node.name} is not a {label}")
    if base.child("DimensionalUnits").text_rows() != list(UNITS):
        found.append(f"DimensionalUnits must be {list(UNITS)}")
    if json.loads(base.child("RunMetadata").text()) != run_metadata(run):
        found.append("RunMetadata does not match run.json")
    names = json.loads(base.child("ZoneNames").text())
    if sorted(names["boundary"].values()) != sorted(z.name for z in mesh_info.boundary_zones):
        found.append("ZoneNames boundary zones differ from the MeshInfo")
    if sorted(names["cell"].values()) != sorted(z.name for z in mesh_info.cell_zones):
        found.append("ZoneNames cell zones differ from the MeshInfo")

    nodes, cells, _ = (int(v) for v in np.asarray(zone.data).ravel())
    if zone.child("ZoneType").text() != "Unstructured":
        found.append("ZoneType must be Unstructured")
    coordinates = zone.child("GridCoordinates")
    found.extend(
        f"Coordinate{axis} must have one value per node"
        for axis in "XYZ"
        if np.asarray(coordinates.child(f"Coordinate{axis}").data).size != nodes
    )
    ngon, nface = zone.child("NGonElements"), zone.child("NFaceElements")
    faces = int(ngon.child("ElementRange").data[1])
    found += _elements_problems(ngon, 22, 1, faces, 1, nodes, signed=False)
    found += _elements_problems(nface, 23, faces + 1, faces + cells, 1, faces, signed=True)

    solution = zone.child("FlowSolution")
    if solution.child("GridLocation").text() != "CellCenter":
        found.append("FlowSolution must be at CellCenter")
    for name in field_names(case):
        values = np.asarray(solution.children[name].data) if name in solution.children else None
        if values is None or values.size != cells or not np.all(np.isfinite(values)):
            found.append(f"FlowSolution {name} must hold {cells} finite values")
    if ("FlowSolutionMean" in zone.children) != (run.averaging is not None):
        found.append("FlowSolutionMean must exist exactly when run.json has averaging")
    periodic = {name for pair in mesh_info.periodic for name in pair.zones}
    zone_bc = zone.children.get("ZoneBC")
    bc_faces: set[int] = set()
    for bc in zone_bc.with_label("BC_t") if zone_bc is not None else []:
        if bc.text() not in SIDS_BC_TYPES:
            found.append(f"BC {bc.name} has BCType {bc.text()!r}, not a SIDS BCType_t value")
        if not _ids_in(bc.child("PointList").data, 1, faces):
            found.append(f"BC {bc.name} PointList must hold face ids")
        bc_faces |= set(np.asarray(bc.child("PointList").data).ravel().tolist())
    nodes_of = {zone_name: node for node, zone_name in names["boundary"].items()}
    expected_bcs = {
        nodes_of[z.name] for z in mesh_info.boundary_zones if z.faces > 0 and z.name not in periodic
    }
    present_bcs = {bc.name for bc in zone_bc.with_label("BC_t")} if zone_bc is not None else set()
    if present_bcs != expected_bcs:
        found.append(f"ZoneBC must hold a BC_t for exactly {sorted(expected_bcs)}")
    for name in periodic:
        family = base.children.get(nodes_of[name])
        if family is None or family.label != "Family_t" or "FamilyBC" in family.children:
            found.append(f"periodic zone {name!r} needs a Family_t without FamilyBC")
    used = np.abs(np.asarray(nface.child("ElementConnectivity").data)).ravel()
    counts = np.bincount(used, minlength=faces + 1)
    exterior = set(np.flatnonzero(counts == 1).tolist())
    found += _periodic_problems(zone, mesh_info, nodes_of, exterior, bc_faces)
    for region in zone.with_label("ZoneSubRegion_t"):
        low, high = (1, faces) if region.name == "WallSolution" else (faces + 1, faces + cells)
        if not _ids_in(region.child("PointList").data, low, high):
            found.append(f"ZoneSubRegion {region.name} PointList is out of range")
    return found


def _vector(node: cgns.Node) -> NDArray[Any] | None:
    """A 3-vector stored in four-byte floats (R4), in either byte order, as float32."""
    array = np.asarray(node.data)
    if node.label != "DataArray_t" or array.dtype.kind != "f" or array.dtype.itemsize != 4:
        return None
    return array.astype(np.float32).ravel() if array.size == 3 else None


def _exact(text: str) -> list[float] | None:
    """Three finite JSON numbers, booleans excluded."""
    value = json.loads(text)
    ok = isinstance(value, list) and len(value) == 3
    if not ok or not all(type(v) in (int, float) and math.isfinite(v) for v in value):
        return None
    return [float(v) for v in value]


def _periodic_problems(
    zone: cgns.Node,
    mesh_info: MeshInfo,
    nodes_of: dict[str, str],
    exterior: set[int],
    bc_faces: set[int],
) -> list[str]:
    """The two reciprocal GridConnectivity_t of every periodic pair (results.md#solutioncgns)."""
    connectivity = zone.children.get("ZoneGridConnectivity")
    expected = {nodes_of[name] for pair in mesh_info.periodic for name in pair.zones}
    present = set(connectivity.children) if connectivity is not None else set()
    if present != expected:
        return [f"ZoneGridConnectivity must hold exactly {sorted(expected)}"]
    if connectivity is None:
        return []
    found = []
    if connectivity.label != "ZoneGridConnectivity_t":
        found.append("ZoneGridConnectivity must be a ZoneGridConnectivity_t")
    claimed: set[int] = set()
    for pair in mesh_info.periodic:
        a, b = pair.zones
        vector = np.array(pair.transform.vector)
        matches = {}
        for side, shift in ((a, vector), (b, -vector)):
            gc = connectivity.child(nodes_of[side])
            what = f"GridConnectivity {gc.name}"
            if (gc.label, gc.text()) != ("GridConnectivity_t", "Zone"):
                found.append(f"{what} must be a GridConnectivity_t of Zone with itself")
            if gc.child("GridConnectivityType").text() != "Abutting1to1":
                found.append(f"{what} must be Abutting1to1")
            if gc.child("GridLocation").text() != "FaceCenter":
                found.append(f"{what} must be at FaceCenter")
            lists = [gc.child(name) for name in ("PointList", "PointListDonor")]
            arrays = [np.asarray(node.data).ravel() for node in lists]
            if any(
                n.label != "IndexArray_t" or x.dtype.kind != "i"
                for n, x in zip(lists, arrays, strict=True)
            ):
                found.append(f"{what} PointList and PointListDonor must be integer IndexArray_t")
                continue
            points, donors = (x.tolist() for x in arrays)
            if not points or len(points) != len(donors) or len(set(points)) != len(points):
                found.append(f"{what} must pair distinct faces one to one")
            if not set(points) | set(donors) <= exterior:
                found.append(f"{what} must pair boundary faces")
            if set(points) & set(donors):
                found.append(f"{what} pairs faces of its own zone")
            if side == a and points != sorted(points):
                found.append(f"{what} PointList must list the faces of {a!r} in mesh order")
            if (bc_faces | claimed) & set(points):
                found.append(f"{what} shares faces with ZoneBC or another periodic zone")
            claimed |= set(points)
            periodic = gc.child("GridConnectivityProperty").child("Periodic")
            if (gc.child("GridConnectivityProperty").label, periodic.label) != (
                "GridConnectivityProperty_t",
                "Periodic_t",
            ):
                found.append(f"{what} must hold GridConnectivityProperty_t/Periodic_t")
            rotation = [_vector(periodic.child(n)) for n in ("RotationCenter", "RotationAngle")]
            if any(r is None or r.any() for r in rotation):
                found.append(f"{what} RotationCenter and RotationAngle must be zero R4 vectors")
            exact = _exact(gc.child("PeriodicTranslation").text())
            single = _vector(periodic.child("Translation"))
            if exact != shift.tolist():
                found.append(f"{what} PeriodicTranslation must be {shift.tolist()}")
            elif single is None or single.tolist() != shift.astype(np.float32).tolist():
                found.append(f"{what} Translation must be its PeriodicTranslation in R4")
            matches[side] = set(zip(points, donors, strict=False))
        if a in matches and b in matches and matches[b] != {(d, p) for p, d in matches[a]}:
            found.append(f"the GridConnectivity of {a!r} and {b!r} are not reciprocal")
    return found


def _elements_problems(
    node: cgns.Node, kind: int, first: int, last: int, low: int, high: int, *, signed: bool
) -> list[str]:
    found = []
    if list(np.asarray(node.data)) != [kind, 0]:
        found.append(f"{node.name} must be element type {kind}")
    if list(np.asarray(node.child("ElementRange").data)) != [first, last]:
        found.append(f"{node.name} ElementRange must be [{first}, {last}]")
    offsets = np.asarray(node.child("ElementStartOffset").data)
    connectivity = np.asarray(node.child("ElementConnectivity").data)
    if (
        offsets.size != last - first + 2
        or offsets[0] != 0
        or np.any(np.diff(offsets) < (3 if not signed else 4))
        or offsets[-1] != connectivity.size
    ):
        found.append(f"{node.name} ElementStartOffset is inconsistent")
    ids = np.abs(connectivity) if signed else connectivity
    if not _ids_in(ids, low, high) or (signed and np.any(connectivity == 0)):
        found.append(f"{node.name} ElementConnectivity is out of range")
    return found


def _ids_in(ids: Any, low: int, high: int) -> bool:
    array = np.asarray(ids)
    return bool(array.size == 0 or (array.min() >= low and array.max() <= high))
