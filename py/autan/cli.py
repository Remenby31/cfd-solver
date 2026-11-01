"""The ``autan`` command line. Every command prints JSON with ``--json``.

Exit status: 0 success, 1 the command found errors (coded diagnostics), 2 usage error.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autan.case import Case, load
from autan.errors import (
    REGISTRY,
    CaseError,
    Diagnostic,
    diagnostic,
    explain,
    reference_page,
    registry_document,
)
from autan.fake import FakeBackend
from autan.meshinfo import MeshInfo
from autan.restart import resolve as resolve_start
from autan.results import Devices, Results, RunInfo
from autan.schema import json_schema as case_json_schema
from autan.schema import require_tags
from autan.solver_backend import SolverBackend
from autan.solver_input import build as build_solver_input
from autan.solver_input import json_schema as solver_input_json_schema
from autan.version import COMMIT, DIRTY, SCHEMAS, __version__

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

    from autan.plan import Plan


def _emit(document: dict[str, Any], as_json: bool, text: str) -> None:
    sys.stdout.write(json.dumps(document, indent=2) + "\n" if as_json else text + "\n")


def _fail(error: CaseError, as_json: bool) -> int:
    if as_json:
        _emit(
            {"ok": False, "diagnostics": [d.model_dump(mode="json") for d in error.diagnostics]},
            True,
            "",
        )
    else:
        sys.stderr.write(str(error) + "\n")
    return 1


def _millions(count: int) -> str:
    return f"{count / 1e6:.1f} M" if count >= 100_000 else f"{count:,}".replace(",", " ")


def _plan_text(result: Plan) -> str:
    mesh = result.mesh
    by_type = mesh["cells_by_type"]
    poly = 100 * by_type["polyhedron"] / mesh["cells"]
    covered = sum(1 for b in result.boundaries if b.condition is not None)
    needing = sum(1 for b in result.boundaries if b.faces > 0)
    lines = [
        f"case        {result.case['name']}  (sha256 {result.case['sha256'][:12]})",
        (
            f"mesh        {mesh['path']} ({mesh['unit']}) · {_millions(mesh['cells'])} cells"
            f" · {poly:.0f} % polyhedra"
            f" · max non-orth. {mesh['quality']['max_non_orthogonality_deg']:g}°"
        ),
        "extent      "
        + " x ".join(
            f"{high - low:.4g}"
            for low, high in zip(mesh["bounds"]["min"], mesh["bounds"]["max"], strict=True)
        )
        + " m",
        f"boundaries  {covered}/{needing} zones covered",
    ]
    width = max([len(b.zone) for b in result.boundaries] + [4])
    for b in result.boundaries:
        kind = b.condition["kind"] if b.condition else "—"
        origin = b.provenance or "missing"
        lines.append(f"  {b.zone:<{width}}  {_millions(b.faces):>9} faces  {kind:<14} {origin}")
    lines.append("cell zones")
    width = max(len(z.zone) for z in result.cell_zones)
    for z in result.cell_zones:
        medium = f"{z.medium['kind']}({z.medium['material']['name']})" if z.medium else "—"
        lines.append(
            f"  {z.zone:<{width}}  {_millions(z.cells):>9} cells  {medium:<24} "
            f"{z.provenance or 'missing'}"
        )
    memory = result.memory
    lines.append(
        f"memory      {memory.bytes_total / 1e9:.1f} GB total · {memory.bytes_per_gpu / 1e9:.1f} "
        f"GB/GPU on {memory.gpus} GPU · {memory.equations} equations ({memory.model})"
    )
    lines += [d.render() for d in result.diagnostics]
    lines.append("plan OK" if result.ok else "plan has errors")
    return "\n".join(lines)


def command_plan(args: argparse.Namespace) -> int:
    try:
        case = _load_quietly(args.case)
        if args.mesh_info is None:
            missing = diagnostic(
                "E0201",
                f"no MeshInfo for mesh {case.mesh!r}: the mesh readers are not built",
                location=case.locate("mesh"),
            )
            return _fail(CaseError([missing]), args.json)
        result = case.plan(MeshInfo.load(args.mesh_info), gpus=args.gpus)
    except CaseError as error:
        return _fail(error, args.json)
    _emit(result.document(), args.json, _plan_text(result))
    return 0 if result.ok else 1


def command_run(args: argparse.Namespace) -> int:
    try:
        case = _load_quietly(args.case)
        backend: FakeBackend | SolverBackend = (
            FakeBackend(iterations=args.iterations, pace=args.pace, diverge_at=args.diverge_at)
            if args.backend == "fake"
            else SolverBackend(
                devices=Devices(kind=args.backend, count=args.device_count),
                deterministic=args.deterministic,
            )
        )
        previous = signal.signal(signal.SIGTERM, _interrupt)
        try:
            results = case.run(
                MeshInfo.load(args.mesh_info),
                backend=backend,
                runs=args.runs,
                start=_start(args),
                mode="warm" if args.warm else "continue",
            )
        finally:
            signal.signal(signal.SIGTERM, previous)
    except CaseError as error:
        return _fail(error, args.json)
    except KeyboardInterrupt:
        sys.stderr.write("run stopped; its directory records status 'stopped'\n")
        return 130
    run = results.run
    text = f"{run.run_id}  {run.status}  {run.iterations} iterations  {results.path}"
    text += "".join(f"\n{d.render()}" for d in run.diagnostics)
    _emit(
        {
            "ok": run.status not in ("diverged", "failed"),
            "path": str(results.path),
            "run": run.document(),
        },
        args.json,
        text,
    )
    return 1 if run.status in ("diverged", "failed") else 0


def command_ui(args: argparse.Namespace) -> int:
    try:
        import autan_ui  # noqa: PLC0415 -- optional dependency, the `ui` extra
    except ImportError:
        missing = diagnostic("E0805", "the autan-ui package is not installed")
        return _fail(CaseError([missing]), as_json=False)
    autan_ui.serve(
        Path(args.runs),
        "127.0.0.1",
        args.port,
        open_browser=not args.no_open,
        meshes=Path(args.meshes) if args.meshes is not None else None,
    )
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    try:
        case = _load_quietly(args.case)
        mesh_info = MeshInfo.load(args.mesh_info)
        warnings = [d for d in case.plan(mesh_info).diagnostics if d.severity == "warning"]
        devices = Devices(kind=args.device_kind, count=args.device_count)
        start = resolve_start(
            case.spec, mesh_info, _start(args), "warm" if args.warm else "continue"
        )
        document = build_solver_input(
            case, mesh_info, devices=devices, deterministic=args.deterministic, start=start
        ).document()
        text = json.dumps(document, indent=2) + "\n"
        if args.output:
            try:
                Path(args.output).write_text(text)
            except OSError as error:
                unwritable = diagnostic("E0806", f"cannot write {args.output}: {error}")
                raise CaseError([unwritable]) from None
        else:
            sys.stdout.write(text)
    except CaseError as error:
        return _fail(error, args.json)
    if warnings:
        sys.stderr.write(
            json.dumps([d.model_dump(mode="json") for d in warnings]) + "\n"
            if args.json
            else "\n".join(d.render() for d in warnings) + "\n"
        )
    return 0


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        message = f"{text} is not a positive integer"
        raise argparse.ArgumentTypeError(message)
    return value


def _start(args: argparse.Namespace) -> Results | None:
    return Results.open(args.start) if args.start else None


def _interrupt(_signal: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _load_quietly(path: str) -> Case:
    """Load a case file with its own prints sent to stderr, so stdout stays our output."""
    with contextlib.redirect_stdout(sys.stderr):
        return load(path)


def _run(command: list[str]) -> str | None:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def command_doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, str]] = []
    found: list[Diagnostic] = []
    checks.append(
        {
            "name": "python",
            "status": "ok",
            "detail": f"{platform.python_version()} ({sys.executable})",
        }
    )

    gpus = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    header = _run(["nvidia-smi"]) or ""
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", header)
    if gpus and gpus.strip():
        rows = [row.strip() for row in gpus.strip().splitlines()]
        detail = f"{len(rows)} x {rows[0]}" + (f" · CUDA {cuda.group(1)}" if cuda else "")
        checks.append({"name": "gpu", "status": "ok", "detail": detail})
    else:
        found.append(diagnostic("W0801", "nvidia-smi is missing or lists no GPU"))
        checks.append({"name": "gpu", "status": "warning", "detail": "no NVIDIA GPU"})

    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    if launcher:
        first = (_run([launcher, "--version"]) or "").strip().splitlines()
        checks.append({"name": "mpi", "status": "ok", "detail": first[0] if first else launcher})
    else:
        found.append(diagnostic("W0802", "neither mpirun nor mpiexec is on the PATH"))
        checks.append({"name": "mpi", "status": "warning", "detail": "not found"})

    found.append(diagnostic("W0803", "this installation contains the SDK only"))
    checks.append({"name": "core", "status": "warning", "detail": "not built"})

    document = {
        "ok": True,
        "checks": checks,
        "diagnostics": [d.model_dump(mode="json") for d in found],
    }
    text = "\n".join(
        [f"{c['name']:<8} {c['status']:<8} {c['detail']}" for c in checks]
        + [d.render() for d in found]
    )
    _emit(document, args.json, text)
    return 0


def command_version(args: argparse.Namespace) -> int:
    document = {
        "autan": __version__,
        "commit": COMMIT,
        "dirty": DIRTY,
        "python": platform.python_version(),
        "schemas": SCHEMAS,
    }
    text = (
        f"autan {__version__} ({COMMIT[:12]}{', dirty' if DIRTY else ''})\n"
        f"python {document['python']}\n"
        + "\n".join(f"schema {name} {ident}" for name, ident in SCHEMAS.items())
    )
    _emit(document, args.json, text)
    return 0


def json_schema(name: str) -> dict[str, Any]:
    """The published JSON Schema of a document: ``case``, ``mesh-info``, ``results`` (run.json)
    or ``solver-input``."""
    if name == "case":
        return case_json_schema()
    if name == "solver-input":
        return solver_input_json_schema()
    models: dict[str, type[BaseModel]] = {"mesh-info": MeshInfo, "results": RunInfo}
    schema = models[name].model_json_schema(by_alias=True, mode="validation")
    require_tags(schema)
    if name == "results":
        # Optional in RunInfo only for writers' convenience; run.json always carries them.
        schema["required"] = [*schema["required"], "start", "convergence", "monitors", "timing"]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMAS[name],
        **schema,
    }


def command_schema(args: argparse.Namespace) -> int:
    sys.stdout.write(json.dumps(json_schema(args.document), indent=2) + "\n")
    return 0


def command_explain(args: argparse.Namespace) -> int:
    if args.all:
        if args.json:
            sys.stdout.write(json.dumps(registry_document(), indent=2) + "\n")
        else:
            sys.stdout.write(reference_page())
        return 0
    code = (args.code or "").upper()
    if code not in REGISTRY:
        sys.stderr.write(f"unknown code {args.code!r}; `autan explain --all` lists every code\n")
        return 2
    entry = REGISTRY[code]
    _emit(
        {
            "code": code,
            "severity": entry.severity,
            "title": entry.title,
            "explanation": entry.explanation,
            "fix": entry.fix,
        },
        args.json,
        explain(code),
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="autan", description="GPU-native CFD solver.")
    commands = root.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("plan", help="check a case against its mesh before running it")
    sub.add_argument("case", help="case file: .py, .toml or .json")
    sub.add_argument("--mesh-info", help="MeshInfo JSON of the case's mesh")
    sub.add_argument("--gpus", type=int, default=1, help="GPUs for the memory estimate")
    sub.add_argument("--json", action="store_true", help="print JSON")
    sub.set_defaults(func=command_plan)

    sub = commands.add_parser("run", help="run a case (only the fake backend is built)")
    sub.add_argument("case", help="case file: .py, .toml or .json")
    sub.add_argument("--mesh-info", required=True, help="MeshInfo JSON of the case's mesh")
    sub.add_argument(
        "--backend",
        choices=["fake", "cpu", "cuda", "hip"],
        default="fake",
        help="fake (synthetic results) or the device kind autan-run runs on (default: fake)",
    )
    sub.add_argument("--device-count", type=_positive, default=1, help="autan-run: devices")
    sub.add_argument("--deterministic", action="store_true", help="autan-run: bitwise mode")
    sub.add_argument("--start", help="run directory to start from (default: the freestream)")
    sub.add_argument("--warm", action="store_true", help="with --start: warm-start, count from 1")
    sub.add_argument("--runs", default="runs", help="directory of run directories")
    sub.add_argument("--iterations", type=int, default=200, help="fake: iterations to write")
    sub.add_argument("--pace", type=float, default=0.0, help="fake: seconds per iteration")
    sub.add_argument("--diverge-at", type=int, help="fake: diverge at this iteration")
    sub.add_argument("--json", action="store_true", help="print JSON")
    sub.set_defaults(func=command_run)

    sub = commands.add_parser(
        "prepare", help="write the solver input (autan.solver-input/1) of a planned case"
    )
    sub.add_argument("case", help="case file: .py, .toml or .json")
    sub.add_argument("--mesh-info", required=True, help="MeshInfo JSON of the case's mesh")
    sub.add_argument("--device-kind", choices=["cpu", "cuda", "hip"], default="cpu")
    sub.add_argument("--device-count", type=_positive, default=1)
    sub.add_argument("--deterministic", action="store_true", help="bitwise-deterministic run")
    sub.add_argument("--start", help="run directory to start from (default: the freestream)")
    sub.add_argument("--warm", action="store_true", help="with --start: warm-start, count from 1")
    sub.add_argument("-o", "--output", help="file to write (default: stdout)")
    sub.add_argument("--json", action="store_true", help="diagnostics as JSON on failure")
    sub.set_defaults(func=command_prepare)

    sub = commands.add_parser("ui", help="browse run directories in the web UI (extra `ui`)")
    sub.add_argument("--runs", default="runs", help="directory of run directories")
    sub.add_argument("--port", type=int, default=8765, help="local port (default 8765)")
    sub.add_argument("--no-open", action="store_true", help="do not open a browser")
    sub.add_argument("--meshes", help="directory of the *.meshinfo.json the case editor plans on")
    sub.set_defaults(func=command_ui)

    sub = commands.add_parser("doctor", help="check GPU, driver, MPI and solver core")
    sub.add_argument("--json", action="store_true", help="print JSON")
    sub.set_defaults(func=command_doctor)

    sub = commands.add_parser("version", help="print versions and schema identifiers")
    sub.add_argument("--json", action="store_true", help="print JSON")
    sub.set_defaults(func=command_version)

    sub = commands.add_parser("explain", help="explain a diagnostic code")
    sub.add_argument("code", nargs="?", help="a code such as E0412")
    sub.add_argument("--all", action="store_true", help="print the reference of every code")
    sub.add_argument("--json", action="store_true", help="print JSON")
    sub.set_defaults(func=command_explain)

    sub = commands.add_parser("schema", help="print the JSON Schema of a document (JSON only)")
    sub.add_argument("document", choices=sorted(SCHEMAS))
    sub.add_argument("--json", action="store_true", help="accepted for uniformity; always JSON")
    sub.set_defaults(func=command_schema)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if getattr(args, "gpus", 1) < 1:
        parser().error("--gpus must be >= 1")
    status: int = args.func(args)
    return status


if __name__ == "__main__":
    sys.exit(main())
