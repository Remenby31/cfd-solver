"""``autan.SolverBackend``: run a case with the solver runtime ``autan-run``.

The SDK plans the case, writes its solver input (docs/contracts/solver-input.md) and
calls ``autan-run --input FILE --runs DIR``; the runtime writes the run directory of
docs/contracts/results.md, which ``autan.Results`` reads back.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from autan.errors import CaseError, Diagnostic, diagnostic
from autan.results import Devices, Results
from autan.solver_input import build

if TYPE_CHECKING:
    from autan.case import Case
    from autan.meshinfo import MeshInfo
    from autan.restart import Start

EXECUTABLE_VARIABLE = "AUTAN_RUN"


LAUNCHER_VARIABLE = "AUTAN_MPIEXEC"


def find_launcher() -> Path | None:
    """``$AUTAN_MPIEXEC`` if set, else ``mpiexec`` or ``mpirun`` on the PATH."""
    explicit = os.environ.get(LAUNCHER_VARIABLE)
    if explicit:
        # A path is taken as given, from the working directory; a bare name from the PATH.
        if os.sep in explicit:
            return Path(explicit).absolute()
        found = shutil.which(explicit)
        return Path(found) if found else Path(explicit).absolute()
    found = shutil.which("mpiexec") or shutil.which("mpirun")
    return Path(found) if found else None


def find_executable() -> Path | None:
    """``$AUTAN_RUN`` if set, else ``autan-run`` on the PATH."""
    explicit = os.environ.get(EXECUTABLE_VARIABLE)
    if explicit:
        return Path(explicit)
    found = shutil.which("autan-run")
    return Path(found) if found else None


@dataclass(frozen=True)
class SolverBackend:
    """Runs a case with ``autan-run`` on ``devices``; ``executable`` defaults to
    :func:`find_executable`. Input errors of the runtime raise ``CaseError`` with its
    coded diagnostics (``E0705``, ``E0706``, ``E0203`` …); a stopped run raises
    ``KeyboardInterrupt`` after the runtime has finalized it."""

    devices: Devices = field(default_factory=lambda: Devices(kind="cpu", count=1))
    executable: Path | None = None

    def run(
        self, case: Case, mesh_info: MeshInfo, runs: Path, start: Start | None = None
    ) -> Results:
        executable = (self.executable or find_executable() or Path()).absolute()
        if not executable.is_file():
            raise CaseError([_not_installed(f"{executable} is not a file")])
        document = build(case, mesh_info, devices=self.devices, start=start).document()
        runs = runs.absolute()
        runs.mkdir(parents=True, exist_ok=True)
        before = {p.name for p in runs.iterdir()}
        with tempfile.TemporaryDirectory(prefix="autan-input-") as folder:
            source = Path(folder) / "solver-input.json"
            source.write_text(json.dumps(document, indent=2) + "\n")
            stop = Path(folder) / "stop"
            command = [
                str(executable),
                "--input",
                str(source),
                "--runs",
                str(runs),
                "--stop-file",
                str(stop),
            ]
            if self.devices.count > 1:  # one rank per device (solver-input.md, devices)
                launcher = find_launcher()
                if launcher is None:
                    message = (
                        f"{self.devices.count} devices need an MPI launcher: set "
                        f"${LAUNCHER_VARIABLE}, or put mpiexec or mpirun on the PATH"
                    )
                    raise CaseError([diagnostic("E0808", message)])
                command = [str(launcher), "-n", str(self.devices.count), *command]
            try:
                # Its own session: the runtime, or the launcher and its local ranks, form one
                # process group that the escalation below can signal as a whole.
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
            except OSError as error:
                if self.devices.count > 1:
                    message = f"the MPI launcher {command[0]} cannot be started: {error}"
                    raise CaseError([diagnostic("E0808", message)]) from None
                raise CaseError([_not_installed(str(error))]) from None
            stdout, stderr, interrupted = _wait(process, stop)
        if interrupted or process.returncode == 130:
            raise KeyboardInterrupt
        if process.returncode == 2:
            raise CaseError(_diagnostics(stderr))
        lines = stdout.strip().splitlines()
        reported = Path(lines[-1]) if lines else None
        if process.returncode not in (0, 1) or reported is None:
            raise CaseError([_failure(process.returncode, stderr)])
        if reported.parent != runs or reported.name in before or not reported.is_dir():
            message = f"autan-run reported {reported}, not a new run directory under {runs}"
            raise CaseError([diagnostic("E0704", message)])
        results = Results.open(reported)
        problems = self._mismatches(results, document)
        expected = {"failed"} if process.returncode == 1 else FINISHED
        if results.run.status not in expected:
            problems.append(f"exit {process.returncode} with status {results.run.status!r}")
        if problems:
            message = "the run autan-run wrote does not match its input: " + "; ".join(problems)
            raise CaseError([diagnostic("E0704", message), *_diagnostics(stderr, fallback=False)])
        return results

    def _mismatches(self, results: Results, document: dict[str, object]) -> list[str]:
        run = results.run
        expected = {
            "case": document["case"]["sha256"],  # type: ignore[index]
            "mesh": document["mesh"]["sha256"],  # type: ignore[index]
            "precision": document["precision"],
            "devices": self.devices,
            "start": document["start"],
        }
        found = {
            "case": run.case.sha256,
            "mesh": run.mesh.sha256,
            "precision": run.precision,
            "devices": run.devices,
            "start": (
                {"run": document["start"]["run"], **run.start.model_dump(mode="json")}  # type: ignore[index]
                if run.start is not None
                else None
            ),
        }
        return [f"{key} differs" for key in expected if expected[key] != found[key]]


FINISHED = frozenset({"converged", "completed", "stopped", "diverged"})
# Seconds the runtime has to finalize a stopped run before it is terminated, then killed.
GRACE = 60.0
# Seconds left to read the pipes once the process group has been killed.
DRAIN = 5.0


def _not_installed(reason: str) -> Diagnostic:
    return diagnostic(
        "E0804",
        f"the solver runtime autan-run cannot be started ({reason}); set "
        f"${EXECUTABLE_VARIABLE} or put it on the PATH",
    )


def _failure(code: int, stderr: str) -> Diagnostic:
    return diagnostic("E0704", f"autan-run exited with {code}: {stderr.strip()}")


def _wait(process: subprocess.Popen[str], stop: Path) -> tuple[str, str, bool]:
    """Wait for the runtime. On SIGINT or SIGTERM of this process, create the stop file so
    that every rank finalizes the run as stopped (a launcher would turn a signal into an
    abort), ignore further interrupts, and terminate then kill its process group only if it
    does not finish within GRACE seconds. The runtime is always reaped before this
    returns."""
    deadline: float | None = None
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        while True:
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except KeyboardInterrupt:
                if deadline is None:
                    deadline = time.monotonic() + GRACE
                    stop.touch()
                continue
            except subprocess.TimeoutExpired:
                stdout, stderr = _terminate(process)
            return stdout, stderr, deadline is not None
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def _terminate(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Terminate, then kill, the runtime's whole process group; a descendant that left the
    group and still holds the pipes cannot keep this waiting more than DRAIN seconds."""
    for number, timeout in ((signal.SIGTERM, GRACE), (signal.SIGKILL, DRAIN)):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, number)
        try:
            return process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
    for pipe in (process.stdout, process.stderr):
        if pipe is not None:
            pipe.close()
    process.wait()
    return "", ""


def _raise_interrupt(_signal: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _diagnostics(stderr: str, *, fallback: bool = True) -> list[Diagnostic]:
    found = []
    for line in stderr.splitlines():
        try:
            found.append(Diagnostic.model_validate(json.loads(line)))
        except ValueError:
            continue
    if not found and fallback:
        found.append(diagnostic("E0705", f"autan-run rejected its input: {stderr.strip()}"))
    return found
