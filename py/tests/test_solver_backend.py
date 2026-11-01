"""The SDK through the real solver runtime: `autan prepare` output into `autan-run`, its
run directory back through `autan.Results`.

`autan-run` as shipped registers no physics; `autan-run-relaxation` (tests/runtime) is the
same runtime with the core's test relaxation physics, which runs every case to the end.
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import autan
import autan.initial_state
import pytest
from autan import cgns
from autan.cli import main

from tests.conftest import ROOT

MESH = ROOT / "tests" / "data" / "mesh" / "hybrid" / "polyMesh"


def _binary(variable: str, built: Path) -> Path | None:
    candidate = os.environ.get(variable) or (str(built) if built.is_file() else None)
    return Path(candidate) if candidate else None


RUNTIME = _binary("AUTAN_RUN", ROOT / "build" / "runtime" / "autan-run")
RELAXATION = _binary(
    "AUTAN_RUN_RELAXATION", ROOT / "build" / "tests" / "runtime" / "autan-run-relaxation"
)
MESH_INFO = _binary("AUTAN_MESH_INFO", ROOT / "build" / "io" / "autan-mesh-info")
pytestmark = pytest.mark.skipif(
    None in (RUNTIME, RELAXATION, MESH_INFO),
    reason="autan-run, autan-run-relaxation and autan-mesh-info not built "
    "(cmake --build build --target autan-run autan-run-relaxation autan-mesh-info)",
)
CASE = (
    "import autan\nfrom autan.units import K, Pa\n"
    f"case = autan.Case('hybrid', mesh={str(MESH)!r})\n"
    "case.physics = autan.Compressible(turbulence=autan.Laminar())\n"
    "case.freestream(mach=0.2, pressure=101325 * Pa, temperature=300 * K)\n"
    "for zone in ('xmin', 'xmax', 'ymax', 'zmin'):\n"
    "    case.boundary(zone).farfield()\n"
)


@pytest.fixture
def hybrid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[autan.Case, autan.MeshInfo]:
    for name in [n for n in os.environ if n.startswith("AUTAN_TEST_")]:
        monkeypatch.delenv(name)
    assert MESH_INFO is not None
    info = subprocess.run(
        [str(MESH_INFO), "--json", str(MESH)], capture_output=True, text=True, check=True
    ).stdout
    path = tmp_path / "hybrid.py"
    path.write_text(CASE)
    meshinfo = tmp_path / "hybrid.meshinfo.json"
    meshinfo.write_text(info)
    return autan.load(path), autan.MeshInfo.load(meshinfo)


def test_the_shipped_runtime_reports_missing_physics(
    hybrid: tuple[autan.Case, autan.MeshInfo], tmp_path: Path
) -> None:
    case, mesh = hybrid
    with pytest.raises(autan.CaseError) as error:
        case.run(mesh, backend=autan.SolverBackend(executable=RUNTIME), runs=tmp_path / "runs")
    [found] = error.value.diagnostics
    assert found.code == "E0706"
    assert found.fix
    assert found.doc.endswith("#e0706")
    assert not any((tmp_path / "runs").iterdir())


@pytest.mark.parametrize("linear_solver", ["multigrid", "gauss-seidel", "lines"])
def test_a_real_run_reads_back_through_the_results_contract(
    hybrid: tuple[autan.Case, autan.MeshInfo], tmp_path: Path, linear_solver: str
) -> None:
    case, mesh = hybrid
    case.time = autan.Steady(linear_solver=linear_solver)  # the runtime's parser accepts it
    backend = autan.SolverBackend(executable=RELAXATION, deterministic=True)
    result = case.run(mesh, backend=backend, runs=tmp_path / "runs")
    run = result.run
    assert run.status in ("converged", "completed")
    assert run.solver.backend == "cpu"
    assert run.case.sha256 == case.spec.hash()
    assert run.mesh.sha256 == mesh.source.sha256
    parts = [
        run.case.sha256,
        run.mesh.sha256,
        run.solver.commit,
        run.precision,
        run.solver.backend,
        run.devices.kind,
        str(run.devices.count),
    ]
    assert run.identity == hashlib.sha256("\n".join(parts).encode()).hexdigest()
    assert result.history.iterations[-1] == run.iterations
    assert result.field("Density").size == mesh.counts.cells


def test_divergence_and_interruption_come_back_as_the_contract_says(
    hybrid: tuple[autan.Case, autan.MeshInfo],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, mesh = hybrid
    monkeypatch.setenv("AUTAN_TEST_DIVERGE_AT", "3")
    result = case.run(
        mesh, backend=autan.SolverBackend(executable=RELAXATION), runs=tmp_path / "diverged"
    )
    assert result.run.status == "diverged"
    assert [d.code for d in result.run.diagnostics] == ["D0107"]


@pytest.mark.usefixtures("hybrid")
def test_cli_runs_the_runtime_and_stops_it_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_path = tmp_path / "hybrid.py"
    info = tmp_path / "hybrid.meshinfo.json"
    monkeypatch.setenv("AUTAN_RUN", str(RELAXATION))
    argv = [
        "run",
        str(case_path),
        "--mesh-info",
        str(info),
        "--backend",
        "cpu",
        "--runs",
        str(tmp_path / "runs"),
        "--json",
    ]
    assert main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run"]["solver"]["backend"] == "cpu"

    # factor 1: the residual never decays, so the run only ends when it is stopped.
    env = {
        **os.environ,
        "AUTAN_RUN": str(RELAXATION),
        "AUTAN_TEST_PAUSE_MS": "20",
        "AUTAN_TEST_FACTOR": "1.0",
    }
    command = [str(Path(sys.executable).with_name("autan")), *argv[:-1]]
    command[command.index(str(tmp_path / "runs"))] = str(tmp_path / "stopped")
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not list(
        (tmp_path / "stopped").glob("run-*/history.jsonl")
    ):
        time.sleep(0.02)
    try:
        assert list((tmp_path / "stopped").glob("run-*/history.jsonl"))
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) == 130
    finally:
        process.kill()
    [run_dir] = (tmp_path / "stopped").glob("run-*")
    assert autan.Results.open(run_dir).run.status == "stopped"


def test_missing_runtime_is_e0804(
    hybrid: tuple[autan.Case, autan.MeshInfo],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, mesh = hybrid
    monkeypatch.setenv("AUTAN_RUN", str(tmp_path / "nowhere"))
    with pytest.raises(autan.CaseError) as error:
        case.run(mesh, backend=autan.SolverBackend(), runs=tmp_path / "runs")
    assert error.value.codes == ["E0804"]


SCRIPT = """
import sys, autan
from pathlib import Path
case = autan.load(sys.argv[1])
mesh = autan.MeshInfo.load(sys.argv[2])
backend = autan.SolverBackend(executable=Path(sys.argv[3]))
try:
    case.run(mesh, backend=backend, runs=Path(sys.argv[4]))
except KeyboardInterrupt:
    sys.exit(130)
"""


@pytest.mark.parametrize("signals", [[signal.SIGTERM], [signal.SIGINT, signal.SIGINT]])
@pytest.mark.usefixtures("hybrid")
def test_the_sdk_never_orphans_the_runtime(signals: list[int], tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    env = {**os.environ, "AUTAN_TEST_PAUSE_MS": "20", "AUTAN_TEST_FACTOR": "1.0"}
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            SCRIPT,
            str(tmp_path / "hybrid.py"),
            str(tmp_path / "hybrid.meshinfo.json"),
            str(RELAXATION),
            str(runs),
        ],
        env=env,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not list(runs.glob("run-*/history.jsonl")):
            time.sleep(0.02)
        for number in signals:
            process.send_signal(number)
            time.sleep(0.05)
        # 130 from the script, or the default handler of a signal that arrives once the
        # run is already finalized.
        assert process.wait(timeout=30) in (130, -signals[-1])
    finally:
        process.kill()
    [run_dir] = runs.glob("run-*")
    assert autan.Results.open(run_dir).run.status == "stopped"
    survivors = subprocess.run(["pgrep", "-f", str(runs)], capture_output=True, check=False)
    assert survivors.stdout == b""  # the runtime was reaped, not orphaned


def _wrapper(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "wrapper.sh"
    script.write_text("#!/bin/sh\n" + body + "\n")
    script.chmod(0o755)
    return script


def test_a_run_not_written_by_this_invocation_is_rejected(
    hybrid: tuple[autan.Case, autan.MeshInfo], tmp_path: Path
) -> None:
    case, mesh = hybrid
    earlier = case.run(
        mesh, backend=autan.SolverBackend(executable=RELAXATION), runs=tmp_path / "runs"
    )
    replay = _wrapper(tmp_path, f"echo {earlier.path}")
    with pytest.raises(autan.CaseError) as error:
        case.run(mesh, backend=autan.SolverBackend(executable=replay), runs=tmp_path / "runs")
    assert error.value.codes == ["E0704"]


def test_exit_status_and_run_status_must_agree(
    hybrid: tuple[autan.Case, autan.MeshInfo], tmp_path: Path
) -> None:
    case, mesh = hybrid
    lying = _wrapper(tmp_path, f'"{RELAXATION}" "$@"; exit 1')
    with pytest.raises(autan.CaseError) as error:
        case.run(mesh, backend=autan.SolverBackend(executable=lying), runs=tmp_path / "runs")
    assert error.value.codes[0] == "E0704"


def test_a_relative_executable_is_the_one_in_the_working_directory(
    hybrid: tuple[autan.Case, autan.MeshInfo], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, mesh = hybrid
    assert RELAXATION is not None
    monkeypatch.chdir(RELAXATION.parent)
    monkeypatch.setenv("AUTAN_RUN", f"./{RELAXATION.name}")
    result = case.run(mesh, backend=autan.SolverBackend(), runs=tmp_path / "runs")
    assert result.run.status in ("converged", "completed")


def test_continue_through_the_runtime_matches_the_uninterrupted_run(
    hybrid: tuple[autan.Case, autan.MeshInfo], tmp_path: Path
) -> None:
    _, mesh = hybrid

    def case(budget: int) -> autan.Case:
        path = tmp_path / f"budget-{budget}.py"
        path.write_text(CASE + f"case.time = autan.Steady(max_iterations={budget})\n")
        return autan.load(path)

    backend = autan.SolverBackend(executable=RELAXATION, deterministic=True)
    whole = case(10000).run(mesh, backend=backend, runs=tmp_path / "whole")
    first = case(10).run(mesh, backend=backend, runs=tmp_path / "parts")
    assert (first.run.status, first.run.iterations) == ("completed", 10)
    rest = case(10000).run(mesh, backend=backend, runs=tmp_path / "parts", start=first)

    def lines(result: autan.Results) -> list[str]:
        return (result.path / "history.jsonl").read_text().splitlines()

    assert lines(first) + lines(rest) == lines(whole)
    assert (rest.run.status, rest.run.iterations) == (whole.run.status, whole.run.iterations)
    assert rest.field("Density").tobytes() == whole.field("Density").tobytes()


PERIODIC_BOX = ROOT / "tests" / "data" / "mesh" / "periodic-box" / "periodic-box.binary.msh"


def _box(tmp_path: Path, turbulence: str) -> tuple[autan.Case, autan.MeshInfo]:
    assert MESH_INFO is not None
    info = tmp_path / "box.meshinfo.json"
    info.write_text(
        subprocess.run(
            [str(MESH_INFO), "--json", str(PERIODIC_BOX)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    case = tmp_path / "box.py"
    case.write_text(
        "import autan\nfrom autan.units import K, Pa\n"
        f"case = autan.Case('box', mesh={str(PERIODIC_BOX)!r})\n"
        f"case.physics = autan.Compressible(turbulence=autan.{turbulence}())\n"
        "case.freestream(mach=0.2, pressure=101325 * Pa, temperature=300 * K)\n"
        "case.boundary('z*').slip()\n"
        "case.monitor(autan.forces('*'), every=7)\n"
    )
    return autan.load(case), autan.MeshInfo.load(info)


def test_a_periodic_run_reads_back_through_the_results_contract(tmp_path: Path) -> None:
    case, mesh = _box(tmp_path, "Laminar")
    result = case.run(
        mesh, backend=autan.SolverBackend(executable=RELAXATION), runs=tmp_path / "runs"
    )
    assert result.run.status == "converged"
    assert result.field("Density").size == mesh.counts.cells  # checks solution.cgns
    zone = cgns.read(result.path / "solution.cgns").child("Base").child("Zone")
    connectivity = zone.child("ZoneGridConnectivity")
    assert {"xmin", "xmax", "ymin", "ymax"} == set(connectivity.children)
    for pair in mesh.periodic:
        a, b = pair.zones
        vector = list(pair.transform.vector)
        side_a, side_b = connectivity.child(a), connectivity.child(b)
        assert json.loads(side_a.child("PeriodicTranslation").text()) == vector
        assert json.loads(side_b.child("PeriodicTranslation").text()) == [-v for v in vector]
        assert side_a.child("PointList").data.size == pair.faces
    # The monitor integrates over the slip walls only, every 7 iterations.
    iterations = result.history.monitor("forces")["iteration"].tolist()
    assert iterations == list(range(7, result.run.iterations + 1, 7))


def test_euler_refuses_a_periodic_mesh(tmp_path: Path) -> None:
    case, mesh = _box(tmp_path, "Inviscid")
    with pytest.raises(autan.CaseError) as error:
        case.run(mesh, backend=autan.SolverBackend(executable=RUNTIME), runs=tmp_path / "runs")
    [found] = error.value.diagnostics
    assert (found.code, found.data.get("path")) == ("E0706", "periodic")  # not missing physics
    assert not any((tmp_path / "runs").iterdir())


def test_the_runtime_records_the_final_monitors(
    hybrid: tuple[autan.Case, autan.MeshInfo], tmp_path: Path
) -> None:
    _, mesh = hybrid
    path = tmp_path / "monitored.py"
    path.write_text(CASE + "case.monitor(autan.forces('*'), every=10)\n")
    result = autan.load(path).run(
        mesh, backend=autan.SolverBackend(executable=RELAXATION), runs=tmp_path / "runs"
    )
    assert (result.run.status, result.run.iterations) == ("converged", 28)  # off cadence
    final = result.run.monitors
    assert final is not None
    assert final["forces"].iteration == 28
    assert result.history.monitor("forces")["iteration"].tolist() == [10, 20]


def test_an_fp32_euler_mms_converges_at_its_round_off_floor(
    hybrid: tuple[autan.Case, autan.MeshInfo], tmp_path: Path
) -> None:
    _, mesh = hybrid
    manufactured = ROOT / "py" / "tests" / "fixtures" / "euler.mms.json"
    digest = hashlib.sha256(manufactured.read_bytes()).hexdigest()
    path = tmp_path / "mms.py"
    path.write_text(
        "import autan\nfrom autan.units import K, Pa, kg, mol, s\n"
        "gas = autan.IdealGas(name='mms', molar_mass=8.314462618 / 287.0 * kg / mol, gamma=1.4,"
        " prandtl=0.72, viscosity=autan.ConstantViscosity(mu=1e-3 * Pa * s))\n"
        f"case = autan.Case('euler-mms', mesh={str(MESH)!r})\n"
        "case.physics = autan.Compressible(turbulence=autan.Inviscid(), gas=gas)\n"
        "case.freestream(mach=0.3, pressure=101325 * Pa, temperature=300 * K)\n"
        f"case.verification = autan.Manufactured(path={str(manufactured)!r}, sha256={digest!r})\n"
    )
    case = autan.load(path)
    assert isinstance(case.spec.time, autan.Steady)
    assert case.spec.time.residual_orders == 8.0  # above the fp32+acc64 plateau
    result = case.run(mesh, backend=autan.SolverBackend(executable=RUNTIME), runs=tmp_path / "runs")
    assert result.run.status == "converged"


HEX = ROOT / "tests" / "data" / "mesh" / "hex" / "polyMesh"


def _cell_centres(solution: Path) -> list[tuple[float, float, float]]:
    """The centre of each cell of a mesh of axis-aligned hexahedra, read from solution.cgns:
    the mean of its eight nodes, which is its centroid."""
    zone = cgns.read(solution).child("Base").child("Zone")
    xyz = [zone.child("GridCoordinates").child(f"Coordinate{a}").data for a in "XYZ"]
    ngon, nface = zone.child("NGonElements"), zone.child("NFaceElements")
    face_offsets = ngon.child("ElementStartOffset").data.tolist()
    face_nodes = ngon.child("ElementConnectivity").data.tolist()
    cell_offsets = nface.child("ElementStartOffset").data.tolist()
    cell_faces = nface.child("ElementConnectivity").data.tolist()
    centres = []
    for c in range(len(cell_offsets) - 1):
        nodes: set[int] = set()
        for face in cell_faces[cell_offsets[c] : cell_offsets[c + 1]]:
            f = abs(face) - 1
            nodes.update(n - 1 for n in face_nodes[face_offsets[f] : face_offsets[f + 1]])
        centres.append(tuple(sum(float(axis[n]) for n in nodes) / len(nodes) for axis in xyz))
    return centres  # type: ignore[return-value]


def test_a_file_written_by_the_sdk_starts_the_real_runtime(tmp_path: Path) -> None:
    assert MESH_INFO is not None
    info = tmp_path / "hex.meshinfo.json"
    info.write_text(
        subprocess.run(
            [str(MESH_INFO), "--json", str(HEX)], capture_output=True, text=True, check=True
        ).stdout
    )
    mesh = autan.MeshInfo.load(info)
    # 27 cells of side 1/3 in the unit cube: centroids at (i + 1/2) / 3, the state a function
    # of position so that a wrong pairing shows; rows in reverse order.
    centroids = [
        ((i + 0.5) / 3, (j + 0.5) / 3, (k + 0.5) / 3)
        for k in range(3)
        for j in range(3)
        for i in range(3)
    ][::-1]

    def state(x: float, y: float, z: float) -> list[float]:
        return [1.0 + x + 2 * y + 4 * z, 10 * x, 0.0, 0.0, 1e5 * (1 + z)]

    digest = autan.initial_state.write(
        tmp_path / "state.json",
        oracle="linear",
        mesh_sha256=mesh.source.sha256,
        centroids=centroids,
        values=[state(*c) for c in centroids],
    )
    case = tmp_path / "hex.py"
    case.write_text(
        "import autan\nfrom autan.units import K, Pa\n"
        f"case = autan.Case('hex', mesh={str(HEX)!r})\n"
        "case.physics = autan.Compressible(turbulence=autan.Laminar())\n"
        "case.freestream(mach=0.2, pressure=101325 * Pa, temperature=300 * K)\n"
        "case.boundary('*').slip()\n"
        f"case.initial_state = autan.StateFile(path='state.json', sha256={digest!r})\n"
    )
    result = autan.load(case).run(
        mesh,
        backend=autan.SolverBackend(executable=RELAXATION, deterministic=True),
        runs=tmp_path / "runs",
    )
    assert (result.path / "states" / f"{digest}.json").is_file()
    density, pressure = result.field("Density"), result.field("Pressure")
    for cell, centre in enumerate(_cell_centres(result.path / "solution.cgns")):
        expected = state(*centre)
        assert density[cell] == pytest.approx(expected[0], rel=1e-12)
        assert pressure[cell] == pytest.approx(expected[4], rel=1e-12)
