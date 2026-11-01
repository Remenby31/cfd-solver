"""Run directories written by the fake backend, read back through ``autan.Results``.

Expected values come from docs/contracts/results.md and case-schema.md (ideal-gas
law, Sutherland's law, the force projections), recomputed here independently.
"""

import json
import math
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import h5py
import jsonschema
import numpy as np
import pytest
from autan import cgns
from autan.cli import json_schema, main
from pydantic import ValidationError

from tests.conftest import CRM_HL, EXAMPLES, FIXTURES, identity_fields, identity_of

R_UNIVERSAL = 8.314462618
CGNSCHECK = shutil.which("cgnscheck")


@pytest.fixture
def crm_run(tmp_path: Path, crm_mesh: autan.MeshInfo) -> autan.Results:
    return autan.load(EXAMPLES / "crm_hl.py").run(
        crm_mesh, backend=autan.FakeBackend(), runs=tmp_path / "runs"
    )


def heat_run(runs: Path, **options: Any) -> autan.Results:
    mesh = autan.MeshInfo.load(FIXTURES / "heat-exchanger.meshinfo.json")
    return autan.load(EXAMPLES / "heat_exchanger.py").run(
        mesh, backend=autan.FakeBackend(**options), runs=runs
    )


def test_run_directory_and_metadata_follow_the_contract(crm_run: autan.Results) -> None:
    path = crm_run.path
    assert sorted(p.name for p in path.iterdir()) == [
        "case.toml",
        "history.jsonl",
        "log.jsonl",
        "mesh-info.json",
        "run.json",
        "solution.cgns",
    ]
    run = json.loads((path / "run.json").read_text())
    assert run["schema"] == "autan.results/1"
    assert re.fullmatch(r"run-\d{8}T\d{6}Z-[0-9a-f]{8}", run["run_id"])
    assert path.name == run["run_id"]
    assert run["status"] == "converged"
    assert run["solver"]["backend"] == "fake"
    assert re.fullmatch(r"[0-9a-f]{40}", run["solver"]["commit"])
    assert run["case"]["sha256"] == autan.load(path / "case.toml").spec.hash()
    assert run["mesh"]["path"] == str((EXAMPLES / "crm-hl.cgns").resolve())
    assert run["identity"] == identity_of(run)
    assert tuple(identity_fields()) == autan.results.IDENTITY
    assert autan.results.run_identity(run) == run["identity"]
    # A count the schema allows as 1.0 is the integer 1; a list is no count.
    as_float = run | {"devices": run["devices"] | {"count": 1.0}}
    assert autan.results.run_identity(as_float) == run["identity"]
    with pytest.raises(ValidationError):
        autan.results.run_identity(run | {"devices": run["devices"] | {"count": [1]}})
    assert run["iterations"] == 100
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", run["finished"])


def test_history_lines_carry_every_equation_and_monitors_at_their_period(
    crm_run: autan.Results,
) -> None:
    lines = [json.loads(line) for line in (crm_run.path / "history.jsonl").read_text().splitlines()]
    assert [line["iteration"] for line in lines] == list(range(1, 101))
    for line in lines:
        assert list(line["residuals"]) == [
            "density",
            "momentum-x",
            "momentum-y",
            "momentum-z",
            "energy",
            "nu-tilde",
        ]
        assert ("forces" in line["monitors"]) == (line["iteration"] % 10 == 0)
        assert line["time"] is None
    first, last = lines[0]["residuals"]["density"], lines[-1]["residuals"]["density"]
    assert isinstance(crm_run.case.time, autan.Steady)
    assert math.log10(first / last) >= crm_run.case.time.residual_orders - 1e-9


def test_force_coefficients_are_the_contract_projections(crm_run: autan.Results) -> None:
    forces = crm_run.forces
    assert list(forces.iteration) == list(range(10, 101, 10))
    alpha = math.radians(19.57)
    e_d = np.array([math.cos(alpha), 0.0, math.sin(alpha)])
    e_l = np.array([-math.sin(alpha), 0.0, math.cos(alpha)])
    rho = crm_run.field("Density")[0]
    speed = float(np.linalg.norm([crm_run.field(f"Velocity{a}")[0] for a in "XYZ"]))
    q_area = 0.5 * rho * speed**2 * 191.84
    force = np.stack([forces.fx, forces.fy, forces.fz])
    np.testing.assert_allclose(forces.cd, e_d @ force / q_area, rtol=1e-12)
    np.testing.assert_allclose(forces.cl, e_l @ force / q_area, rtol=1e-12)
    np.testing.assert_allclose(forces.cs, np.cross(e_l, e_d) @ force / q_area, atol=1e-12)
    np.testing.assert_allclose(forces.cmy, forces.my / (q_area * 7.0), rtol=1e-12)


def test_crm_hl_freestream_fields_follow_ideal_gas_and_sutherland(crm_run: autan.Results) -> None:
    temperature, mach, re_number, length = 288.15, 0.20, 5.49e6, 7.0
    r = R_UNIVERSAL / 0.0289647
    mu = 1.716e-5 * (temperature / 273.15) ** 1.5 * (273.15 + 110.4) / (temperature + 110.4)
    speed = mach * math.sqrt(1.4 * r * temperature)
    rho = re_number * mu / (speed * length)
    np.testing.assert_allclose(crm_run.field("Density"), rho, rtol=1e-12)
    np.testing.assert_allclose(crm_run.field("Pressure"), rho * r * temperature, rtol=1e-12)
    np.testing.assert_allclose(crm_run.field("Temperature"), temperature)
    alpha = math.radians(19.57)
    np.testing.assert_allclose(crm_run.field("VelocityX"), speed * math.cos(alpha), rtol=1e-12)
    np.testing.assert_allclose(crm_run.field("VelocityZ"), speed * math.sin(alpha), rtol=1e-12)
    np.testing.assert_allclose(crm_run.field("TurbulentSANuTilde"), 3.0 * mu / rho, rtol=1e-12)
    faces, pressure = crm_run.wall("Pressure")
    assert faces.size > 0
    np.testing.assert_allclose(pressure, rho * r * temperature, rtol=1e-12)


def test_heat_exchanger_zones_carry_their_own_media(tmp_path: Path) -> None:
    result = heat_run(tmp_path)
    pressure, temperature = 101325.0, 300.0
    for zone, molar_mass in (("air", 0.0289647), ("co2", 0.0440095)):
        cells = result.cell_zone(zone)
        assert cells.size > 0
        expected = pressure / (R_UNIVERSAL / molar_mass * temperature)
        np.testing.assert_allclose(result.field("Density")[cells], expected, rtol=1e-12)
    plate = result.cell_zone("plate")
    np.testing.assert_allclose(result.field("Density")[plate], 8933.0)
    np.testing.assert_allclose(result.field("VelocityX")[plate], 0.0)
    assert set(result.history.equations) >= {"k", "omega"}
    cells = np.concatenate([result.cell_zone(z) for z in ("air", "co2", "plate")])
    assert sorted(cells.tolist()) == list(range(result.field("Density").size))


@pytest.mark.skipif(CGNSCHECK is None, reason="cgnscheck (CGNS library tools) not installed")
@pytest.mark.parametrize("example", ["crm_hl", "heat_exchanger"])
def test_solution_passes_the_cgns_library_checker(example: str, tmp_path: Path) -> None:
    result = (
        heat_run(tmp_path, iterations=5)
        if example == "heat_exchanger"
        else autan.load(EXAMPLES / "crm_hl.py").run(
            autan.MeshInfo.load(FIXTURES / "crm-hl.meshinfo.json"),
            backend=autan.FakeBackend(iterations=5),
            runs=tmp_path,
        )
    )
    assert CGNSCHECK is not None
    done = subprocess.run(
        [CGNSCHECK, str(result.path / "solution.cgns")], capture_output=True, text=True, check=False
    )
    assert "ERROR" not in done.stdout + done.stderr
    assert done.returncode == 0


def test_cgns_tree_layout(crm_run: autan.Results) -> None:
    tree = cgns.read(crm_run.path / "solution.cgns")
    base = tree.child("Base")
    zone = base.child("Zone")
    _, cells, _ = np.asarray(zone.data).ravel()
    faces = int(zone.child("NGonElements").child("ElementRange").data[1])
    assert list(zone.child("NFaceElements").child("ElementRange").data) == [
        faces + 1,
        faces + cells,
    ]
    names = json.loads(base.child("ZoneNames").text())
    assert set(names["boundary"].values()) == {z.name for z in crm_run.mesh_info.boundary_zones}
    bc_types = {b.name: b.text() for b in zone.child("ZoneBC").with_label("BC_t")}
    assert bc_types["wing.main"] == "BCWallViscousHeatFlux"
    assert bc_types["farfield"] == "BCFarfield"
    assert bc_types["symmetry"] == "BCSymmetryPlane"
    metadata = json.loads(base.child("RunMetadata").text())
    assert metadata["identity"] == crm_run.run.identity


def test_cgns_node_names_avoid_collisions(
    tmp_path: Path, mesh_variant: Callable[..., autan.MeshInfo], write_case: Callable[[str], Path]
) -> None:
    long_name = "a-boundary-zone-name-longer-than-32"

    def edit(document: dict[str, Any]) -> None:
        document["boundary_zones"][0]["name"] = long_name
        document["boundary_zones"][1]["name"] = "fluid"
        document["boundary_zones"][2]["name"] = "ZoneBC"

    mesh = mesh_variant("crm-hl", edit)
    body = CRM_HL.replace('case.boundary("wing.*").wall()', 'case.boundary("*").wall()')
    body = body.replace('case.boundary("symmetry").symmetry()\n', "")
    case = autan.load(write_case(body.replace('forces("wing.*"', 'forces("fuselage"')))
    result = case.run(mesh, backend=autan.FakeBackend(iterations=2), runs=tmp_path)
    names = json.loads(
        cgns.read(result.path / "solution.cgns").child("Base").child("ZoneNames").text()
    )
    assert names["boundary"]["boundary-0000"] == long_name
    assert names["boundary"]["boundary-0001"] == "fluid"
    assert names["cell"]["cell-zone-0000"] == "fluid"
    assert names["boundary"]["ZoneBC"] == "ZoneBC"


def test_divergence_is_reported_with_its_cell(tmp_path: Path, crm_mesh: autan.MeshInfo) -> None:
    result = autan.load(EXAMPLES / "crm_hl.py").run(
        crm_mesh, backend=autan.FakeBackend(iterations=100, diverge_at=40), runs=tmp_path
    )
    assert result.run.status == "diverged"
    [found] = result.run.diagnostics
    assert found.code == "D0107"
    assert found.severity == "divergence"
    assert found.data["iteration"] == 40
    assert 0 <= found.data["cell"] < result.field("Density").size
    assert found.data["zone"] == "fluid"
    assert result.history.iterations[-1] == 39


def test_unsteady_run_records_time_and_average(
    tmp_path: Path, write_case: Callable[[str], Path], crm_mesh: autan.MeshInfo
) -> None:
    body = CRM_HL.replace("autan.SA()", "autan.DDES(base=autan.SA())") + (
        "from autan.units import s\ncase.time = autan.Unsteady(steps=40, dt=1e-4 * s, average_from=10)\n"
    )
    result = autan.load(write_case(body)).run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path)
    assert result.run.status == "completed"
    assert result.run.time == pytest.approx(40 * 1e-4)
    assert result.run.averaging is not None
    assert (result.run.averaging.from_step, result.run.averaging.to_step) == (11, 40)
    lines = [json.loads(line) for line in (result.path / "history.jsonl").read_text().splitlines()]
    assert lines[4]["time"] == pytest.approx(5e-4)
    np.testing.assert_array_equal(result.field("Density", mean=True), result.field("Density"))


def test_runs_of_one_identity_produce_identical_data(
    tmp_path: Path, crm_mesh: autan.MeshInfo
) -> None:
    case = autan.load(EXAMPLES / "crm_hl.py")
    first = case.run(crm_mesh, backend=autan.FakeBackend(iterations=20), runs=tmp_path)
    second = case.run(crm_mesh, backend=autan.FakeBackend(iterations=20), runs=tmp_path)
    assert first.run.run_id != second.run.run_id
    assert first.run.identity == second.run.identity
    assert (first.path / "history.jsonl").read_bytes() == (
        second.path / "history.jsonl"
    ).read_bytes()
    for name in ("Density", "VelocityX", "Pressure", "TurbulentSANuTilde"):
        assert first.field(name).tobytes() == second.field(name).tobytes()


def test_live_run_is_readable_at_every_moment(tmp_path: Path, crm_mesh: autan.MeshInfo) -> None:
    case = autan.load(EXAMPLES / "crm_hl.py")
    worker = threading.Thread(
        target=case.run,
        args=(crm_mesh,),
        kwargs={"backend": autan.FakeBackend(pace=0.005), "runs": tmp_path},
    )
    worker.start()
    live_lines: set[int] = set()
    while worker.is_alive():
        for path in tmp_path.glob("run-*"):
            assert {"run.json", "case.toml", "mesh-info.json"} <= {p.name for p in path.iterdir()}
            opened = autan.Results.open(path)
            if opened.run.status == "running":
                live_lines.add(len(opened.history.iterations))
    worker.join()
    assert len(live_lines - {0, 100}) > 1
    [path] = tmp_path.glob("run-*")
    assert autan.Results.open(path).run.status == "converged"


@pytest.mark.parametrize(
    ("damage", "code"),
    [
        (lambda p: (p / "run.json").write_text('{"schema": "autan.results/2"}'), "E0701"),
        (
            lambda p: (p / "case.toml").write_text(
                (p / "case.toml").read_text().replace("mach = 0.2", "mach = 0.3")
            ),
            "E0701",
        ),
        (lambda p: (p / "solution.cgns").unlink(), "E0701"),
        (
            lambda p: (p / "history.jsonl").write_text(
                "\n".join(reversed((p / "history.jsonl").read_text().splitlines()))
            ),
            "E0701",
        ),
    ],
)
def test_reader_rejects_non_conforming_directories(
    damage: Callable[[Path], object], code: str, crm_run: autan.Results
) -> None:
    damage(crm_run.path)
    with pytest.raises(autan.CaseError) as error:
        autan.Results.open(crm_run.path)
    assert error.value.codes == [code]


def test_diverged_run_without_solution_opens_for_its_history(
    tmp_path: Path, crm_mesh: autan.MeshInfo
) -> None:
    result = autan.load(EXAMPLES / "crm_hl.py").run(
        crm_mesh, backend=autan.FakeBackend(iterations=30, diverge_at=10), runs=tmp_path
    )
    (result.path / "solution.cgns").unlink()
    reopened = autan.Results.open(result.path)
    assert reopened.history.iterations[-1] == 9
    with pytest.raises(autan.CaseError) as error:
        reopened.field("Density")
    assert error.value.codes == ["E0703"]


def test_run_needs_an_explicit_backend_and_a_valid_plan(
    tmp_path: Path, crm_mesh: autan.MeshInfo, mesh_variant: Callable[..., autan.MeshInfo]
) -> None:
    case = autan.load(EXAMPLES / "crm_hl.py")
    with pytest.raises(autan.CaseError) as error:
        case.run(crm_mesh, runs=tmp_path)
    assert error.value.codes == ["E0804"]

    def untype(document: dict[str, Any]) -> None:
        document["boundary_zones"][4]["type"] = None

    with pytest.raises(autan.CaseError) as error:
        case.run(mesh_variant("crm-hl", untype), backend=autan.FakeBackend(), runs=tmp_path)
    assert error.value.codes == ["E0412"]
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_results_lookup_errors(crm_run: autan.Results) -> None:
    with pytest.raises(autan.CaseError) as error:
        crm_run.history.monitor("lift")
    assert error.value.codes == ["E0702"]
    with pytest.raises(autan.CaseError) as error:
        crm_run.field("Vorticity")
    assert error.value.codes == ["E0703"]


def test_cli_run_reports_divergence_with_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "run",
        str(EXAMPLES / "crm_hl.py"),
        "--mesh-info",
        str(FIXTURES / "crm-hl.meshinfo.json"),
        "--runs",
        str(tmp_path),
        "--json",
    ]
    assert main([*argv, "--iterations", "20", "--diverge-at", "5"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["run"]["status"] == "diverged"
    assert out["run"]["diagnostics"][0]["code"] == "D0107"
    assert autan.Results.open(out["path"]).run.run_id == out["run"]["run_id"]
    assert main([*argv, "--iterations", "20"]) == 0


def test_dimensional_units_follow_the_cgns_hdf5_mapping(crm_run: autan.Results) -> None:
    # CGNS/HDF5 mapping: a C1 array of Fortran dimensions [32, 5] is stored as HDF5 [5, 32].
    with h5py.File(crm_run.path / "solution.cgns", "r") as handle:
        raw = handle["Base/DimensionalUnits/ data"][()]
    assert raw.shape == (5, 32)
    names = [bytes(row.astype(np.uint8)).decode().rstrip() for row in raw]
    assert names == ["Kilogram", "Meter", "Second", "Kelvin", "Radian"]


def test_identity_alone_fixes_the_data(tmp_path: Path, crm_mesh: autan.MeshInfo) -> None:
    case = autan.load(EXAMPLES / "crm_hl.py")
    short = case.run(
        crm_mesh, backend=autan.FakeBackend(iterations=20, diverge_at=11), runs=tmp_path
    )
    long = case.run(
        crm_mesh, backend=autan.FakeBackend(iterations=40, diverge_at=11), runs=tmp_path
    )
    assert short.run.identity == long.run.identity
    assert (short.path / "history.jsonl").read_bytes() == (long.path / "history.jsonl").read_bytes()
    assert short.field("Density").tobytes() == long.field("Density").tobytes()


@pytest.mark.parametrize(
    ("options", "extra", "status"),
    [
        ({}, "", "converged"),
        ({"iterations": 30}, "", "stopped"),
        ({}, "case.time = autan.Steady(max_iterations=1)\n", "completed"),
        ({}, "case.time = autan.Steady(max_iterations=50)\n", "completed"),
    ],
)
def test_status_follows_the_convergence_criterion(
    options: dict[str, Any],
    extra: str,
    status: str,
    write_case: Callable[[str], Path],
    crm_mesh: autan.MeshInfo,
) -> None:
    result = autan.load(write_case(CRM_HL + extra)).run(
        crm_mesh, backend=autan.FakeBackend(**options), runs=write_case("").parent / "runs"
    )
    assert result.run.status == status
    density = result.history.residual("density")
    assert extra or options or math.log10(density[0] / density[-1]) >= 8.0 - 1e-9
    if status != "converged":
        assert density.size < 2 or math.log10(density[0] / density[-1]) < 8.0


def test_each_zone_uses_its_own_medium(tmp_path: Path) -> None:
    result = heat_run(tmp_path)
    temperature, pressure, speed = (
        300.0,
        101325.0,
        0.1 * math.sqrt(1.4 * R_UNIVERSAL / 0.0289647 * 300.0),
    )
    co2 = result.cell_zone("co2")
    rho = pressure / (R_UNIVERSAL / 0.0440095 * temperature)
    mu = 1.370e-5 * (temperature / 273.15) ** 1.5 * (273.15 + 222.0) / (temperature + 222.0)
    k = 1.5 * (0.001 * speed) ** 2
    np.testing.assert_allclose(result.field("TurbulentEnergyKinetic")[co2], k, rtol=1e-12)
    np.testing.assert_allclose(
        result.field("TurbulentDissipationRate")[co2], rho * k / (0.009 * mu), rtol=1e-12
    )
    plate = result.cell_zone("plate")
    np.testing.assert_allclose(result.field("TurbulentDissipationRate")[plate], 0.0)


def test_incompressible_zone_override_keeps_its_density(
    write_case: Callable[[str], Path], tmp_path: Path
) -> None:
    body = (
        "import autan\nfrom autan.units import Pa, kg, m, s\n"
        "water = autan.ConstantDensityFluid(name='water', density=998 * kg / m**3,"
        " viscosity=1e-3 * Pa * s)\n"
        "oil = autan.ConstantDensityFluid(name='oil', density=870 * kg / m**3,"
        " viscosity=0.03 * Pa * s)\n"
        "case = autan.Case('c', mesh='crm-hl.cgns')\n"
        "case.physics = autan.Incompressible(turbulence=autan.Laminar(), fluid=water)\n"
        "case.freestream(velocity=2 * m / s)\n"
        "case.boundary('*').slip()\n"
    )
    mesh = autan.MeshInfo.load(FIXTURES / "crm-hl.meshinfo.json")
    result = autan.load(write_case(body + "case.cell_zone('fluid').fluid(oil)\n")).run(
        mesh, backend=autan.FakeBackend(), runs=tmp_path / "runs"
    )
    np.testing.assert_allclose(result.field("Density"), 870.0)


def test_interrupted_run_is_stopped_with_its_solution(
    tmp_path: Path, example: Callable[[str], Path]
) -> None:
    command = [
        str(Path(sys.executable).with_name("autan")),
        "run",
        str(example("crm_hl.py")),
        "--mesh-info",
        str(FIXTURES / "crm-hl.meshinfo.json"),
        "--runs",
        str(tmp_path / "runs"),
        "--pace",
        "0.05",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not list((tmp_path / "runs").glob("run-*/history.jsonl")):
        time.sleep(0.02)
    time.sleep(0.2)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=20) == 130
    [path] = (tmp_path / "runs").glob("run-*")
    result = autan.Results.open(path)
    assert result.run.status == "stopped"
    assert result.run.iterations == result.history.iterations[-1] > 0
    assert result.field("Density").size > 0


def test_awkward_zone_names_get_safe_node_names(
    tmp_path: Path, mesh_variant: Callable[..., autan.MeshInfo], write_case: Callable[[str], Path]
) -> None:
    names = ["x" * 33, "boundary-0000", "Zone", "entrée", "ZoneNames"]

    def edit(document: dict[str, Any]) -> None:
        for zone, name in zip(document["boundary_zones"], names, strict=False):
            zone["name"] = name

    body = CRM_HL.replace('case.boundary("wing.*").wall()', 'case.boundary("*").wall()')
    body = body.replace('case.boundary("symmetry").symmetry()\n', "")
    case = autan.load(write_case(body.replace('forces("wing.*"', 'forces("symmetry"')))
    result = case.run(
        mesh_variant("crm-hl", edit), backend=autan.FakeBackend(iterations=2), runs=tmp_path
    )
    mapping = json.loads(
        cgns.read(result.path / "solution.cgns").child("Base").child("ZoneNames").text()
    )["boundary"]
    assert sorted(mapping.values()) == sorted([*names, "symmetry", "farfield"])
    assert all(n.isascii() and len(n) <= 32 for n in mapping)
    assert len(set(mapping)) == len(mapping)


def _rewrite(path: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    document = json.loads((path / "run.json").read_text())
    edit(document)
    (path / "run.json").write_text(json.dumps(document))


def _drop_last_line(path: Path) -> None:
    lines = (path / "history.jsonl").read_text().splitlines()
    (path / "history.jsonl").write_text("\n".join(lines[:-1]) + "\n")


@pytest.mark.parametrize(
    "damage",
    [
        lambda p: _rewrite(p, lambda d: d.update(identity="0" * 64)),
        lambda p: _rewrite(p, lambda d: d.update(started="yesterday")),
        lambda p: _rewrite(p, lambda d: d["mesh"].update(path="crm-hl.cgns")),
        lambda p: _rewrite(p, lambda d: d.update(precision="fp64")),
        lambda p: _rewrite(p, lambda d: d.update(status="diverged")),
        lambda p: _rewrite(p, lambda d: d.update(finished=None)),
        lambda p: (p / "history.jsonl").unlink(),
        _drop_last_line,
        lambda p: (p / "history.jsonl").write_text(
            (p / "history.jsonl").read_text().replace('"energy"', '"enthalpy"')
        ),
        lambda p: (p / "history.jsonl").write_text(
            (p / "history.jsonl").read_text().replace("1.0, ", "NaN, ", 1)
        ),
        lambda p: (p / "history.jsonl").write_text((p / "history.jsonl").read_text()[:-20]),
    ],
)
def test_reader_checks_every_contract_invariant(
    damage: Callable[[Path], object], crm_run: autan.Results
) -> None:
    damage(crm_run.path)
    with pytest.raises(autan.CaseError) as error:
        autan.Results.open(crm_run.path)
    assert error.value.codes == ["E0701"]


def _shrink_density(path: Path) -> None:
    with h5py.File(path, "r+") as handle:
        data = handle["Base/Zone/FlowSolution/Density/ data"]
        values = data[()]
        del handle["Base/Zone/FlowSolution/Density/ data"]
        handle["Base/Zone/FlowSolution/Density"].create_dataset(" data", data=values[:1])


@pytest.mark.parametrize(
    "damage",
    [
        lambda path: path.write_bytes(b"not a CGNS file"),
        lambda path: h5py.File(path, "w").close(),
        _shrink_density,
    ],
)
def test_solution_is_checked_on_first_use(
    damage: Callable[[Path], object], crm_run: autan.Results
) -> None:
    damage(crm_run.path / "solution.cgns")
    reopened = autan.Results.open(crm_run.path)
    assert reopened.history.iterations[-1] == reopened.run.iterations
    with pytest.raises(autan.CaseError) as error:
        reopened.field("Density")
    assert set(error.value.codes) == {"E0701"}


def test_unsteady_metadata_and_mean_describe_the_run(
    write_case: Callable[[str], Path], crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    body = CRM_HL.replace("autan.SA()", "autan.DDES(base=autan.SA())") + (
        "from autan.units import s\n"
        "case.time = autan.Unsteady(steps=4, dt=1e-3 * s, average_from=2)\n"
    )
    case = autan.load(write_case(body))
    full = case.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path / "a")
    metadata = json.loads(
        cgns.read(full.path / "solution.cgns").child("Base").child("RunMetadata").text()
    )
    assert metadata["time"] == pytest.approx(4e-3)
    assert metadata["averaging"] == full.run.document()["averaging"]
    early = case.run(crm_mesh, backend=autan.FakeBackend(iterations=1), runs=tmp_path / "b")
    assert early.run.status == "stopped"
    assert early.run.averaging is None
    with pytest.raises(autan.CaseError):
        early.field("Density", mean=True)


def test_log_records_the_run_and_its_diagnostics(
    crm_run: autan.Results, tmp_path: Path, crm_mesh: autan.MeshInfo
) -> None:
    lines = [json.loads(line) for line in (crm_run.path / "log.jsonl").read_text().splitlines()]
    assert all(
        re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", line["time"]) for line in lines
    )
    assert lines[0]["iteration"] is None
    assert lines[-1]["iteration"] == crm_run.run.iterations
    assert "W0205" in {line["code"] for line in lines}  # the plan's warning, logged at start
    assert {line["source"] for line in lines} == {"fake"}
    assert len(crm_run.log) == len(lines)

    diverged = autan.load(EXAMPLES / "crm_hl.py").run(
        crm_mesh, backend=autan.FakeBackend(iterations=30, diverge_at=10), runs=tmp_path
    )
    [found] = diverged.run.diagnostics
    [logged] = [line for line in diverged.log if line.code == "D0107"]
    assert (logged.level, logged.data, logged.iteration) == ("error", found.data, 10)


def _rewrite_log(path: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    lines = [json.loads(line) for line in (path / "log.jsonl").read_text().splitlines()]
    edit(lines[0])
    (path / "log.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))


@pytest.mark.parametrize(
    "damage",
    [
        lambda p: (p / "log.jsonl").unlink(),
        lambda p: (p / "log.jsonl").write_text(""),
        lambda p: _rewrite_log(p, lambda line: line.update(level="debug")),
        lambda p: _rewrite_log(p, lambda line: line.update(code="E9999")),
        lambda p: _rewrite_log(p, lambda line: line.update(time="2026-11-01T01:02:03Z")),
        lambda p: _rewrite_log(p, lambda line: line.pop("source")),
        lambda p: (p / "log.jsonl").write_text((p / "log.jsonl").read_text()[:-1]),
        lambda p: _rewrite_log(p, lambda line: line.update(time="2026-99-99T99:99:99.000Z")),
        lambda p: _rewrite_log(p, lambda line: line.update(code="D0107", level="info")),
        lambda p: _rewrite_log(p, lambda line: line.update(iteration=5)),
        lambda p: _rewrite_log(
            p, lambda line: line.update(code="W0205", level="warning", data={"iteration": 3})
        ),
        lambda p: (p / "log.jsonl").write_text(
            (p / "log.jsonl")
            .read_text()
            .replace('"data": {}', '"data": {"x": NaN}', 1)
            .replace('"data":{}', '"data":{"x":NaN}', 1)
        ),
    ],
)
def test_reader_checks_the_log(damage: Callable[[Path], object], crm_run: autan.Results) -> None:
    damage(crm_run.path)
    with pytest.raises(autan.CaseError) as error:
        autan.Results.open(crm_run.path)
    assert error.value.codes == ["E0701"]


def test_a_diagnostic_of_run_json_must_be_logged(tmp_path: Path, crm_mesh: autan.MeshInfo) -> None:
    diverged = autan.load(EXAMPLES / "crm_hl.py").run(
        crm_mesh, backend=autan.FakeBackend(iterations=30, diverge_at=10), runs=tmp_path
    )
    lines = (diverged.path / "log.jsonl").read_text().splitlines()
    kept = [line for line in lines if json.loads(line)["code"] != "D0107"]
    (diverged.path / "log.jsonl").write_text("\n".join(kept) + "\n")
    with pytest.raises(autan.CaseError) as error:
        autan.Results.open(diverged.path)
    assert error.value.codes == ["E0701"]


def test_log_lines_are_framed_on_bytes(crm_run: autan.Results) -> None:
    path = crm_run.path / "log.jsonl"
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["message"] = "line separator \u2028 inside a message"
    path.write_text(json.dumps(first, ensure_ascii=False) + "\n" + "\n".join(lines[1:]) + "\n")
    assert autan.Results.open(crm_run.path).log[0].message == first["message"]

    document = json.loads((crm_run.path / "run.json").read_text())
    document.update(status="running", finished=None, monitors=None)
    (crm_run.path / "run.json").write_text(json.dumps(document))
    with path.open("ab") as log:
        log.write('{"message": "half a character \u00e9'.encode()[:-1])
    assert len(autan.Results.open(crm_run.path).log) == len(lines)


def test_the_writer_refuses_lines_the_reader_would_reject(crm_run: autan.Results) -> None:
    from autan.results import RunWriter  # noqa: PLC0415

    writer = RunWriter.__new__(RunWriter)
    writer.path = crm_run.path
    with pytest.raises(ValueError, match="registered"):
        writer.log("error", "x", source="solver", code="E9999")
    with pytest.raises(ValueError, match="level"):
        writer.log("info", "x", source="solver", code="D0107")


@pytest.mark.parametrize(
    ("options", "every", "status", "iteration"),
    [
        ({}, 10, "converged", 100),  # on cadence: the last history line carries it
        ({}, 7, "converged", 100),  # off cadence
        ({"iterations": 0}, 10, "stopped", 0),  # the initial state
        ({"diverge_at": 10, "iterations": 30}, 10, "diverged", None),
    ],
)
def test_run_json_records_the_final_monitors(
    options: dict[str, Any],
    every: int,
    status: str,
    iteration: int | None,
    write_case: Callable[..., Path],
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
) -> None:
    body = CRM_HL.replace("every=10", f"every={every}")
    result = autan.load(write_case(body)).run(
        crm_mesh, backend=autan.FakeBackend(**options), runs=tmp_path
    )
    assert result.run.status == status
    final = result.run.monitors
    if iteration is None:
        assert final is None
        return
    assert final is not None
    assert final["forces"].iteration == iteration
    history = result.path / "history.jsonl"  # absent before the first iteration
    lines = history.read_text().splitlines() if history.exists() else []
    last = json.loads(lines[-1])["monitors"] if lines else {}
    if iteration % every == 0 and "forces" in last:
        assert final["forces"].model_extra == last["forces"]


@pytest.mark.parametrize(
    "damage",
    [
        lambda d: d.pop("monitors"),
        lambda d: d["monitors"]["forces"].update(iteration=99),
        lambda d: d["monitors"]["forces"].update(cl=d["monitors"]["forces"]["cl"] + 1),
        lambda d: d.update(monitors=None),
        lambda d: d["monitors"]["forces"].update(fx=10**400),  # beyond the double range
    ],
)
def test_the_reader_checks_the_final_monitors(
    damage: Callable[[dict[str, Any]], object], crm_run: autan.Results
) -> None:
    document = json.loads((crm_run.path / "run.json").read_text())
    damage(document)
    (crm_run.path / "run.json").write_text(json.dumps(document))
    with pytest.raises(autan.CaseError) as error:
        autan.Results.open(crm_run.path)
    assert error.value.codes == ["E0701"]


@pytest.mark.parametrize(
    "record", [{}, {"iteration": 100, "fx": "1.0"}, {"iteration": "100", "fx": 1.0}]
)
def test_the_published_schema_types_the_final_monitors(
    record: dict[str, Any], crm_run: autan.Results
) -> None:
    document = json.loads((crm_run.path / "run.json").read_text())
    validator = jsonschema.Draft202012Validator(json_schema("results"))
    validator.validate(document)
    document["monitors"] = {"forces": record}
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(document)


def test_a_final_monitor_that_overflows_fails_the_run(
    write_case: Callable[..., Path], crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    body = CRM_HL.replace(
        'autan.forces("wing.*", ref_area=191.84 * m**2)',
        'autan.forces("wing.*", ref_area=191.84 * m**2, moment_center=(1e308 * m, 1e308 * m, 1e308 * m))',
    )
    assert body != CRM_HL
    result = autan.load(write_case(body)).run(
        crm_mesh, backend=autan.FakeBackend(iterations=5), runs=tmp_path
    )
    assert (result.run.status, result.run.monitors) == ("failed", None)
    assert [d.code for d in result.run.diagnostics] == ["E0704"]


def test_run_metadata_is_run_json_without_its_final_keys(crm_run: autan.Results) -> None:
    path = crm_run.path / "solution.cgns"
    with h5py.File(path, "r+") as handle:
        node = handle["Base/RunMetadata"]
        metadata = json.loads(bytes(node[" data"][...]).decode())
        metadata["monitors"] = None
        del node[" data"]
        node.create_dataset(
            " data", data=np.frombuffer(json.dumps(metadata).encode(), dtype=np.int8)
        )
    with pytest.raises(autan.CaseError) as error:
        autan.Results.open(crm_run.path).field("Density")
    assert error.value.codes == ["E0701"]


def _set_convergence(path: Path, value: object) -> None:
    document = json.loads((path / "run.json").read_text())
    if value == "absent":
        del document["convergence"]
    else:
        document["convergence"] = value
    (path / "run.json").write_text(json.dumps(document))


def test_convergence_floors_belong_to_a_steady_fp32_run(
    crm_run: autan.Results, write_case: Callable[..., Path], crm_mesh: autan.MeshInfo
) -> None:
    names = autan.results.residual_names(crm_run.case)
    floors = {"floor": dict.fromkeys(names, 1e-6) | {names[0]: 0.0}}  # a zero scale: no floor
    _set_convergence(crm_run.path, floors)
    assert autan.Results.open(crm_run.path).run.convergence is not None
    run_json = json.loads((crm_run.path / "run.json").read_text())
    del run_json["convergence"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(json_schema("results")).validate(run_json)
    for value in ("absent", {"floor": dict.fromkeys(names[:-1], 1e-6)}):
        _set_convergence(crm_run.path, value)
        with pytest.raises(autan.CaseError) as error:
            autan.Results.open(crm_run.path)
        assert error.value.codes == ["E0701"]

    fp64 = autan.load(
        write_case(CRM_HL.replace("autan.SA())", 'autan.SA(), precision="fp64")'), name="fp64.py")
    )
    result = fp64.run(crm_mesh, backend=autan.FakeBackend(iterations=2), runs=crm_run.path.parent)
    assert result.run.convergence is None
    _set_convergence(result.path, floors)
    with pytest.raises(autan.CaseError) as error:
        autan.Results.open(result.path)
    assert error.value.codes == ["E0701"]
