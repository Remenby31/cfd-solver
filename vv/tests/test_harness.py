"""The verification harness on run directories from real producers.

- `autan.FakeBackend` runs (the SDK's producer) against the freestream oracle computed independently
  from case-schema.md: the fake writes the freestream state, so errors are rounding.
- Runs written through `autan`'s writers (`autan_vv.runs`) on V&V meshes, with fields equal to an
  oracle times (1 + C h^p g(x)): the harness must recover p, pass the gate for p = 2 and fail it for
  p = 1, and refuse families or runs that do not fit.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

import autan
import numpy as np
import pytest
from autan.results import field_names
from autan.units import K, Pa, kg, m, mol, s
from autan_vv.cli import main
from autan_vv.errors import VVError
from autan_vv.gates import GATES, Gate
from autan_vv.harness import UNIVERSAL_GAS_CONSTANT, read_geometry, verify
from autan_vv.meshes.facemesh import FaceMesh
from autan_vv.meshes.families import cartesian, generate_family
from autan_vv.meshes.fluent import write_fluent
from autan_vv.oracles.registry import get_oracle
from autan_vv.runs import mesh_info, write_run

pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")
ROOT = Path(__file__).resolve().parents[2]
Meshes = list[tuple[FaceMesh, Path, str]]


def _gas(gas_constant: float, viscosity: float = 10.0, prandtl: float = 0.72) -> autan.IdealGas:
    return autan.IdealGas(
        name="manufactured",
        molar_mass=UNIVERSAL_GAS_CONSTANT / gas_constant * kg / mol,
        gamma=1.4,
        prandtl=prandtl,
        viscosity=autan.ConstantViscosity(mu=viscosity * Pa * s),
    )


def _case(
    path: Path,
    oracle_name: str,
    *,
    mach: float = 0.3,
    precision: str = "fp64",
    time: float | None = None,
    edit: Callable[[autan.Case], None] | None = None,
) -> autan.Case:
    oracle = get_oracle(oracle_name)
    case = autan.Case(f"verify-{oracle_name}", mesh=path)
    p = oracle.parameters
    gas = _gas(oracle.gas_constant, p.get("viscosity", 10.0), p.get("prandtl", 0.72))
    turbulence = {"navier-stokes": autan.Laminar(), "rans-sa": autan.SA()}.get(
        oracle.physics, autan.Inviscid()
    )
    case.physics = autan.Compressible(turbulence=turbulence, precision=precision, gas=gas)  # type: ignore[arg-type]
    case.freestream(mach=mach, pressure=1e5 * Pa, temperature=300 * K)
    if time is not None:
        case.time = autan.Unsteady(steps=1, dt=time * s)
    case.boundary("*").farfield()
    if edit is not None:
        edit(case)
    return case


def _h(mesh: FaceMesh, axes: tuple[int, ...]) -> float:
    span = np.prod([np.ptp(mesh.points[:, a]) for a in set(range(3)) - set(axes)])
    return float((mesh.geometry.cell_volume.sum() / span / mesh.n_cells) ** (1 / len(axes)))


def _runs(
    tmp_path: Path,
    meshes: Meshes,
    oracle_name: str,
    order: float,
    *,
    amplitude: float = 0.05,
    time: float | None = None,
    field_orders: dict[str, float] | None = None,
    **case_options: object,
) -> list[Path]:
    """Runs whose fields are the oracle times (1 + amplitude h^order g(x)): errors scale as h^order,
    or as h^field_orders[name] for the fields it names."""
    oracle = get_oracle(oracle_name)
    paths = []
    for mesh, path, fmt in meshes:
        case = _case(path, oracle_name, time=time, **case_options)  # type: ignore[arg-type]
        x = mesh.geometry.cell_centroid
        g = 1.0 + 0.5 * np.sin(0.2 * x[:, 0] + 0.1 * x[:, 1]) * np.cos(x[:, 2])
        exact = oracle.fields(x, time)
        h = _h(mesh, oracle.axes)
        orders = field_orders or {}
        fields = {
            k: exact[k] * (1.0 + amplitude * h ** orders.get(k, order) * g)
            for k in field_names(case.spec)
        }
        info = mesh_info(mesh, path, fmt)
        paths.append(write_run(tmp_path / "runs", case, mesh, info, fields, time=time).path)
    return paths


def _family(tmp_path: Path, kind: str, levels: tuple[int, ...]) -> Meshes:
    family, meshes = generate_family(kind, levels, tmp_path / kind)
    fmt = "fluent" if kind in ("poly", "agglomerated") else "gmsh"
    return [
        (mesh, tmp_path / kind / lv.file, fmt)
        for lv, mesh in zip(family.levels, meshes, strict=True)
    ]


def _planar(tmp_path: Path, levels: tuple[int, ...], half: float = 10.0) -> Meshes:
    """One-layer Cartesian meshes of [-half, half]^2 x [0, 1], written as Fluent files."""
    out = []
    for n in levels:
        mesh = cartesian((n, n, 1), (-half, -half, 0.0), (half, half, 1.0))
        path = tmp_path / f"planar-{n}.msh"
        write_fluent(mesh, path, "fluid")
        out.append((mesh, path, "fluent"))
    return out


@pytest.mark.parametrize("kind", ["hex", "poly", "hybrid"])
def test_geometry_rebuilt_from_solution_cgns_matches_the_mesh(tmp_path: Path, kind: str) -> None:
    meshes = _family(tmp_path, kind, (4,))
    (run,) = _runs(tmp_path, meshes, "mms-euler", 2.0)
    rebuilt = read_geometry(autan.Results.open(run))
    original = meshes[0][0].geometry
    np.testing.assert_allclose(rebuilt.geometry.cell_volume, original.cell_volume, rtol=1e-13)
    np.testing.assert_allclose(
        rebuilt.geometry.cell_centroid, original.cell_centroid, rtol=0, atol=1e-14
    )


@pytest.mark.parametrize(("order", "passes"), [(2.0, True), (1.0, False)])
def test_order_gate_recovers_the_order_of_the_error(
    tmp_path: Path, order: float, passes: bool
) -> None:
    runs = _runs(tmp_path, _family(tmp_path, "hex", (4, 6, 8, 12)), "mms-euler", order)
    report = verify(runs, "mms-euler", "euler-mms-order")
    for name in ("rho", "rhou", "rhoE", "Pressure"):
        fit = report["fits"][name]["L2"]
        assert fit["order"] == pytest.approx(order, abs=0.02)
        assert fit["pairwise_orders"][-1] == pytest.approx(order, abs=0.03)
    verdict = report["verdict"]
    assert verdict["pass"] is passes
    assert verdict["evidence"] is False  # these runs carry backend "fake"
    assert [r["n_cells"] for r in report["runs"]] == [64, 216, 512, 1728]
    assert all(
        len(r["mesh_sha256"]) == 64 and len(r["solver_commit"]) == 40 for r in report["runs"]
    )
    assert all(not autan.Results.open(p).run.deterministic for p in runs)


@pytest.mark.parametrize(
    ("oracle", "gate"),
    [
        ("mms-euler", "euler-mms-order"),
        ("mms-navier-stokes", "ns-mms-order"),
        ("mms-rans-sa", "rans-sa-mms-order"),
    ],
)
@pytest.mark.parametrize(("order", "investigate"), [(2.4, False), (2.8, True)])
def test_mms_order_gates_warn_above_their_upper_bound(
    tmp_path: Path, oracle: str, gate: str, order: float, investigate: bool
) -> None:
    runs = _runs(tmp_path, _family(tmp_path, "hex", (4, 6, 8, 12)), oracle, order)
    verdict = verify(runs, oracle, gate)["verdict"]
    assert verdict["pass"], verdict["checks"]
    assert "rho: order above 2.1" in verdict["warnings"]
    assert verdict["investigate"] is investigate


# Oracle, gate, and a run field together with a gated field it determines.
MMS_GATES = [
    ("mms-euler", "euler-mms-order", ("Density", "rho")),
    ("mms-navier-stokes", "ns-mms-order", ("Density", "rho")),
    ("mms-rans-sa", "rans-sa-mms-order", ("TurbulentSANuTilde", "TurbulentSANuTilde")),
]


@pytest.mark.parametrize(("oracle", "gate", "_field"), MMS_GATES)
def test_mms_order_gates_need_errors_decreasing_with_refinement(
    tmp_path: Path, oracle: str, gate: str, _field: tuple[str, str]
) -> None:
    # The second level's error, 2.5 times too large, exceeds the first's; the orders stay >= 1.9.
    meshes = _family(tmp_path, "hex", (4, 6, 8, 12, 16))
    runs = _runs(tmp_path / "a", meshes, oracle, 2.0)
    (louder,) = _runs(tmp_path / "b", meshes[1:2], oracle, 2.0, amplitude=0.125)
    verdict = verify([runs[0], louder, *runs[2:]], oracle, gate)["verdict"]
    rho = next(c for c in verdict["checks"] if c.get("field") == "rho")
    assert min(rho["fitted_order"], rho["finest_pair_order"]) >= 1.9
    assert rho["errors_decrease"] is False
    assert not rho["pass"]
    # A middle level under the rounding floor does not hide the next level's error rising again.
    (silent,) = _runs(tmp_path / "c", meshes[2:3], oracle, 2.0, amplitude=1e-14)
    verdict = verify([*runs[:2], silent, *runs[3:]], oracle, gate)["verdict"]
    rho = next(c for c in verdict["checks"] if c.get("field") == "rho")
    assert rho["errors_decrease"] is False
    assert not verdict["pass"]


@pytest.mark.parametrize(("oracle", "gate", "field"), MMS_GATES)
def test_mms_order_gates_check_every_listed_field(
    tmp_path: Path, oracle: str, gate: str, field: tuple[str, str]
) -> None:
    written, gated = field
    meshes = _family(tmp_path, "hex", (4, 6, 8, 12))
    runs = _runs(tmp_path, meshes, oracle, 2.0, field_orders={written: 1.5})
    verdict = verify(runs, oracle, gate)["verdict"]
    assert not verdict["pass"]
    assert not next(c for c in verdict["checks"] if c.get("field") == gated)["pass"]


@pytest.mark.parametrize("source", ["planar", "vortex-hex", "vortex-prism"])
def test_vortex_gate_checks_exactly_zero_components_and_its_prerequisites(
    tmp_path: Path, source: str
) -> None:
    # The gate's own mesh families (autan-vv meshes vortex-*) refine one domain, as it requires.
    if source == "planar":
        meshes = _planar(tmp_path, (16, 24, 32, 48))
    else:
        meshes = _family(tmp_path, source, (8, 16, 32, 64))
    runs = _runs(tmp_path, meshes, "isentropic-vortex", 2.0, amplitude=0.005, time=40.0)
    verdict = verify(runs, "isentropic-vortex", "vortex-order")["verdict"]
    rhow = next(c for c in verdict["checks"] if c.get("field") == "rhow")
    assert rhow["identically_zero"]
    assert verdict["pass"], verdict["checks"]
    early = _runs(tmp_path / "early", meshes, "isentropic-vortex", 2.0, amplitude=0.005, time=4.0)
    verdict = verify(early, "isentropic-vortex", "vortex-order")["verdict"]
    assert not verdict["pass"]
    assert not next(c for c in verdict["checks"] if c.get("prerequisite") == "time")["pass"]


def test_levels_under_the_rounding_floor_do_not_count(tmp_path: Path) -> None:
    meshes = _family(tmp_path, "hex", (4, 6, 8, 12))
    runs = _runs(tmp_path, meshes, "mms-euler", 2.0, amplitude=1e-6, precision="fp32+acc64")
    verdict = verify(runs, "mms-euler", "euler-mms-order")["verdict"]
    assert not verdict["pass"]
    assert all("rounding floor" in c["reason"] for c in verdict["checks"])


def test_freestream_preservation_on_fake_backend_runs(tmp_path: Path) -> None:
    ((mesh, path, _),) = _family(tmp_path, "hex", (3,))
    case = autan.Case("fake-freestream", mesh=path)
    case.physics = autan.Compressible(turbulence=autan.SA(), precision="fp64")
    case.freestream(
        mach=0.4, re=2.0e6, ref_length=1.0 * m, temperature=250 * K, aoa=3.0 * autan.units.deg
    )
    case.boundary("*").farfield()
    results = case.run(
        mesh_info(mesh, path, "gmsh"), backend=autan.FakeBackend(), runs=tmp_path / "runs"
    )
    verdict = verify([results.path], "freestream", "freestream-preservation")["verdict"]
    checked = {c["field"] for c in verdict["checks"]}
    assert {"Temperature", "TurbulentSANuTilde", "VelocityY"} <= checked
    assert verdict["pass"], verdict["checks"]
    assert not verdict["evidence"]
    assert any("fake" in reason for reason in verdict["not_evidence_because"])


def test_freestream_preservation_fails_on_a_wrong_temperature(tmp_path: Path) -> None:
    ((mesh, path, fmt),) = _family(tmp_path, "hex", (3,))
    case = _case(path, "mms-euler")
    speed = 0.3 * np.sqrt(1.4 * 287.0 * 300.0)
    values = {
        "Density": 1e5 / (287.0 * 300.0),
        "VelocityX": speed,
        "VelocityY": 0.0,
        "VelocityZ": 0.0,
        "Pressure": 1e5,
        "Temperature": 600.0,
    }
    fields = {k: np.full(mesh.n_cells, v) for k, v in values.items()}
    run = write_run(tmp_path / "runs", case, mesh, mesh_info(mesh, path, fmt), fields).path
    verdict = verify([run], "freestream", "freestream-preservation")["verdict"]
    assert not next(c for c in verdict["checks"] if c["field"] == "Temperature")["pass"]
    assert all(c["pass"] for c in verdict["checks"] if c["field"] != "Temperature")


def test_runs_that_do_not_fit_the_oracle_or_each_other_are_rejected(tmp_path: Path) -> None:
    meshes = _family(tmp_path, "hex", (4, 6, 8, 12))
    runs = _runs(tmp_path, meshes, "mms-euler", 2.0)
    other = _runs(tmp_path / "other", meshes[:1], "mms-euler", 2.0, mach=0.5)
    calls: list[tuple[Callable[[], object], str]] = [
        (lambda: verify(runs[:1], "ringleb"), "V0724"),
        (lambda: verify([runs[0], runs[0]], "mms-euler"), "V0727"),
        (lambda: verify([*runs[1:], other[0]], "mms-euler"), "V0727"),
        (lambda: verify(runs[:3], "mms-euler", "euler-mms-order"), "V0729"),
        (lambda: verify(runs, "mms-euler", "ringleb-order"), "V0728"),
        (lambda: verify(runs, "mms-euler", "no-such-gate"), "V0728"),
    ]
    for call, code in calls:
        with pytest.raises(VVError) as err:
            call()
        assert err.value.code == code


def test_viscous_oracles_check_the_material_law(tmp_path: Path) -> None:
    mesh = cartesian((1, 4, 1))
    path = tmp_path / "couette.msh"
    write_fluent(mesh, path, "fluid")

    def sutherland(case: autan.Case) -> None:
        physics = case.spec.physics
        law = autan.Sutherland(mu_ref=1.8e-5 * Pa * s, t_ref=300 * K, s=110.4 * K)
        gas = physics.gas.model_copy(update={"viscosity": law})  # type: ignore[union-attr]
        case.physics = autan.Compressible(turbulence=autan.Laminar(), precision="fp64", gas=gas)

    (run,) = _runs(tmp_path, [(mesh, path, "fluent")], "couette", 2.0, edit=sutherland)
    with pytest.raises(VVError) as err:
        verify([run], "couette", "couette-velocity-exact")
    assert err.value.code == "V0724"


def test_two_dimensional_oracles_need_one_layer_meshes(tmp_path: Path) -> None:
    thin = cartesian((4, 4, 4), (-10.0, -10.0, 0.0), (10.0, 10.0, 1e-10))
    path = tmp_path / "thin.msh"
    write_fluent(thin, path, "fluid")
    (run,) = _runs(tmp_path, [(thin, path, "fluent")], "isentropic-vortex", 2.0, time=40.0)
    with pytest.raises(VVError) as err:
        verify([run], "isentropic-vortex")
    assert err.value.code == "V0723"


def test_fixture_writer_refuses_outputs_it_cannot_write(tmp_path: Path) -> None:
    ((mesh, path, fmt),) = _family(tmp_path, "hex", (3,))
    case = _case(path, "mms-euler")
    case.monitor(autan.forces("*", ref_area=1.0 * m**2), every=1)
    with pytest.raises(VVError) as err:
        write_run(tmp_path / "runs", case, mesh, mesh_info(mesh, path, fmt), {})
    assert err.value.code == "V0731"
    assert not (tmp_path / "runs").exists()


def test_cli_verify_writes_a_strict_report_and_maps_errors_to_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = [
        str(p) for p in _runs(tmp_path, _family(tmp_path, "hex", (4, 6, 8, 12)), "mms-euler", 1.0)
    ]
    out = tmp_path / "verdict.json"
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "verify",
                *runs,
                "--against",
                "mms-euler",
                "--gate",
                "euler-mms-order",
                "--out",
                str(out),
            ]
        )
    assert exit_info.value.code == 1
    assert "gate euler-mms-order: FAIL" in capsys.readouterr().out

    def reject(token: str) -> None:
        raise AssertionError(token)

    assert json.loads(out.read_text(), parse_constant=reject)["schema"] == "autan.vv-verdict/1"
    assert main(["verify", str(tmp_path / "missing"), "--against", "mms-euler"]) == 2


def test_every_gate_is_stated_in_acceptance_md() -> None:
    text = (ROOT / "docs" / "validation" / "acceptance.md").read_text()
    for gate in GATES.values():
        row = next((line for line in text.splitlines() if f"`{gate.id}`" in line), None)
        assert row is not None, gate.id
        numbers = {round(float(v), 9) for v in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", row)}
        if gate.measure == "order":
            centred = {
                round((gate.lower + gate.upper) / 2, 9),
                round((gate.upper - gate.lower) / 2, 9),
            }
            assert {gate.lower, gate.upper} <= numbers or centred <= numbers, gate.id
        else:
            assert gate.upper in numbers, gate.id


def test_a_non_finite_error_fails_the_gate_instead_of_being_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A gate on entropy, the field a negative density makes non-finite.
    gate = Gate("entropy-order", ("mms-euler",), "order", "L2", ("Entropy",), 1.9, 2.1, "M2")
    monkeypatch.setitem(GATES, gate.id, gate)
    meshes = _family(tmp_path, "hex", (4, 6, 8, 12, 16))
    runs = _runs(tmp_path, meshes[1:], "mms-euler", 2.0)
    mesh, path, fmt = meshes[0]
    case = _case(path, "mms-euler")
    exact = get_oracle("mms-euler").fields(mesh.geometry.cell_centroid)
    fields = {k: exact[k].copy() for k in field_names(case.spec)}
    fields["Density"][0] = -1.0  # negative density: NaN entropy on the coarsest level
    bad = write_run(tmp_path / "runs", case, mesh, mesh_info(mesh, path, fmt), fields).path
    report = verify([bad, *runs], "mms-euler", "euler-mms-order")
    assert report["runs"][0]["errors"]["Entropy"]["L2"] is None
    assert report["fits"]["Entropy"]["L2"]["non_finite"]
    with pytest.raises(SystemExit):
        main(
            [
                "verify",
                str(bad),
                *map(str, runs),
                "--against",
                "mms-euler",
                "--gate",
                "entropy-order",
            ]
        )


def test_families_must_refine_one_domain(tmp_path: Path) -> None:
    meshes = []
    for n, side in ((4, 0.04), (6, 0.03), (8, 0.02), (12, 0.01)):
        mesh = cartesian((n, n, n), upper=(side, side, side))
        path = tmp_path / f"box-{n}.msh"
        write_fluent(mesh, path, "fluid")
        meshes.append((mesh, path, "fluent"))
    with pytest.raises(VVError) as err:
        verify(_runs(tmp_path, meshes, "mms-euler", 2.0), "mms-euler")
    assert err.value.code == "V0727"


def test_cell_zone_materials_are_checked_too(tmp_path: Path) -> None:
    mesh = cartesian((1, 4, 1))
    path = tmp_path / "couette.msh"
    write_fluent(mesh, path, "fluid")

    def override(case: autan.Case) -> None:
        gas = case.spec.physics.gas  # type: ignore[union-attr]
        case.cell_zone("fluid").fluid(gas.model_copy(update={"prandtl": 0.9}))

    (run,) = _runs(tmp_path, [(mesh, path, "fluent")], "couette", 2.0, edit=override)
    with pytest.raises(VVError) as err:
        verify([run], "couette")
    assert err.value.code == "V0724"


def test_couette_velocity_gate_checks_every_component(tmp_path: Path) -> None:
    mesh = cartesian((1, 4, 1))
    path = tmp_path / "couette.msh"
    write_fluent(mesh, path, "fluid")
    case = _case(path, "couette")
    exact = get_oracle("couette").fields(mesh.geometry.cell_centroid)
    fields = {k: exact[k].copy() for k in field_names(case.spec)}
    fields["VelocityY"] += 10.0
    run = write_run(tmp_path / "runs", case, mesh, mesh_info(mesh, path, "fluent"), fields).path
    verdict = verify([run], "couette", "couette-velocity-exact")["verdict"]
    assert not next(c for c in verdict["checks"] if c["field"] == "VelocityY")["pass"]
    assert next(c for c in verdict["checks"] if c["field"] == "VelocityX")["pass"]


def test_malformed_fields_and_degenerate_boxes_are_coded_errors(tmp_path: Path) -> None:
    ((mesh, path, fmt),) = _family(tmp_path, "hex", (2,))
    case = _case(path, "mms-euler")
    fields = {k: np.ones(mesh.n_cells) for k in field_names(case.spec)}
    fields["Pressure"] = np.ones((2, mesh.n_cells // 2))
    with pytest.raises(VVError) as err:
        write_run(tmp_path / "runs", case, mesh, mesh_info(mesh, path, fmt), fields)
    assert err.value.code == "V0731"
    with pytest.raises(VVError) as err:
        cartesian((1, 1, 1), upper=(1e-13, 1e-13, 0.0))
    assert err.value.code == "V0302"


def test_cli_reports_fields_without_a_fit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    meshes = _planar(tmp_path, (16, 24, 32, 48))
    runs = _runs(tmp_path, meshes, "isentropic-vortex", 2.0, amplitude=0.005, time=40.0)
    assert (
        main(
            ["verify", *map(str, runs), "--against", "isentropic-vortex", "--gate", "vortex-order"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "VelocityZ: L2 order not measured" in out
    assert "gate vortex-order: PASS" in out


def test_freestream_refuses_cell_zone_gas_overrides(tmp_path: Path) -> None:
    ((mesh, path, _),) = _family(tmp_path, "hex", (3,))
    case = autan.Case("fake-freestream", mesh=path)
    case.physics = autan.Compressible(turbulence=autan.SA(), precision="fp64")
    case.freestream(mach=0.4, pressure=1e5 * Pa, temperature=250 * K)
    case.boundary("*").farfield()
    heavy = case.spec.physics.gas.model_copy(update={"molar_mass": 0.058 * kg / mol})  # type: ignore[union-attr]
    case.cell_zone("fluid").fluid(heavy)
    results = case.run(
        mesh_info(mesh, path, "gmsh"), backend=autan.FakeBackend(), runs=tmp_path / "runs"
    )
    with pytest.raises(VVError) as err:
        verify([results.path], "freestream", "freestream-preservation")
    assert err.value.code == "V0724"


def test_ringleb_family_with_its_face_states_meets_the_gate(tmp_path: Path) -> None:
    # Each level imposes its own face-state file (another path and hash per level) on a curved
    # domain whose polygonal bounds move with refinement: still one problem and one domain.
    import hashlib  # noqa: PLC0415

    family, meshes = generate_family("ringleb", (6, 8, 12, 16), tmp_path / "ringleb")
    oracle = get_oracle("ringleb")
    paths = []
    for lv, mesh in zip(family.levels, meshes, strict=True):
        path = tmp_path / "ringleb" / lv.file
        states = path.with_suffix(".states.json")
        sha = hashlib.sha256(states.read_bytes()).hexdigest()

        case = autan.Case("verify-ringleb", mesh=path)
        case.physics = autan.Compressible(
            turbulence=autan.Inviscid(), gas=_gas(oracle.gas_constant)
        )
        case.freestream(mach=0.3, pressure=1e5 * Pa, temperature=300 * K)
        for zone in ("k-min", "k-max", "inflow", "outflow"):
            case.boundary(zone).oracle_state(path=str(states), sha256=sha)
        for zone in ("back", "front"):
            case.boundary(zone).symmetry()
        x = mesh.geometry.cell_centroid
        exact = oracle.fields(x, None)
        h = _h(mesh, oracle.axes)
        g = 1.0 + 0.5 * np.sin(0.2 * x[:, 0] + 0.1 * x[:, 1])
        fields = {k: exact[k] * (1.0 + 0.05 * h**2.0 * g) for k in field_names(case.spec)}
        paths.append(
            write_run(tmp_path / "runs", case, mesh, mesh_info(mesh, path, "gmsh"), fields).path
        )
    verdict = verify(paths, "ringleb", "ringleb-order")["verdict"]
    assert verdict["pass"], verdict["checks"]


def test_fp32_agreement_measures_the_relative_difference_in_units_of_eps32(tmp_path: Path) -> None:
    # The fp32 runs hold the fp64 density times (1 + delta): its relative L2 difference is delta
    # whatever the mesh, so the reported value is delta / 2^-23, compared with k.
    from autan_vv.agreement import EPS32, agreement  # noqa: PLC0415

    meshes = _family(tmp_path, "hex", (4, 6))
    for delta in (100.0 * EPS32, 600.0 * EPS32):
        out = tmp_path / f"d{delta:.3g}"
        double = _runs(out, meshes, "mms-euler", 2.0)
        single = []
        for mesh, path, fmt in meshes:
            case = _case(path, "mms-euler", precision="fp32+acc64")
            x = mesh.geometry.cell_centroid
            exact = get_oracle("mms-euler").fields(x, None)
            h = _h(mesh, (0, 1, 2))
            g = 1.0 + 0.5 * np.sin(0.2 * x[:, 0] + 0.1 * x[:, 1]) * np.cos(x[:, 2])
            fields = {k: exact[k] * (1.0 + 0.05 * h**2.0 * g) for k in field_names(case.spec)}
            fields["Density"] = fields["Density"] * (1.0 + delta)
            info = mesh_info(mesh, path, fmt)
            single.append(write_run(out / "single", case, mesh, info, fields).path)
        report = agreement(double, single)
        for pair in report["pairs"]:
            assert pair["fields"]["Density"] == pytest.approx(delta / EPS32, rel=1e-9)
            assert pair["fields"]["Pressure"] == 0.0
        assert report["pass"] is (delta / EPS32 <= 500.0)
