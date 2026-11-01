"""The solver input through `autan prepare`; expected values recomputed from the contracts."""

import hashlib
import json
import math
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import autan.solver_input
import jsonschema
import pytest
import rfc8785
from autan.cli import json_schema, main
from pydantic import ValidationError

from tests.conftest import CRM_HL, FIXTURES

R_UNIVERSAL = 8.314462618


def prepare(
    capsys: pytest.CaptureFixture[str], case: Path, mesh: str, *extra: str
) -> tuple[int, Any]:
    status = main(["prepare", str(case), "--mesh-info", str(FIXTURES / mesh), *extra])
    out = capsys.readouterr().out
    return status, json.loads(out) if status == 0 else out


def test_crm_hl_input_is_resolved_and_closed_in_si(
    capsys: pytest.CaptureFixture[str], example: Callable[[str], Path]
) -> None:
    status, document = prepare(
        capsys,
        example("crm_hl.py"),
        "crm-hl.meshinfo.json",
        "--device-kind",
        "cuda",
        "--device-count",
        "8",
    )
    assert status == 0
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    assert document["devices"] == {"kind": "cuda", "count": 8}
    assert document["reproducibility"] == "run"  # the case's default (case-schema.md#physics)

    # The case hash is SHA-256 of the RFC 8785 form of the (fully defaulted) case document.
    canonical = rfc8785.dumps(tomllib.loads(document["case"]["toml"]))
    assert hashlib.sha256(canonical).hexdigest() == document["case"]["sha256"]

    temperature, mach, re_number, length = 288.15, 0.2, 5.49e6, 7.0
    r = R_UNIVERSAL / 0.0289647
    mu = 1.716e-5 * (temperature / 273.15) ** 1.5 * (273.15 + 110.4) / (temperature + 110.4)
    speed = mach * math.sqrt(1.4 * r * temperature)
    rho = re_number * mu / (speed * length)
    alpha = math.radians(19.57)
    fs = document["freestream"]
    assert fs["density"] == pytest.approx(rho, rel=1e-12)
    assert fs["pressure"] == pytest.approx(rho * r * temperature, rel=1e-12)
    assert fs["velocity"] == pytest.approx(
        [speed * math.cos(alpha), 0.0, speed * math.sin(alpha)], rel=1e-12
    )
    assert fs["aoa"] == pytest.approx(alpha, rel=1e-12)
    assert fs["nu_tilde"] == pytest.approx(3.0 * mu / rho, rel=1e-12)
    assert (fs["k"], fs["omega"]) == (None, None)

    kinds = {b["zone"]: b["kind"] for b in document["boundaries"]}
    assert kinds == {
        "wing.slat": "wall",
        "wing.main": "wall",
        "wing.flap": "wall",
        "fuselage": "wall",
        "empennage": "wall",
        "symmetry": "symmetry",
        "farfield": "farfield",
    }
    [forces] = document["monitors"]
    assert forces["zones"] == ["wing.slat", "wing.main", "wing.flap"]
    assert forces["ref_area"] == pytest.approx(191.84)
    assert forces["ref_length"] == pytest.approx(7.0)
    assert document["time"] == {
        "kind": "steady",
        "max_iterations": 10000,
        "residual_orders": 8.0,
        "linear_solver": "multigrid",
    }


def test_heat_exchanger_media_and_inlets_in_si(
    capsys: pytest.CaptureFixture[str], example: Callable[[str], Path]
) -> None:
    status, document = prepare(capsys, example("heat_exchanger.py"), "heat-exchanger.meshinfo.json")
    assert status == 0
    media = {z["zone"]: z["medium"] for z in document["cell_zones"]}
    assert media["co2"]["gas_constant"] == pytest.approx(R_UNIVERSAL / 0.0440095, rel=1e-12)
    assert media["co2"]["viscosity"] == {
        "law": "sutherland",
        "mu_ref": 1.370e-5,
        "t_ref": 273.15,
        "s": 222.0,
    }
    assert media["plate"] == {
        "kind": "solid",
        "name": "copper",
        "density": 8933.0,
        "specific_heat": 385.0,
        "conductivity": 401.0,
    }
    inlet = next(b for b in document["boundaries"] if b["zone"] == "air-inlet")
    assert (inlet["kind"], inlet["total_pressure"], inlet["total_temperature"]) == (
        "inlet",
        pytest.approx(103000.0),
        pytest.approx(400.0),
    )
    outlet = next(b for b in document["boundaries"] if b["zone"] == "co2-outlet")
    assert outlet["static_pressure"] == pytest.approx(101325.0)
    turbulence = document["physics"]["turbulence"]
    assert (turbulence["intensity"], turbulence["viscosity_ratio"]) == (0.001, 0.009)


def test_plan_errors_block_the_input(
    capsys: pytest.CaptureFixture[str], write_case: Callable[[str], Path], tmp_path: Path
) -> None:
    document = json.loads((FIXTURES / "crm-hl.meshinfo.json").read_text())
    document["boundary_zones"][4]["type"] = None
    info = tmp_path / "untyped.meshinfo.json"
    info.write_text(json.dumps(document))
    case = write_case((FIXTURES.parent.parent / "examples" / "crm_hl.py").read_text())
    assert main(["prepare", str(case), "--mesh-info", str(info), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert [d["code"] for d in out["diagnostics"]] == ["E0412"]


def _valid(
    capsys: pytest.CaptureFixture[str], example: Callable[[str], Path], name: str, mesh: str
) -> dict[str, Any]:
    status, document = prepare(capsys, example(name), mesh)
    assert status == 0
    result: dict[str, Any] = document
    return result


def _set_boundary(document: dict[str, Any], **values: Any) -> None:
    document["boundaries"][0].update(values)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(extra=1),
        lambda d: d.pop("schema"),
        lambda d: d["time"].pop("kind"),
        lambda d: d["monitors"][0].pop("kind"),
        lambda d: _set_boundary(d, kind="inlet"),
        lambda d: _set_boundary(d, static_pressure=1.0),
        lambda d: d["physics"]["turbulence"].update(nu_tilde_ratio=None),
        lambda d: d["physics"]["turbulence"].update(intensity=0.1),
        lambda d: d.update(
            time={"kind": "unsteady", "steps": 10, "dt": None, "cfl": None, "average_from": None}
        ),
        lambda d: d.update(
            time={"kind": "unsteady", "steps": 10, "dt": 1e-3, "cfl": 1.0, "average_from": None}
        ),
        lambda d: d["physics"]["turbulence"].update(
            model="wmles", wall_model="equilibrium", nu_tilde_ratio=None
        ),
    ],
)
def test_schema_and_reader_reject_the_same_documents(
    edit: Callable[[dict[str, Any]], None],
    capsys: pytest.CaptureFixture[str],
    example: Callable[[str], Path],
) -> None:
    document = _valid(capsys, example, "crm_hl.py", "crm-hl.meshinfo.json")
    validator = jsonschema.Draft202012Validator(json_schema("solver-input"))
    validator.validate(document)
    autan.solver_input.SolverInput.model_validate(document)
    edit(document)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(document)
    with pytest.raises(ValidationError):
        autan.solver_input.SolverInput.model_validate(document)


def test_each_zone_starts_in_its_own_medium(
    capsys: pytest.CaptureFixture[str], example: Callable[[str], Path]
) -> None:
    document = _valid(capsys, example, "heat_exchanger.py", "heat-exchanger.meshinfo.json")
    initial = {z["zone"]: z["initial"] for z in document["cell_zones"]}
    pressure, temperature = 101325.0, 300.0
    rho = pressure / (R_UNIVERSAL / 0.0440095 * temperature)
    mu = 1.370e-5 * (temperature / 273.15) ** 1.5 * (273.15 + 222.0) / (temperature + 222.0)
    speed = 0.1 * math.sqrt(1.4 * R_UNIVERSAL / 0.0289647 * temperature)
    k = 1.5 * (0.001 * speed) ** 2
    assert initial["co2"]["density"] == pytest.approx(rho, rel=1e-12)
    assert initial["co2"]["pressure"] == pytest.approx(pressure)
    assert initial["co2"]["omega"] == pytest.approx(rho * k / (0.009 * mu), rel=1e-12)
    assert initial["plate"] == {"kind": "solid", "temperature": temperature}


def test_prepare_reports_plan_warnings_and_bad_arguments(
    capsys: pytest.CaptureFixture[str], example: Callable[[str], Path], tmp_path: Path
) -> None:
    case = example("crm_hl.py")
    case.write_text(case.read_text() + 'case.boundary("fuselage").slip()\n')
    argv = ["prepare", str(case), "--mesh-info", str(FIXTURES / "crm-hl.meshinfo.json")]
    assert main([*argv, "--json"]) == 0
    warnings = json.loads(capsys.readouterr().err)
    assert [w["code"] for w in warnings] == ["W0415"]
    with pytest.raises(SystemExit) as exit_:
        main([*argv, "--device-count", "0"])
    assert exit_.value.code == 2
    assert main([*argv, "--json", "--output", str(tmp_path)]) == 1
    assert [d["code"] for d in json.loads(capsys.readouterr().out)["diagnostics"]] == ["E0806"]


@pytest.mark.parametrize("name", ["case", "mesh-info", "results", "solver-input"])
def test_each_published_schema_has_its_identifier(name: str) -> None:
    schema = json_schema(name)
    assert schema["$id"] == autan.version.SCHEMAS[name]
    assert schema["properties"]["schema"]["const"] == autan.version.SCHEMAS[name]


@pytest.mark.parametrize("mode", ["none", "run", "ranks"])
def test_reproducibility_reaches_the_runtime_and_the_hash(
    mode: str,
    write_case: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    body = CRM_HL.replace("autan.SA())", f"autan.SA(), reproducibility={mode!r})")
    case = write_case(body, name=f"{mode}.py")
    spec = autan.load(case).spec
    copy = tmp_path / "copy.toml"
    copy.write_text(autan.load(case).to_toml())
    assert autan.load(copy).spec == spec
    assert (spec.hash() == autan.load(write_case(CRM_HL, name="default.py")).spec.hash()) == (
        mode == "run"  # the default
    )
    assert main(["prepare", str(case), "--mesh-info", str(FIXTURES / "crm-hl.meshinfo.json")]) == 0
    document = json.loads(capsys.readouterr().out)
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    assert document["reproducibility"] == mode


def test_the_runtime_mode_is_the_case_s(
    write_case: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    case = write_case(CRM_HL)
    assert main(["prepare", str(case), "--mesh-info", str(FIXTURES / "crm-hl.meshinfo.json")]) == 0
    document = json.loads(capsys.readouterr().out)
    document["reproducibility"] = "none"  # the embedded case says "run"
    with pytest.raises(ValidationError, match="reproducibility"):
        autan.solver_input.SolverInput.model_validate(document)


def test_the_linear_solver_is_a_steady_choice(
    write_case: Callable[..., Path], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    body = CRM_HL + "case.time = autan.Steady(linear_solver='gauss-seidel')\n"
    case = write_case(body)
    copy = tmp_path / "copy.toml"
    copy.write_text(autan.load(case).to_toml())
    assert autan.load(copy).spec == autan.load(case).spec
    assert main(["prepare", str(case), "--mesh-info", str(FIXTURES / "crm-hl.meshinfo.json")]) == 0
    document = json.loads(capsys.readouterr().out)
    validator = jsonschema.Draft202012Validator(json_schema("solver-input"))
    validator.validate(document)
    assert document["time"]["linear_solver"] == "gauss-seidel"
    unsteady = {"kind": "unsteady", "steps": 10, "dt": 1e-3, "cfl": None, "average_from": None}
    with pytest.raises(jsonschema.ValidationError):  # an unsteady run has no linear solver choice
        validator.validate(document | {"time": unsteady | {"linear_solver": "multigrid"}})
