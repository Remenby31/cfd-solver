"""The oracle-state boundary condition and its face-state file (face-states.md)."""

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import autan.solver_input
import jsonschema
import pytest
from autan import cgns
from autan.cli import json_schema, main

from tests.conftest import CRM_HL, mesh_document

LAMINAR = CRM_HL.replace("autan.SA()", "autan.Laminar()")
VARIABLES = ["density", "velocity-x", "velocity-y", "velocity-z", "pressure"]


def states(**changes: Any) -> dict[str, Any]:
    document = {
        "schema": "autan.face-states/1",
        "oracle": "test",
        "mesh_sha256": mesh_document("crm-hl")["source"]["sha256"],
        "variables": VARIABLES,
        "zones": {
            "farfield": {
                "faces": 2,
                "centroids": [[0, 0, 0], [1, 0, 0]],
                "values": [[1.2, 30, 0, 0, 1e5], [1.2, 30, 0, 0, 1e5]],
            }
        },
    }
    document.update(changes)
    return document


def zone(rows: int, value: float = 1.2, **changes: object) -> dict[str, Any]:
    """The farfield zone with ``rows`` faces, its first density ``value``, then ``changes``."""
    values = [[value if i == 0 else 1.2, 30, 0, 0, 1e5] for i in range(rows)]
    entry = {"faces": rows, "centroids": [[i, 0, 0] for i in range(rows)], "values": values}
    return {"farfield": entry | changes}


@pytest.fixture
def two_faces(mesh_variant: Callable[..., autan.MeshInfo], tmp_path: Path) -> Path:
    def edit(document: dict[str, Any]) -> None:
        for zone in document["boundary_zones"]:
            if zone["name"] == "farfield":
                zone.update(faces=2, type=None, native_type=None)

    path = tmp_path / "two.meshinfo.json"
    path.write_text(json.dumps(mesh_variant("crm-hl", edit).document()))
    return path


def oracle_case(
    write_case: Callable[..., Path],
    content: dict[str, Any] | bytes,
    digest: str | None = None,
    body: str = LAMINAR,
) -> Path:
    data = content if isinstance(content, bytes) else json.dumps(content).encode()
    case = write_case(
        body + "case.boundary('farfield').oracle_state(path='ringleb.json', "
        f"sha256={digest or hashlib.sha256(data).hexdigest()!r})\n"
    )
    (case.parent / "ringleb.json").write_bytes(data)
    return case


def test_an_oracle_condition_reaches_the_solver_and_the_run(
    write_case: Callable[..., Path],
    two_faces: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case = oracle_case(write_case, states())
    data = (case.parent / "ringleb.json").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    mesh = autan.MeshInfo.load(two_faces)
    result = autan.load(case).plan(mesh)
    assert result.diagnostics == ()
    farfield = next(b for b in result.boundaries if b.zone == "farfield")
    assert (farfield.condition or {})["kind"] == "oracle-state"

    assert main(["prepare", str(case), "--mesh-info", str(two_faces)]) == 0
    document = json.loads(capsys.readouterr().out)
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    [boundary] = [b for b in document["boundaries"] if b["zone"] == "farfield"]
    assert boundary["states"] == {
        "path": str((case.parent / "ringleb.json").resolve()),
        "sha256": digest,
    }

    run = autan.load(case).run(
        mesh, backend=autan.FakeBackend(iterations=2), runs=tmp_path / "runs"
    )
    assert (run.path / "states" / f"{digest}.json").read_bytes() == data
    zone_bc = cgns.read(run.path / "solution.cgns").child("Base").child("Zone").child("ZoneBC")
    assert zone_bc.child("farfield").text() == "BCDirichlet"
    assert run.field("Density").size > 0
    (run.path / "states" / f"{digest}.json").write_text("{}")
    with pytest.raises(autan.CaseError) as error:
        autan.Results.open(run.path)
    assert error.value.codes == ["E0701"]


@pytest.mark.parametrize(
    ("content", "digest", "code"),
    [
        (b"{}", "0" * 64, "E0420"),
        (states(schema="autan.mms/1"), None, "E0420"),
        (states(variables=["density", "pressure"]), None, "E0420"),
        (states(mesh_sha256="1" * 64), None, "E0422"),
        (states(zones={}), None, "E0423"),
        (states(zones=zone(3)), None, "E0423"),
        (states(zones=zone(2, centroids=[[], None])), None, "E0420"),
        (states(zones=zone(2, values=[{}, "bad"])), None, "E0420"),
        (states(zones=zone(2, faces="2")), None, "E0420"),
        (states(zones=zone(2, value=float("nan"))), None, "E0420"),
        (states(zones=zone(2, value=-1.2)), None, "E0420"),
    ],
)
def test_the_face_state_file_is_checked(
    content: dict[str, Any] | bytes,
    digest: str | None,
    code: str,
    write_case: Callable[..., Path],
    two_faces: Path,
) -> None:
    case = oracle_case(write_case, content, digest)
    codes = {d.code for d in autan.load(case).plan(autan.MeshInfo.load(two_faces)).diagnostics}
    assert code in codes
    assert codes <= {code, "E0423"}


def test_a_missing_file_is_e0420(write_case: Callable[..., Path], two_faces: Path) -> None:
    case = oracle_case(write_case, states())
    (case.parent / "ringleb.json").unlink()
    codes = [d.code for d in autan.load(case).plan(autan.MeshInfo.load(two_faces)).diagnostics]
    assert codes == ["E0420"]


WATER = (
    "import autan\nfrom autan.units import Pa, kg, m, s\n"
    "water = autan.ConstantDensityFluid(name='water', density=998 * kg / m**3,"
    " viscosity=1e-3 * Pa * s)\n"
    "case = autan.Case('c', mesh='crm-hl.cgns')\n"
    "case.physics = autan.Incompressible(turbulence=autan.Laminar(), fluid=water)\n"
    "case.freestream(velocity=2 * m / s)\n"
)


@pytest.mark.parametrize("body", [WATER, CRM_HL])
def test_oracle_states_need_compressible_physics_without_turbulence_model(
    body: str, write_case: Callable[..., Path], two_faces: Path
) -> None:
    case = oracle_case(write_case, states(), body=body)
    codes = [d.code for d in autan.load(case).plan(autan.MeshInfo.load(two_faces)).diagnostics]
    assert "E0414" in codes


def _boundary(document: dict[str, Any], name: str) -> dict[str, Any]:
    [boundary] = [b for b in document["boundaries"] if b["zone"] == name]
    found: dict[str, Any] = boundary
    return found


def _borrow_states(document: dict[str, Any]) -> None:
    # Another zone takes the farfield rule's states: its own rule gives none.
    far = _boundary(document, "farfield")
    _boundary(document, "fuselage").update(kind="oracle-state", states=far["states"])


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: _boundary(d, "farfield")["states"].update(sha256="0" * 64),
        _borrow_states,
        lambda d: _boundary(d, "farfield").update(rule=None),
        lambda d: _boundary(d, "fuselage").update(rule=_boundary(d, "farfield")["rule"]),
        lambda d: _boundary(d, "farfield").update(rule=99),
    ],
)
def test_the_solver_input_states_are_those_of_the_case(
    edit: Callable[[dict[str, Any]], None],
    write_case: Callable[..., Path],
    two_faces: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case = oracle_case(write_case, states())
    assert main(["prepare", str(case), "--mesh-info", str(two_faces)]) == 0
    document = json.loads(capsys.readouterr().out)
    autan.solver_input.SolverInput.model_validate(document)
    edit(document)
    with pytest.raises(ValueError, match="boundary"):
        autan.solver_input.SolverInput.model_validate(document)


def test_a_face_state_path_must_be_a_regular_file(
    write_case: Callable[..., Path], two_faces: Path
) -> None:
    case = oracle_case(write_case, states())
    (case.parent / "ringleb.json").unlink()
    os.mkfifo(case.parent / "ringleb.json")  # no writer: a blocking read would hang
    codes = [d.code for d in autan.load(case).plan(autan.MeshInfo.load(two_faces)).diagnostics]
    assert codes == ["E0420"]


def test_both_validators_reject_oracle_states_with_a_turbulence_model(
    write_case: Callable[..., Path], two_faces: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = oracle_case(write_case, states())
    assert main(["prepare", str(case), "--mesh-info", str(two_faces)]) == 0
    document = json.loads(capsys.readouterr().out)
    document["physics"]["turbulence"].update(model="sa", nu_tilde_ratio=3.0)
    document["freestream"]["nu_tilde"] = 4.5e-5
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    with pytest.raises(ValueError, match="oracle-state"):
        autan.solver_input.SolverInput.model_validate(document)
