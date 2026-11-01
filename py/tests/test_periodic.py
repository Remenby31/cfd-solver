"""Periodic pairs of boundary zones: a mesh fact (mesh-info.md), never a case condition
(case-schema.md#boundary-resolution)."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import autan.solver_input
import h5py
import jsonschema
import numpy as np
import pydantic
import pytest
from autan import cgns
from autan.cli import json_schema, main

from tests.conftest import CRM_HL

PAIR = {
    "zones": ["fuselage", "empennage"],
    "transform": {"kind": "translation", "vector": [0.0, 2.5, 0.0]},
    "faces": 95000,
}


def periodic(document: dict[str, Any], *pairs: dict[str, Any]) -> None:
    """Make fuselage and empennage periodic zones listed with ``pairs`` (PAIR by default);
    one ``{}`` drops the ``periodic`` key."""
    for zone in document["boundary_zones"]:
        if zone["name"] in ("fuselage", "empennage"):
            zone.update(type="periodic", faces=95000)
    document["periodic"] = list(pairs or [PAIR])
    if pairs == ({},):
        del document["periodic"]


@pytest.fixture
def mesh(mesh_variant: Callable[..., autan.MeshInfo], tmp_path: Path) -> Path:
    path = tmp_path / "periodic.meshinfo.json"
    path.write_text(json.dumps(mesh_variant("crm-hl", periodic).document()))
    return path


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: periodic(d, PAIR | {"zones": ["fuselage", "fuselage"]}),
        lambda d: periodic(d, PAIR | {"zones": ["fuselage", "farfield"]}),
        lambda d: periodic(d, PAIR | {"faces": 94000}),
        lambda d: periodic(d, PAIR | {"transform": {"kind": "translation", "vector": [0, 0, 0]}}),
        lambda d: periodic(d, PAIR | {"transform": {"kind": "rotation", "vector": [0, 1, 0]}}),
        lambda d: periodic(d, PAIR, PAIR),
        lambda d: periodic(d, {}),
    ],
)
def test_periodic_pairs_must_match_their_zones(
    edit: Callable[[dict[str, Any]], None], mesh_variant: Callable[..., autan.MeshInfo]
) -> None:
    with pytest.raises(autan.CaseError) as error:
        mesh_variant("crm-hl", edit)
    assert error.value.codes == ["E0202"]


BAD_PAIRS = [
    {"zones": ["fuselage", "fuselage"]},
    {"zones": ["", "empennage"]},
    {"transform": {"kind": "translation", "vector": [0, 0.0, 0]}},
]


@pytest.mark.parametrize("change", BAD_PAIRS)
def test_both_schemas_reject_a_malformed_pair(
    change: dict[str, Any],
    mesh: Path,
    write_case: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    info = json.loads(mesh.read_text())
    assert main(["prepare", str(write_case(CRM_HL)), "--mesh-info", str(mesh)]) == 0
    prepared = json.loads(capsys.readouterr().out)
    for kind, document in (("mesh-info", info), ("solver-input", prepared)):
        document["periodic"][0].update(change)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(json_schema(kind)).validate(document)
    with pytest.raises(autan.CaseError):
        autan.MeshInfo.from_document(info, autan.Location())
    with pytest.raises(pydantic.ValidationError):
        autan.solver_input.SolverInput.model_validate(prepared)


def test_a_zone_in_two_solver_input_pairs_is_rejected(
    mesh: Path, write_case: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["prepare", str(write_case(CRM_HL)), "--mesh-info", str(mesh)]) == 0
    document = json.loads(capsys.readouterr().out)
    document["periodic"].append(document["periodic"][0])
    with pytest.raises(ValueError, match="two periodic pairs"):
        autan.solver_input.SolverInput.model_validate(document)


def test_a_periodic_pair_resolves_from_the_mesh(
    write_case: Callable[..., Path], mesh: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = write_case(CRM_HL)
    plan = autan.load(case).plan(autan.MeshInfo.load(mesh))
    assert plan.ok, plan.diagnostics
    resolved = {b.zone: (b.condition, b.provenance) for b in plan.boundaries}
    assert resolved["fuselage"] == ({"kind": "periodic", "partner": "empennage"}, "mesh")
    assert resolved["empennage"] == ({"kind": "periodic", "partner": "fuselage"}, "mesh")

    assert main(["prepare", str(case), "--mesh-info", str(mesh)]) == 0
    document = json.loads(capsys.readouterr().out)
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    assert document["periodic"] == [{k: PAIR[k] for k in ("zones", "transform")}]
    assert {"fuselage", "empennage"}.isdisjoint(b["zone"] for b in document["boundaries"])

    document["boundaries"].append({**document["boundaries"][0], "zone": "fuselage"})
    with pytest.raises(ValueError, match="periodic"):
        autan.solver_input.SolverInput.model_validate(document)


def test_a_wildcard_skips_periodic_zones_and_a_name_is_e0424(
    write_case: Callable[..., Path], mesh: Path
) -> None:
    info = autan.MeshInfo.load(mesh)
    every = autan.load(write_case(CRM_HL + "case.boundary('*').slip()\n", name="every.py"))
    resolved = {b.zone: b.condition for b in every.plan(info).boundaries}
    assert resolved["fuselage"] == {"kind": "periodic", "partner": "empennage"}

    named = autan.load(write_case(CRM_HL + "case.boundary('fuselage').slip()\n"))
    assert [d.code for d in named.plan(info).diagnostics if d.severity == "error"] == ["E0424"]


def test_a_wildcard_that_spells_a_periodic_zone_is_still_a_wildcard(
    mesh_variant: Callable[..., autan.MeshInfo], write_case: Callable[..., Path]
) -> None:
    def rename(document: dict[str, Any]) -> None:
        periodic(document, PAIR | {"zones": ["fuse*", "empennage"]})
        document["boundary_zones"][3]["name"] = "fuse*"

    info = mesh_variant("crm-hl", rename)
    case = autan.load(write_case(CRM_HL + "case.boundary('fuse*').slip()\n"))
    codes = [d.code for d in case.plan(info).diagnostics if d.severity == "error"]
    assert codes == ["E0410"]  # it matches no other zone; not E0424


# results.md#solutioncgns: the translation is the MeshInfo's, in mesh units, times mesh.scale.
@pytest.mark.parametrize(("unit", "scale"), [("m", 1.0), ("in", 0.0254)])
def test_periodic_pairs_are_written_as_reciprocal_connectivities(
    unit: str, scale: float, write_case: Callable[..., Path], mesh: Path, tmp_path: Path
) -> None:
    body = CRM_HL.replace('mesh="crm-hl.cgns"', f'mesh="crm-hl.cgns", mesh_unit="{unit}"')
    case = autan.load(write_case(body))
    result = case.run(
        autan.MeshInfo.load(mesh), backend=autan.FakeBackend(iterations=2), runs=tmp_path
    )
    assert result.field("Density").size > 0  # reads and checks solution.cgns
    zone = cgns.read(result.path / "solution.cgns").child("Base").child("Zone")
    assert "fuselage" not in zone.child("ZoneBC").children
    side_a = zone.child("ZoneGridConnectivity").child("fuselage")
    side_b = zone.child("ZoneGridConnectivity").child("empennage")
    assert json.loads(side_a.child("PeriodicTranslation").text()) == [0.0, 2.5 * scale, 0.0]
    assert json.loads(side_b.child("PeriodicTranslation").text()) == [-0.0, -2.5 * scale, -0.0]
    pairs = set(
        zip(
            side_a.child("PointList").data.ravel().tolist(),
            side_a.child("PointListDonor").data.ravel().tolist(),
            strict=True,
        )
    )
    assert pairs
    assert all(a != b for a, b in pairs)


def _retype(text: Any, old: str, new: str) -> None:
    text[list(text[...]).index(ord(old))] = ord(new)


def _replace(node: Any, change: Callable[[Any], Any]) -> None:
    data = change(node[" data"][...])
    del node[" data"]
    node.create_dataset(" data", data=data)


def _set_text(node: Any, text: str) -> None:
    _replace(node, lambda _: np.frombuffer(text.encode(), dtype=np.int8))


def _reverse_side_a(group: Any) -> None:
    for name in ("PointList", "PointListDonor"):
        _replace(group[f"fuselage/{name}"], lambda a: a.ravel()[::-1].reshape(a.shape))


def _edit(path: Path, change: Callable[[Any], None]) -> None:
    with h5py.File(path / "solution.cgns", "r+") as handle:
        change(handle["Base/Zone/ZoneGridConnectivity"])


@pytest.mark.parametrize(
    "change",
    [
        lambda g: g["empennage/PointListDonor/ data"].__setitem__(slice(0, 1), 1),
        lambda g: g["fuselage/GridConnectivityProperty/Periodic/Translation/ data"].__setitem__(
            1, 2.4
        ),
        lambda g: _retype(g["fuselage/PeriodicTranslation/ data"], "2", "3"),  # [0, 3.5, 0]
        lambda g: g.__delitem__("empennage"),
        lambda g: g["fuselage/GridConnectivityProperty/Periodic/RotationAngle/ data"].__setitem__(
            0, 0.125
        ),
        _reverse_side_a,  # the same correspondence, not in mesh order
        lambda g: _replace(g["fuselage/PointList"], lambda a: a + 0.5),  # not integer ids
        lambda g: _set_text(g["fuselage/PeriodicTranslation"], "[false, 2.5, false]"),
        lambda g: g.parent.copy(
            g.parent["ZoneBC"][next(iter(g.parent["ZoneBC"]))], "ZoneBC/fuselage"
        ),
    ],
)
def test_the_reader_checks_the_periodic_layout(
    change: Callable[[Any], None], write_case: Callable[..., Path], mesh: Path, tmp_path: Path
) -> None:
    case = autan.load(write_case(CRM_HL))
    result = case.run(
        autan.MeshInfo.load(mesh), backend=autan.FakeBackend(iterations=2), runs=tmp_path
    )
    _edit(result.path, change)
    with pytest.raises(autan.CaseError) as error:
        autan.Results.open(result.path).field("Density")
    assert error.value.codes == ["E0701"]


def test_a_big_endian_r4_translation_is_valid(
    write_case: Callable[..., Path], mesh: Path, tmp_path: Path
) -> None:
    case = autan.load(write_case(CRM_HL))
    result = case.run(
        autan.MeshInfo.load(mesh), backend=autan.FakeBackend(iterations=2), runs=tmp_path
    )
    _edit(
        result.path,
        lambda g: _replace(
            g["fuselage/GridConnectivityProperty/Periodic/Translation"], lambda a: a.astype(">f4")
        ),
    )
    assert autan.Results.open(result.path).field("Density").size > 0


def test_a_monitor_on_a_periodic_zone_is_e0603(
    write_case: Callable[..., Path], mesh: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    info = autan.MeshInfo.load(mesh)
    named = autan.load(write_case(CRM_HL + "case.monitor(autan.forces('fuselage', name='f'))\n"))
    assert [d.code for d in named.plan(info).diagnostics if d.severity == "error"] == ["E0603"]
    every = write_case(
        CRM_HL + "case.monitor(autan.forces('f*', name='f'))\n", name="every.py"
    )  # farfield and the periodic fuselage
    assert main(["prepare", str(every), "--mesh-info", str(mesh)]) == 0
    document = json.loads(capsys.readouterr().out)
    [monitor] = [m for m in document["monitors"] if m["name"] == "f"]
    assert monitor["zones"] == ["farfield"]
    monitor["zones"] = ["farfield", "fuselage"]
    with pytest.raises(pydantic.ValidationError):
        autan.solver_input.SolverInput.model_validate(document)
    monitor["zones"] = []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)

    only = autan.load(write_case(CRM_HL + "case.monitor(autan.forces('fuse*', name='f'))\n"))
    assert [d.code for d in only.plan(info).diagnostics if d.severity == "error"] == ["E0601"]


def test_a_monitor_wildcard_that_spells_a_periodic_zone_is_still_a_wildcard(
    mesh_variant: Callable[..., autan.MeshInfo], write_case: Callable[..., Path]
) -> None:
    def rename(document: dict[str, Any]) -> None:
        periodic(document, PAIR | {"zones": ["fuse*", "empennage"]})
        document["boundary_zones"][3]["name"] = "fuse*"

    case = autan.load(write_case(CRM_HL + "case.monitor(autan.forces('fuse*', name='f'))\n"))
    codes = [d.code for d in case.plan(mesh_variant("crm-hl", rename)).diagnostics]
    assert [c for c in codes if c.startswith("E")] == ["E0601"]
