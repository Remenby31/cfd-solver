"""MeshInfo documents through the public reader."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import jsonschema
import pytest
from autan.cli import json_schema
from autan.meshinfo import mesh_sha256

from tests.conftest import FIXTURES, ROOT, mesh_document


@pytest.mark.parametrize("name", ["crm-hl", "heat-exchanger"])
def test_fixtures_are_valid_for_the_reader_and_the_published_schema(name: str) -> None:
    mesh = autan.MeshInfo.load(FIXTURES / f"{name}.meshinfo.json")
    assert mesh.document() == mesh_document(name)
    jsonschema.Draft202012Validator(json_schema("mesh-info")).validate(mesh_document(name))


def _zones(d: dict[str, Any]) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = d["boundary_zones"]
    return zones


def _grouped(
    document: dict[str, Any],
    groups: dict[str, list[str]],
    *,
    sort: bool = True,
    source: bool = True,
) -> None:
    names = sorted(groups) if sort else list(groups)
    document["groups"] = [{"name": name, "zones": groups[name]} for name in names]
    if source:
        document["groups_source"] = {"path": "m.zones.json", "sha256": "0" * 64, "provenance": {}}


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(schema="autan.mesh-info/2"),
        lambda d: d.update(extra=1),
        lambda d: d["source"].update(sha256="ABC"),
        lambda d: d["counts"]["cells_by_type"].update(hexa=1),
        lambda d: d["cell_zones"][0].update(cells=1),
        lambda d: _zones(d).append(dict(_zones(d)[0])),
        lambda d: _zones(d)[0].update(type="periodic"),
        lambda d: d["interfaces"].append({"zones": ["fluid", "nowhere"], "faces": 1}),
        lambda d: d["interfaces"].append({"zones": ["fluid", "fluid"], "faces": 1}),
        lambda d: d["quality"].update(max_non_orthogonality_deg=181.0),
        lambda d: d.pop("quality"),
        lambda d: d.pop("bounds"),
        lambda d: d["bounds"]["min"].__setitem__(0, 1e3),  # min above max
        lambda d: d["bounds"]["max"].pop(),
        # zone groups: refused by the MeshInfo reader too
        lambda d: _grouped(d, {"wing": ["nowhere"]}),
        lambda d: _grouped(d, {"wing": []}),
        lambda d: _grouped(d, {"wing": ["wing.main", "wing.slat"]}),  # not in MeshInfo order
        lambda d: _grouped(d, {"wing": ["wing.main", "wing.main"]}),
        lambda d: _grouped(d, {"*wing": ["wing.main"]}),
        lambda d: _grouped(d, {"-wing": ["wing.main"]}),
        lambda d: _grouped(d, {"b": ["wing.main"], "a": ["wing.main"]}, sort=False),
        lambda d: _grouped(d, {"wing": ["wing.main"]}, source=False),
    ],
)
def test_invalid_documents_are_rejected_with_e0202(
    edit: Callable[[dict[str, Any]], None], mesh_variant: Callable[..., autan.MeshInfo]
) -> None:
    with pytest.raises(autan.CaseError) as error:
        mesh_variant("crm-hl", edit)
    assert set(error.value.codes) == {"E0202"}


def test_unreadable_file_is_e0202(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{")
    with pytest.raises(autan.CaseError) as error:
        autan.MeshInfo.load(path)
    assert error.value.codes == ["E0202"]


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d["counts"].update(nodes=True),
        lambda d: d["cell_zones"][0].update(cells="48200000"),
        lambda d: d["cell_zones"].append(
            {"name": "core", "type": None, "native_type": None, "cells": 1}
        ),
    ],
)
def test_counts_are_strict_integers(
    edit: Callable[[dict[str, Any]], None], mesh_variant: Callable[..., autan.MeshInfo]
) -> None:
    with pytest.raises(autan.CaseError) as error:
        mesh_variant("crm-hl", edit)
    assert set(error.value.codes) == {"E0202"}


def test_interface_faces_must_fit_in_the_interior(
    mesh_variant: Callable[..., autan.MeshInfo],
) -> None:
    def edit(document: dict[str, Any]) -> None:
        document["interfaces"][0]["faces"] = document["counts"]["faces"]

    with pytest.raises(autan.CaseError) as error:
        mesh_variant("heat-exchanger", edit)
    assert error.value.codes == ["E0202"]


def test_undecodable_file_is_e0202(tmp_path: Path) -> None:
    path = tmp_path / "binary.json"
    path.write_bytes(b"\xff\xfe{")
    with pytest.raises(autan.CaseError) as error:
        autan.MeshInfo.load(path)
    assert error.value.codes == ["E0202"]


def test_an_openfoam_case_directory_is_its_polymesh(tmp_path: Path) -> None:
    # mesh.md, Readers: a case directory holding constant/polyMesh names that mesh.
    polymesh = ROOT / "tests" / "data" / "mesh" / "hybrid" / "polyMesh"
    shutil.copytree(polymesh, tmp_path / "case" / "constant" / "polyMesh")
    assert mesh_sha256(tmp_path / "case") == mesh_sha256(polymesh)
