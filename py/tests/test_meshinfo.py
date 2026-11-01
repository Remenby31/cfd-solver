"""MeshInfo documents through the public reader."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import jsonschema
import pytest
from autan.cli import json_schema

from tests.conftest import FIXTURES, mesh_document


@pytest.mark.parametrize("name", ["crm-hl", "heat-exchanger"])
def test_fixtures_are_valid_for_the_reader_and_the_published_schema(name: str) -> None:
    mesh = autan.MeshInfo.load(FIXTURES / f"{name}.meshinfo.json")
    assert mesh.document() == mesh_document(name)
    jsonschema.Draft202012Validator(json_schema("mesh-info")).validate(mesh_document(name))


def _zones(d: dict[str, Any]) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = d["boundary_zones"]
    return zones


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
