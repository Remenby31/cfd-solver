import copy
import hashlib
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = ROOT / "py" / "examples"
CRM_HL = (EXAMPLES / "crm_hl.py").read_text()
HEAT_EXCHANGER = (EXAMPLES / "heat_exchanger.py").read_text()

# Stand-ins for the example meshes: their SHA-256 is the source.sha256 of the fixtures.
MESH_FILES = {"crm-hl.cgns": b"crm-hl fixture", "heat-exchanger.msh": b"heat-exchanger fixture"}


def identity_of(run: dict[str, Any]) -> str:
    """``identity`` as results.md#identity lists its fields, read from that page."""
    page = (ROOT / "docs" / "contracts" / "results.md").read_text()
    section = page.split("### Identity\n", 1)[1].split("\n## ", 1)[0]
    fields = re.findall(r"^\d+\. `([a-z0-9_.]+)`$", section, flags=re.MULTILINE)
    parts = []
    for path in fields:
        if path.startswith("start.") and run["start"] is None:
            continue
        value: Any = run
        for key in path.split("."):
            value = value[key]
        parts.append(str(value))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def mesh_document(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((FIXTURES / f"{name}.meshinfo.json").read_text())
    return document


def place_meshes(directory: Path) -> None:
    for name, content in MESH_FILES.items():
        (directory / name).write_bytes(content)


@pytest.fixture
def crm_mesh() -> autan.MeshInfo:
    return autan.MeshInfo.load(FIXTURES / "crm-hl.meshinfo.json")


@pytest.fixture
def mesh_variant() -> Callable[..., autan.MeshInfo]:
    """A fixture MeshInfo after ``edit`` mutates its JSON document."""

    def build(name: str, edit: Callable[[dict[str, Any]], None]) -> autan.MeshInfo:
        document = copy.deepcopy(mesh_document(name))
        edit(document)
        return autan.MeshInfo.from_document(document, autan.Location(file=name))

    return build


@pytest.fixture
def write_case(tmp_path: Path) -> Callable[..., Path]:
    """Write a case file next to the example mesh stand-ins."""
    place_meshes(tmp_path)

    def write(body: str, name: str = "case.py") -> Path:
        path = tmp_path / name
        path.write_text(body)
        return path

    return write


@pytest.fixture
def example(tmp_path: Path) -> Callable[[str], Path]:
    """Copy an example case next to the example mesh stand-ins."""
    place_meshes(tmp_path)

    def copy_example(name: str) -> Path:
        return Path(shutil.copy(EXAMPLES / name, tmp_path / name))

    return copy_example
