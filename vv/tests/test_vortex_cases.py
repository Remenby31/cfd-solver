"""The isentropic-vortex cases of a vortex mesh family (autan-vv vortex-cases)."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import autan
import pytest
from autan import initial_state
from autan.meshinfo import mesh_sha256
from autan_vv.cli import main

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")


def test_every_level_solves_one_case_from_its_own_initial_state(tmp_path: Path) -> None:
    # One period of the vortex (t = 40) with the finest level's step on every level, zmin and zmax
    # symmetric, each level started from the oracle on its own mesh.
    main(["meshes", "vortex-prism", "--levels", "8,16", "--out", str(tmp_path / "mesh")])
    main(["vortex-cases", "--family", str(tmp_path / "mesh"), "--out", str(tmp_path / "cases")])
    documents = []
    for n in (8, 16):
        path = tmp_path / "cases" / f"vortex-prism-n{n:03d}.toml"
        case = autan.load(path)
        spec = case.spec
        assert spec.time.kind == "unsteady"
        assert math.isclose(spec.time.steps * spec.time.dt.to("s").magnitude, 40.0, rel_tol=1e-15)
        assert {r.zones for r in spec.boundaries} == {"zmin", "zmax"}
        assert {r.condition.kind for r in spec.boundaries} == {"symmetry"}
        state = case.resolve(spec.initial_state.path)
        assert spec.initial_state.sha256 == hashlib.sha256(state.read_bytes()).hexdigest()
        document = initial_state.parse(state.read_bytes())
        assert not isinstance(document, str), document
        assert document.mesh_sha256 == mesh_sha256(case.resolve(spec.mesh))
        assert document.cells == 2 * n * n
        dump = spec.model_dump(mode="json")
        for key in ("name", "mesh", "initial_state"):
            dump.pop(key)
        documents.append(dump)
    assert documents[0] == documents[1]


def test_a_mesh_file_other_than_its_manifest_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The initial state is computed on the regenerated level: a file the manifest does not record
    # would be paired with centroids that are not its own.
    main(["meshes", "vortex-hex", "--levels", "8", "--out", str(tmp_path / "mesh")])
    main(["meshes", "vortex-hex", "--levels", "16", "--out", str(tmp_path / "other")])
    (tmp_path / "mesh" / "vortex-hex-n008.msh").write_bytes(
        (tmp_path / "other" / "vortex-hex-n016.msh").read_bytes()
    )
    family, out = str(tmp_path / "mesh"), str(tmp_path / "c")
    assert main(["vortex-cases", "--family", family, "--out", out]) == 2
    assert "V0805" in capsys.readouterr().err
