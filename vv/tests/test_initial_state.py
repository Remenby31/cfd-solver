"""The initial-state exporter: an oracle at the cell centroids of a mesh, as the SDK reads it."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
from autan import initial_state
from autan.meshinfo import mesh_sha256
from autan_vv.initial_state import write_initial_state
from autan_vv.meshes.families import cartesian
from autan_vv.oracles.inviscid import isentropic_vortex

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(("offset", "side"), [(0.0, 20.0), (1e7, 1.0)])
@pytest.mark.parametrize("directory", [False, True])
def test_the_vortex_is_written_at_every_cell_centroid(
    tmp_path: Path, offset: float, side: float, directory: bool
) -> None:
    # A mesh far from the origin, as a translated domain or projected coordinates give, keeps
    # centroids within the solver's pairing tolerance; a polyMesh directory is hashed as
    # mesh-info.md hashes it.
    half = 0.5 * side
    lower, upper = (offset - half, offset - half, 0.0), (offset + half, offset + half, 1.0)
    mesh = cartesian((4, 4, 1), lower, upper)
    if directory:
        mesh_file = tmp_path / "polyMesh"
        mesh_file.mkdir()
        for name in ("points", "faces", "owner", "neighbour", "boundary"):
            (mesh_file / name).write_text(f"{name} of the mesh the case names\n")
    else:
        mesh_file = tmp_path / "box.msh"
        mesh_file.write_bytes(b"the mesh the case names")
    out = tmp_path / "vortex.json"
    oracle = isentropic_vortex()

    digest = write_initial_state(oracle, mesh, mesh_file, out)

    data = out.read_bytes()
    assert digest == hashlib.sha256(data).hexdigest()
    document = initial_state.parse(data)
    assert not isinstance(document, str), document
    assert document.oracle == "isentropic-vortex"
    assert document.mesh_sha256 == mesh_sha256(mesh_file)
    # Cell (i, j) has its centroid at offset - side / 2 + side / 4 (i + 1/2) along x and y, 0.5
    # along z; the solver pairs within 1e-9 of the largest extent (initial-state.md).
    centres = offset - half + 0.25 * side * (np.arange(4) + 0.5)
    expected = np.array(sorted((x, y, 0.5) for x in centres for y in centres))
    rows = np.array(document.centroids)
    order = np.lexsort((np.round(rows[:, 1] - offset, 3), np.round(rows[:, 0] - offset, 3)))
    np.testing.assert_allclose(rows[order], expected, rtol=0, atol=1e-9 * side)
    state = oracle.evaluate(rows, None)
    names = ("Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure")
    np.testing.assert_array_equal(
        np.array(document.values), np.stack([state[n] for n in names], axis=1)
    )
