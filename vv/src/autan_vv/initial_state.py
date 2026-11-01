"""Initial-state files (``autan.initial-state/1``, docs/contracts/initial-state.md) from an oracle
evaluated at the cell centroids of a mesh."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
from autan import initial_state
from autan.meshinfo import mesh_sha256

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.meshes.facemesh import FaceMesh
    from autan_vv.oracles import Oracle

_FIELDS = ("Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure")


def write_initial_state(
    oracle: Oracle, mesh: FaceMesh, mesh_file: Path, out: Path, time: float | None = None
) -> str:
    """Write to `out` the state of `oracle` at `time` (its start without one) at every cell
    centroid of `mesh`, the mesh `mesh_file` holds (a file or a polyMesh directory), and return
    the file's SHA-256 for the case. The state is the oracle's value at the centroid, with no
    projection onto the cell."""
    # Centroids computed about a point of the mesh, then moved back: about the origin, a mesh far
    # from it loses the digits the solver's pairing tolerance (1e-9 of the extent) needs.
    origin = mesh.points[0]
    local = dataclasses.replace(mesh, points=mesh.points - origin)
    centroids = local.geometry.cell_centroid + origin
    fields = oracle.evaluate(centroids, time)
    values = np.stack([np.broadcast_to(fields[name], (mesh.n_cells,)) for name in _FIELDS], axis=1)
    return initial_state.write(
        out,
        oracle=oracle.name,
        mesh_sha256=mesh_sha256(mesh_file),
        centroids=centroids.tolist(),
        values=values.tolist(),
    )
