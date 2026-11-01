"""MeshInfo ``autan.mesh-info/1``: the mesh summary emitted by the core (docs/contracts/mesh-info.md)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator

from autan.errors import CaseError, Location, diagnostic
from autan.units import fail

if TYPE_CHECKING:
    from collections.abc import Iterable

SCHEMA_ID = "autan.mesh-info/1"

BoundaryType = Literal[
    "wall", "thermal-wall", "slip", "symmetry", "farfield", "inlet", "outlet", "periodic"
]
Name = Annotated[str, Field(min_length=1)]
Count = Annotated[int, Field(strict=True, ge=0)]
Positive = Annotated[int, Field(strict=True, ge=1)]
POLYMESH_FILES = ("points", "faces", "owner", "neighbour", "boundary", "cellZones")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Source(_Strict):
    path: Name
    format: Literal["cgns", "gmsh", "fluent", "polymesh"]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]


class CellsByType(_Strict):
    tetra: Count
    pyramid: Count
    prism: Count
    hexa: Count
    polyhedron: Count


class Counts(_Strict):
    cells: Positive
    faces: Positive
    nodes: Positive
    cells_by_type: CellsByType


Point = tuple[
    Annotated[float, Field(strict=True, allow_inf_nan=False)],
    Annotated[float, Field(strict=True, allow_inf_nan=False)],
    Annotated[float, Field(strict=True, allow_inf_nan=False)],
]


class Bounds(_Strict):
    min: Point
    max: Point


class BoundaryZone(_Strict):
    name: Name
    type: BoundaryType | None
    native_type: str | None
    faces: Count


class CellZone(_Strict):
    name: Name
    type: Literal["fluid", "solid"] | None
    native_type: str | None
    cells: Positive


class Interface(_Strict):
    zones: tuple[Name, Name]
    faces: Positive


Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]


def _distinct(zones: tuple[str, str]) -> tuple[str, str]:
    if zones[0] == zones[1]:
        message = "a periodic pair is two distinct zones"
        raise ValueError(message)
    return zones


def _non_zero(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    if not any(vector):
        message = "a periodic translation is not zero"
        raise ValueError(message)
    return vector


PairZones = Annotated[
    tuple[Name, Name], AfterValidator(_distinct), Field(json_schema_extra={"uniqueItems": True})
]
ZERO = {"prefixItems": [{"const": 0}] * 3}


class Translation(_Strict):
    kind: Literal["translation"]
    vector: Annotated[
        tuple[Finite, Finite, Finite],
        AfterValidator(_non_zero),
        Field(json_schema_extra={"not": ZERO}),
    ]


class PeriodicPair(_Strict):
    zones: PairZones
    transform: Translation
    faces: Positive


def shared_zones(pairs: Iterable[tuple[str, str]]) -> set[str]:
    """Zones that appear in more than one of ``pairs``."""
    names = [name for pair in pairs for name in pair]
    return {name for name in names if names.count(name) > 1}


class Quality(_Strict):
    max_non_orthogonality_deg: Annotated[float, Field(strict=True, ge=0, le=180)]
    max_faces_per_cell: Annotated[int, Field(strict=True, ge=4)]


class MeshInfo(_Strict):
    """Summary of a mesh: counts, typed boundary and cell zones, interfaces, quality, hash."""

    schema_: Literal["autan.mesh-info/1"] = Field(alias="schema")
    source: Source
    counts: Counts
    bounds: Bounds
    boundary_zones: tuple[BoundaryZone, ...]
    cell_zones: Annotated[tuple[CellZone, ...], Field(min_length=1)]
    interfaces: tuple[Interface, ...]
    periodic: tuple[PeriodicPair, ...]
    quality: Quality

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        problems = []
        if any(low > high for low, high in zip(self.bounds.min, self.bounds.max, strict=True)):
            problems.append("bounds.min exceeds bounds.max")
        if sum(self.counts.cells_by_type.model_dump().values()) != self.counts.cells:
            problems.append("counts.cells_by_type does not sum to counts.cells")
        if sum(zone.cells for zone in self.cell_zones) != self.counts.cells:
            problems.append("cell_zones[].cells does not sum to counts.cells")
        boundary = sum(zone.faces for zone in self.boundary_zones)
        if boundary > self.counts.faces:
            problems.append("boundary_zones[].faces exceeds counts.faces")
        if sum(i.faces for i in self.interfaces) > self.counts.faces - boundary:
            problems.append("interfaces[].faces exceeds the interior face count")
        for label, zones in (
            ("boundary_zones", self.boundary_zones),
            ("cell_zones", self.cell_zones),
        ):
            names = [zone.name for zone in zones]
            if len(set(names)) != len(names):
                problems.append(f"{label} has duplicate names")
        cell_names = {zone.name for zone in self.cell_zones}
        pairs = [interface.zones for interface in self.interfaces]
        for a, b in pairs:
            if a >= b:
                problems.append(f"interface {[a, b]} is not two distinct sorted names")
            if a not in cell_names or b not in cell_names:
                problems.append(f"interface {[a, b]} names an unknown cell zone")
        if len(set(pairs)) != len(pairs):
            problems.append("interfaces lists a pair twice")
        problems += self._periodic_problems()
        if problems:
            fail("E0202", "{problems}", problems="; ".join(problems))
        return self

    def _periodic_problems(self) -> list[str]:
        faces = {zone.name: zone.faces for zone in self.boundary_zones}
        periodic = {zone.name for zone in self.boundary_zones if zone.type == "periodic"}
        paired = [name for pair in self.periodic for name in pair.zones]
        problems = []
        for pair in self.periodic:
            a, b = pair.zones
            if not {a, b} <= periodic:
                problems.append(f"periodic pair {[a, b]} is not two periodic zones")
            elif faces[a] != pair.faces or faces[b] != pair.faces:
                problems.append(f"periodic pair {[a, b]} has another face count than its zones")
        if sorted(paired) != sorted(periodic):
            problems.append("each periodic zone must be in exactly one periodic pair")
        return problems

    @classmethod
    def load(cls, path: str | Path) -> MeshInfo:
        """Read a MeshInfo JSON file; any defect raises ``CaseError`` with ``E0202``."""
        location = Location(file=str(path))
        try:
            document = json.loads(Path(path).read_text())
        except (OSError, ValueError) as error:
            raise CaseError(
                [diagnostic("E0202", f"cannot read {path}: {error}", location=location)]
            ) from None
        return cls.from_document(document, location)

    @classmethod
    def from_document(cls, document: object, location: Location) -> MeshInfo:
        if isinstance(document, dict) and document.get("schema") != SCHEMA_ID:
            raise CaseError(
                [
                    diagnostic(
                        "E0202",
                        f"schema is {document.get('schema')!r}, expected {SCHEMA_ID!r}",
                        location=location,
                    )
                ]
            )
        try:
            return cls.model_validate(document)
        except ValidationError as error:
            raise CaseError(
                diagnostic(
                    "E0202",
                    f"{'.'.join(map(str, e['loc'])) or 'mesh-info'}: {e['msg']}",
                    location=location,
                )
                for e in error.errors(include_url=False)
            ) from None

    def document(self) -> dict[str, object]:
        return self.model_dump(mode="json", by_alias=True)


def mesh_sha256(path: Path) -> str:
    """The mesh hash of a mesh file, or of a polyMesh directory (mesh-info.md, ``source.sha256``)."""
    digest = hashlib.sha256()
    if path.is_dir():
        for name in POLYMESH_FILES:
            member = path / name
            if member.exists():
                data = member.read_bytes()
                digest.update(f"{name}\n{len(data)}\n".encode() + data)
        return digest.hexdigest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
