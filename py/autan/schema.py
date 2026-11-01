"""The case document ``autan.case/1`` as pydantic models (docs/contracts/case-schema.md)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import pint
import rfc8785
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from autan.errors import CaseError, Diagnostic, Location, diagnostic
from autan.units import K, deg, fail, m, quantity, si, validate_number

SCHEMA_ID = "autan.case/1"
# Largest integer RFC 8785 canonicalizes exactly (IEEE 754 double, 2**53 - 1).
MAX_INTEGER = 2**53 - 1

Length = Annotated[pint.Quantity, *quantity("[length]", positive=True)]
Area = Annotated[pint.Quantity, *quantity("[length] ** 2", positive=True)]
Speed = Annotated[pint.Quantity, *quantity("[length] / [time]", positive=True)]
Duration = Annotated[pint.Quantity, *quantity("[time]", positive=True)]
Temperature = Annotated[pint.Quantity, *quantity("[temperature]", positive=True)]
Pressure = Annotated[pint.Quantity, *quantity("[pressure]", positive=True)]
Density = Annotated[pint.Quantity, *quantity("[mass] / [length] ** 3", positive=True)]
DynamicViscosity = Annotated[pint.Quantity, *quantity("[mass] / [length] / [time]", positive=True)]
MolarMass = Annotated[pint.Quantity, *quantity("[mass] / [substance]", positive=True)]
SpecificHeat = Annotated[
    pint.Quantity, *quantity("[length] ** 2 / [time] ** 2 / [temperature]", positive=True)
]
Conductivity = Annotated[
    pint.Quantity, *quantity("[mass] * [length] / [time] ** 3 / [temperature]", positive=True)
]
Angle = Annotated[pint.Quantity, *quantity("[angle]")]
Coordinate = Annotated[pint.Quantity, *quantity("[length]")]
Velocity = Annotated[pint.Quantity, *quantity("[length] / [time]")]
PositiveNumber = Annotated[
    float, Field(gt=0), BeforeValidator(lambda v: validate_number(v, positive=True))
]
Gamma = Annotated[float, Field(gt=1), BeforeValidator(lambda v: validate_number(v, positive=True))]
Count = Annotated[int, Field(strict=True, ge=1, le=MAX_INTEGER)]
Index = Annotated[int, Field(strict=True, ge=0, le=MAX_INTEGER)]
Pattern = Annotated[str, Field(min_length=1)]
Name = Annotated[str, Field(min_length=1)]

_RANGE_ERRORS = frozenset({"greater_than", "greater_than_equal", "less_than", "less_than_equal"})
_TAGS = ("kind", "model", "law")


def _caller_location() -> Location:
    frame = sys._getframe(1)
    package = str(Path(__file__).parent)
    while frame is not None and frame.f_code.co_filename.startswith(package):
        frame = frame.f_back  # type: ignore[assignment]
    if frame is None:
        return Location()
    return Location(file=frame.f_code.co_filename, line=frame.f_lineno)


def document_path(parts: tuple[int | str, ...], document: object) -> str:
    """Document path of a pydantic error location, without the union tags pydantic inserts.

    ``("boundaries", 0, "condition", "wall", "temperature")`` -> ``"boundaries[0].condition
    .temperature"`` when the condition's ``kind`` is ``"wall"``.
    """
    text, node = "", document
    for part in parts:
        if isinstance(node, BaseModel):
            node = node.model_dump(by_alias=True)
        if (
            isinstance(node, dict)
            and part not in node
            and any(node.get(tag) == part for tag in _TAGS)
        ):
            continue
        if isinstance(part, int):
            text += f"[{part}]"
            node = node[part] if isinstance(node, list | tuple) and part < len(node) else None
        else:
            text += f".{part}" if text else part
            node = node.get(part) if isinstance(node, dict) else None
    return text


def diagnostics_from(
    error: ValidationError, location: Location, document: object
) -> list[Diagnostic]:
    """Coded diagnostics of a pydantic error: ``autan.E####`` types keep their code, range
    violations are ``E0304``, cross-field problems expand to one diagnostic each, the rest
    is ``E0003``."""
    found = []
    for item in error.errors(include_url=False):
        path = document_path(item["loc"], document)
        if item["type"] == "autan.consistency":
            problems = (item.get("ctx") or {})["problems"]
        elif item["type"].startswith("autan."):
            problems = [
                (item["type"].removeprefix("autan."), path, f"{path or 'case'}: {item['msg']}")
            ]
        elif item["type"] in _RANGE_ERRORS:
            problems = [("E0304", path, f"{path}: {item['msg']}")]
        else:
            problems = [("E0003", path, f"{path or 'case'}: {item['msg']}")]
        for code, where, message in problems:
            full = ".".join(p for p in (location.path, where) if p) or None
            found.append(
                diagnostic(code, message, location=location.model_copy(update={"path": full}))
            )
    return found


class Spec(BaseModel):
    """Base of every case model: immutable, strict about unknown keys, coded errors."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True, revalidate_instances="always"
    )

    def __init__(self, /, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as error:
            raise CaseError(diagnostics_from(error, _caller_location(), data)) from None

    # Nested validation must not run this conversion, only direct construction does:
    # pydantic calls a model's __init__ during validation unless it is marked as the base one.
    __init__.__pydantic_base_init__ = True  # type: ignore[attr-defined]


# --- materials ---------------------------------------------------------------------------


class Sutherland(Spec):
    law: Literal["sutherland"] = "sutherland"
    mu_ref: DynamicViscosity
    t_ref: Temperature
    s: Temperature


class ConstantViscosity(Spec):
    law: Literal["constant"] = "constant"
    mu: DynamicViscosity


Viscosity = Annotated[Sutherland | ConstantViscosity, Field(discriminator="law")]


class IdealGas(Spec):
    """A calorically perfect gas: molar mass, ratio of specific heats, Prandtl number, viscosity."""

    kind: Literal["ideal-gas"] = "ideal-gas"
    name: Name
    molar_mass: MolarMass
    gamma: Gamma
    prandtl: PositiveNumber
    viscosity: Viscosity


class ConstantDensityFluid(Spec):
    """A fluid of constant density and viscosity, for incompressible flow."""

    kind: Literal["constant-density-fluid"] = "constant-density-fluid"
    name: Name
    density: Density
    viscosity: DynamicViscosity


class Solid(Spec):
    """A solid for conjugate heat transfer: density, specific heat, thermal conductivity."""

    kind: Literal["solid"] = "solid"
    name: Name
    density: Density
    specific_heat: SpecificHeat
    conductivity: Conductivity


FluidMaterial = Annotated[IdealGas | ConstantDensityFluid, Field(discriminator="kind")]


# --- physics -----------------------------------------------------------------------------


class Inviscid(Spec):
    """Euler equations."""

    model: Literal["inviscid"] = "inviscid"


class Laminar(Spec):
    """Navier-Stokes equations without a turbulence model."""

    model: Literal["laminar"] = "laminar"


class SA(Spec):
    """Spalart-Allmaras, negative variant (SA-neg); freestream and inlets get nu~ = ratio * nu."""

    model: Literal["sa"] = "sa"
    nu_tilde_ratio: PositiveNumber = 3.0


class SST(Spec):
    """Menter k-omega SST; freestream and inlets get k from ``intensity``, omega from mu_t/mu."""

    model: Literal["sst"] = "sst"
    intensity: PositiveNumber = 0.001
    viscosity_ratio: PositiveNumber = 0.009


class DDES(Spec):
    """Delayed detached-eddy simulation on an SA or SST base model; needs an unsteady run."""

    model: Literal["ddes"] = "ddes"
    base: Annotated[SA | SST, Field(discriminator="model")] = Field(default_factory=SST)


class WMLES(Spec):
    """Wall-modelled large-eddy simulation; needs an unsteady run."""

    model: Literal["wmles"] = "wmles"
    wall_model: Literal["equilibrium"] = "equilibrium"


Turbulence = Annotated[Inviscid | Laminar | SA | SST | DDES | WMLES, Field(discriminator="model")]
Precision = Literal["fp32+acc64", "fp64"]
# What bits a run guarantees to repeat (case-schema.md#physics).
Reproducibility = Literal["none", "run", "ranks"]


def _air() -> IdealGas:
    from autan.gases import AIR  # noqa: PLC0415 -- autan.gases imports this module

    return AIR.model_copy(deep=True)


class Compressible(Spec):
    """Density-based compressible flow; ``gas`` is the default medium of fluid zones."""

    kind: Literal["compressible"] = "compressible"
    turbulence: Turbulence
    precision: Precision = "fp32+acc64"
    reproducibility: Reproducibility = "run"
    gas: IdealGas = Field(default_factory=_air)


class Incompressible(Spec):
    """Pressure-based incompressible flow; ``fluid`` is the default medium of fluid zones."""

    kind: Literal["incompressible"] = "incompressible"
    turbulence: Turbulence
    precision: Precision = "fp32+acc64"
    reproducibility: Reproducibility = "run"
    fluid: ConstantDensityFluid


Physics = Annotated[Compressible | Incompressible, Field(discriminator="kind")]


class Freestream(Spec):
    mach: PositiveNumber | None = None
    velocity: Speed | None = None
    aoa: Angle = Field(default_factory=lambda: 0.0 * deg)
    sideslip: Angle = Field(default_factory=lambda: 0.0 * deg)
    temperature: Temperature | None = None
    pressure: Pressure | None = None
    re: PositiveNumber | None = None
    ref_length: Length | None = None


# --- time --------------------------------------------------------------------------------


class Steady(Spec):
    """Local time stepping until every residual drops ``residual_orders`` orders, at most
    ``max_iterations`` iterations; an implicit physics solves each step's linear system with
    ``linear_solver``."""

    kind: Literal["steady"] = "steady"
    max_iterations: Count = 10000
    residual_orders: PositiveNumber = 8.0
    linear_solver: Literal["multigrid", "gauss-seidel"] = "multigrid"


class Unsteady(Spec):
    """Time-accurate run of ``steps`` steps, with either a fixed ``dt`` or a ``cfl`` target."""

    kind: Literal["unsteady"] = "unsteady"
    steps: Count
    dt: Duration | None = None
    cfl: PositiveNumber | None = None
    average_from: Index | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.dt is None) == (self.cfl is None):
            fail("E0003", "give exactly one of dt or cfl")
        if self.average_from is not None and self.average_from >= self.steps:
            fail("E0304", "average_from must be < steps")
        return self


Time = Annotated[Steady | Unsteady, Field(discriminator="kind")]


# --- boundaries, cell zones, monitors ----------------------------------------------------


class Wall(Spec):
    kind: Literal["wall"] = "wall"
    temperature: Temperature | None = None


class Slip(Spec):
    kind: Literal["slip"] = "slip"


class Symmetry(Spec):
    kind: Literal["symmetry"] = "symmetry"


class PointVortex(Spec):
    """The point-vortex correction of a 2D farfield: the far field of a vortex at ``center``
    whose circulation is that of the lift on the wall and slip zones matching ``surfaces``."""

    surfaces: Pattern
    center: tuple[Coordinate, Coordinate, Coordinate]


class Farfield(Spec):
    kind: Literal["farfield"] = "farfield"
    point_vortex: PointVortex | None = None


class Inlet(Spec):
    kind: Literal["inlet"] = "inlet"
    total_pressure: Pressure
    total_temperature: Temperature


class VelocityInlet(Spec):
    kind: Literal["velocity-inlet"] = "velocity-inlet"
    velocity: Speed


class Outlet(Spec):
    kind: Literal["outlet"] = "outlet"
    static_pressure: Pressure


class OracleState(Spec):
    """The state of every boundary face from an ``autan.face-states/1`` file exported for
    this mesh (docs/contracts/face-states.md), fixed by its SHA-256."""

    kind: Literal["oracle-state"] = "oracle-state"
    path: Name
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]


Condition = Annotated[
    Wall | Slip | Symmetry | Farfield | Inlet | VelocityInlet | Outlet | OracleState,
    Field(discriminator="kind"),
]


class BoundaryRule(Spec):
    zones: Pattern
    condition: Condition


class State(Spec):
    """Initial state of a fluid zone: pressure (absolute for compressible flow, gauge for
    incompressible), temperature (compressible only) and velocity in mesh axes."""

    pressure: Annotated[pint.Quantity, *quantity("[pressure]")]
    temperature: Temperature | None = None
    velocity: tuple[Velocity, Velocity, Velocity]


class Fluid(Spec):
    kind: Literal["fluid"] = "fluid"
    material: FluidMaterial
    initial: State | None = None


class SolidZone(Spec):
    kind: Literal["solid"] = "solid"
    material: Solid


Medium = Annotated[Fluid | SolidZone, Field(discriminator="kind")]


class CellZoneRule(Spec):
    zones: Pattern
    medium: Medium


class Forces(Spec):
    """Integrated forces and moments over the boundary zones matching ``surfaces``."""

    kind: Literal["forces"] = "forces"
    name: Name = "forces"
    surfaces: Pattern
    every: Count = 1
    ref_area: Area | None = None
    moment_center: tuple[Coordinate, Coordinate, Coordinate] = Field(
        default_factory=lambda: (0.0 * m, 0.0 * m, 0.0 * m)
    )


Monitor = Annotated[Forces, Field(discriminator="kind")]


# --- verification -----------------------------------------------------------------------


class Manufactured(Spec):
    """A manufactured solution: an ``autan.mms/1`` file whose source term the solver adds
    and whose exact fields give the data of every boundary zone."""

    path: Name
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]


class Verification(Spec):
    manufactured: Manufactured


class StateFile(Spec):
    """The initial state of every fluid cell, from an ``autan.initial-state/1`` file written
    for the case's mesh (docs/contracts/initial-state.md), fixed by its SHA-256."""

    path: Name
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]


# --- case --------------------------------------------------------------------------------


class CaseSpec(Spec):
    """A complete case document, ``autan.case/1``."""

    schema_: Literal["autan.case/1"] = Field(default="autan.case/1", alias="schema")
    name: Name
    mesh: Name
    physics: Physics
    freestream: Freestream
    time: Time = Field(default_factory=Steady)
    boundaries: tuple[BoundaryRule, ...] = ()
    cell_zones: tuple[CellZoneRule, ...] = ()
    monitors: tuple[Monitor, ...] = ()
    verification: Verification | None = None
    initial_state: StateFile | None = None

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if isinstance(self.physics, Compressible) and self.freestream.temperature is None:
            filled = self.freestream.model_copy(update={"temperature": 288.15 * K})
            object.__setattr__(self, "freestream", filled)
        problems = consistency(self)
        if problems:
            fail("consistency", "{problems}", problems=problems)
        return self

    @classmethod
    def from_document(cls, document: dict[str, Any], location: Location) -> CaseSpec:
        """Validate a JSON/TOML tree; every problem is reported as a coded diagnostic."""
        version = document.get("schema")
        if version != SCHEMA_ID:
            where = location.model_copy(update={"path": "schema"})
            message = f"schema is {version!r}, expected {SCHEMA_ID!r}"
            raise CaseError([diagnostic("E0004", message, location=where)])
        try:
            return cls.model_validate(document)
        except ValidationError as error:
            raise CaseError(diagnostics_from(error, location, document)) from None

    def document(self) -> dict[str, Any]:
        """The case as a JSON-compatible tree, defaults filled in."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def hash(self) -> str:
        """SHA-256 of the RFC 8785 canonical JSON (docs/contracts/case-schema.md#case-hash)."""
        return hashlib.sha256(rfc8785.dumps(self.document())).hexdigest()


def consistency(spec: CaseSpec) -> list[tuple[str, str, str]]:
    """Rules spanning several fields, as (code, document path, message)."""
    found: list[tuple[str, str, str]] = []
    fs, physics = spec.freestream, spec.physics
    if isinstance(physics, Compressible):
        if fs.mach is None:
            found.append(("E0306", "freestream.mach", "compressible freestream needs mach"))
        if fs.velocity is not None:
            found.append(
                ("E0306", "freestream.velocity", "compressible freestream takes mach, not velocity")
            )
        if (fs.pressure is None) == (fs.re is None):
            found.append(
                (
                    "E0306",
                    "freestream",
                    "compressible freestream needs exactly one of pressure or re",
                )
            )
        if fs.re is not None and fs.ref_length is None:
            found.append(("E0306", "freestream.ref_length", "re needs ref_length"))
    else:
        if fs.velocity is None:
            found.append(
                ("E0306", "freestream.velocity", "incompressible freestream needs velocity")
            )
        found.extend(
            ("E0306", f"freestream.{key}", f"incompressible freestream takes no {key}")
            for key in ("mach", "re", "pressure", "temperature")
            if getattr(fs, key) is not None
        )
    compressible = isinstance(physics, Compressible)
    for index, rule in enumerate(spec.cell_zones):
        state = rule.medium.initial if isinstance(rule.medium, Fluid) else None
        if state is None:
            continue
        where = f"cell_zones[{index}].medium.initial"
        if compressible and (state.temperature is None or si(state.pressure, "Pa") <= 0):
            found.append(
                (
                    "E0507",
                    where,
                    "a compressible initial state needs a temperature and a pressure > 0",
                )
            )
        if not compressible and state.temperature is not None:
            found.append(("E0507", where, "an incompressible initial state takes no temperature"))
    if spec.initial_state is not None:
        zones = [
            f"cell_zones[{i}].medium.initial"
            for i, rule in enumerate(spec.cell_zones)
            if isinstance(rule.medium, Fluid) and rule.medium.initial is not None
        ]
        if not compressible or not isinstance(physics.turbulence, Inviscid | Laminar):
            found.append(
                (
                    "E0512",
                    "initial_state",
                    "an initial-state file needs compressible inviscid or laminar physics",
                )
            )
        found.extend(
            ("E0512", where, "a zone takes no initial state with an initial-state file")
            for where in zones
        )
    if spec.verification is not None and spec.boundaries:
        found.append(
            (
                "E0416",
                "boundaries",
                "a manufactured solution gives every boundary its data; remove the boundary rules",
            )
        )
    if spec.verification is not None and isinstance(spec.time, Unsteady):
        found.append(
            ("E0417", "time", "manufactured solutions are steady: verification needs a steady time")
        )
    if isinstance(physics.turbulence, DDES | WMLES) and not isinstance(spec.time, Unsteady):
        found.append(
            ("E0305", "time", f"{physics.turbulence.model} needs case.time = autan.Unsteady(...)")
        )
    return found


def require_tags(schema: dict[str, Any]) -> None:
    """Discriminators and ``schema`` are required in documents, even where Python has defaults."""
    if schema.get("type") == "object" and "properties" in schema:
        tags = [k for k, v in schema["properties"].items() if "const" in v or k == "schema"]
        schema["required"] = sorted(set(schema.get("required", [])) | set(tags))
    for value in schema.values():
        if isinstance(value, dict):
            require_tags(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    require_tags(item)


def json_schema() -> dict[str, Any]:
    """The published JSON Schema of ``autan.case/1``, cross-field rules included."""
    schema = CaseSpec.model_json_schema(by_alias=True, mode="validation")
    require_tags(schema)

    def physics(kind: str) -> dict[str, Any]:
        return {"properties": {"physics": {"properties": {"kind": {"const": kind}}}}}

    def freestream(rule: dict[str, Any]) -> dict[str, Any]:
        return {"properties": {"freestream": rule}}

    def absent(*keys: str) -> dict[str, Any]:
        return {"not": {"anyOf": [{"required": [k]} for k in keys]}}

    def zone_states(rule: dict[str, Any]) -> dict[str, Any]:
        initial = {"properties": {"initial": {"anyOf": [{"type": "null"}, rule]}}}
        return {"properties": {"cell_zones": {"items": {"properties": {"medium": initial}}}}}

    schema["allOf"] = [
        {
            "if": {"required": ["initial_state"]},
            "then": {
                "properties": {
                    "physics": {
                        "properties": {
                            "kind": {"const": "compressible"},
                            "turbulence": {
                                "properties": {"model": {"enum": ["inviscid", "laminar"]}}
                            },
                        }
                    },
                    **zone_states({"not": {}})["properties"],
                },
            },
        },
        {"if": physics("compressible"), "then": zone_states({"required": ["temperature"]})},
        {
            "if": physics("incompressible"),
            "then": zone_states({"not": {"required": ["temperature"]}}),
        },
        {
            "if": {
                "required": ["verification"],
                "properties": {"verification": {"type": "object"}},
            },
            "then": {"properties": {"boundaries": {"maxItems": 0}}},
        },
        {
            "if": physics("compressible"),
            "then": freestream(
                {
                    "required": ["mach"],
                    **absent("velocity"),
                    "oneOf": [
                        {"required": ["pressure"], **absent("re")},
                        {"required": ["re", "ref_length"], **absent("pressure")},
                    ],
                }
            ),
        },
        {
            "if": physics("incompressible"),
            "then": freestream(
                {"required": ["velocity"], **absent("mach", "re", "pressure", "temperature")}
            ),
        },
        {
            "if": {
                "properties": {
                    "physics": {
                        "properties": {
                            "turbulence": {"properties": {"model": {"enum": ["ddes", "wmles"]}}}
                        }
                    }
                }
            },
            "then": {
                "required": ["time"],
                "properties": {"time": {"properties": {"kind": {"const": "unsteady"}}}},
            },
        },
    ]
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_ID, **schema}
