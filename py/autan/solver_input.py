"""``autan.solver-input/1``: the resolved, SI-only input of the solver runtime ``autan-run``.

The SDK builds it after a plan without errors (docs/contracts/solver-input.md): every
boundary and cell zone resolved, every quantity converted to SI, the freestream state
closed. The runtime applies no unit, default or resolution rule of its own.
"""

from __future__ import annotations

import math
import tomllib
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from autan.errors import CaseError, Location
from autan.meshinfo import (
    PairZones,
    Translation,
    shared_zones,
)
from autan.results import (
    WOKEN_SCHEMA,
    Devices,
    woken_problem,
)
from autan.schema import (
    DDES,
    SA,
    SST,
    WMLES,
    CaseSpec,
    Compressible,
    ConstantDensityFluid,
    ConstantViscosity,
    Farfield,
    Fluid,
    IdealGas,
    Medium,
    OracleState,
    Solid,
    SolidZone,
    Steady,
    Sutherland,
    require_tags,
)
from autan.units import parse, si

if TYPE_CHECKING:
    from autan.case import Case
    from autan.meshinfo import MeshInfo
    from autan.restart import Start
    from autan.schema import State

SCHEMA_ID = "autan.solver-input/1"
GAS_CONSTANT = 8.314462618  # J/(mol K), CODATA 2018 exact value

Number = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Vector = tuple[Number, Number, Number]
_MEDIUM: TypeAdapter[Fluid | SolidZone] = TypeAdapter(Medium)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# Values each boundary kind requires; every other value is null ("wall" may have temperature).
BOUNDARY_VALUES: dict[str, frozenset[str]] = {
    "wall": frozenset(),
    "slip": frozenset(),
    "symmetry": frozenset(),
    "farfield": frozenset(),
    "inlet": frozenset({"total_pressure", "total_temperature"}),
    "velocity-inlet": frozenset({"velocity"}),
    "outlet": frozenset({"static_pressure"}),
    "manufactured": frozenset(),
    "oracle-state": frozenset({"states"}),
}
OPTIONAL_VALUES: dict[str, frozenset[str]] = {
    "wall": frozenset({"temperature"}),
    "farfield": frozenset({"point_vortex"}),
}
BOUNDARY_KEYS = (
    "temperature",
    "total_pressure",
    "total_temperature",
    "velocity",
    "static_pressure",
    "states",
    "point_vortex",
)
# The face states carry no turbulence variable (face-states.md).
ORACLE_MODELS = ("inviscid", "laminar")
# Turbulence parameters each (model, base) requires; every other parameter is null.
TURBULENCE_VALUES: dict[tuple[str, str | None], frozenset[str]] = {
    ("inviscid", None): frozenset(),
    ("laminar", None): frozenset(),
    ("sa", None): frozenset({"nu_tilde_ratio"}),
    ("sst", None): frozenset({"intensity", "viscosity_ratio"}),
    ("ddes", "sa"): frozenset({"base", "nu_tilde_ratio"}),
    ("ddes", "sst"): frozenset({"base", "intensity", "viscosity_ratio"}),
    ("wmles", None): frozenset({"wall_model"}),
}
TURBULENCE_KEYS = ("base", "wall_model", "nu_tilde_ratio", "intensity", "viscosity_ratio")


class CaseRef(_Strict):
    name: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]
    toml: str


def _absolute(path: str) -> str:
    if "\0" in path or not Path(path).is_absolute():
        message = f"{path!r} is not an absolute path"
        raise ValueError(message)
    return path


AbsolutePath = Annotated[
    str,
    Field(min_length=1, pattern=r"^(/|[A-Za-z]:[\\/])[^\u0000]*$"),
    AfterValidator(_absolute),
]


class MeshRef(_Strict):
    path: AbsolutePath
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]


class SutherlandLaw(_Strict):
    law: Literal["sutherland"]
    mu_ref: Positive
    t_ref: Positive
    s: Positive


class ConstantLaw(_Strict):
    law: Literal["constant"]
    mu: Positive


class Gas(_Strict):
    kind: Literal["ideal-gas"]
    name: str
    gas_constant: Positive
    gamma: Annotated[float, Field(gt=1)]
    prandtl: Positive
    viscosity: Annotated[SutherlandLaw | ConstantLaw, Field(discriminator="law")]


class Liquid(_Strict):
    kind: Literal["constant-density-fluid"]
    name: str
    density: Positive
    viscosity: Positive


class SolidMedium(_Strict):
    kind: Literal["solid"]
    name: str
    density: Positive
    specific_heat: Positive
    conductivity: Positive


MediumInput = Annotated[Gas | Liquid | SolidMedium, Field(discriminator="kind")]


class TurbulenceInput(_Strict):
    model: Literal["inviscid", "laminar", "sa", "sst", "ddes", "wmles"]
    base: Literal["sa", "sst"] | None
    wall_model: Literal["equilibrium"] | None
    nu_tilde_ratio: Positive | None
    intensity: Positive | None
    viscosity_ratio: Positive | None

    @model_validator(mode="after")
    def _values_match_model(self) -> Self:
        expected = TURBULENCE_VALUES.get((self.model, self.base))
        given = {k for k in TURBULENCE_KEYS if getattr(self, k) is not None}
        if expected is None or given != expected:
            message = f"turbulence {self.model!r} takes exactly {sorted(expected or ())}"
            raise ValueError(message)
        return self


class PhysicsInput(_Strict):
    kind: Literal["compressible", "incompressible"]
    turbulence: TurbulenceInput


class FreestreamInput(_Strict):
    density: Positive
    pressure: Number
    temperature: Positive | None
    velocity: Vector
    speed: Positive
    viscosity: Positive
    mach: Positive | None
    re: Positive | None
    ref_length: Positive | None
    aoa: Number
    sideslip: Number
    nu_tilde: Positive | None
    k: Positive | None
    omega: Positive | None


class StatesRef(_Strict):
    path: AbsolutePath
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]


Coordinate = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class PointVortexInput(_Strict):
    zones: Annotated[tuple[str, ...], Field(min_length=1)]
    center: tuple[Coordinate, Coordinate, Coordinate]


class BoundaryInput(_Strict):
    zone: str
    kind: Literal[
        "wall",
        "slip",
        "symmetry",
        "farfield",
        "inlet",
        "velocity-inlet",
        "outlet",
        "manufactured",
        "oracle-state",
    ]
    temperature: Positive | None
    total_pressure: Positive | None
    total_temperature: Positive | None
    velocity: Positive | None
    static_pressure: Positive | None
    states: StatesRef | None
    point_vortex: PointVortexInput | None

    @model_validator(mode="after")
    def _values_match_kind(self) -> Self:
        required = BOUNDARY_VALUES[self.kind]
        given = {k for k in BOUNDARY_KEYS if getattr(self, k) is not None}
        allowed = required | OPTIONAL_VALUES.get(self.kind, frozenset())
        if not required <= given <= allowed:
            message = f"boundary {self.kind!r} takes {sorted(required)} (and {sorted(allowed)})"
            raise ValueError(message)
        return self


class FluidInitial(_Strict):
    kind: Literal["fluid"]
    density: Positive
    pressure: Number
    temperature: Positive | None
    velocity: Vector
    nu_tilde: Positive | None
    k: Positive | None
    omega: Positive | None


class SolidInitial(_Strict):
    kind: Literal["solid"]
    temperature: Positive


class CellZoneInput(_Strict):
    zone: str
    medium: MediumInput
    initial: Annotated[FluidInitial | SolidInitial, Field(discriminator="kind")]

    @model_validator(mode="after")
    def _initial_matches_medium(self) -> Self:
        if (self.medium.kind == "solid") != (self.initial.kind == "solid"):
            message = "a solid zone has a solid initial state, a fluid zone a fluid one"
            raise ValueError(message)
        return self


class SteadyInput(_Strict):
    kind: Literal["steady"]
    max_iterations: Annotated[int, Field(ge=1)]
    residual_orders: Positive
    linear_solver: Literal["multigrid", "gauss-seidel", "lines"]


class UnsteadyInput(_Strict):
    kind: Literal["unsteady"]
    steps: Annotated[int, Field(ge=1)]
    dt: Positive | None
    cfl: Positive | None
    average_from: Annotated[int, Field(ge=0)] | None

    @model_validator(mode="after")
    def _one_step_rule(self) -> Self:
        if (self.dt is None) == (self.cfl is None):
            message = "give exactly one of dt or cfl"
            raise ValueError(message)
        if self.average_from is not None and self.average_from >= self.steps:
            message = "average_from must be < steps"
            raise ValueError(message)
        return self


class ForcesInput(_Strict):
    kind: Literal["forces"]
    name: str
    every: Annotated[int, Field(ge=1)]
    zones: Annotated[tuple[str, ...], Field(min_length=1)]
    ref_area: Positive | None
    ref_length: Positive | None
    moment_center: Vector


class FileInput(_Strict):
    path: AbsolutePath
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]


class PeriodicInput(_Strict):
    zones: PairZones
    transform: Translation


class StartInput(_Strict):
    run: AbsolutePath
    run_id: Annotated[str, Field(pattern=r"^run-\d{8}T\d{6}Z-[0-9a-f]{8}$")]
    identity: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]
    mode: Literal["continue", "warm"]
    iterations: Annotated[int, Field(strict=True, ge=1)]
    time: Number | None
    residual_reference: dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]] | None
    woken: list[str] | None
    solution_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]

    @model_validator(mode="after")
    def _woken_names(self) -> Self:
        problem = woken_problem(self.residual_reference, self.woken)
        if problem:
            raise ValueError(problem)
        return self


class SolverInput(_Strict):
    """The input document of ``autan-run`` (docs/contracts/solver-input.md); SI throughout."""

    schema_: Literal["autan.solver-input/1"] = Field(alias="schema")
    case: CaseRef
    mesh: MeshRef
    precision: Literal["fp32+acc64", "fp64"]
    deterministic: bool
    devices: Devices
    physics: PhysicsInput
    freestream: FreestreamInput
    boundaries: tuple[BoundaryInput, ...]
    periodic: tuple[PeriodicInput, ...]
    cell_zones: tuple[CellZoneInput, ...]
    time: Annotated[SteadyInput | UnsteadyInput, Field(discriminator="kind")]
    monitors: tuple[ForcesInput, ...]
    verification: FileInput | None
    initial_state: FileInput | None
    start: StartInput | None

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        turbulence, fs = self.physics.turbulence, self.freestream
        if turbulence.model in ("ddes", "wmles") and self.time.kind != "unsteady":
            message = f"{turbulence.model} needs an unsteady time"
            raise ValueError(message)
        if shared_zones(pair.zones for pair in self.periodic):
            message = "a zone is in two periodic pairs"
            raise ValueError(message)
        periodic = {zone for pair in self.periodic for zone in pair.zones}
        if periodic & {b.zone for b in self.boundaries}:
            message = "a periodic zone takes no boundary condition"
            raise ValueError(message)
        walls = {b.zone for b in self.boundaries if b.kind in ("wall", "slip")}
        if any(not set(b.point_vortex.zones) <= walls for b in self.boundaries if b.point_vortex):
            message = "a point vortex integrates over wall and slip zones only"
            raise ValueError(message)
        if periodic & {zone for m in self.monitors for zone in m.zones}:
            message = "a monitor integrates over no periodic zone"
            raise ValueError(message)
        sa = turbulence.nu_tilde_ratio is not None
        sst = turbulence.intensity is not None
        if (fs.nu_tilde is not None) != sa or (fs.k is not None, fs.omega is not None) != (
            sst,
            sst,
        ):
            message = "freestream turbulence values do not match the turbulence model"
            raise ValueError(message)
        try:
            case = CaseSpec.from_document(tomllib.loads(self.case.toml), Location(path="case.toml"))
        except (CaseError, ValueError) as error:
            message = f"case.toml is not a valid case: {error}"
            raise ValueError(message) from None
        if case.hash() != self.case.sha256:
            message = "case.sha256 is not the hash of case.toml"
            raise ValueError(message)
        initial = case.initial_state.sha256 if case.initial_state else None
        if initial != (self.initial_state.sha256 if self.initial_state else None):
            message = "initial_state differs from the case's initial_state"
            raise ValueError(message)
        if self.initial_state is not None and self.start is not None:
            message = "an initial-state file and a start exclude each other"
            raise ValueError(message)
        if self.initial_state is not None and (self.physics.kind, turbulence.model) not in {
            ("compressible", m) for m in ORACLE_MODELS
        }:
            message = "an initial-state file needs compressible inviscid or laminar physics"
            raise ValueError(message)
        recorded = case.verification.manufactured.sha256 if case.verification else None
        if recorded != (self.verification.sha256 if self.verification else None):
            message = "verification differs from the case's verification"
            raise ValueError(message)
        if any(b.kind == "oracle-state" for b in self.boundaries) and (
            self.physics.kind,
            turbulence.model,
        ) not in {("compressible", m) for m in ORACLE_MODELS}:
            message = "oracle-state needs compressible inviscid or laminar physics"
            raise ValueError(message)
        for boundary in self.boundaries:
            rules = [r.condition for r in case.boundaries if fnmatchcase(boundary.zone, r.zones)]
            oracle = rules[0] if len(rules) == 1 and isinstance(rules[0], OracleState) else None
            if (boundary.states and boundary.states.sha256) != (oracle and oracle.sha256):
                message = f"boundary {boundary.zone!r}: states differ from the case's rule"
                raise ValueError(message)
        manufactured = {b.kind == "manufactured" for b in self.boundaries}
        if manufactured and manufactured != {self.verification is not None}:
            message = "with verification every boundary is manufactured; without, none is"
            raise ValueError(message)
        compressible = self.physics.kind == "compressible"
        if (fs.temperature is not None) != compressible:
            message = "freestream temperature is given exactly for compressible physics"
            raise ValueError(message)
        fluid = "ideal-gas" if compressible else "constant-density-fluid"
        if any(z.medium.kind not in (fluid, "solid") for z in self.cell_zones):
            message = f"{self.physics.kind} fluid zones need a {fluid} medium"
            raise ValueError(message)
        return self

    def document(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


def viscosity(gas: IdealGas, temperature: float) -> float:
    """Dynamic viscosity of ``gas`` at ``temperature`` (K), in Pa s."""
    law = gas.viscosity
    if isinstance(law, ConstantViscosity):
        return si(law.mu, "Pa * s")
    assert isinstance(law, Sutherland)
    t_ref, s = si(law.t_ref, "K"), si(law.s, "K")
    # Sutherland (1893): mu = mu_ref (T / T_ref)^(3/2) (T_ref + S) / (T + S)
    mu_ref = si(law.mu_ref, "Pa * s")
    return mu_ref * math.pow(temperature / t_ref, 1.5) * (t_ref + s) / (temperature + s)


def gas_constant(gas: IdealGas) -> float:
    """Specific gas constant R = R_u / M, in J/(kg K)."""
    return GAS_CONSTANT / si(gas.molar_mass, "kg / mol")


def freestream_state(spec: CaseSpec) -> FreestreamInput:
    """Close the freestream state of case-schema.md#freestream in SI."""
    fs, physics = spec.freestream, spec.physics
    alpha, beta = si(fs.aoa, "rad"), si(fs.sideslip, "rad")
    direction = (
        math.cos(alpha) * math.cos(beta),
        0.0 - math.sin(beta),
        math.sin(alpha) * math.cos(beta),
    )
    temperature: float | None
    if isinstance(physics, Compressible):
        assert fs.temperature is not None
        assert fs.mach is not None
        gas = physics.gas
        temperature = si(fs.temperature, "K")
        r = gas_constant(gas)
        speed = fs.mach * math.sqrt(gas.gamma * r * temperature)
        mu = viscosity(gas, temperature)
        if fs.re is not None:
            assert fs.ref_length is not None
            density = fs.re * mu / (speed * si(fs.ref_length, "m"))
            pressure = density * r * temperature
        else:
            assert fs.pressure is not None
            pressure = si(fs.pressure, "Pa")
            density = pressure / (r * temperature)
    else:
        assert fs.velocity is not None
        temperature, pressure = None, 0.0
        density = si(physics.fluid.density, "kg / m**3")
        mu = si(physics.fluid.viscosity, "Pa * s")
        speed = si(fs.velocity, "m / s")
    turbulence = turbulence_input(spec)
    nu_tilde = k = omega = None
    if turbulence.nu_tilde_ratio is not None:
        nu_tilde = turbulence.nu_tilde_ratio * mu / density
    if turbulence.intensity is not None and turbulence.viscosity_ratio is not None:
        k = 1.5 * (turbulence.intensity * speed) ** 2
        omega = density * k / (turbulence.viscosity_ratio * mu)
    return FreestreamInput(
        density=density,
        pressure=pressure,
        temperature=temperature,
        velocity=(speed * direction[0], speed * direction[1], speed * direction[2]),
        speed=speed,
        viscosity=mu,
        mach=fs.mach,
        re=fs.re,
        ref_length=si(fs.ref_length, "m") if fs.ref_length is not None else None,
        aoa=alpha,
        sideslip=beta,
        nu_tilde=nu_tilde,
        k=k,
        omega=omega,
    )


def turbulence_input(spec: CaseSpec) -> TurbulenceInput:
    turbulence = spec.physics.turbulence
    base = turbulence.base if isinstance(turbulence, DDES) else turbulence
    return TurbulenceInput(
        model=turbulence.model,
        base=turbulence.base.model if isinstance(turbulence, DDES) else None,
        wall_model=turbulence.wall_model if isinstance(turbulence, WMLES) else None,
        nu_tilde_ratio=base.nu_tilde_ratio if isinstance(base, SA) else None,
        intensity=base.intensity if isinstance(base, SST) else None,
        viscosity_ratio=base.viscosity_ratio if isinstance(base, SST) else None,
    )


def medium_input(medium: Fluid | SolidZone) -> MediumInput:
    material = medium.material
    if isinstance(material, Solid):
        return SolidMedium(
            kind="solid",
            name=material.name,
            density=si(material.density, "kg / m**3"),
            specific_heat=si(material.specific_heat, "J / kg / K"),
            conductivity=si(material.conductivity, "W / m / K"),
        )
    if isinstance(material, ConstantDensityFluid):
        return Liquid(
            kind="constant-density-fluid",
            name=material.name,
            density=si(material.density, "kg / m**3"),
            viscosity=si(material.viscosity, "Pa * s"),
        )
    law = material.viscosity
    viscosity_law: SutherlandLaw | ConstantLaw = (
        SutherlandLaw(
            law="sutherland",
            mu_ref=si(law.mu_ref, "Pa * s"),
            t_ref=si(law.t_ref, "K"),
            s=si(law.s, "K"),
        )
        if isinstance(law, Sutherland)
        else ConstantLaw(law="constant", mu=si(law.mu, "Pa * s"))
    )
    return Gas(
        kind="ideal-gas",
        name=material.name,
        gas_constant=gas_constant(material),
        gamma=material.gamma,
        prandtl=material.prandtl,
        viscosity=viscosity_law,
    )


def zone_viscosity(medium: Gas | Liquid, temperature: float | None) -> float:
    """Dynamic viscosity of a resolved fluid medium, in Pa s."""
    if isinstance(medium, Liquid):
        return medium.viscosity
    assert temperature is not None
    law = medium.viscosity
    if isinstance(law, ConstantLaw):
        return law.mu
    # Sutherland (1893): mu = mu_ref (T / T_ref)^(3/2) (T_ref + S) / (T + S)
    ratio = math.pow(temperature / law.t_ref, 1.5)
    return law.mu_ref * ratio * (law.t_ref + law.s) / (temperature + law.s)


def initial_state(
    medium: MediumInput,
    freestream: FreestreamInput,
    turbulence: TurbulenceInput,
    given: State | None = None,
) -> FluidInitial | SolidInitial:
    """Initial state of a cell zone (solver-input.md#initial-state): the zone's ``given``
    state, or the freestream pressure, temperature and velocity, closed in the zone's own
    medium. Turbulence values use the freestream speed as reference."""
    if isinstance(medium, SolidMedium):
        assert freestream.temperature is not None
        return SolidInitial(kind="solid", temperature=freestream.temperature)
    if given is None:
        pressure, temperature, velocity = (
            freestream.pressure,
            freestream.temperature,
            freestream.velocity,
        )
    else:
        pressure = si(given.pressure, "Pa")
        temperature = si(given.temperature, "K") if given.temperature is not None else None
        velocity = (
            si(given.velocity[0], "m / s"),
            si(given.velocity[1], "m / s"),
            si(given.velocity[2], "m / s"),
        )
    if isinstance(medium, Gas):
        assert temperature is not None
        density = pressure / (medium.gas_constant * temperature)
    else:
        density = medium.density
    nu = zone_viscosity(medium, temperature) / density
    k = (
        1.5 * (turbulence.intensity * freestream.speed) ** 2
        if turbulence.intensity is not None
        else None
    )
    return FluidInitial(
        kind="fluid",
        density=density,
        pressure=pressure,
        temperature=temperature,
        velocity=velocity,
        nu_tilde=(
            turbulence.nu_tilde_ratio * nu if turbulence.nu_tilde_ratio is not None else None
        ),
        k=k,
        omega=(
            k / (turbulence.viscosity_ratio * nu)
            if k is not None and turbulence.viscosity_ratio is not None
            else None
        ),
    )


def given_state(medium: Fluid | SolidZone) -> State | None:
    return medium.initial if isinstance(medium, Fluid) else None


def _si(condition: dict[str, Any], key: str, unit: str) -> float | None:
    return float(parse(condition[key], unit).to(unit).magnitude) if key in condition else None


def _point_vortex(
    spec: CaseSpec, resolution: Any, mesh_info: MeshInfo, plan: Any
) -> PointVortexInput | None:
    """The resolved point vortex of a farfield zone's rule, in SI."""
    from autan.plan import vortex_zones  # noqa: PLC0415 -- plan imports this module

    rule = spec.boundaries[resolution.rule] if resolution.rule is not None else None
    vortex = rule.condition.point_vortex if rule and isinstance(rule.condition, Farfield) else None
    if vortex is None:
        return None
    zones, _ = vortex_zones(vortex.surfaces, mesh_info, plan.boundaries)
    return PointVortexInput(
        zones=tuple(zones),
        center=(si(vortex.center[0], "m"), si(vortex.center[1], "m"), si(vortex.center[2], "m")),
    )


def build(
    case: Case,
    mesh_info: MeshInfo,
    *,
    devices: Devices,
    deterministic: bool,
    start: Start | None = None,
) -> SolverInput:
    """The solver input of ``case`` on ``mesh_info``; plan errors raise ``CaseError``."""
    plan = case.plan(mesh_info)
    if not plan.ok:
        raise CaseError(d for d in plan.diagnostics if d.severity == "error")
    spec = case.spec
    boundaries = []
    for resolution in plan.boundaries:
        condition = resolution.condition
        if condition is None or condition["kind"] == "periodic":
            continue
        boundaries.append(
            BoundaryInput(
                zone=resolution.zone,
                kind=condition["kind"],
                temperature=_si(condition, "temperature", "K"),
                total_pressure=_si(condition, "total_pressure", "Pa"),
                total_temperature=_si(condition, "total_temperature", "K"),
                velocity=_si(condition, "velocity", "m / s"),
                static_pressure=_si(condition, "static_pressure", "Pa"),
                states=(
                    StatesRef(path=str(case.resolve(condition["path"])), sha256=condition["sha256"])
                    if condition["kind"] == "oracle-state"
                    else None
                ),
                point_vortex=_point_vortex(spec, resolution, mesh_info, plan),
            )
        )
    freestream, turbulence = freestream_state(spec), turbulence_input(spec)
    media = [(z.zone, _MEDIUM.validate_python(z.medium)) for z in plan.cell_zones]
    cell_zones = tuple(
        CellZoneInput(
            zone=zone,
            medium=medium_input(rule),
            initial=initial_state(medium_input(rule), freestream, turbulence, given_state(rule)),
        )
        for zone, rule in media
    )
    periodic = {zone for pair in mesh_info.periodic for zone in pair.zones}
    zone_names = [b.zone for b in plan.boundaries if b.zone not in periodic]
    monitors = tuple(
        ForcesInput(
            kind="forces",
            name=m.name,
            every=m.every,
            zones=tuple(n for n in zone_names if fnmatchcase(n, m.surfaces)),
            ref_area=si(m.ref_area, "m**2") if m.ref_area is not None else None,
            ref_length=(
                si(spec.freestream.ref_length, "m")
                if spec.freestream.ref_length is not None
                else None
            ),
            moment_center=(
                si(m.moment_center[0], "m"),
                si(m.moment_center[1], "m"),
                si(m.moment_center[2], "m"),
            ),
        )
        for m in spec.monitors
    )
    time: SteadyInput | UnsteadyInput = (
        SteadyInput(
            kind="steady",
            max_iterations=spec.time.max_iterations,
            residual_orders=spec.time.residual_orders,
            linear_solver=spec.time.linear_solver,
        )
        if isinstance(spec.time, Steady)
        else UnsteadyInput(
            kind="unsteady",
            steps=spec.time.steps,
            dt=si(spec.time.dt, "s") if spec.time.dt is not None else None,
            cfl=spec.time.cfl,
            average_from=spec.time.average_from,
        )
    )
    return SolverInput(
        schema_=SCHEMA_ID,
        case=CaseRef(name=spec.name, sha256=spec.hash(), toml=case.to_toml()),
        mesh=MeshRef(path=str(case.mesh_path), sha256=mesh_info.source.sha256),
        precision=spec.physics.precision,
        deterministic=deterministic,
        devices=devices,
        physics=PhysicsInput(kind=spec.physics.kind, turbulence=turbulence),
        freestream=freestream,
        boundaries=tuple(boundaries),
        periodic=tuple(
            PeriodicInput(zones=p.zones, transform=p.transform) for p in mesh_info.periodic
        ),
        cell_zones=cell_zones,
        time=time,
        monitors=monitors,
        start=(
            StartInput(run=str(start.run), **start.info.model_dump()) if start is not None else None
        ),
        verification=(
            FileInput(
                path=str(case.resolve(spec.verification.manufactured.path)),
                sha256=spec.verification.manufactured.sha256,
            )
            if spec.verification is not None
            else None
        ),
        initial_state=(
            FileInput(
                path=str(case.resolve(spec.initial_state.path)), sha256=spec.initial_state.sha256
            )
            if spec.initial_state is not None
            else None
        ),
    )


def json_schema() -> dict[str, Any]:
    """The published JSON Schema of ``autan.solver-input/1``, with the value rules of
    BOUNDARY_VALUES and TURBULENCE_VALUES and the cross-field rules of the contract."""
    schema = SolverInput.model_json_schema(by_alias=True, mode="validation")
    require_tags(schema)
    defs = schema["$defs"]

    def exactly(
        keys: frozenset[str], all_keys: tuple[str, ...], optional: frozenset[str]
    ) -> dict[str, Any]:
        return {
            "properties": {
                k: ({"not": {"type": "null"}} if k in keys else {"type": "null"})
                for k in all_keys
                if k not in optional
            }
        }

    defs["BoundaryInput"]["allOf"] = [
        {
            "if": {"properties": {"kind": {"const": kind}}},
            "then": exactly(values, BOUNDARY_KEYS, OPTIONAL_VALUES.get(kind, frozenset())),
        }
        for kind, values in BOUNDARY_VALUES.items()
    ]
    defs["TurbulenceInput"]["anyOf"] = [
        {
            "properties": {
                "model": {"const": model},
                "base": {"type": "null"} if base is None else {"const": base},
                **exactly(values - {"base"}, TURBULENCE_KEYS[1:], frozenset())["properties"],
            }
        }
        for (model, base), values in TURBULENCE_VALUES.items()
    ]
    defs["StartInput"]["oneOf"] = [
        {"properties": {"residual_reference": {"type": "null"}, "woken": {"type": "null"}}},
        {"properties": {"residual_reference": {"type": "object"}, "woken": WOKEN_SCHEMA}},
    ]
    defs["UnsteadyInput"]["oneOf"] = [
        {"properties": {"dt": {"type": "number"}, "cfl": {"type": "null"}}},
        {"properties": {"dt": {"type": "null"}, "cfl": {"type": "number"}}},
    ]

    def boundary_kinds(rule: dict[str, Any]) -> dict[str, Any]:
        return {"properties": {"boundaries": {"items": {"properties": {"kind": rule}}}}}

    schema["allOf"] = [
        {
            "if": {"properties": {"initial_state": {"type": "object"}}},
            "then": {
                "properties": {
                    "start": {"type": "null"},
                    "physics": {
                        "properties": {
                            "kind": {"const": "compressible"},
                            "turbulence": {"properties": {"model": {"enum": list(ORACLE_MODELS)}}},
                        }
                    },
                }
            },
        },
        {
            "if": {
                "properties": {
                    "physics": {
                        "not": {
                            "properties": {
                                "kind": {"const": "compressible"},
                                "turbulence": {
                                    "properties": {"model": {"enum": list(ORACLE_MODELS)}}
                                },
                            }
                        }
                    }
                }
            },
            "then": boundary_kinds({"not": {"const": "oracle-state"}}),
        },
        {
            "if": {"properties": {"verification": {"type": "object"}}},
            "then": boundary_kinds({"const": "manufactured"}),
        },
        {
            "if": {"properties": {"verification": {"type": "null"}}},
            "then": boundary_kinds({"not": {"const": "manufactured"}}),
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
            "then": {"properties": {"time": {"properties": {"kind": {"const": "unsteady"}}}}},
        },
    ]
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_ID, **schema}
