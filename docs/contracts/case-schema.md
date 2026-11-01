# Contract: case schema

A case is a typed document that says what to compute on which mesh. Python code builds it (`autan.Case`), TOML and JSON store it, the UI and agents edit it, the solver consumes it. Schema identifier: `autan.case/1`. The JSON Schema is generated from the SDK (`autan schema case`) and is the machine-readable form of this page.

## Example

The CRM-HL case of `docs/design/dx.html`, as Python and as the TOML it serializes to. The Python is in the canonical form `case.to_python()` writes, which `autan.load` reads back into the same case: only values that differ from their default, quantities in the unit the case carries, library materials by name, numbers in their shortest round-trip form. The reference area is the HLPW-4/5 semispan value, 297 360 in²:

```python
import autan
from autan.units import deg, m

case = autan.Case("crm-hl", mesh="crm-hl.cgns")
case.physics = autan.Compressible(turbulence=autan.SA())
case.freestream(mach=0.2, aoa=19.57 * deg, re=5490000.0, ref_length=7.0 * m)
case.boundary("wing.*").wall()
case.boundary("symmetry").symmetry()
case.monitor(autan.forces("wing.*", ref_area=191.84 * m**2), every=10)
```

```toml
schema = "autan.case/1"
name = "crm-hl"
mesh = "crm-hl.cgns"
mesh_unit = "m"

[physics]
kind = "compressible"
precision = "fp32+acc64"
turbulence = { model = "sa", nu_tilde_ratio = 3.0 }
gas = { kind = "ideal-gas", name = "air", molar_mass = "0.0289647 kg / mol", gamma = 1.4, prandtl = 0.72, viscosity = { law = "sutherland", mu_ref = "1.716e-05 Pa * s", t_ref = "273.15 K", s = "110.4 K" } }

[freestream]
mach = 0.2
aoa = "19.57 deg"
sideslip = "0.0 deg"
re = 5490000.0
ref_length = "7.0 m"
temperature = "288.15 K"

[time]
kind = "steady"
max_iterations = 10000
residual_orders = 8.0
linear_solver = "multigrid"

[[boundaries]]
zones = "wing.*"
condition = { kind = "wall" }

[[boundaries]]
zones = "symmetry"
condition = { kind = "symmetry" }

[[monitors]]
kind = "forces"
name = "forces"
surfaces = "wing.*"
every = 10
ref_area = "191.84 m ** 2"
```

## Quantities

A dimensional value is written as a string `"<number> <unit>"`, for example `"7.0 m"`, `"101325.0 Pa"`, `"1.716e-05 Pa * s"`. Units are parsed by [Pint](https://pint.readthedocs.io) with its default registry. Each field declares a dimension; a value of another dimension is rejected (`E0301`), a bare number in a dimensional field is rejected (`E0302`), an unknown unit is rejected (`E0303`). Angles need an angle unit (`deg`, `rad`): a bare `19.57` is ambiguous and rejected. Dimensionless fields (`mach`, `re`, `gamma`, `prandtl`) are plain numbers.

Values keep the unit they were written in; consumers convert to SI. The Python builders take a Pint quantity or its text. A quantity built in Python is held as its text reads back: a 64-bit float in Pint's short unit, so the case, its document and its hash agree. A unit whose text does not read back with the field's dimension (exponents are written with six significant digits) is rejected (`E0301`). Fields marked *> 0* reject zero and negative values (`E0304`); temperatures are checked after conversion to kelvin.

## Tree

Every table rejects unknown keys. Discriminated unions are tagged by `kind` (or `model`, `law`); a document always writes the tag, even where the Python builder fills it in. Integers are strict (a string or a boolean is rejected) and at most 2⁵³ − 1, the largest integer RFC 8785 represents exactly.

### Case

| Key | Type | Default |
|---|---|---|
| `schema` | `"autan.case/1"` | required (`E0004` otherwise) |
| `name` | non-empty string | required |
| `mesh` | path, relative to the case file | required |
| `mesh_unit` | unit of length of the mesh file's coordinates, as Pint writes it (`"m"`, `"mm"`, `"in"`); never read from the file. Only the mesh's coordinates, and what MeshInfo reports from the file (`periodic[].transform`), are in this unit; every other length of the case carries its own. `E0301` for a unit of another dimension, `E0303` for text that is not a unit | `"m"` |
| `physics` | [Physics](#physics) | required |
| `freestream` | [Freestream](#freestream) | required |
| `time` | [Time](#time) | `{kind = "steady"}` |
| `boundaries` | list of [BoundaryRule](#boundaries) | `[]` |
| `cell_zones` | list of [CellZoneRule](#cell-zones) | `[]` |
| `monitors` | list of [Monitor](#monitors) | `[]` |
| `verification` | [Verification](#verification) | absent |
| `initial_state` | [Initial state file](#initial-state-file) | absent |

### Physics

`kind = "compressible"`: density-based solver.

| Key | Type | Default |
|---|---|---|
| `turbulence` | [Turbulence](#turbulence) | required |
| `precision` | `"fp32+acc64"` · `"fp64"` | `"fp32+acc64"` |
| `gas` | [IdealGas](#materials) | air |

`kind = "incompressible"`: pressure-based solver.

| Key | Type | Default |
|---|---|---|
| `turbulence` | [Turbulence](#turbulence) | required |
| `precision` | `"fp32+acc64"` · `"fp64"` | `"fp32+acc64"` |
| `fluid` | [ConstantDensityFluid](#materials) | required |

`precision = "fp32+acc64"` computes in FP32 and accumulates solution updates, norms and geometry in FP64 (see `docs/PLAN.md`). `"fp64"` computes everything in FP64.

`gas` and `fluid` are the default medium of every fluid cell zone.

### Turbulence

| `model` | Options | Notes |
|---|---|---|
| `"inviscid"` | — | Euler equations. |
| `"laminar"` | — | Navier-Stokes without a turbulence model. |
| `"sa"` | `nu_tilde_ratio` > 0, default 3.0 | Spalart-Allmaras, negative variant (SA-neg). |
| `"sst"` | `intensity` > 0, default 0.001; `viscosity_ratio` > 0, default 0.009 | Menter k-ω SST. |
| `"ddes"` | `base`: an `sa` or `sst` table with its options, default `{model = "sst"}` | Delayed detached-eddy simulation. Needs `time.kind = "unsteady"` (`E0305`). |
| `"wmles"` | `wall_model = "equilibrium"` (default) | Wall-modelled LES. Needs `time.kind = "unsteady"` (`E0305`). |

The options set the turbulence state of the freestream and of every inlet, from the local laminar viscosity ν = μ/ρ and speed V: SA takes ν̃ = `nu_tilde_ratio` · ν; SST takes k = 1.5 (`intensity` · V)² and ω = ρ k / (`viscosity_ratio` · μ). The defaults follow the NASA Turbulence Modeling Resource: ν̃/ν = 3 for SA, μt/μ = 0.009 for SST; the SST intensity default (0.1 %) is our choice.

### Freestream

The freestream state defines farfield conditions and the initial state. It is expressed with the physics' default medium.

| Key | Type | Compressible | Incompressible |
|---|---|---|---|
| `mach` | number > 0 | required | forbidden |
| `velocity` | speed > 0 | forbidden | required |
| `aoa` | angle | default `"0.0 deg"` | default `"0.0 deg"` |
| `sideslip` | angle | default `"0.0 deg"` | default `"0.0 deg"` |
| `temperature` | temperature > 0 | default `"288.15 K"` | forbidden |
| `pressure` | pressure > 0 | exactly one of `pressure` or `re` | forbidden |
| `re` | number > 0, Reynolds number based on `ref_length` | exactly one of `pressure` or `re` | forbidden |
| `ref_length` | length > 0 | required with `re` | optional |

A violation of this table is `E0306`. With `re`, the static pressure follows from the ideal-gas law and the gas viscosity at `temperature`.

The flow direction in mesh axes is (cos α cos β, −sin β, sin α cos β) with α = `aoa`, β = `sideslip`.

### Time

| `kind` | Keys |
|---|---|
| `"steady"` | `max_iterations` integer ≥ 1, default 10000; `residual_orders` number > 0, default 8.0; `linear_solver` `"multigrid"` · `"gauss-seidel"`, default `"multigrid"` |
| `"unsteady"` | `steps` integer ≥ 1; exactly one of `dt` (time > 0) or `cfl` (number > 0); `average_from` integer in [0, `steps`), optional |

A steady run uses local time stepping and has converged when every residual of [results.md](results.md#history) has dropped `residual_orders` orders of magnitude below its value at iteration 1 (of the original run, for a continued one); it stops after `max_iterations` otherwise.

`linear_solver` chooses how an implicit physics solves each step's linear system: `"multigrid"`, agglomeration multigrid, or `"gauss-seidel"`, symmetric block Gauss-Seidel over a colouring of the whole mesh, the same algorithm on any number of ranks. A physics states which it runs where (the Euler physics: `"multigrid"` on one rank only; `E0706` otherwise); an explicit physics ignores it.

In `fp32+acc64`, round-off stops the residuals before they can drop 8 orders. So an equation also counts as converged once its residual reaches its floor C·ε₃₂·Φ_k, whichever comes first. The constants are C = 0.5 and ε₃₂ = 2⁻²³. Φ_k = f_k·√(Σ_i A_i²/N) is the flux scale of equation k: A_i is the surface of cell i (all its faces), the sum and N are those of the history residual, and f_k is the face flux magnitude of the freestream. With q∞ = |v∞| + c∞ and c∞ = √(γ p∞/ρ∞), f_k is ρ∞ q∞ for density, ρ∞ q∞² for each momentum, and ρ∞ H∞ q∞ for energy, with H∞ = c∞²/(γ−1) + |v∞|²/2. C puts the floor at least 4.3 times above the round-off plateaus measured on a NACA 0012 slice (M 0.5), manufactured solutions and Ni's transonic bump (M 0.675). A physics that gives no flux scale has no floor. The floors of a run are in its `run.json` ([results.md](results.md#runjson)). An equation whose residual at iteration 1 is below the floor, 1e-12 times the largest residual of any equation at iteration 1, is identically zero (the transverse momentum of a one-cell 2D slice, for example) and is exempt as long as its residual stays at or below the floor. The first time it rises above the floor, the runtime reports `W0710` and the equation must meet the rule like the others for the rest of the run, continuations included.

With `average_from`, an unsteady run also writes the time average of every cell field over steps `average_from + 1` to `steps`, each step weighted by its time step (see [results.md](results.md#solutioncgns)).

Other numerical settings (flux scheme, limiter, CFL ramp) are not part of `autan.case/1`: the solver version fixes them, and the solver commit recorded by every run identifies them.

### Boundaries

A rule applies a condition to every boundary zone whose name matches `zones`, a glob pattern (`*`, `?`, `[…]`, case-sensitive, matched against the whole name).

| `condition.kind` | Keys | Physics |
|---|---|---|
| `"wall"` | `temperature` (temperature > 0), optional: isothermal if set, adiabatic otherwise | viscous; with `inviscid` turbulence use `slip` (`E0414`) |
| `"slip"` | — | all |
| `"symmetry"` | — | all |
| `"farfield"` | — (uses the freestream); optional `point_vortex = {surfaces, center}` (pattern over boundary zones; three lengths) | all |
| `"inlet"` | `total_pressure` (pressure > 0), `total_temperature` (temperature > 0) | compressible (`E0414` otherwise) |
| `"velocity-inlet"` | `velocity` (speed > 0) | incompressible (`E0414` otherwise) |
| `"outlet"` | `static_pressure` (pressure > 0) | all |
| `"oracle-state"` | `path` (relative to the case file) and `sha256` of an `autan.face-states/1` file exported for this mesh ([face-states.md](face-states.md)): every face of the zone takes its state | compressible, inviscid or laminar (`E0414` otherwise) |

A farfield's `point_vortex` corrects the far-field state of a 2D lifting body: the farfield takes the velocity of a point vortex at `center`, whose circulation is that of the lift on the boundary zones matched by `surfaces` (wall and slip zones only; wildcards skip periodic zones, `E0425` otherwise). The physics states which flows it accepts (the Euler physics: steady case, no sideslip, subsonic freestream; `E0706` otherwise).

`inlet`, `velocity-inlet` and `outlet` are subsonic conditions. Inlet flow is normal to the boundary face. Where the outflow is supersonic, the outlet extrapolates every variable and ignores `static_pressure`. The solver reports inlet faces with supersonic or reversed flow and outlet faces with reversed flow (codes defined with the M2 boundary conditions).

### Cell zones

A rule assigns a medium to every cell zone whose name matches `zones` (same glob rules).

| `medium.kind` | Keys |
|---|---|
| `"fluid"` | `material`: an [IdealGas](#materials) (compressible) or a [ConstantDensityFluid](#materials) (incompressible); `E0504` otherwise; `initial` (optional): the state the zone starts from, `{pressure, temperature, velocity}` — pressure (absolute > 0 and temperature > 0 for compressible flow; gauge, without temperature, for incompressible flow, `E0507`), velocity three speeds in mesh axes; without it the zone starts from the freestream |
| `"solid"` | `material`: a [Solid](#materials); compressible physics only (`E0506`), since `autan.case/1` gives incompressible flow no energy equation |

### Materials

Materials are stored in full, never by reference to a library entry, so the case hash covers every property. `autan.gases` and `autan.materials` provide presets with cited sources.

| `kind` | Keys |
|---|---|
| `"ideal-gas"` | `name`; `molar_mass` (kg/mol-dimension > 0); `gamma` > 1; `prandtl` > 0; `viscosity`: `{law = "sutherland", mu_ref, t_ref, s}` or `{law = "constant", mu}` (dynamic viscosity > 0, temperatures > 0) |
| `"constant-density-fluid"` | `name`; `density` > 0; `viscosity` (dynamic viscosity > 0) |
| `"solid"` | `name`; `density` > 0; `specific_heat` > 0; `conductivity` > 0 |

### Monitors

| `kind` | Keys |
|---|---|
| `"forces"` | `name` (unique among monitors, default `"forces"`, `E0602`); `surfaces` glob over boundary zones, periodic zones excepted (naming one is `E0603`); `every` integer ≥ 1, default 1; `ref_area` area > 0, optional; `moment_center` three lengths, default origin |

Coefficients are reported only when `ref_area` is set; moment coefficients also need `freestream.ref_length`. The force definition and the recorded quantities are in [results.md](results.md#history).

### Verification

`verification = { manufactured = { path, sha256 } }` makes the case a verification case against a manufactured solution: `path` names an `autan.mms/1` file written by `autan-vv` (relative to the case file, like `mesh`), `sha256` is the SHA-256 of its bytes. The solver adds the solution's source term and every boundary zone takes the solution's exact data, so the case has no boundary rules (`E0416`). `autan plan` checks the file: missing `E0207`, different hash `E0206`, not an `autan.mms/1` document `E0208` ([mms.md](../validation/mms.md)); and checks it against the case (`E0209`): the solution's declared `parameters` (`gamma`, `gas_constant`, `viscosity`, `prandtl`) must equal the case gas (constant viscosity), its declared `model` must be compressible flow with the case's turbulence model (`inviscid`, `laminar` or `sa`; `sa-noft2` has no case counterpart), and the case has no cell zone rules (one fluid medium). Manufactured solutions are steady, so a verification case is steady (`E0417`). In Python: `case.verification = autan.Manufactured(path="euler.mms.json", sha256="…")`.

### Initial state file

`initial_state = { path, sha256 }` starts every fluid cell from an `autan.initial-state/1` file written for the case's mesh ([initial-state.md](initial-state.md)): `path` is relative to the case file, like `mesh`, and `sha256` is the SHA-256 of its bytes. `autan plan` checks the file (`E0508` to `E0510`). The file carries no turbulence variable, so the case is compressible, inviscid or laminar. It replaces the fluid zones' `initial`, which the case leaves out, and a run of the case starts from it, not from a previous run (`E0512` otherwise). In Python: `case.initial_state = autan.StateFile(path="vortex.json", sha256="…")`.

## Boundary resolution

The solver needs one condition per boundary zone of the mesh. The SDK resolves them from the case and the mesh's [MeshInfo](mesh-info.md), and records where each comes from (its **provenance**):

A zone of a periodic pair (MeshInfo `periodic`, [mesh-info.md](mesh-info.md)) takes no condition: the solver pairs its faces with those of its partner. Its provenance is `mesh` and `autan plan` shows `{"kind": "periodic", "partner": …}`. A rule naming it is an error (`E0424`); patterns with wildcards skip periodic zones. The rest of this section concerns the other zones.

With `verification`, every boundary zone takes the manufactured solution's data, provenance `verification`, and the steps below do not apply.

1. Each rule must match at least one zone (`E0410`). A zone matched by two rules is ambiguous (`E0411`): narrow the patterns.
2. A zone matched by a rule takes the rule's condition. Provenance `code`. If the mesh typed the zone with a different kind, the override is reported (`W0415`).
3. Otherwise the zone's mesh type applies. Provenance `mesh`. `wall` becomes an adiabatic no-slip wall, or a slip wall with `inviscid` turbulence; `slip`, `symmetry` and `farfield` map one to one. `thermal-wall`, `inlet` and `outlet` need values that only the case can give (`E0413`).
4. Otherwise the zone has no condition (`E0412`), unless it has no faces.

Cell zones resolve the same way: rules (`E0501` no match, `E0502` ambiguous), then the mesh type (`fluid` takes the physics' default medium; `solid` needs a material, `E0503`), then, for a zone the mesh leaves untyped, a fluid with the physics' default medium (provenance `default`). Two fluid zones with different materials that share an interface would need species transport, which `autan.case/1` does not provide (`E0505`).

Monitor surfaces must match at least one boundary zone (`E0601`).

`autan plan` prints this resolution; `autan plan --json` returns it.

## Case hash

The case hash is the SHA-256 of the case document with every default filled in, canonicalized by the JSON Canonicalization Scheme ([RFC 8785](https://www.rfc-editor.org/rfc/rfc8785)) and encoded in UTF-8. Non-finite numbers cannot occur: the schema rejects them. Two documents that differ only in formatting, key order or TOML/JSON syntax have the same hash; writing the same value in another unit changes it.

## Diagnostics

Every problem is reported as a diagnostic, in this JSON shape wherever diagnostics appear (`autan plan --json`, `run.json`, the SDK's `CaseError`):

```json
{"code": "E0412", "severity": "error", "message": "boundary zone 'empennage' has no condition",
 "location": {"file": "crm-hl.py", "line": 4, "path": "boundaries"},
 "fix": "case.boundary('empennage').wall()", "doc": "https://remenby31.github.io/autan/sdk/errors/#e0412",
 "data": {"zone": "empennage"}}
```

| Field | Meaning |
|---|---|
| `code` | `E####` (error, blocks the run), `W####` (warning) or `D####` (divergence during a run). `E0101`–`E0199` belong to the mesh readers ([mesh.md](mesh.md)). |
| `severity` | `error`, `warning` or `divergence`, given by the code's letter. |
| `message` | Human-readable; never parse it. |
| `location` | `null`, or any of `file`, `line` (1-based), `path` (document path such as `boundaries[2].condition`). |
| `fix`, `doc` | A suggested correction, and the code's documentation page. |
| `data` | Machine-readable details. Mesh entities use `zone` (name), `cell` and `face` (0-based indices in the mesh's cell and face order, the order of `NFaceElements` and `NGonElements` in [results.md](results.md#solutioncgns)), `iteration`. |

Codes are registered in `autan.errors`; their reference page `docs/sdk/errors.md` and the machine-readable export [`diagnostics.json`](diagnostics.json) (`autan explain --all --json`: severity, title, fix and doc of every code, embedded by the C++ runtime) are generated from it.
