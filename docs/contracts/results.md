# Contract: results

A run writes one directory. Its layout, the run metadata and the CGNS solution layout are the contract between the solver (and the fake backend), the results API (`autan.Results`), the UI and V&V. Schema identifier: `autan.results/1`.

## Run directory

```
<run_id>/
  run.json         run metadata, written at start and rewritten at every status change
  case.toml        the case document as run (case-schema.md), every default filled in
  mesh-info.json   the MeshInfo of the mesh read (mesh-info.md)
  history.jsonl    one line per recorded iteration, appended during the run
  log.jsonl        one line per event, appended during the run (see Log)
  manufactured.json  verification runs only: the manufactured solution, byte for byte
  states/          one <sha256>.json per face-state file of an oracle-state condition and for the initial-state file, byte for byte
  solution.cgns    solution (CGNS/HDF5), written when the run ends
```

The writer prepares the directory under a name starting with `.`, writes `run.json`, `case.toml`, `mesh-info.json`, the first line of `log.jsonl` and, for a verification case, `manufactured.json` (whose SHA-256 is the case's `verification.manufactured.sha256`), and the files of `states/` into it, then renames it to `<run_id>/`, failing if that exists; readers ignore names starting with `.`. `history.jsonl` exists from the first recorded iteration; each line is written in one piece, but while `status` is `running` the last line may still be incomplete and readers skip it. `solution.cgns` is written under a temporary name and renamed, at the end of the run and, as a checkpoint, while it runs; readers do not use it while the run is `running`. It is required when `status` is `converged`, `completed` or `stopped`, written if possible when `diverged` (the last state, for diagnosis), and optional when `failed`. Readers reject a directory whose `run.json` has another `schema` value, and open a run without a solution for its metadata and history.

`case.toml` keeps `mesh` as the case wrote it, since it is part of the case hash. The mesh actually read is `run.json` `mesh.path`, an absolute path, identified by `mesh.sha256`.

## run.json

```json
{
  "schema": "autan.results/1",
  "run_id": "run-20261031T214012Z-7f3a9c21",
  "identity": "4be1…",
  "status": "converged",
  "case": {"name": "crm-hl", "sha256": "…"},
  "mesh": {"path": "/data/crm-hl.cgns", "sha256": "…", "groups_sha256": null},
  "solver": {"version": "0.1.0", "commit": "d769576…", "backend": "fake"},
  "precision": "fp32+acc64",
  "deterministic": true,
  "devices": {"kind": "cpu", "count": 1},
  "started": "2026-10-31T21:40:12Z",
  "finished": "2026-10-31T21:54:40Z",
  "iterations": 3000,
  "time": null,
  "averaging": null,
  "start": null,
  "convergence": null,
  "monitors": {"forces": {"iteration": 3000, "fx": 1.2e4, "fy": 0.0, "fz": 9.8e4, "mx": 0.0, "my": -3.1e4, "mz": 0.0, "cl": 2.04, "cd": 0.31, "cs": 0.0, "cmx": 0.0, "cmy": -0.4, "cmz": 0.0}},
  "timing": {"wall_seconds": 868.2, "solve_seconds": 851.7, "iterations": 3000, "ns_per_cell_iteration": 5.9},
  "diagnostics": []
}
```

| Field | Meaning |
|---|---|
| `run_id` | Unique name of this execution and of its directory: `run-`, `started` as `YYYYMMDDThhmmssZ`, `-`, 8 random lowercase hex characters. |
| `identity` | What the run computes: SHA-256 of `case.sha256`, `mesh.sha256`, `mesh.groups_sha256` (the empty string for null), `solver.commit`, `precision`, `solver.backend`, `devices.kind`, `devices.count` and, for a run with a `start`, `start.identity`, `start.mode`, `start.iterations`, `start.solution_sha256`, joined by `\n`, UTF-8. |
| `status` | `running` · `converged` (convergence criterion of [case-schema.md](case-schema.md#time) met) · `completed` (iteration or step budget reached without convergence, or every unsteady step done) · `stopped` (stopped by user code, signal or an execution cap) · `diverged` (carries a `D` diagnostic) · `failed` (carries `E0704`). |
| `case.sha256` | Case hash as defined in [case-schema.md](case-schema.md#case-hash). |
| `mesh.path`, `mesh.sha256`, `mesh.groups_sha256` | Absolute path of the mesh read, `source.sha256` of its MeshInfo, and `groups_source.sha256` of its MeshInfo, null without zone groups: the zones a case's `group` selected. |
| `solver.version`, `solver.commit` | Version and full 40-character git commit of the solver build. |
| `solver.backend` | `cuda` · `hip` · `cpu` · `fake`. Results of the `fake` backend are synthetic and must never be used as reference data. |
| `precision` | Precision as run, from the case. |
| `deterministic` | Whether the run used the bitwise-deterministic mode. |
| `devices` | `kind` (`cuda` · `hip` · `cpu`) and `count`, the number of ranks ([solver-input.md](solver-input.md)). |
| `started`, `finished` | UTC, ISO 8601, second precision, `Z` suffix. `finished` is `null` while running. |
| `iterations` | Last completed iteration (pseudo-time iterations for steady runs, time steps for unsteady ones). |
| `time` | Physical time reached in seconds (unsteady), `null` (steady). |
| `averaging` | `null`, or `{"from_step", "to_step", "from_time", "to_time"}`: the interval actually averaged by an unsteady run with `average_from`, once at least one step past `average_from` is done; a continued run averages from the original `average_from`. |
| `start` | `null`, or where the run started from: `{run_id, identity, mode, iterations, time, solution_sha256, residual_reference, woken}` of the previous run ([solver-input.md](solver-input.md)); `residual_reference` is the residuals at iteration 1 of the original run of a steady `continue` chain, keyed by the equation names of [History](#history) with the values of that history line, `null` otherwise; `woken`, given exactly with `residual_reference`, lists in sorted order the identically zero equations of that reference that rose above their floor anywhere in the chain ([case-schema.md](case-schema.md#time)), so that a continuation judges convergence as the uninterrupted run would. The key is always present. A `continue` run carries on the previous count: its first history line is `start.iterations + 1` and `iterations` is the total; a `warm` run counts from 1. |
| `convergence` | `{"floor": {<equation>: value}}`, the round-off floor of every equation of [History](#history) (≥ 0, in its units; 0 when the physics gives that equation no scale) of a steady `fp32+acc64` run whose physics gives flux scales ([case-schema.md](case-schema.md#time)); `null` otherwise, and while a `running` run has not yet built its physics. The key is always present. |
| `monitors` | The record of every monitor of the case on the run's final state, `{<name>: {iteration, <the quantities of [History](#history)>}}`, `iteration` being `iterations`, whether or not the last iteration was a multiple of the monitor's `every`; equal to the last history line when that line carries the monitor. Given for `converged`, `completed` and `stopped` runs (a run stopped before its first iteration: the initial state, iteration 0); `null` while `running` and for `diverged` and `failed` runs. The key is always present. |
| `timing` | `{wall_seconds, solve_seconds, iterations, ns_per_cell_iteration}`: the time from the start of the runtime to the end of the run, the time of the physics' steps and averaging alone (no reading or partitioning, no history, log or solution writes; the slowest rank's), the iterations of this run (for a `continue`, those after `start.iterations`) and `solve_seconds` × 10⁹ / (cells of the mesh × `iterations`). `null` while `running`, for a run with no iteration, for a run that failed on several ranks, and for the `fake` backend. Information about one run, outside `identity` and the bitwise guarantee; performance is measured with `bench/` ([infra/bench.md](../infra/bench.md)). The key is always present. |
| `diagnostics` | Diagnostics raised during the run, in the shape defined in [case-schema.md](case-schema.md#diagnostics); a `diverged` run carries its `D` code here. |

Two runs with the same `identity` and `deterministic: true` that stop at the same iteration produce bitwise-identical `history.jsonl` lines and `solution.cgns` data arrays (`log.jsonl` excepted: it carries wall-clock times).

## History

`history.jsonl` holds one JSON object per line, in increasing `iteration`:

```json
{"iteration": 10, "time": null, "residuals": {"density": 1.2e-3, "momentum-x": 4.1e-4, "momentum-y": 2.0e-4, "momentum-z": 3.3e-4, "energy": 2.9e-3, "nu-tilde": 7.7e-5}, "monitors": {"forces": {"fx": 1.2e4, "fy": 0.0, "fz": 9.8e4, "mx": 0.0, "my": -3.1e4, "mz": 0.0, "cl": 2.04, "cd": 0.31, "cs": 0.0, "cmx": 0.0, "cmy": -0.4, "cmz": 0.0}}}
```

- `time`: physical time in seconds for unsteady runs, `null` for steady runs.
- `residuals`: one value per solved equation, every line carrying every equation. Names: `density`, `momentum-x`, `momentum-y`, `momentum-z`, `energy` (compressible) or `pressure` (the continuity defect), `momentum-x`, `momentum-y`, `momentum-z` (incompressible), then `nu-tilde` (SA, DDES on SA) or `k`, `omega` (SST, DDES on SST).
- Residual value: R_i is the volume-integrated residual of cell i for that equation, in SI units of the conserved quantity per second (kg/s for density, N for momentum, W for energy, ρν̃, ρk, ρω equations likewise). Steady runs evaluate the spatial operator at the current iterate; unsteady runs evaluate the physical-time defect at the last inner iteration of the step. The value is √(Σ R_i² / N) over the N fluid cells of the whole mesh, each counted once (halo copies excluded), accumulated in FP64. Solid cells are excluded.
- `monitors`: one entry per monitor recorded at this iteration (iteration divisible by the monitor's `every`), keyed by monitor name. A line may carry no monitor.

Forces monitor. Over the faces of the matched boundary zones, with n the unit normal pointing out of the fluid domain, p∞ the freestream static pressure and τ the viscous stress tensor, the force the fluid exerts on the surface and its moment about `moment_center` x_c are

F = ∫ [(p − p∞) n − τ·n] dA,  M = ∫ (x − x_c) × [(p − p∞) n − τ·n] dA,

reported as `fx` `fy` `fz` (N) and `mx` `my` `mz` (N·m) in mesh axes. With `ref_area` set, `cd`, `cs`, `cl` are F projected on the drag direction e_D = (cos α cos β, −sin β, sin α cos β), the side direction e_S = e_L × e_D and the lift direction e_L = (−sin α, 0, cos α), divided by q∞ `ref_area`, where q∞ = ½ ρ∞ V∞²; with `freestream.ref_length` also set, `cmx` `cmy` `cmz` are M divided by q∞ `ref_area` `ref_length`.

## Log

`log.jsonl` holds one JSON object per line in UTF-8, lines separated by LF and in the order they were written, appended during the run like `history.jsonl` (written in one piece and flushed; while the run is `running` the last line may be incomplete and readers skip it; once it ends every line is complete):

```json
{"time": "2026-11-01T01:02:03.456Z", "iteration": 812, "level": "error", "source": "solver", "message": "divergence at iteration 812 near cell 3418772", "code": "D0107", "data": {"iteration": 812, "cell": 3418772, "zone": "fluid"}}
```

| Key | Content |
|---|---|
| `time` | Wall-clock UTC, ISO 8601 with milliseconds and `Z`. |
| `iteration` | The iteration the event belongs to, `null` before the first (mesh reading, partitioning). |
| `level` | `info`, `warning` or `error`. |
| `source` | The component that logs: a short identifier matching `[a-z][a-z0-9-]*` (`solver`, `linear`, `io`, `mpi`, `fake`, …). |
| `message` | Human-readable; never parse it. |
| `code` | A registered diagnostic code, as in [case-schema.md](case-schema.md#diagnostics), or `null`. |
| `data` | Machine-readable details, `{}` when there are none: the diagnostic's data when `code` is set. |

The first line logs the start of the run (`iteration` null), and the writer logs the end with the final status. A line with a `code` has the level of its letter (`E` and `D`: `error`, `W`: `warning`) and, when its `data` has an `iteration`, the same `iteration`. Every diagnostic of `run.json` also has a line with the same `code` and `data`, written before `run.json` refers to it. Times are wall-clock and need not increase: the order of the lines is the order of the events. The log is outside the bitwise guarantee of [run.json](#runjson).

## solution.cgns

CGNS/HDF5, written against the CGNS SIDS 4.x and readable by the CGNS library (`cgnscheck` reports no error). All values in SI.

```
/CGNSLibraryVersion                      4.4
/Base                    CGNSBase_t      CellDimension 3, PhysicalDimension 3
  DataClass                              Dimensional
  DimensionalUnits                       Kilogram Meter Second Kelvin Radian
  RunMetadata            Descriptor_t    run.json without status, finished, iterations, monitors, timing, diagnostics (JSON)
  ZoneNames              Descriptor_t    {"boundary": {<node name>: <zone name>}, "cell": {…}} (JSON)
  <family>               Family_t        one per boundary zone and per cell zone
    FamilyBC             FamilyBC_t      boundary zones with a condition: BCType of that condition
  Zone                   Zone_t          Unstructured; [nodes, cells, 0]
    ZoneType                             Unstructured
    GridCoordinates                      CoordinateX, CoordinateY, CoordinateZ (R8, m: the mesh's times mesh.scale)
    NGonElements         Elements_t      NGON_n, ElementRange [1, F], ElementStartOffset, ElementConnectivity (I8)
    NFaceElements        Elements_t      NFACE_n, ElementRange [F+1, F+C], ElementStartOffset, ElementConnectivity (I8, signed face ids)
    ZoneBC               ZoneBC_t        absent when every boundary zone with faces is periodic
      <boundary zone>    BC_t            BCType, GridLocation FaceCenter, PointList (NGON ids), FamilyName
    ZoneGridConnectivity ZoneGridConnectivity_t  with periodic pairs only
      <periodic zone>    GridConnectivity_t  donor "Zone"; Abutting1to1, FaceCenter, PointList, PointListDonor (NGON ids), PeriodicTranslation
    <cell zone>          ZoneSubRegion_t GridLocation CellCenter, PointList (NFACE ids), FamilyName
    FlowSolution         FlowSolution_t  GridLocation CellCenter, one R8 DataArray per field
    FlowSolutionMean     FlowSolution_t  same fields, time average (unsteady runs with average_from)
    RestartState         FlowSolution_t  optional, CellCenter, R8 arrays the physics needs to restart (earlier time levels…); readers ignore it
    WallSolution         ZoneSubRegion_t GridLocation FaceCenter, PointList of every wall and slip face, one R8 DataArray per field
```

There is one `Zone_t` holding the whole mesh, every cell as a polyhedron (`NFACE_n`), every face as a polygon (`NGON_n`). Cell i (0-based, the index used by diagnostics) is element F + 1 + i; face j is element 1 + j. In `NFaceElements`, a positive face id means the face normal points out of the cell. Cell zones are `ZoneSubRegion_t`; boundary zones are `BC_t`, except the zones of a periodic pair (below). Interfaces between cell zones are interior faces and need no node. `FlowSolution` holds each cell's discrete unknown, located at the cell centroid: compare it with point values of a reference solution there.

Node names. A zone keeps its name as node name when it has 1 to 32 printable ASCII characters, contains no `/`, is not `.` or `..`, is unique among all boundary and cell zone names, is not itself of the form `boundary-NNNN` or `cell-zone-NNNN`, differs from the other children of `Base` listed above and, for cell zones, from the other children of `Zone`. Otherwise its node name is `boundary-NNNN` or `cell-zone-NNNN`, NNNN being its 0-based index in the MeshInfo list, zero-padded to 4 digits. `ZoneNames` maps every node name to the zone name.

BCType of each resolved condition: adiabatic `wall` → `BCWallViscousHeatFlux` (zero heat flux), isothermal `wall` → `BCWallViscousIsothermal`; `slip` → `BCWallInviscid`; `symmetry` → `BCSymmetryPlane`; `farfield` → `BCFarfield`; `inlet` and `velocity-inlet` → `BCInflowSubsonic`; `outlet` → `BCOutflowSubsonic`; `manufactured` (verification cases) → `UserDefined`; `oracle-state` → `BCDirichlet`. The node values are the SIDS strings (`BCTypeName` in the CGNS library), not the C enumerator names.

Periodic pairs ([mesh-info.md](mesh-info.md), `periodic`). Each pair `[a, b]` is two reciprocal `GridConnectivity_t` of `Zone` with itself, named after the node of each side's zone; the zones of a pair have no `BC_t`, and their `Family_t` no `FamilyBC`. The side-a connection has, as `PointList`, the faces of zone a in mesh order and, as `PointListDonor`, the face of zone b each one maps to. The side-b connection holds the same correspondence the other way round, in any order. Its `GridConnectivityProperty/Periodic` (`Periodic_t`) has `RotationCenter` and `RotationAngle` zero and `Translation` in R4, as CGNS requires; the descriptor `PeriodicTranslation` holds the translation `[x, y, z]` exactly, as JSON doubles, and is the authority: `transform.vector` of the pair times `mesh.scale` (m) for side a, its opposite for side b, with `Translation` its rounding to R4. No face is both in a connection and in a `BC_t`.

`FlowSolution` fields (SIDS names):

| Field | Unit | Written for |
|---|---|---|
| `Density` | kg/m³ | all cells (solid cells: the solid density) |
| `VelocityX`, `VelocityY`, `VelocityZ` | m/s | all cells (0 in solid cells) |
| `Pressure` | Pa | all cells (0 in solid cells) |
| `Temperature` | K | compressible physics |
| `TurbulentSANuTilde` | m²/s | SA and DDES on SA |
| `TurbulentEnergyKinetic`, `TurbulentDissipationRate` | m²/s², 1/s | SST and DDES on SST (ω) |

`WallSolution` is a `ZoneSubRegion_t` because the CGNS library does not read a face-centred `FlowSolution_t` on unstructured zones; it is absent when the mesh has no wall or slip face. Its fields, on wall and slip faces: `Pressure` (Pa, at the face) and `WallShearStressX`, `WallShearStressY`, `WallShearStressZ` (Pa, the tangential traction −τ·n the fluid exerts on the wall, n as in [History](#history); 0 on slip faces). Cp and Cf follow from these and q∞.

## Results API

`autan.Results.open(path)` reads a run directory and checks its metadata, case, MeshInfo, history and log against this contract (`E0701` otherwise); `results.log` holds the log lines; it reads `solution.cgns` only on the first `field`, `wall` or `cell_zone` call, which checks it the same way, so listing many runs stays cheap. `results.run` returns the run metadata, `results.case` the case, `results.mesh_info` the MeshInfo, `results.history` the residual and monitor time series, `results.forces` the forces monitor when there is exactly one, `results.field(name, mean=False)` a cell-centred field as a NumPy array, `results.wall(name)` a wall field with its face indices, `results.cell_zone(name)` the cell indices of a cell zone.
