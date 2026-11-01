# `autan run`

`autan run` plans a case, refuses it on any error, then writes a run directory as specified in [results.md](../contracts/results.md) under `DIR` (default `./runs`).

```text
autan run CASE --mesh-info MESHINFO [--backend fake|cpu|cuda|hip] [--device-count N] [--runs DIR] [--json]
```

In Python: `case.run(mesh_info, backend=..., runs="runs")`, which returns `autan.Results`.

## Solver backend

`--backend cpu|cuda|hip` (Python: `autan.SolverBackend(devices=autan.Devices(kind="cuda", count=8))`) writes the case's [solver input](../contracts/solver-input.md) and runs the solver runtime `autan-run`, found through `$AUTAN_RUN` or the PATH (`E0804` if neither). With `--device-count N` > 1 it starts N ranks, one per device, through the MPI launcher of `$AUTAN_MPIEXEC`, or `mpiexec` then `mpirun` on the PATH (`E0808` if none). The runtime's input errors come back as their own coded diagnostics. `autan-run` runs inviscid compressible cases of a single ideal gas, without solid zones, with `slip`, `symmetry`, `farfield`, `inlet` and `outlet` conditions (and manufactured ones in verification cases), on meshes without periodic pairs, on the CPU, in `fp64` and `fp32+acc64`; anything else ends with `E0706` before any run directory exists.

`autan-run` is not in the wheel yet: [build it from source](../getting-started/build.md) and set `$AUTAN_RUN`.

## Fake backend

`--backend fake` (the default; Python: `autan.FakeBackend()`) writes contract-conforming, deterministic, **synthetic** results so the UI and V&V can be built against the real formats; `run.json` records `"backend": "fake"`, and these results must never be used as reference data.

| What | Fake backend |
|---|---|
| Geometry | A unit cube of n³ hexahedra written as polyhedra, n ≥ 8 and set by the MeshInfo zone counts; boundary zones are consecutive groups of its boundary faces (the two zones of a periodic pair get as many faces each, paired in order, without the pair's translation relating them: the box is not periodic), cell zones are slabs along x. Zone names, conditions and media are the case's. |
| Fields | The freestream state of [case-schema.md](../contracts/case-schema.md#freestream) in every fluid cell (each zone with its own gas), the solid density in solid cells, zero wall shear. An initial-state file is checked and archived; its values are not used. |
| History | Residuals that reach `residual_orders` orders of decay at iteration 100, where a steady run converges (`completed` if `max_iterations` ends it first); forces approaching a thin-airfoil lift, projected as the contract defines. One line per iteration, flushed as written. The trajectory depends only on the case and the mesh, so runs with the same identity agree line for line. Unsteady runs without `dt` use 1 ms steps. |
| Options | `--iterations N` stops the run after N iterations (or steps), status `stopped`; `--pace S` sleeps S seconds per iteration to imitate a live run; `--diverge-at N` stops at iteration N with status `diverged` and `D0107`. |

## Stopping a run and exit status

Ctrl-C or SIGTERM stops a run: it is finalized as `stopped`, with its solution (written by the SDK for the fake backend, by `autan-run` for the solver backend, to which the SDK passes the signal on), and the command exits with 130. Otherwise the exit status is 0, or 1 when the plan has errors or the run diverged or failed.
