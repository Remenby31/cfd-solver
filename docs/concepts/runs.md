# Runs

A run is one execution of a case on a mesh. It writes one directory with its results and what identifies them: the case as run, the mesh hash and the solver commit. The solver runtime `autan-run` and the SDK's `fake` backend write the same layout, the [results contract](../contracts/results.md); `autan.Results` reads it.

## Run states

![Run states](../assets/diagrams/run-lifecycle.svg)

`autan plan` runs first. If it finds an error, nothing is written; nor does `autan-run` write anything for physics it does not run (`E0706`). Otherwise the run starts in `running` and ends in one state:

| State | Meaning |
|---|---|
| `converged` | Every residual dropped by the requested number of orders. |
| `completed` | The iteration or time-step budget ran out first. |
| `stopped` | You, a signal or an execution cap stopped it. |
| `diverged` | The solution became non-finite or non-physical; a `D` diagnostic names the cell. |
| `failed` | The backend hit an unexpected error, `E0704`. |

## The run directory

| File | Written |
|---|---|
| `run.json` | At start, then at every state change. |
| `case.toml` | At start: the case as run, every default filled in. |
| `mesh-info.json` | At start: the MeshInfo of the mesh read. |
| `history.jsonl` | One line per recorded iteration, during the run. |
| `solution.cgns` | At the end, before the state leaves `running`; see the diagram for which end states have one. |

You can open a run while it is `running`: the history is readable, the solution is not.

## How runs are identified

Every run records the mesh hash, the case hash, the solver commit and the precision. Their combination, with the backend and the devices, is the run's `identity` ([Identity](../contracts/results.md#identity)). With `physics.reproducibility = "run"` or `"ranks"`, two runs of the solver with the same identity that stop at the same iteration write the same history and solution data, bit for bit. [run.json](../contracts/results.md#runjson) defines each field.
