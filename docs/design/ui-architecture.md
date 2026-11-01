# Web UI architecture

The web UI of milestone M10 (`docs/PLAN.md`): a local application served by `autan ui`, never a cloud service. It shows every run and study whoever started it (UI, SDK, CLI, agent), prepares cases as `autan.case/1` code, follows runs live, renders 3D fields and compares runs. Visual and interaction direction: [`ui-inspiration/README.md`](ui-inspiration/README.md). Code: `ui/`.

## Shape

```
browser ── HTTP + SSE ──▶ autan_ui (FastAPI, Python) ──▶ <runs>/<run_id>/   (results.md)
  React app (static, served by the same process)        surface extraction (NumPy) → binary
```

- **One process, one port, loopback only.** Requests whose `Host` is not the served address are refused (DNS rebinding). `autan_ui.serve(runs, host, port)` serves the built app and the API. `autan ui [--runs DIR] [--port N] [--no-open]` in the SDK CLI is a thin call to it; the default runs directory is `./runs`, the same one `autan run` writes.
- **The registry is the runs directory.** Every directory under `<runs>` that the SDK's reader (`autan.Results.open`) accepts is a run, whoever wrote it; names starting with `.` are directories still being written. There is no database and no UI-side state: a run started from a notebook appears in the UI the moment its directory does. Directories that do not conform are listed as rejected with the reason, never hidden.
- **POSIX only.** Every file of a run or of the built app is opened beneath a pinned directory descriptor without following links (`dir_fd`, `O_NOFOLLOW`), so the server runs on Linux and macOS; on Windows, through WSL.
- **Heavy data stays on the server.** Volume fields (tens of millions of cells) never reach the browser. The server extracts what is displayed (wall surfaces, slices, iso-surfaces, probes) and sends triangles and node values. Slices and probes are server-side NumPy operations, so the browser renderer only needs to draw triangle surfaces fast.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/runs` | `{runs: [run.json…], rejected: [{dir, reason}], summaries: {run_id: summary}, solutions: [run_id]}`, newest first; `solutions` are the ended runs with a `solution.cgns`, whose solution is a result ([results.md](../contracts/results.md)). A summary is read incrementally from `history.jsonl`: the last complete `iteration` (run.json is rewritten only when the status changes), `orders` the slowest equation not yet converged has dropped since iteration 1 by the steady rule of [case-schema.md](../contracts/case-schema.md#time) (identically zero equations exempt, fp32+acc64 equations converged at their `convergence.floor`; the case's target once every equation has converged), their `trend` (≤ 48 points), the case's `target_orders` and `max_iterations`, and the `settings` studies usually sweep (`aoa`, `sideslip`, `mach`, `re`, `turbulence`, `precision`, as the case writes them). |
| `GET /api/runs/events` | SSE: `runs` (snapshot), then `run` (upsert, the new run.json), `removed` (run id), `rejected` (the whole list), `summaries` and `solutions` as they change |
| `GET /api/runs/{id}` | `{run, case, case_toml, mesh_info, outputs}`: run.json, case.toml as JSON and as text, mesh-info.json and the files of an accepted run |
| `GET /api/runs/{id}/plan` | `autan plan --json` of the stored case against the stored MeshInfo, for the run's GPU count |
| `GET /api/runs/{id}/history/events` | SSE: `run`, then `history` and `log` batches (history.jsonl and log.jsonl records; each log line also carries `doc`, the documentation page of its `code`). The event id is the JSON map of byte offsets reached, so a reconnecting `EventSource` resumes without loss or duplicates. A line still being written is held until its newline while the run is running; after it ends, an unterminated line ends the stream with `invalid`. Unknown runs are 404 and rejected ones 409 with the reason, before any stream starts. |
| `GET /api/runs/history/events?run=…&run=…` | The same events for up to 8 runs over one connection (a browser opens at most six per host), payloads `{run_id, data}`; the event id is the JSON map of byte offsets. Used by Compare. |
| `GET /api/runs/{id}/surface` | The wall and slip zones of `solution.cgns` as an `autan-ui.surface/1` message, fields per node: `Cp` = (p − p∞)/q∞, `Cf` = \|τ\|/q∞ (none for an inviscid case), `Pressure` (Pa), with q∞ from the stored case's freestream; face values (`WallSolution`) averaged to nodes with face-area weights, polygons fan-triangulated. 404 without a solution. Opened from a run with *Open in Analysis* (`/viewer?run=…&field=…`); `/viewer` alone lists the ended runs with a solution (`solutions` of `/api/runs`). |
| `GET /api/meshes` | The MeshInfo files (`*.meshinfo.json`) of the `--meshes` directory a case can be planned against: name, source, counts, boundary and cell zones |
| `GET /api/materials` | The SDK's library gases and solids (`autan.gases`, `autan.materials`): library, the name a case writes them by (`CO2`), and their document; the editor offers them for the physics gas and cell zones |
| `GET /api/meshes/{name}/surface` | The boundary zones of a listed mesh as an `autan-ui.surface/1` message, from the `autan.surface/1` file `autan-mesh-info --surface` wrote next to its MeshInfo ([mesh.md](../contracts/mesh.md#boundary-surface)), checked against that MeshInfo (hash, zones, faces); polygons triangulated as for walls. 404 without the file, 422 when it breaks the contract |
| `POST /api/cases/render` | `{case, mesh?}` → `{ok, python, toml, plan, diagnostics}`: the case document as the SDK reads it, its canonical SDK code (`Case.to_python`) and case.toml, and its plan against a listed MeshInfo; or the SDK's coded diagnostics. Planned from a copy that names no file the server can reach (mesh and every `path` repointed into an empty scratch directory). 404 for an unlisted mesh |
| `GET /api/surfaces/synthetic?triangles=N` | A synthetic wing surface for the viewer benchmark, shown only by `/viewer?bench=1` or `?triangles=N` |

Live updates come from file-system events (`watchfiles`, 50 ms debounce) and a 1 s timeout: every wake-up rescans, so a change made before the watcher started is still reported. A directory is checked by `autan.Results.open` again only when its `run.json`, `case.toml`, `mesh-info.json` or `solution.cgns` changes; history.jsonl is re-read from the last offset. SSE rather than WebSocket: traffic is server → browser only, `EventSource` reconnects by itself, and FastAPI ships SSE natively.

### Surface message `autan-ui.surface/1`

An internal format between `autan_ui` and the web app (not a project contract): a little-endian `u32` header length, a UTF-8 JSON header padded to 4 bytes, then the buffers.

```json
{"schema": "autan-ui.surface/1", "origin": [x, y, z], "nodes": N, "triangles": T,
 "zones": [{"name", "first_triangle", "triangles"}],
 "fields": [{"name", "location": "node", "min", "max"}],
 "buffers": [{"name": "positions" | "indices" | "field:<name>", "dtype": "f32" | "u32", "offset", "length"}]}
```

Positions are FP32 relative to `origin` (FP64), so a model far from the mesh origin keeps sub-micrometre resolution. Triangles are sorted by zone, so a zone is one contiguous range: one draw group, one pick id. The browser wraps the buffers in typed-array views without copying and hands them to the GPU. Field values are per node; face-centred wall data (`WallSolution`) is averaged to nodes with face-area weights. Arrow IPC was considered and rejected: a columnar table format and its library for what are three flat arrays.

## Stack

Chosen on the project's criteria: fluid 3D on 5–20 M faces, live data, keyboard-first UX, local-only, permissive licences, a small number of durable dependencies.

| Concern | Choice | Licence | Why, and what was rejected |
|---|---|---|---|
| Server | FastAPI 0.162 + uvicorn + watchfiles, Python ≥ 3.12, own `uv.lock` | MIT, BSD-3, MIT | Same language as the SDK, so `autan ui` is a function call and the server checks run directories with `autan.Results.open`; native SSE. |
| Package manager | Bun 1.3, text `bun.lock` | MIT | Fast installs, one binary; Vite and Vitest run unchanged. |
| Build | Vite 8 (Rolldown) + TypeScript 7 (native compiler) | MIT, Apache-2.0 | Sub-second builds and type checks. |
| UI | React 19 | MIT | Largest ecosystem for accessible primitives, palette and charts. |
| Routing, server state | TanStack Router (typed search params) + TanStack Query | MIT | Filters, camera and displayed field live in the URL (design principle 7) with types; SSE events write straight into the Query cache. |
| Styling | Tailwind v4 over `tokens.css` (`@theme inline`) | MIT | Utilities read the design tokens; no second source of values. Geist via `@fontsource-variable` (OFL-1.1), self-hosted. |
| Primitives | Base UI 1.9, lucide-react icons | MIT, ISC | Unstyled, accessible components (menus, tabs, collapsibles, tooltips, hover cards, toasts, toggles, meters) styled by data attributes; `ui/src/components/ui/` builds the component library on them (below). |
| Palette | cmdk | MIT | ⌘K with fuzzy ranking and grouped results. |
| Motion | Motion 14 | MIT | Layout and presence animations (rows entering the registry); follows the system's reduced-motion setting. |
| Charts, long lists | uPlot, TanStack Virtual | MIT | Canvas time series built for large, appended data, log axes, 50 kB; the log renders only the visible lines of a 100 k-line history. |
| 3D | three.js 0.186 `WebGPURenderer`, wrapped in one imperative `SurfaceViewer` class | MIT | Measured below. No react-three-fiber: the scene is one mesh per surface, and a declarative scene graph adds a reconciler for nothing. vtk.js rejected (below). |
| Lint, format | Biome 2.6 (repository-wide, with the docs site) | MIT/Apache-2.0 | One tool for TS, JSON and CSS. |
| Tests | Vitest 5, Playwright 1.64 on headless Chrome | MIT, Apache-2.0 | Playwright drives the real server, real files and the real GPU. |

`WebGPURenderer` falls back to WebGL2 where WebGPU is missing; the fallback is logged and shown in the viewer status line.

### 3D renderer: three.js against vtk.js

Same synthetic surface (swept NACA 0012 wing, structured triangulation, one node field), same 1600 × 1000 canvas, camera orbiting, headless Chrome 156 on an Apple M4 Pro (Metal), vsync off. Frame time is measured to GPU completion (`queue.onSubmittedWorkDone` on WebGPU, a 1-pixel `readPixels` on WebGL2), median of 180 frames after 20 warm-up frames. First frame (ms) runs from the surface message received to the first frame completed on the GPU, including geometry set-up (one bounding-sphere pass) and shader compilation. three.js renders with 4× MSAA, vtk.js with its defaults.

| Triangles | three.js WebGPU: first / frame | three.js WebGL2: first / frame | vtk.js 38 WebGL: first / frame |
|---|---|---|---|
| 1 M | 287 / 1.0 ms | 172 / 1.9 ms | 376 / 1.2 ms |
| 5 M | 190 / 2.4 ms | 100 / 3.1 ms | 1158 / 2.1 ms |
| 10 M | 457 / 3.3 ms | 223 / 4.2 ms | 2246 / 3.1 ms |
| 20 M | 782 / 5.3 ms | 413 / 6.2 ms | 4550 / 5.3 ms |

Once loaded, all three draw 20 M triangles in about 5–6 ms, far under a 16.7 ms frame. vtk.js takes 6–11× longer to show the first frame at 20 M, because it rebuilds its own vertex buffers on the CPU, and that cost returns at every change of surface or snapshot during a live run. Its WebGPU backend did not initialise in this setup. vtk.js's strengths (cutters, contour filters, transfer functions) are server-side operations in this architecture.

Transfer of the 20 M-triangle message (400 MB, loopback): 1.0–1.5 s; at 5 M triangles the whole path from request to first frame is about 0.4 s. The measurement uses a structured triangulation with good vertex locality; an unstructured surface will be slower to draw and is re-measured when `solution.cgns` surfaces are served.

Reproduce: `cd ui && bun run bench:viewer` writes one JSON per case to `ui/bench/results/`. The vtk.js comparison was a one-off prototype and is not kept.

### Studies and Compare

There is no study entity in `autan.results/1` yet: Home groups runs by case name and calls a group a study when its runs differ in the swept settings of their summaries, labelled with the range (`α 7.05–21.47 deg`). On Home, `x` selects the focused row, `Space` previews it, `c` compares the selection (also from a group header). `/compare?runs=[…]` (≤ 8, in selection order, A first) overlays residual drop, CL and CD against iteration with a shared cursor, draws the lift and drag polars of the converged runs with one forces monitor giving CL and CD: one series per problem solved but for the angle of attack (the same stored case settings, as written, without name, time budget and monitors; the same mesh hash; the same coefficient monitor but for its name and interval), every run a point in its run's colour with a table of values, and each run left out named with the reason; then lists the case settings that differ, with where each run ran from run.json (`run.devices`, `run.backend`, `run.solver` commit, `run.deterministic`).

### Case editor

`/cases/new` builds a case field by field against a MeshInfo of `--meshes`; beside it, the SDK's own `case.py` (`Case.to_python`) and `case.toml` of the case as typed, and its plan. The editor never writes code: the case document holds only what was set and the SDK fills its defaults, so its code is the SDK's by construction. The mesh's boundary is drawn when its `autan.surface/1` file is listed, its zones listed otherwise; each zone takes the colour of the condition the plan gives it (from a rule or from the mesh's type), the others a neutral tone, with a legend; a zone clicked gets the boundary rule whose pattern covers it, or a new one. After a change of mesh, the rules that match none of its zones are offered for removal. Once the plan is ready, the case is copied as `case.py` or saved as `<name>.toml`. Settings the SDK requires and that are still empty are listed in one line, not as errors; a value set and refused shows its code. The freestream is given by a Reynolds number or by a pressure (an inviscid flow by a pressure), and zones without a condition are one line of the plan. *Edit a copy* on a run page reads that run's case into the form; the SDK's documents of the run's case and of the form's are compared path by path, and what the form cannot say yet (an unsteady time, a custom material) is listed as what the copy changes.

### Component library

`ui/src/components/ui/` rebuilds the patterns of Vercel's public Geist design system on Base UI and `tokens.css`; no Vercel code or asset is used. Each component states its rule in one line at its definition.

| Geist pattern | Component | Used on |
|---|---|---|
| Button (primary, secondary, tertiary; 32/36/40 px) | `Button` | Run header, empty states |
| Badge (subtle tones) | `Badge` | Plan provenance, tab counts, fake backend |
| Status Dot | `StateGlyph`, `Status` (glyph, label, duration) | Home rows, run header, ⌘K |
| Table (interactive rows) | Home run list (rows are links), `PlanTable`, `MonitorTable` | Home, run page |
| Collapse | `Collapse` (count or state on the right) | Run sections, rejected directories |
| Entity | `Entity` | Outputs |
| Gauge | `Gauge` | Iteration budget, residual drop towards the target |
| Loading Dots, Skeleton | `LoadingDots`, `Skeleton` | Plan loading, Home loading, reconnecting stream |
| Menu | `Menu`, `MenuItem`, `MenuCheckboxItem` | Row actions, filter chips |
| Command Menu | `CommandPalette` (groups, preview, key bar) | ⌘K everywhere |
| Toast | `ToastProvider`, `useToast` | Copy confirmations |
| Tooltip | `Tooltip` | Hashes, fake backend, disabled actions |
| Relative Time Card | `RelativeTime` (UTC and local on hover) | Started, outputs |
| Empty State | `EmptyState` | No runs, no match |
| Snippet | `Snippet` | Empty Home, Case tab |
| Tabs (secondary) | `TabList` (sliding indicator) | Run page |
| Note | `Note` | Diagnostics, server errors |
| Keyboard Input | `Kbd` | Find, search, palette |
| Input (search) | `SearchInput`, `FilterChip` | Home filters |
| Toggle | `LiveToggle`, theme toggle group | Log, sidebar |
| Middle Truncate | `MiddleTruncate` | Mesh hash, solver commit |
| Checkbox | `Checkbox` | Run selection on Home |
| Sheet | `Peek` (Space on a row) | Home |
| Description | `MetaGrid` | Run facts |

### Viewer

- Geometry: `position`, `index` and the displayed field as a `float` attribute; no normals (flat shading from screen-space derivatives), so a 20 M-triangle surface costs 400 MB of buffers and one CPU pass (the bounding sphere) before the upload.
- Colour: the field is normalised by uniforms and looked up in a 256-texel colormap texture; changing the range or the colormap is a uniform update, not a buffer rebuild. A field that changes sign gets a range symmetric about zero, so the neutral colour marks zero and the legend shows it. Cool–warm (Moreland 2009) for signed fields; viridis for magnitudes comes with the first real fields.
- Picking: each zone is drawn with its id as colour into a 1 × 1 render target under the cursor, then read back: one GPU pass whatever the triangle count.
- Rendering is on demand (camera or state change), not a continuous loop.
- Buffer limits: the WebGPU device is requested with the adapter's `maxBufferSize`, since 10 M+ triangles exceed the 256 MiB default.

## Develop

```sh
cd ui
bun install --frozen-lockfile
uv sync --locked --project server
scripts/sample-runs.sh runs --live &                               # fake-backend runs, one live
scripts/real-runs.sh runs --live &                                 # runs of the real runtime (relaxation physics); needs the C++ build
uv run --project server python -m autan_ui --runs runs --no-open    # API on :8765
bun run dev                                                          # app on :5173, proxies /api
```

Checks: `bun run check` (Biome, `tsc`), `bun run test` (Vitest), `bun run e2e` (Playwright, builds the app and starts the server on synthetic runs), `uv run --project server pytest`, `bun run bench:viewer`.

Sample and test runs come from the SDK's fake backend (`autan run`, `autan.FakeBackend`), the real producer of `autan.results/1`, so the tests send its output through this server.
