# Diagnostic codes

Generated from `py/autan/errors.py` by `autan explain --all`; do not edit by hand.

`E` codes are errors and block the run, `W` codes are warnings, `D` codes report a divergence detected during a run. Codes `E0101` to `E0199` belong to the mesh readers ([mesh.md](../contracts/mesh.md)).

## E0001

**The case file cannot be read** (error)

The file does not exist, is not valid Python or TOML, or raised an exception while running. Python case files run as scripts; the traceback points at the failing line.

Fix: Fix the file at the reported location.

## E0002

**The case file does not define exactly one case** (error)

A Python case file must create exactly one `autan.Case`, at module level.

Fix: Create one case: `case = autan.Case("name", mesh="mesh.cgns")`.

## E0003

**The case document does not match the schema** (error)

A field is missing, has the wrong type, or is not part of `autan.case/1`. The location gives the path of the field in the document. See docs/contracts/case-schema.md.

Fix: Correct the field named in the message.

## E0004

**Unsupported case schema version** (error)

The document's `schema` key is not `autan.case/1`.

Fix: Set `schema = "autan.case/1"`.

## E0101

**Inconsistent mesh array sizes** (error)

Input arrays disagree on a size, or CSR offsets are not monotone.

Fix: Regenerate the mesh with the meshing tool; the file or partition is corrupt.

## E0102

**Mesh index out of range** (error)

A node, cell or zone index is out of range.

Fix: Regenerate the mesh; an index points past the end of its array.

## E0103

**Face with fewer than 3 nodes** (error)

A face has fewer than 3 nodes.

Fix: Remove or repair the degenerate face in the meshing tool.

## E0104

**Face with a repeated node** (error)

A face lists a node twice.

Fix: Repair the face in the meshing tool so each node appears once.

## E0105

**Face with the same cell on both sides** (error)

A face has the same cell on both sides.

Fix: Repair the mesh connectivity in the meshing tool.

## E0106

**Cell with too few faces** (error)

An owned cell has fewer than 4 faces, or a ghost cell none.

Fix: Repair the cell in the meshing tool, or regenerate the partition.

## E0107

**Cell not closed** (error)

Invariant 5 fails for a cell.

Fix: Repair the cell in the meshing tool: its faces must enclose a volume.

## D0107

**Divergence** (divergence)

The solution diverged: a residual or a state became non-finite or non-physical. `data` gives the iteration, the cell (0-based) and its cell zone.

Fix: Lower the CFL number or use a more dissipative limiter; check the mesh quality near the cell.

## E0108

**Face shared by more than two cells** (error)

More than two cells list the same face.

Fix: Repair non-manifold faces in the meshing tool.

## E0109

**Face does not match its zone's role** (error)

A face does not match its zone's role, or labels put one zone on both interior and boundary faces.

Fix: Put interior and boundary faces in separate zones in the meshing tool.

## E0110

**Duplicate zone name** (error)

Two face zones or two cell zones share a name.

Fix: Rename one of the zones in the meshing tool; `interior`, `unassigned` and `fluid` are reserved.

## E0111

**Labelled face not found** (error)

A labelled face matches no cell face.

Fix: Regenerate the mesh; a zone references a face that does not exist.

## E0112

**Empty mesh** (error)

No cell, face or point.

Fix: Check that the file is the intended mesh and was exported completely.

## E0113

**Mesh too large** (error)

A count exceeds the `index_t` range.

Fix: Partition the mesh, or use a build with 64-bit indices.

## E0114

**Inverted cell** (error)

Two cells list their common face with the same orientation (one is inverted).

Fix: Repair the inverted cell in the meshing tool.

## E0115

**Face labelled twice** (error)

A face carries two labels.

Fix: Assign each face to one zone in the meshing tool.

## E0117

**Duplicate or negative global cell id** (error)

Two cells share a global id, or one is negative.

Fix: Regenerate the partition; global cell ids must be unique and non-negative.

## E0118

**Invalid periodic pair** (error)

A periodic pair is declared wrongly: zones out of range, empty or not boundary zones of kind periodic, a zone paired with itself or in two pairs, a boundary zone of kind periodic in no pair, or a translation that is zero or not finite.

Fix: Fix the periodic declaration in the meshing tool: each periodic zone in exactly one pair, with a non-zero translation.

## E0119

**Periodic faces do not match** (error)

The faces of a periodic pair do not match one to one under its translation.

Fix: Regenerate the mesh with conforming periodic faces, and check the translation.

## E0120

**Degenerate face** (error)

A face has zero or non-finite area.

Fix: Repair the face in the meshing tool; it has no area.

## E0121

**Non-positive cell volume** (error)

An owned cell has zero, negative or non-finite volume.

Fix: Repair the cell in the meshing tool; it is inverted or flat.

## E0130

**Mesh file cannot be opened** (error)

The file or directory cannot be opened or read.

Fix: Check the path and the file permissions.

## E0131

**Unknown mesh format** (error)

The content matches no supported format.

Fix: Export the mesh as CGNS, Gmsh, Fluent .msh or an OpenFOAM polyMesh.

## E0132

**Malformed mesh file** (error)

The file violates its format specification (syntax, truncation, inconsistent counts).

Fix: Re-export the mesh; the file does not follow its format specification.

## E0133

**Unsupported mesh feature** (error)

The file is valid but uses a feature the reader does not support yet; the message names it.

Fix: Export the mesh without the feature named in the message.

## E0134

**Mesh refers to another file** (error)

The mesh refers to another file (CGNS link); meshes must be self-contained.

Fix: Export a self-contained mesh (no CGNS links to other files).

## E0135

**Entity in several zones** (error)

An entity belongs to several zones in the source (Gmsh entity in two physical groups, face in two OpenFOAM face zones).

Fix: Assign each entity to one zone in the meshing tool.

## E0136

**Zone groups file unreadable** (error)

The mesh's `<mesh>.zones.json` is not JSON, or not an `autan.zone-groups/1` document (missing or unknown key, wrong type).

Fix: Fix the file against docs/contracts/mesh-info.md, or remove it.

## E0137

**Zone groups file describes another mesh** (error)

The `mesh_sha256` of the mesh's `<mesh>.zones.json` differs from the mesh hash, or the file the runtime reads differs from the solver input's `mesh.groups_sha256`.

Fix: Regenerate the groups for this mesh, then the MeshInfo and the solver input.

## E0138

**Bad zone group** (error)

A group of the mesh's `<mesh>.zones.json` names a boundary zone the mesh does not have (names are exact, spaces included), lists a zone twice, is empty, or has a name outside `[A-Za-z0-9_.-]` or starting with `-`.

Fix: Correct the group against the zone names of the mesh's MeshInfo.

## E0139

**Zone group defined twice** (error)

Two groups of the mesh's `<mesh>.zones.json` have the same name.

Fix: Give each group one entry.

## E0140

**Invalid partition request** (error)

A part count, rank, part index or cell weight is out of range, or an array has the wrong size.

Fix: Ask for 1 to N parts (N = number of cells), one rank per part, and a weight of at least 1 per cell.

## E0141

**Mesh already partitioned** (error)

The mesh to partition already has ghost cells.

Fix: Partition the global mesh, not one of its partitions.

## E0142

**Partitioner unavailable or failed** (error)

The partitioner (METIS, Scotch) is not built or reports an error.

Fix: Use a build with the partitioner enabled, or choose the other partitioning method.

## E0143

**Empty partition** (error)

A part received no cell.

Fix: Run on fewer ranks, or relax the cell weights.

## E0144

**Inconsistent halo** (error)

The communicator size differs from the partition count, or two ranks disagree on what they exchange.

Fix: Launch one MPI rank per part, each with the partition extracted for its own rank.

## E0145

**Device halo exchange unsupported** (error)

A halo exchange on device memory without CUDA support in the build or in MPI.

Fix: Use a build with CUDA enabled and a CUDA-aware MPI, such as the dev image.

## E0146

**Several ranks without MPI** (error)

A launcher started several ranks of a build without MPI.

Fix: Run this build on one rank, or use a build with MPI, such as the dev image.

## E0201

**No mesh information** (error)

Planning a case needs the MeshInfo of its mesh (docs/contracts/mesh-info.md). The C++ mesh readers are not part of this build, so it must be passed explicitly.

Fix: Pass the MeshInfo JSON: `autan plan case.py --mesh-info mesh.meshinfo.json`.

## E0202

**The MeshInfo document is invalid** (error)

The JSON does not match `autan.mesh-info/1`, or its counts are inconsistent (zone counts that do not sum to the cell count, duplicate names, unknown interface zones).

Fix: Regenerate the MeshInfo with the mesh reader of the same solver version.

## E0203

**The MeshInfo describes another mesh file** (error)

The MeshInfo `source.sha256` differs from the SHA-256 of the case's mesh file, so it describes another mesh or another version of it.

Fix: Pass the MeshInfo of the case's mesh, or change `mesh` in the case.

## W0204

**Highly non-orthogonal cells** (warning)

The maximum face non-orthogonality exceeds 70°, the threshold above which OpenFOAM's checkMesh reports severely non-orthogonal faces. Gradients and viscous fluxes lose accuracy and implicit solves may stall.

Fix: Improve the mesh near the worst cells, or lower the CFL number.

## W0205

**Mesh file not found** (warning)

The case's mesh file does not exist where the case says, so `plan` cannot check that the MeshInfo describes it (by comparing the mesh hash).

Fix: Put the mesh at the path the case gives, relative to the case file.

## E0206

**Manufactured solution differs from its hash** (error)

The SHA-256 of the manufactured-solution file named by `verification.manufactured` differs from the `sha256` the case records, so the case would verify against another solution.

Fix: Regenerate the file, or update the case's sha256 to the file the study uses.

## E0207

**Manufactured solution not found** (error)

The file named by `verification.manufactured.path` does not exist, relative to the case file.

Fix: Export the manufactured solution with autan-vv and give its path.

## E0208

**Not a manufactured solution** (error)

The file named by `verification.manufactured.path` is not JSON with `"schema": "autan.mms/1"` (docs/validation/mms.md).

Fix: Point the case at a manufactured solution exported by autan-vv.

## E0209

**Manufactured solution does not match the case** (error)

The constants the manufactured solution declares (gamma, gas constant, viscosity, Prandtl number) differ from the case's gas, its declared model is not compressible flow with the case's turbulence model, or the case has cell zone rules a single-medium solution cannot cover. The message names the difference.

Fix: Build the case gas from the solution's parameters, or export the solution the case needs.

## E0210

**The MeshInfo's zone groups describe another groups file** (error)

The SHA-256 of the mesh's `<mesh>.zones.json` differs from the MeshInfo `groups_source.sha256`, or the file exists and the MeshInfo has no groups: the groups the case selects would not be the ones the MeshInfo lists.

Fix: Regenerate the MeshInfo with `autan-mesh-info` after changing the groups file.

## W0211

**Zone groups file not found** (warning)

The MeshInfo has zone groups but the `<mesh>.zones.json` next to the case's mesh does not exist, so `plan` cannot check that they still describe it.

Fix: Put the groups file next to the mesh.

## E0301

**Quantity of the wrong dimension** (error)

The value has a dimension other than the one the field expects, for example a speed given for the Mach number.

Fix: Pass a value of the expected dimension, with a unit from `autan.units`.

## E0302

**Missing unit** (error)

A dimensional field received a bare number. Angles also need a unit, because degrees and radians cannot be told apart.

Fix: Multiply by a unit: `19.57 * deg`, `7.0 * m`.

## E0303

**Unknown unit** (error)

The text could not be parsed as `<number> <unit>` with Pint's default registry.

Fix: Write the quantity as `"<number> <unit>"`, for example `"101325 Pa"`.

## E0304

**Value out of range** (error)

The value must be finite, and strictly positive where the schema says `> 0` (temperatures are compared in kelvin).

Fix: Use a physically valid value.

## E0305

**Turbulence model needs an unsteady run** (error)

DDES and WMLES resolve turbulent structures in time and cannot run steady.

Fix: Set `case.time = autan.Unsteady(steps=..., dt=...)`.

## E0306

**Freestream does not match the physics** (error)

Compressible flow needs `mach` and exactly one of `pressure` or `re` (with `ref_length`); incompressible flow needs `velocity` and takes no `mach`, `re`, `pressure` or `temperature`.

Fix: Set the freestream parameters required by the physics.

## E0410

**Boundary rule matches no zone** (error)

The pattern of `case.boundary(...)` matches no boundary zone of the mesh.

Fix: Use one of the zone names listed by `autan plan`, or a glob that matches some.

## E0411

**Boundary zone matched by several rules** (error)

Two `case.boundary(...)` patterns match the same zone; neither takes precedence.

Fix: Narrow the patterns so each zone matches one rule.

## E0412

**Boundary zone without a condition** (error)

The zone is not typed in the mesh file and no rule of the case matches it.

Fix: Give it a condition: `case.boundary("name").wall()` (or `slip()`, `symmetry()`, `farfield()`, `inlet(...)`, `outlet(...)`).

## E0413

**Mesh boundary type needs values** (error)

The mesh types the zone as a thermal wall, an inlet or an outlet, which needs values only the case can provide.

Fix: Set them: `case.boundary("name").outlet(static_pressure=101325 * Pa)`.

## E0414

**Boundary condition not available for this physics** (error)

`inlet` (total conditions) is compressible only, `velocity-inlet` incompressible only, a no-slip `wall` needs viscous flow, and `oracle-state` needs compressible inviscid or laminar physics.

Fix: Use the condition matching the physics: `slip()` for inviscid walls, `inlet(...)` or `velocity_inlet(...)`.

## W0415

**Case overrides the mesh boundary type** (warning)

The case applies a condition of another kind than the type stored in the mesh file. This is allowed; the warning makes the override visible.

Fix: Check that the override is intended.

## E0416

**Boundary rules with a manufactured solution** (error)

A case verified against a manufactured solution takes the data of every boundary zone from the exact solution; a `case.boundary(...)` rule would contradict it.

Fix: Remove the boundary rules from the verification case.

## E0417

**Unsteady verification case** (error)

Manufactured solutions are steady, and a run starting from the freestream would keep an initial-condition error at fixed time; `autan.case/1` verification cases are steady.

Fix: Set `case.time = autan.Steady(...)` in the verification case.

## E0420

**Face-state file missing or invalid** (error)

The file of an `oracle_state` condition cannot be read, its SHA-256 differs from the case's, or it is not a valid `autan.face-states/1` document: its variables, a row of 3 finite coordinates and 5 finite values per face, density and pressure > 0 (docs/contracts/face-states.md).

Fix: Export the face states with autan-vv for this mesh and give their path and sha256.

## E0421

**Face states do not match the boundary faces** (error)

The solver could not pair every boundary face of the zone with one face state by centroid, within 1e-9 times the largest extent of the zone's bounding box in the whole mesh.

Fix: Export the face states from the mesh the case uses.

## E0422

**Face states exported for another mesh** (error)

The file's `mesh_sha256` differs from the mesh hash of the case's MeshInfo.

Fix: Export the face states for the mesh the case uses.

## E0423

**Face states do not cover the zone** (error)

A boundary zone that takes the `oracle_state` condition is missing from the file, or the file gives another number of faces than the mesh has in that zone.

Fix: Export the face states for every zone the condition applies to.

## E0424

**Condition on a periodic zone** (error)

A boundary rule names a zone of a periodic pair; the mesh pairs its faces with those of its partner, so it takes no condition. Patterns with wildcards skip periodic zones.

Fix: Remove the rule, or change the mesh if the zone is not meant to be periodic.

## E0425

**Point-vortex surfaces match no wall** (error)

The `surfaces` of a farfield's `point_vortex` match no boundary zone, name a zone of a periodic pair, or match zones that are not wall or slip zones: the vortex's circulation comes from the lift on the body. Patterns with wildcards skip periodic zones.

Fix: Give a pattern that matches the walls of the lifting body, e.g. `surfaces="wing.*"`.

## E0426

**Zone group pattern matches no group** (error)

A selection's `group` (a glob over the zone group names of the mesh's MeshInfo) matches no group. Groups come from the mesh's `<mesh>.zones.json`.

Fix: Use one of the group names the message lists, or select zones by name.

## E0501

**Cell zone rule matches no zone** (error)

The pattern of `case.cell_zone(...)` matches no cell zone of the mesh.

Fix: Use one of the cell zone names listed by `autan plan`.

## E0502

**Cell zone matched by several rules** (error)

Two `case.cell_zone(...)` patterns match the same zone.

Fix: Narrow the patterns so each zone matches one rule.

## E0503

**Solid zone without a material** (error)

The mesh declares the zone solid and the case gives it no material.

Fix: Set it: `case.cell_zone("name").solid(autan.materials.COPPER)`.

## E0504

**Material does not match the physics** (error)

Compressible fluid zones need an ideal gas; incompressible fluid zones need a constant-density fluid.

Fix: Use a material of the right kind for the physics.

## E0505

**Fluids of different materials in contact** (error)

Two fluid zones with different materials share an interface. Mixing them needs species transport, which `autan.case/1` does not provide.

Fix: Use the same material on both sides, or separate them by a solid zone.

## E0506

**Solid zone with incompressible physics** (error)

Conjugate heat transfer needs an energy equation in the fluid, which incompressible flow does not solve in `autan.case/1`.

Fix: Use `autan.Compressible(...)`, or remove the solid zone from the mesh.

## E0507

**Initial state does not match the physics** (error)

A compressible fluid zone starts from an absolute pressure > 0, a temperature and a velocity; an incompressible one from a gauge pressure and a velocity, without temperature.

Fix: Give the state the physics needs: `autan.State(pressure=..., temperature=..., velocity=(u, v, w))`.

## E0508

**Initial-state file missing or invalid** (error)

The file of `initial_state` cannot be read, is not a regular file, its SHA-256 differs from the case's, or it is not a valid `autan.initial-state/1` document: its variables, a row of 3 finite coordinates and 5 finite values per cell, density and pressure > 0 (docs/contracts/initial-state.md).

Fix: Write the initial state with `autan.initial_state.write` for this mesh, and give its path and the SHA-256 it returns.

## E0509

**Initial state written for another mesh** (error)

The file's `mesh_sha256` differs from the mesh hash of the case's MeshInfo.

Fix: Write the initial state for the mesh the case uses.

## E0510

**Initial state does not cover the fluid cells** (error)

The file gives another number of rows than the mesh has fluid cells.

Fix: Write one row per fluid cell of the mesh, at its centroid.

## E0511

**Initial states do not match the cells** (error)

The solver could not pair every fluid cell with exactly one row of the file by centroid, within 1e-9 times the largest extent of the mesh's bounding box, or two cells lie within that tolerance of each other; `data` names the cells.

Fix: Write the initial state from the cell centroids of the mesh the case uses.

## E0512

**Initial-state file not available for this case** (error)

An initial-state file needs compressible inviscid or laminar physics (it carries no turbulence variable), replaces the fluid zones' `initial` states, and cannot be combined with a start from a previous run, which gives the initial state itself.

Fix: Remove the zones' `initial`, the start, or the file; or use a compressible inviscid or laminar case.

## E0601

**Monitor surfaces match no boundary zone** (error)

The `surfaces` pattern of a monitor matches no boundary zone of the mesh.

Fix: Use a pattern that matches the zones to integrate over.

## E0602

**Duplicate monitor name** (error)

Two monitors share a name; results are keyed by monitor name.

Fix: Name them: `autan.forces("wing.*", name="wing")`.

## E0603

**Monitor on a periodic zone** (error)

A monitor names a zone of a periodic pair: its faces are paired with those of its partner, so they carry no load. Patterns with wildcards skip periodic zones.

Fix: Integrate over the walls instead, or change the mesh if the zone is not meant to be periodic.

## E0701

**The run directory does not conform to autan.results/1** (error)

A file is missing or unreadable, `run.json` has another schema, the stored case or MeshInfo does not match the hashes of `run.json`, or the history is out of order. See docs/contracts/results.md.

Fix: Open a directory written by a backend of this solver version.

## E0702

**No such monitor** (error)

The history has no monitor of that name, or `results.forces` was used on a case with zero or several forces monitors.

Fix: Use `results.history.monitor(name)` with a name from `results.history.monitors`.

## E0703

**No such solution data** (error)

The run has no `solution.cgns` (it failed or diverged before writing one), or the requested field is not in it.

Fix: Check `results.run.status`, and use a field name listed in the message.

## E0704

**The run failed** (error)

The backend stopped on an unexpected error; the message names it. The run directory keeps the history written so far.

Fix: Report the message with the run directory.

## E0705

**Invalid solver input** (error)

`autan-run` rejected its input: an unknown or missing key, a wrong type or schema, or a value that the boundary kind or turbulence model requires or forbids. `data.path` is the path of the key. See docs/contracts/solver-input.md.

Fix: Regenerate the input with `autan prepare` from the same Autan version as the runtime.

## E0706

**Physics not in this solver build** (error)

The solver build has no implementation for the requested combination of physics, turbulence model, precision and device kind, or for the case's media, zones and conditions (a solid zone, several gases, a condition the physics lacks); `data` names the combination or the zone.

Fix: Use a build that includes it, or change the case or the devices.

## E0707

**Previous run cannot be resumed** (error)

A start needs a previous run that ended converged, completed or stopped after at least one iteration and holds a solution that conforms to its run.json; diverged and failed runs, and runs without a sound solution, cannot be resumed.

Fix: Start from another run, or run the case from the freestream.

## E0708

**Start does not match the case** (error)

`continue` needs the previous run's case unchanged except its iteration or step budget, and a budget above the iterations already done; `warm` needs the same physics and transported variables. The message names the difference.

Fix: Change the case, or use `warm` to start a different case from that state.

## E0709

**Start solution differs** (error)

The solution of the run to start from no longer has the SHA-256 the solver input records: the run directory changed after the input was written.

Fix: Prepare the input again from the run as it is now.

## W0710

**Exempt equation left its floor** (warning)

An equation that was identically zero at iteration 1 (below 1e-12 times the largest initial residual) rose above that floor, so it must now drop `residual_orders` like the others (case-schema.md#time). `data` gives the equation, iteration, residual and floor.

Fix: Check the boundary conditions and the mesh: a symmetric or 2D problem whose symmetry breaks during the run usually means a boundary condition breaks it.

## W0801

**No NVIDIA GPU found** (warning)

`nvidia-smi` is missing or lists no GPU. CUDA runs are not possible on this machine.

Fix: Install the NVIDIA driver, or run on a GPU machine.

## W0802

**No MPI launcher found** (warning)

Neither `mpirun` nor `mpiexec` is on the PATH. Multi-GPU runs need MPI.

Fix: Install a GPU-aware MPI (Open MPI or MPICH built with CUDA support).

## W0803

**Solver core not built** (warning)

This installation contains the SDK only: cases can be planned, and run only with the `fake` backend, which writes synthetic results.

Fix: Build the C++ core (see README) to run real computations.

## E0804

**No solver backend** (error)

`case.run()` needs a backend, and the solver backend needs the runtime `autan-run`, found through `$AUTAN_RUN` or the PATH.

Fix: Pass one: `case.run(mesh_info, backend=autan.SolverBackend())` with autan-run installed, or `autan.FakeBackend()` for synthetic results.

## E0805

**Web UI not installed** (error)

`autan ui` needs the optional `autan-ui` package, which this installation lacks.

Fix: Install the extra: `uv add 'autan[ui]'` (or `uv sync --extra ui` in a checkout).

## E0806

**Cannot write the output file** (error)

The file given to `-o`/`--output` cannot be written: missing directory, a directory in its place, or no permission.

Fix: Give a writable file path.

## E0807

**No GPU device** (error)

The input asks for `devices.kind = cuda` and this build of autan-run drives CUDA, but the process sees no CUDA device (`data.visible`, `data.needed`).

Fix: Run on a host with a GPU visible to the process (`docker --gpus`), or use `devices.kind = cpu`.

## E0808

**No MPI launcher** (error)

A run on several devices starts one rank per device through an MPI launcher, found through `$AUTAN_MPIEXEC`, or `mpiexec` then `mpirun` on the PATH.

Fix: Install a GPU-aware MPI (Open MPI or MPICH built with CUDA support), set `$AUTAN_MPIEXEC`, or run on one device.

## E1001

**Unsupported kernel construct** (error)

A body uses a statement, expression or call outside the subset.

Fix: Rewrite the body with the DSL subset of docs/design/kernel-generator.md.

## E1002

**Argument is not a DSL type** (error)

An argument has no annotation, or one that is not a role type.

Fix: Annotate every argument with a role type (Geom, State, Residual, Param, ...).

## E1003

**Invalid loop binding** (error)

Loop bindings do not match the body's arguments, their types, or the loop kind.

Fix: Make the loop's bindings match the body's arguments, their types and the loop kind.

## E1004

**Invalid loop output** (error)

A loop's outputs do not fit its kind (`Inc` signs, `Write`, accumulators).

Fix: Give the loop outputs that fit its kind (Inc with signs, Write, accumulators).

## E1005

**Invalid control flow in a kernel** (error)

A name is assigned in one branch only, `return` is in one branch only, or code follows `return`.

Fix: Assign each name and return on every branch, and put no code after return.

## E1006

**Type mismatch in a kernel** (error)

Shapes or kinds of operands disagree.

Fix: Make the operand shapes and kinds agree.

## E1007

**rel() of a computed value** (error)

`rel()` receives something other than `Geom` arguments as loaded.

Fix: Call rel() on Geom arguments as loaded, not on computed values.

## E1008

**Inc target is not an accumulation** (error)

An `Inc` target does not have role `Residual`.

Fix: Accumulate with Inc only into Residual datasets.

## E1009

**Invalid kernel module** (error)

A kernel module cannot be imported or declares no loop.

Fix: Fix the module so it imports and declares at least one loop.

## E1010

**Unknown backend** (error)

`--backends` names an unknown backend.

Fix: Use a backend listed by the generator's --help.

## E1011

**Unknown policy** (error)

`--policies` names an unknown policy.

Fix: Use a policy listed by the generator's --help.

## E1012

**Differentiation not supported** (error)

A loop with several output datasets is differentiated, or a boundary loop writes face data computed from active data without frozen=True.

Fix: Differentiate loops with a single output dataset; declare boundary-state loops frozen=True to drop their derivatives knowingly.

## E1013

**Policy not available on the backend** (error)

A precision policy is requested on a backend that cannot run it (`fp64` on Metal).

Fix: Use fp32+acc64 or fp32 on Metal; run fp64 on the CPU, CUDA or HIP backends.

## E1100

**Generated loop failed** (error)

A generated loop or the spike driver fails without a more specific code.

Fix: Report the message with the kernel and the input that triggered it.

## E1101

**GPU runtime error** (error)

A CUDA or HIP call or kernel launch fails.

Fix: Check the driver and the device with `autan doctor`, then report the message.

## E1102

**Unknown precision policy** (error)

A driver receives an unknown policy name.

Fix: Use one of the precision policies the driver lists.

## E1103

**Unknown race strategy** (error)

A driver receives an unknown strategy name.

Fix: Use one of the race strategies the driver lists.

## E1104

**Face scratch too narrow** (error)

Store-gather scratch has fewer components per face than the loop's outputs.

Fix: Regenerate the loop so its scratch holds every output component per face.

## E1105

**Incidence scratch too narrow** (error)

The deterministic adjoint of a cell reduction has fewer scratch components per incidence than the values it sends to neighbours.

Fix: Size the schedule's incidence scratch for the widest reduction adjoint of the module.

## E1106

**Incidence twins missing** (error)

The deterministic adjoint of a cell reduction runs on a topology without incidence twins.

Fix: Set Topology::incidence_twin from autan::kernels::incidence_twins(mesh) and copy it to the device.

## E1107

**Metal runtime error** (error)

A Metal call, library compilation or command buffer fails.

Fix: Check that the Mac has an Apple GPU with `autan doctor`, then report the message.

## E1108

**Strategy not available** (error)

A face-loop strategy needs atomics the backend lacks for the policy's accumulation type.

Fix: Use a deterministic strategy (coloured, store-gather, gather-recompute) or fp32.

## E1109

**Strategy not rank-independent** (error)

A face or boundary loop on a rank-independent topology uses a strategy whose order follows the local numbering (scatter, atomic, warp-aggregated, coloured).

Fix: Use store-gather or gather-recompute, whose per-cell order is the global one, or build the topology without the global order.
