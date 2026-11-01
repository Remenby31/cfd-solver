"""Diagnostic codes: one registry, rendered for humans, JSON and the reference page.

A code is ``E####`` (error: blocks the run), ``W####`` (warning) or ``D####``
(divergence detected during a run). ``docs/sdk/errors.md`` is generated from
``REGISTRY`` by ``autan explain --all``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

DOC_URL = "https://remenby31.github.io/autan/sdk/errors/"

Severity = Literal["error", "warning", "divergence"]
_SEVERITY: dict[str, Severity] = {"E": "error", "W": "warning", "D": "divergence"}


@dataclass(frozen=True)
class Code:
    code: str
    title: str
    explanation: str
    fix: str

    @property
    def severity(self) -> Severity:
        return _SEVERITY[self.code[0]]


_CODES = [
    Code(
        "E0001",
        "The case file cannot be read",
        "The file does not exist, is not valid Python or TOML, or raised an exception while "
        "running. Python case files run as scripts; the traceback points at the failing line.",
        "Fix the file at the reported location.",
    ),
    Code(
        "E0002",
        "The case file does not define exactly one case",
        "A Python case file must create exactly one `autan.Case`, at module level.",
        'Create one case: `case = autan.Case("name", mesh="mesh.cgns")`.',
    ),
    Code(
        "E0003",
        "The case document does not match the schema",
        "A field is missing, has the wrong type, or is not part of `autan.case/1`. The location "
        "gives the path of the field in the document. See docs/contracts/case-schema.md.",
        "Correct the field named in the message.",
    ),
    Code(
        "E0004",
        "Unsupported case schema version",
        "The document's `schema` key is not `autan.case/1`.",
        'Set `schema = "autan.case/1"`.',
    ),
    Code(
        "E0101",
        "Inconsistent mesh array sizes",
        "Input arrays disagree on a size, or CSR offsets are not monotone.",
        "Regenerate the mesh with the meshing tool; the file or partition is corrupt.",
    ),
    Code(
        "E0102",
        "Mesh index out of range",
        "A node, cell or zone index is out of range.",
        "Regenerate the mesh; an index points past the end of its array.",
    ),
    Code(
        "E0103",
        "Face with fewer than 3 nodes",
        "A face has fewer than 3 nodes.",
        "Remove or repair the degenerate face in the meshing tool.",
    ),
    Code(
        "E0104",
        "Face with a repeated node",
        "A face lists a node twice.",
        "Repair the face in the meshing tool so each node appears once.",
    ),
    Code(
        "E0105",
        "Face with the same cell on both sides",
        "A face has the same cell on both sides.",
        "Repair the mesh connectivity in the meshing tool.",
    ),
    Code(
        "E0106",
        "Cell with too few faces",
        "An owned cell has fewer than 4 faces, or a ghost cell none.",
        "Repair the cell in the meshing tool, or regenerate the partition.",
    ),
    Code(
        "E0107",
        "Cell not closed",
        "Invariant 5 fails for a cell.",
        "Repair the cell in the meshing tool: its faces must enclose a volume.",
    ),
    Code(
        "E0108",
        "Face shared by more than two cells",
        "More than two cells list the same face.",
        "Repair non-manifold faces in the meshing tool.",
    ),
    Code(
        "E0109",
        "Face does not match its zone's role",
        "A face does not match its zone's role, or labels put one zone on both interior and boundary faces.",
        "Put interior and boundary faces in separate zones in the meshing tool.",
    ),
    Code(
        "E0110",
        "Duplicate zone name",
        "Two face zones or two cell zones share a name.",
        "Rename one of the zones in the meshing tool; `interior`, `unassigned` and `fluid` are reserved.",
    ),
    Code(
        "E0111",
        "Labelled face not found",
        "A labelled face matches no cell face.",
        "Regenerate the mesh; a zone references a face that does not exist.",
    ),
    Code(
        "E0112",
        "Empty mesh",
        "No cell, face or point.",
        "Check that the file is the intended mesh and was exported completely.",
    ),
    Code(
        "E0113",
        "Mesh too large",
        "A count exceeds the `index_t` range.",
        "Partition the mesh, or use a build with 64-bit indices.",
    ),
    Code(
        "E0114",
        "Inverted cell",
        "Two cells list their common face with the same orientation (one is inverted).",
        "Repair the inverted cell in the meshing tool.",
    ),
    Code(
        "E0115",
        "Face labelled twice",
        "A face carries two labels.",
        "Assign each face to one zone in the meshing tool.",
    ),
    Code(
        "E0117",
        "Duplicate or negative global cell id",
        "Two cells share a global id, or one is negative.",
        "Regenerate the partition; global cell ids must be unique and non-negative.",
    ),
    Code(
        "E0118",
        "Invalid periodic pair",
        "A periodic pair is declared wrongly: zones out of range, empty or not boundary zones of kind "
        "periodic, a zone paired with itself or in two pairs, a boundary zone of kind periodic in "
        "no pair, or a translation that is zero or not finite.",
        "Fix the periodic declaration in the meshing tool: each periodic zone in exactly one "
        "pair, with a non-zero translation.",
    ),
    Code(
        "E0119",
        "Periodic faces do not match",
        "The faces of a periodic pair do not match one to one under its translation.",
        "Regenerate the mesh with conforming periodic faces, and check the translation.",
    ),
    Code(
        "E0120",
        "Degenerate face",
        "A face has zero or non-finite area.",
        "Repair the face in the meshing tool; it has no area.",
    ),
    Code(
        "E0121",
        "Non-positive cell volume",
        "An owned cell has zero, negative or non-finite volume.",
        "Repair the cell in the meshing tool; it is inverted or flat.",
    ),
    Code(
        "E0130",
        "Mesh file cannot be opened",
        "The file or directory cannot be opened or read.",
        "Check the path and the file permissions.",
    ),
    Code(
        "E0131",
        "Unknown mesh format",
        "The content matches no supported format.",
        "Export the mesh as CGNS, Gmsh, Fluent .msh or an OpenFOAM polyMesh.",
    ),
    Code(
        "E0132",
        "Malformed mesh file",
        "The file violates its format specification (syntax, truncation, inconsistent counts).",
        "Re-export the mesh; the file does not follow its format specification.",
    ),
    Code(
        "E0133",
        "Unsupported mesh feature",
        "The file is valid but uses a feature the reader does not support yet; the message names it.",
        "Export the mesh without the feature named in the message.",
    ),
    Code(
        "E0134",
        "Mesh refers to another file",
        "The mesh refers to another file (CGNS link); meshes must be self-contained.",
        "Export a self-contained mesh (no CGNS links to other files).",
    ),
    Code(
        "E0135",
        "Entity in several zones",
        "An entity belongs to several zones in the source (Gmsh entity in two physical groups, face in two OpenFOAM face zones).",
        "Assign each entity to one zone in the meshing tool.",
    ),
    Code(
        "E0136",
        "Zone groups file unreadable",
        "The mesh's `<mesh>.zones.json` is not JSON, or not an `autan.zone-groups/1` document "
        "(missing or unknown key, wrong type).",
        "Fix the file against docs/contracts/mesh-info.md, or remove it.",
    ),
    Code(
        "E0137",
        "Zone groups file describes another mesh",
        "The `mesh_sha256` of the mesh's `<mesh>.zones.json` differs from the mesh hash, or the "
        "file the runtime reads differs from the solver input's `mesh.groups_sha256`.",
        "Regenerate the groups for this mesh, then the MeshInfo and the solver input.",
    ),
    Code(
        "E0138",
        "Bad zone group",
        "A group of the mesh's `<mesh>.zones.json` names a boundary zone the mesh does not "
        "have (names are exact, spaces included), lists a zone twice, is empty, or has a name "
        "outside `[A-Za-z0-9_.-]` or starting with `-`.",
        "Correct the group against the zone names of the mesh's MeshInfo.",
    ),
    Code(
        "E0139",
        "Zone group defined twice",
        "Two groups of the mesh's `<mesh>.zones.json` have the same name.",
        "Give each group one entry.",
    ),
    Code(
        "E0140",
        "Invalid partition request",
        "A part count, rank, part index or cell weight is out of range, or an array has the wrong size.",
        "Ask for 1 to N parts (N = number of cells), one rank per part, and a weight of at least 1 per cell.",
    ),
    Code(
        "E0141",
        "Mesh already partitioned",
        "The mesh to partition already has ghost cells.",
        "Partition the global mesh, not one of its partitions.",
    ),
    Code(
        "E0142",
        "Partitioner unavailable or failed",
        "The partitioner (METIS, Scotch) is not built or reports an error.",
        "Use a build with the partitioner enabled, or choose the other partitioning method.",
    ),
    Code(
        "E0143",
        "Empty partition",
        "A part received no cell.",
        "Run on fewer ranks, or relax the cell weights.",
    ),
    Code(
        "E0144",
        "Inconsistent halo",
        "The communicator size differs from the partition count, or two ranks disagree on what they exchange.",
        "Launch one MPI rank per part, each with the partition extracted for its own rank.",
    ),
    Code(
        "E0145",
        "Device halo exchange unsupported",
        "A halo exchange on device memory without CUDA support in the build or in MPI.",
        "Use a build with CUDA enabled and a CUDA-aware MPI, such as the dev image.",
    ),
    Code(
        "E0146",
        "Several ranks without MPI",
        "A launcher started several ranks of a build without MPI.",
        "Run this build on one rank, or use a build with MPI, such as the dev image.",
    ),
    Code(
        "E0201",
        "No mesh information",
        "Planning a case needs the MeshInfo of its mesh (docs/contracts/mesh-info.md). The "
        "C++ mesh readers are not part of this build, so it must be passed explicitly.",
        "Pass the MeshInfo JSON: `autan plan case.py --mesh-info mesh.meshinfo.json`.",
    ),
    Code(
        "E0202",
        "The MeshInfo document is invalid",
        "The JSON does not match `autan.mesh-info/1`, or its counts are inconsistent (zone "
        "counts that do not sum to the cell count, duplicate names, unknown interface zones).",
        "Regenerate the MeshInfo with the mesh reader of the same solver version.",
    ),
    Code(
        "E0203",
        "The MeshInfo describes another mesh file",
        "The MeshInfo `source.sha256` differs from the SHA-256 of the case's mesh file, so it describes another mesh or another version of it.",
        "Pass the MeshInfo of the case's mesh, or change `mesh` in the case.",
    ),
    Code(
        "W0205",
        "Mesh file not found",
        "The case's mesh file does not exist where the case says, so `plan` cannot check "
        "that the MeshInfo describes it (by comparing the mesh hash).",
        "Put the mesh at the path the case gives, relative to the case file.",
    ),
    Code(
        "E0210",
        "The MeshInfo's zone groups describe another groups file",
        "The SHA-256 of the mesh's `<mesh>.zones.json` differs from the MeshInfo "
        "`groups_source.sha256`, or the file exists and the MeshInfo has no groups: the groups "
        "the case selects would not be the ones the MeshInfo lists.",
        "Regenerate the MeshInfo with `autan-mesh-info` after changing the groups file.",
    ),
    Code(
        "W0211",
        "Zone groups file not found",
        "The MeshInfo has zone groups but the `<mesh>.zones.json` next to the case's mesh does "
        "not exist, so `plan` cannot check that they still describe it.",
        "Put the groups file next to the mesh.",
    ),
    Code(
        "E0206",
        "Manufactured solution differs from its hash",
        "The SHA-256 of the manufactured-solution file named by `verification.manufactured` "
        "differs from the `sha256` the case records, so the case would verify against "
        "another solution.",
        "Regenerate the file, or update the case's sha256 to the file the study uses.",
    ),
    Code(
        "E0207",
        "Manufactured solution not found",
        "The file named by `verification.manufactured.path` does not exist, relative to the "
        "case file.",
        "Export the manufactured solution with autan-vv and give its path.",
    ),
    Code(
        "E0208",
        "Not a manufactured solution",
        "The file named by `verification.manufactured.path` is not JSON with "
        '`"schema": "autan.mms/1"` (docs/validation/mms.md).',
        "Point the case at a manufactured solution exported by autan-vv.",
    ),
    Code(
        "E0209",
        "Manufactured solution does not match the case",
        "The constants the manufactured solution declares (gamma, gas constant, viscosity, "
        "Prandtl number) differ from the case's gas, its declared model is not compressible "
        "flow with the case's turbulence model, or the case has cell zone rules a "
        "single-medium solution cannot cover. The message names the difference.",
        "Build the case gas from the solution's parameters, or export the solution the case needs.",
    ),
    Code(
        "W0204",
        "Highly non-orthogonal cells",
        "The maximum face non-orthogonality exceeds 70°, the threshold above which "
        "OpenFOAM's checkMesh reports severely non-orthogonal faces. Gradients and viscous "
        "fluxes lose accuracy and implicit solves may stall.",
        "Improve the mesh near the worst cells, or lower the CFL number.",
    ),
    Code(
        "E0301",
        "Quantity of the wrong dimension",
        "The value has a dimension other than the one the field expects, for example a speed "
        "given for the Mach number.",
        "Pass a value of the expected dimension, with a unit from `autan.units`.",
    ),
    Code(
        "E0302",
        "Missing unit",
        "A dimensional field received a bare number. Angles also need a unit, because "
        "degrees and radians cannot be told apart.",
        "Multiply by a unit: `19.57 * deg`, `7.0 * m`.",
    ),
    Code(
        "E0303",
        "Unknown unit",
        "The text could not be parsed as `<number> <unit>` with Pint's default registry.",
        'Write the quantity as `"<number> <unit>"`, for example `"101325 Pa"`.',
    ),
    Code(
        "E0304",
        "Value out of range",
        "The value must be finite, and strictly positive where the schema says `> 0` "
        "(temperatures are compared in kelvin).",
        "Use a physically valid value.",
    ),
    Code(
        "E0305",
        "Turbulence model needs an unsteady run",
        "DDES and WMLES resolve turbulent structures in time and cannot run steady.",
        "Set `case.time = autan.Unsteady(steps=..., dt=...)`.",
    ),
    Code(
        "E0306",
        "Freestream does not match the physics",
        "Compressible flow needs `mach` and exactly one of `pressure` or `re` (with "
        "`ref_length`); incompressible flow needs `velocity` and takes no `mach`, `re`, "
        "`pressure` or `temperature`.",
        "Set the freestream parameters required by the physics.",
    ),
    Code(
        "E0410",
        "Boundary rule matches no zone",
        "The pattern of `case.boundary(...)` matches no boundary zone of the mesh.",
        "Use one of the zone names listed by `autan plan`, or a glob that matches some.",
    ),
    Code(
        "E0411",
        "Boundary zone matched by several rules",
        "Two `case.boundary(...)` patterns match the same zone; neither takes precedence.",
        "Narrow the patterns so each zone matches one rule.",
    ),
    Code(
        "E0412",
        "Boundary zone without a condition",
        "The zone is not typed in the mesh file and no rule of the case matches it.",
        'Give it a condition: `case.boundary("name").wall()` (or `slip()`, `symmetry()`, '
        "`farfield()`, `inlet(...)`, `outlet(...)`).",
    ),
    Code(
        "E0413",
        "Mesh boundary type needs values",
        "The mesh types the zone as a thermal wall, an inlet or an outlet, which needs values "
        "only the case can provide.",
        'Set them: `case.boundary("name").outlet(static_pressure=101325 * Pa)`.',
    ),
    Code(
        "E0414",
        "Boundary condition not available for this physics",
        "`inlet` (total conditions) is compressible only, `velocity-inlet` incompressible "
        "only, a no-slip `wall` needs viscous flow, and `oracle-state` needs compressible "
        "inviscid or laminar physics.",
        "Use the condition matching the physics: `slip()` for inviscid walls, `inlet(...)` or "
        "`velocity_inlet(...)`.",
    ),
    Code(
        "W0415",
        "Case overrides the mesh boundary type",
        "The case applies a condition of another kind than the type stored in the mesh file. "
        "This is allowed; the warning makes the override visible.",
        "Check that the override is intended.",
    ),
    Code(
        "E0416",
        "Boundary rules with a manufactured solution",
        "A case verified against a manufactured solution takes the data of every boundary zone "
        "from the exact solution; a `case.boundary(...)` rule would contradict it.",
        "Remove the boundary rules from the verification case.",
    ),
    Code(
        "E0417",
        "Unsteady verification case",
        "Manufactured solutions are steady, and a run starting from the freestream would keep "
        "an initial-condition error at fixed time; `autan.case/1` verification cases are "
        "steady.",
        "Set `case.time = autan.Steady(...)` in the verification case.",
    ),
    Code(
        "E0420",
        "Face-state file missing or invalid",
        "The file of an `oracle_state` condition cannot be read, its SHA-256 differs from the "
        "case's, or it is not a valid `autan.face-states/1` document: its variables, a row of "
        "3 finite coordinates and 5 finite values per face, density and pressure > 0 "
        "(docs/contracts/face-states.md).",
        "Export the face states with autan-vv for this mesh and give their path and sha256.",
    ),
    Code(
        "E0421",
        "Face states do not match the boundary faces",
        "The solver could not pair every boundary face of the zone with one face state by "
        "centroid, within 1e-9 times the largest extent of the zone's bounding box in the whole mesh.",
        "Export the face states from the mesh the case uses.",
    ),
    Code(
        "E0422",
        "Face states exported for another mesh",
        "The file's `mesh_sha256` differs from the mesh hash of the case's MeshInfo.",
        "Export the face states for the mesh the case uses.",
    ),
    Code(
        "E0423",
        "Face states do not cover the zone",
        "A boundary zone that takes the `oracle_state` condition is missing from the file, or "
        "the file gives another number of faces than the mesh has in that zone.",
        "Export the face states for every zone the condition applies to.",
    ),
    Code(
        "E0424",
        "Condition on a periodic zone",
        "A boundary rule names a zone of a periodic pair; the mesh pairs its faces with those "
        "of its partner, so it takes no condition. Patterns with wildcards skip periodic zones.",
        "Remove the rule, or change the mesh if the zone is not meant to be periodic.",
    ),
    Code(
        "E0425",
        "Point-vortex surfaces match no wall",
        "The `surfaces` of a farfield's `point_vortex` match no boundary zone, name a zone of "
        "a periodic pair, or match zones that are not wall or slip zones: the vortex's "
        "circulation comes from the lift on the body. Patterns with wildcards skip periodic "
        "zones.",
        'Give a pattern that matches the walls of the lifting body, e.g. `surfaces="wing.*"`.',
    ),
    Code(
        "E0426",
        "Zone group pattern matches no group",
        "A selection's `group` (a glob over the zone group names of the mesh's MeshInfo) "
        "matches no group. Groups come from the mesh's `<mesh>.zones.json`.",
        "Use one of the group names the message lists, or select zones by name.",
    ),
    Code(
        "E0501",
        "Cell zone rule matches no zone",
        "The pattern of `case.cell_zone(...)` matches no cell zone of the mesh.",
        "Use one of the cell zone names listed by `autan plan`.",
    ),
    Code(
        "E0502",
        "Cell zone matched by several rules",
        "Two `case.cell_zone(...)` patterns match the same zone.",
        "Narrow the patterns so each zone matches one rule.",
    ),
    Code(
        "E0503",
        "Solid zone without a material",
        "The mesh declares the zone solid and the case gives it no material.",
        'Set it: `case.cell_zone("name").solid(autan.materials.COPPER)`.',
    ),
    Code(
        "E0504",
        "Material does not match the physics",
        "Compressible fluid zones need an ideal gas; incompressible fluid zones need a "
        "constant-density fluid.",
        "Use a material of the right kind for the physics.",
    ),
    Code(
        "E0505",
        "Fluids of different materials in contact",
        "Two fluid zones with different materials share an interface. Mixing them needs "
        "species transport, which `autan.case/1` does not provide.",
        "Use the same material on both sides, or separate them by a solid zone.",
    ),
    Code(
        "E0506",
        "Solid zone with incompressible physics",
        "Conjugate heat transfer needs an energy equation in the fluid, which incompressible "
        "flow does not solve in `autan.case/1`.",
        "Use `autan.Compressible(...)`, or remove the solid zone from the mesh.",
    ),
    Code(
        "E0507",
        "Initial state does not match the physics",
        "A compressible fluid zone starts from an absolute pressure > 0, a temperature and a "
        "velocity; an incompressible one from a gauge pressure and a velocity, without "
        "temperature.",
        "Give the state the physics needs: `autan.State(pressure=..., temperature=..., "
        "velocity=(u, v, w))`.",
    ),
    Code(
        "E0508",
        "Initial-state file missing or invalid",
        "The file of `initial_state` cannot be read, is not a regular file, its SHA-256 differs "
        "from the case's, or it is not a valid `autan.initial-state/1` document: its "
        "variables, a row of 3 finite coordinates and 5 finite values per cell, density and "
        "pressure > 0 (docs/contracts/initial-state.md).",
        "Write the initial state with `autan.initial_state.write` for this mesh, and give "
        "its path and the SHA-256 it returns.",
    ),
    Code(
        "E0509",
        "Initial state written for another mesh",
        "The file's `mesh_sha256` differs from the mesh hash of the case's MeshInfo.",
        "Write the initial state for the mesh the case uses.",
    ),
    Code(
        "E0510",
        "Initial state does not cover the fluid cells",
        "The file gives another number of rows than the mesh has fluid cells.",
        "Write one row per fluid cell of the mesh, at its centroid.",
    ),
    Code(
        "E0511",
        "Initial states do not match the cells",
        "The solver could not pair every fluid cell with exactly one row of the file by "
        "centroid, within 1e-9 times the largest extent of the mesh's bounding box, or two "
        "cells lie within that tolerance of each other; `data` names the cells.",
        "Write the initial state from the cell centroids of the mesh the case uses.",
    ),
    Code(
        "E0512",
        "Initial-state file not available for this case",
        "An initial-state file needs compressible inviscid or laminar physics (it carries no "
        "turbulence variable), replaces the fluid zones' `initial` states, and cannot be "
        "combined with a start from a previous run, which gives the initial state itself.",
        "Remove the zones' `initial`, the start, or the file; or use a compressible inviscid "
        "or laminar case.",
    ),
    Code(
        "E0601",
        "Monitor surfaces match no boundary zone",
        "The `surfaces` pattern of a monitor matches no boundary zone of the mesh.",
        "Use a pattern that matches the zones to integrate over.",
    ),
    Code(
        "E0602",
        "Duplicate monitor name",
        "Two monitors share a name; results are keyed by monitor name.",
        'Name them: `autan.forces("wing.*", name="wing")`.',
    ),
    Code(
        "E0603",
        "Monitor on a periodic zone",
        "A monitor names a zone of a periodic pair: its faces are paired with those of its "
        "partner, so they carry no load. Patterns with wildcards skip periodic zones.",
        "Integrate over the walls instead, or change the mesh if the zone is not meant to be "
        "periodic.",
    ),
    Code(
        "E0701",
        "The run directory does not conform to autan.results/1",
        "A file is missing or unreadable, `run.json` has another schema, the stored case or "
        "MeshInfo does not match the hashes of `run.json`, or the history is out of order. "
        "See docs/contracts/results.md.",
        "Open a directory written by a backend of this solver version.",
    ),
    Code(
        "E0702",
        "No such monitor",
        "The history has no monitor of that name, or `results.forces` was used on a case "
        "with zero or several forces monitors.",
        "Use `results.history.monitor(name)` with a name from `results.history.monitors`.",
    ),
    Code(
        "E0703",
        "No such solution data",
        "The run has no `solution.cgns` (it failed or diverged before writing one), or the "
        "requested field is not in it.",
        "Check `results.run.status`, and use a field name listed in the message.",
    ),
    Code(
        "E0704",
        "The run failed",
        "The backend stopped on an unexpected error; the message names it. The run directory "
        "keeps the history written so far.",
        "Report the message with the run directory.",
    ),
    Code(
        "E0705",
        "Invalid solver input",
        "`autan-run` rejected its input: an unknown or missing key, a wrong type or schema, or "
        "a value that the boundary kind or turbulence model requires or forbids. `data.path` "
        "is the path of the key. See docs/contracts/solver-input.md.",
        "Regenerate the input with `autan prepare` from the same Autan version as the runtime.",
    ),
    Code(
        "E0706",
        "Physics not in this solver build",
        "The solver build has no implementation for the requested combination of physics, "
        "turbulence model, precision and device kind, or for the case's media, zones and "
        "conditions (a solid zone, several gases, a condition the physics lacks); `data` names "
        "the combination or the zone.",
        "Use a build that includes it, or change the case or the devices.",
    ),
    Code(
        "E0707",
        "Previous run cannot be resumed",
        "A start needs a previous run that ended converged, completed or stopped after at least one "
        "iteration and holds a solution that conforms to its run.json; diverged and failed runs, "
        "and runs without a sound solution, cannot be resumed.",
        "Start from another run, or run the case from the freestream.",
    ),
    Code(
        "E0708",
        "Start does not match the case",
        "`continue` needs the previous run's case unchanged except its iteration or step "
        "budget, and a budget above the iterations already done; `warm` needs the same physics "
        "and transported variables. The message names the difference.",
        "Change the case, or use `warm` to start a different case from that state.",
    ),
    Code(
        "E0709",
        "Start solution differs",
        "The solution of the run to start from no longer has the SHA-256 the solver input "
        "records: the run directory changed after the input was written.",
        "Prepare the input again from the run as it is now.",
    ),
    Code(
        "W0710",
        "Exempt equation left its floor",
        "An equation that was identically zero at iteration 1 (below 1e-12 times the largest "
        "initial residual) rose above that floor, so it must now drop `residual_orders` like "
        "the others (case-schema.md#time). `data` gives the equation, iteration, residual and "
        "floor.",
        "Check the boundary conditions and the mesh: a symmetric or 2D problem whose symmetry "
        "breaks during the run usually means a boundary condition breaks it.",
    ),
    Code(
        "D0107",
        "Divergence",
        "The solution diverged: a residual or a state became non-finite or non-physical. "
        "`data` gives the iteration, the cell (0-based) and its cell zone.",
        "Lower the CFL number or use a more dissipative limiter; check the mesh quality "
        "near the cell.",
    ),
    Code(
        "E1001",
        "Unsupported kernel construct",
        "A body uses a statement, expression or call outside the subset.",
        "Rewrite the body with the DSL subset of docs/design/kernel-generator.md.",
    ),
    Code(
        "E1002",
        "Argument is not a DSL type",
        "An argument has no annotation, or one that is not a role type.",
        "Annotate every argument with a role type (Geom, State, Residual, Param, ...).",
    ),
    Code(
        "E1003",
        "Invalid loop binding",
        "Loop bindings do not match the body's arguments, their types, or the loop kind.",
        "Make the loop's bindings match the body's arguments, their types and the loop kind.",
    ),
    Code(
        "E1004",
        "Invalid loop output",
        "A loop's outputs do not fit its kind (`Inc` signs, `Write`, accumulators).",
        "Give the loop outputs that fit its kind (Inc with signs, Write, accumulators).",
    ),
    Code(
        "E1005",
        "Invalid control flow in a kernel",
        "A name is assigned in one branch only, `return` is in one branch only, or code follows `return`.",
        "Assign each name and return on every branch, and put no code after return.",
    ),
    Code(
        "E1006",
        "Type mismatch in a kernel",
        "Shapes or kinds of operands disagree.",
        "Make the operand shapes and kinds agree.",
    ),
    Code(
        "E1007",
        "rel() of a computed value",
        "`rel()` receives something other than `Geom` arguments as loaded.",
        "Call rel() on Geom arguments as loaded, not on computed values.",
    ),
    Code(
        "E1008",
        "Inc target is not an accumulation",
        "An `Inc` target does not have role `Residual`.",
        "Accumulate with Inc only into Residual datasets.",
    ),
    Code(
        "E1009",
        "Invalid kernel module",
        "A kernel module cannot be imported or declares no loop.",
        "Fix the module so it imports and declares at least one loop.",
    ),
    Code(
        "E1010",
        "Unknown backend",
        "`--backends` names an unknown backend.",
        "Use a backend listed by the generator's --help.",
    ),
    Code(
        "E1011",
        "Unknown policy",
        "`--policies` names an unknown policy.",
        "Use a policy listed by the generator's --help.",
    ),
    Code(
        "E1012",
        "Differentiation not supported",
        "A loop with several output datasets is differentiated, or a boundary loop writes face data computed from active data without frozen=True.",
        "Differentiate loops with a single output dataset; declare boundary-state loops frozen=True to drop their derivatives knowingly.",
    ),
    Code(
        "E1013",
        "Policy not available on the backend",
        "A precision policy is requested on a backend that cannot run it (`fp64` on Metal).",
        "Use fp32+acc64 or fp32 on Metal; run fp64 on the CPU, CUDA or HIP backends.",
    ),
    Code(
        "E1100",
        "Generated loop failed",
        "A generated loop or the spike driver fails without a more specific code.",
        "Report the message with the kernel and the input that triggered it.",
    ),
    Code(
        "E1101",
        "GPU runtime error",
        "A CUDA or HIP call or kernel launch fails.",
        "Check the driver and the device with `autan doctor`, then report the message.",
    ),
    Code(
        "E1102",
        "Unknown precision policy",
        "A driver receives an unknown policy name.",
        "Use one of the precision policies the driver lists.",
    ),
    Code(
        "E1103",
        "Unknown race strategy",
        "A driver receives an unknown strategy name.",
        "Use one of the race strategies the driver lists.",
    ),
    Code(
        "E1104",
        "Face scratch too narrow",
        "Store-gather scratch has fewer components per face than the loop's outputs.",
        "Regenerate the loop so its scratch holds every output component per face.",
    ),
    Code(
        "E1105",
        "Incidence scratch too narrow",
        "The deterministic adjoint of a cell reduction has fewer scratch components per incidence than the values it sends to neighbours.",
        "Size the schedule's incidence scratch for the widest reduction adjoint of the module.",
    ),
    Code(
        "E1106",
        "Incidence twins missing",
        "The deterministic adjoint of a cell reduction runs on a topology without incidence twins.",
        "Set Topology::incidence_twin from autan::kernels::incidence_twins(mesh) and copy it to the device.",
    ),
    Code(
        "E1107",
        "Metal runtime error",
        "A Metal call, library compilation or command buffer fails.",
        "Check that the Mac has an Apple GPU with `autan doctor`, then report the message.",
    ),
    Code(
        "E1108",
        "Strategy not available",
        "A face-loop strategy needs atomics the backend lacks for the policy's accumulation type.",
        "Use a deterministic strategy (coloured, store-gather, gather-recompute) or fp32.",
    ),
    Code(
        "E1109",
        "Strategy not rank-independent",
        "A face or boundary loop on a rank-independent topology uses a strategy whose order follows the local numbering (scatter, atomic, warp-aggregated, coloured).",
        "Use store-gather or gather-recompute, whose per-cell order is the global one, or build the topology without the global order.",
    ),
    Code(
        "W0801",
        "No NVIDIA GPU found",
        "`nvidia-smi` is missing or lists no GPU. CUDA runs are not possible on this machine.",
        "Install the NVIDIA driver, or run on a GPU machine.",
    ),
    Code(
        "W0802",
        "No MPI launcher found",
        "Neither `mpirun` nor `mpiexec` is on the PATH. Multi-GPU runs need MPI.",
        "Install a GPU-aware MPI (Open MPI or MPICH built with CUDA support).",
    ),
    Code(
        "W0803",
        "Solver core not built",
        "This installation contains the SDK only: cases can be planned, and run only with "
        "the `fake` backend, which writes synthetic results.",
        "Build the C++ core (see README) to run real computations.",
    ),
    Code(
        "E0804",
        "No solver backend",
        "`case.run()` needs a backend, and the solver backend needs the runtime `autan-run`, "
        "found through `$AUTAN_RUN` or the PATH.",
        "Pass one: `case.run(mesh_info, backend=autan.SolverBackend())` with autan-run "
        "installed, or `autan.FakeBackend()` for synthetic results.",
    ),
    Code(
        "E0805",
        "Web UI not installed",
        "`autan ui` needs the optional `autan-ui` package, which this installation lacks.",
        "Install the extra: `uv add 'autan[ui]'` (or `uv sync --extra ui` in a checkout).",
    ),
    Code(
        "E0806",
        "Cannot write the output file",
        "The file given to `-o`/`--output` cannot be written: missing directory, a directory "
        "in its place, or no permission.",
        "Give a writable file path.",
    ),
    Code(
        "E0807",
        "No GPU device",
        "The input asks for `devices.kind = cuda` and this build of autan-run drives CUDA, but "
        "the process sees no CUDA device (`data.visible`, `data.needed`).",
        "Run on a host with a GPU visible to the process (`docker --gpus`), or use "
        "`devices.kind = cpu`.",
    ),
    Code(
        "E0808",
        "No MPI launcher",
        "A run on several devices starts one rank per device through an MPI launcher, found "
        "through `$AUTAN_MPIEXEC`, or `mpiexec` then `mpirun` on the PATH.",
        "Install a GPU-aware MPI (Open MPI or MPICH built with CUDA support), set "
        "`$AUTAN_MPIEXEC`, or run on one device.",
    ),
]

REGISTRY: dict[str, Code] = {code.code: code for code in _CODES}


def doc_url(code: str) -> str:
    return f"{DOC_URL}#{code.lower()}"


class Location(BaseModel):
    """Where a diagnostic points: a file, a line in it, and/or a path in the case document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str | None = None
    line: int | None = None
    path: str | None = None

    def __str__(self) -> str:
        where = self.file or "<case>"
        if self.line is not None:
            where += f":{self.line}"
        if self.path:
            where += f" ({self.path})"
        return where


class Diagnostic(BaseModel):
    """One problem, with its registered code, location, suggested fix and documentation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: Severity
    message: str
    location: Location | None = None
    fix: str
    doc: str
    data: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        lines = [f"{self.severity}[{self.code}]: {self.message}"]
        if self.location is not None:
            lines.append(f"  --> {self.location}")
        lines.append(f"   = fix : {self.fix}")
        lines.append(f"   = doc : {self.doc}")
        return "\n".join(lines)


def diagnostic(
    code: str,
    message: str,
    *,
    location: Location | None = None,
    fix: str | None = None,
    **data: Any,
) -> Diagnostic:
    """Build a diagnostic for a registered code; ``fix`` overrides the generic fix."""
    entry = REGISTRY[code]
    return Diagnostic(
        code=code,
        severity=entry.severity,
        message=message,
        location=location,
        fix=fix or entry.fix,
        doc=doc_url(code),
        data=data,
    )


class CaseError(Exception):
    """Raised when a case cannot be built, loaded, planned or run; carries its diagnostics."""

    def __init__(self, diagnostics: Iterable[Diagnostic]) -> None:
        self.diagnostics = list(diagnostics)
        super().__init__("\n".join(d.render() for d in self.diagnostics))

    @property
    def codes(self) -> list[str]:
        return [d.code for d in self.diagnostics]


def registry_document() -> dict[str, Any]:
    """Every registered code as JSON, the form other components embed
    (docs/contracts/diagnostics.json)."""
    return {
        "schema": "autan.diagnostics/1",
        "codes": {
            entry.code: {
                "severity": entry.severity,
                "title": entry.title,
                "fix": entry.fix,
                "doc": doc_url(entry.code),
            }
            for entry in sorted(REGISTRY.values(), key=lambda c: c.code[1:])
        },
    }


def reference_page() -> str:
    """The Markdown reference of every registered code."""
    lines = [
        "# Diagnostic codes",
        "",
        "Generated from `py/autan/errors.py` by `autan explain --all`; do not edit by hand.",
        "",
        (
            "`E` codes are errors and block the run, `W` codes are warnings, `D` codes report "
            "a divergence detected during a run. Codes `E0101` to `E0199` belong to the mesh "
            "readers ([mesh.md](../contracts/mesh.md))."
        ),
        "",
    ]
    for entry in sorted(REGISTRY.values(), key=lambda c: c.code[1:]):
        lines += [explain(entry.code), ""]
    return "\n".join(lines)


def explain(code: str) -> str:
    """The Markdown documentation of one registered code."""
    entry = REGISTRY[code]
    return "\n".join(
        [
            f"## {entry.code}",
            "",
            f"**{entry.title}** ({entry.severity})",
            "",
            entry.explanation,
            "",
            f"Fix: {entry.fix}",
        ]
    )
