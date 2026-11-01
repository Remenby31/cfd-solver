# Contract: MeshInfo

`MeshInfo` is the JSON summary of a mesh that the C++ core emits after reading it. The SDK consumes it to resolve boundary and cell zones, to check coverage and to estimate memory, without loading the mesh itself. Schema identifier: `autan.mesh-info/1`. The JSON Schema is generated from the SDK (`autan schema mesh-info`).

Producer: the mesh readers in `io/` (owner: CFD - Noyau maillage). Consumer: `py/autan` (owner: CFD - SDK & contrats). Until the core emits it, the SDK reads MeshInfo files passed explicitly (`autan plan --mesh-info FILE`).

## Document

```json
{
  "schema": "autan.mesh-info/1",
  "source": {"path": "crm-hl.cgns", "format": "cgns", "sha256": "9f2c…"},
  "counts": {
    "cells": 48200000, "faces": 151000000, "nodes": 52000000,
    "cells_by_type": {"tetra": 964000, "pyramid": 0, "prism": 13014000, "hexa": 0, "polyhedron": 34222000}
  },
  "bounds": {"min": [-1.2e3, 0.0, -2.1e2], "max": [2.9e3, 1.16e3, 5.6e2]},
  "boundary_zones": [
    {"name": "wing.main", "type": "wall", "native_type": "BCWall", "faces": 812000},
    {"name": "empennage", "type": null, "native_type": null, "faces": 95000}
  ],
  "cell_zones": [{"name": "fluid", "type": null, "native_type": null, "cells": 48200000}],
  "interfaces": [],
  "periodic": [],
  "groups": [{"name": "wing.main", "zones": ["wing.main"]}],
  "groups_source": {"path": "crm-hl.cgns.zones.json", "sha256": "4b1a…", "provenance": {"url": "https://…"}},
  "quality": {"max_non_orthogonality_deg": 62.0, "max_faces_per_cell": 214}
}
```

| Field | Type | Meaning |
|---|---|---|
| `schema` | `"autan.mesh-info/1"` | Contract version. Any other value is rejected. |
| `source.path` | string | Mesh file path as given to the reader. |
| `source.format` | `"cgns"` · `"gmsh"` · `"fluent"` · `"polymesh"` | Reader used. |
| `source.sha256` | 64 lowercase hex chars | The **mesh hash** recorded by every run: SHA-256 of the mesh file bytes. For a polyMesh directory, SHA-256 of the concatenation, for each file in the order `points`, `faces`, `owner`, `neighbour`, `boundary`, `cellZones` (when present), of `<name>\n<byte length in decimal>\n<bytes>`. |
| `counts.cells`, `counts.faces`, `counts.nodes` | integer ≥ 1 | Global counts after reading, before partitioning. `faces` counts each face once, interior and boundary. |
| `counts.cells_by_type` | object, keys `tetra` `pyramid` `prism` `hexa` `polyhedron`, integers ≥ 0 | Cells by shape. A cell is `polyhedron` unless it is exactly one of the four standard shapes. The values sum to `counts.cells`. |
| `bounds.min`, `bounds.max` | three numbers each | The least and greatest x, y and z over the mesh's nodes, in the file's units: the mesh's extent before the case's `mesh_unit` scales it. |
| `boundary_zones[]` | list, names unique | Every boundary face belongs to exactly one zone. Boundary faces the file leaves outside any group form a zone named `unassigned` with `type: null`; a file that also names a zone `unassigned` is rejected by the reader. |
| `boundary_zones[].name` | non-empty string | Zone name from the file (Fluent zone name, Gmsh physical name, CGNS `BC_t` or family name). |
| `boundary_zones[].type` | normalized type or `null` | See the mapping below. `null` means the file carries no usable type. |
| `boundary_zones[].native_type` | string or `null` | The type exactly as written in the file, for display. |
| `boundary_zones[].faces` | integer ≥ 0 | Face count. A zone the file declares without faces is kept so its name stays visible; it needs no condition. |
| `cell_zones[]` | list, names unique, at least one | Every cell belongs to exactly one zone. A file without cell zones yields one zone named `fluid` with `type: null`; cells outside every zone (Gmsh cells outside any 3D physical group) go to that zone. |
| `cell_zones[].type` | `"fluid"` · `"solid"` · `null` | Medium declared by the file. |
| `cell_zones[].native_type` | string or `null` | As written in the file. |
| `cell_zones[].cells` | integer ≥ 1 | Cell count. The values sum to `counts.cells`. |
| `interfaces[]` | list | One entry per pair of cell zones that share at least one face. |
| `interfaces[].zones` | two distinct cell-zone names, sorted | The pair. Each pair appears once. |
| `interfaces[].faces` | integer ≥ 1 | Shared faces. |
| `periodic[]` | list | One entry per periodic pair of boundary zones; each zone of type `periodic` is in exactly one pair, and no other zone is. |
| `periodic[].zones` | two distinct boundary-zone names | The pair `[a, b]`: the transform maps the faces of `a` onto those of `b`, face for face. |
| `periodic[].transform` | `{"kind": "translation", "vector": [x, y, z]}` | `x_b = x_a + vector`, in mesh units. Other kinds (rotation) are reserved. |
| `periodic[].faces` | integer ≥ 1 | Faces of each zone of the pair. |
| `groups[]` | list, sorted by `name` | The zone groups of the mesh's sidecar ([Zone groups](#zone-groups)); `[]` without one. |
| `groups[].name` | string matching `[A-Za-z0-9_.][A-Za-z0-9_.-]*` | Group name, unique. |
| `groups[].zones` | non-empty list of `boundary_zones` names | The group's zones, in `boundary_zones` order; groups may share zones. |
| `groups_source` | `null` or `{path, sha256, provenance}` | The sidecar read: its path, next to the mesh file with symlinks followed, the SHA-256 of its bytes, its `provenance` object. |
| `quality.max_non_orthogonality_deg` | number in [0, 180] | Max over interior faces of the angle between the face normal and the vector joining the two cell centroids; over boundary faces, between the normal and the vector from the cell centroid to the face centroid. |
| `quality.max_faces_per_cell` | integer ≥ 4 | Largest face count of any cell. |

Unknown fields are rejected. Readers fill every field; there are no optional fields. A mesh must be self-contained: the mesh hash covers one file (or one polyMesh directory), so readers reject CGNS links to other files. Reader errors (malformed files, external links, invalid topology or geometry) use the codes `E0101`–`E0199` defined in [mesh.md](mesh.md).

## Zone groups

A mesh whose file does not name its parts (the HLPW-5 CRM-HL grid has 668 zones `TRI_bdy N`) can carry a sidecar next to it, `<mesh file>.zones.json`, or `<directory>.zones.json` for a polyMesh directory (symlinks followed: next to the file itself), which `autan-mesh-info` reads into `groups` (a CGNS family is already one zone, [mesh.md](mesh.md#readers)). It is never inferred from the mesh and never modifies it:

```json
{"schema": "autan.zone-groups/1", "mesh_sha256": "9f2c…", "provenance": {"url": "https://…", "note": "from the mapbc"}, "groups": {"wing.main": ["TRI_bdy 12", "TRI_bdy 13"]}}
```

The four keys are required and no other is allowed; `provenance` maps keys to strings, shown as given. `E0136`: the file cannot be read, is not JSON, or breaks this schema (a key missing, unknown or given twice, a wrong type). `E0137`: `mesh_sha256` is not the mesh hash. `E0138`: a zone that is not a boundary zone name (exact, spaces included), a zone twice in one group, an empty group, or a name outside `[A-Za-z0-9_.][A-Za-z0-9_.-]*`. `E0139`: a group name given twice. The run records the sidecar's hash (solver-input `mesh.groups_sha256`) and archives the groups with its MeshInfo.

## Normalized boundary types

| Normalized | Fluent `.msh` | CGNS `BCType_t` | polyMesh | Gmsh |
|---|---|---|---|---|
| `wall` | `wall` | `BCWall`, `BCWallViscous` | `wall` | — |
| `thermal-wall` | — | `BCWallViscousHeatFlux`, `BCWallViscousIsothermal` | — | — |
| `slip` | — | `BCWallInviscid` | — | — |
| `symmetry` | `symmetry` | `BCSymmetryPlane` | `symmetry`, `symmetryPlane` | — |
| `farfield` | `pressure-far-field` | `BCFarfield` | — | — |
| `inlet` | `velocity-inlet`, `pressure-inlet`, `mass-flow-inlet` | `BCInflow`, `BCInflowSubsonic`, `BCInflowSupersonic`, `BCTunnelInflow` | — | — |
| `outlet` | `pressure-outlet`, `outflow` | `BCOutflow`, `BCOutflowSubsonic`, `BCOutflowSupersonic`, `BCTunnelOutflow` | — | — |
| `periodic` | — | a zone of a periodic `GridConnectivity_t` pair | — | the groups of a `$Periodic` surface link |
| `null` | every other type | every other type, `FamilySpecified` resolved to the family's type first | every other type (`patch`, `empty`, `cyclic`, …) | every other physical group |

`periodic` marks the two zones of a periodic pair (`periodic[]`), whose faces the core matches ([mesh.md](mesh.md#periodic-pairs)); a periodic zone takes no condition.

`wall` carries no thermal condition: the case resolves it to an adiabatic wall. `thermal-wall` declares one whose value the case must give, so it is never resolved silently.

Fluent `interior` zones are interior faces, not boundary zones. A Fluent cell zone of type `fluid` or `solid` maps to that type; CGNS, polyMesh and Gmsh cell zones map to `null`.

How the SDK turns these types into boundary conditions is part of the case contract: [case-schema.md](case-schema.md#boundary-resolution).
