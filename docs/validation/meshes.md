# Mesh families

Source: `vv/src/autan_vv/meshes/`. Generate with `autan-vv meshes <kind> --levels 8,16,32,64 [--perturbation 0.3] [--seed S] --out DIR`.

## Kinds

All meshes fill the unit cube [0, 1]³; level `n` has about `n` cells per edge.

| Kind | Cells | Construction | File |
|---|---|---|---|
| `tet` | tetrahedra | Gmsh Delaunay, size 1/n | MSH 4.1 ASCII |
| `hex` | hexahedra | Gmsh transfinite, n³ | MSH 4.1 ASCII |
| `prism` | prisms | Gmsh unstructured triangles in xy, n layers in z | MSH 4.1 ASCII |
| `hybrid` | hex, pyramid, tet, prism | Structured n³ grid in four slabs along z: hexahedra; a transition layer where each hexahedron becomes a pyramid on its bottom quad plus ten tetrahedra; Kuhn tetrahedra; prisms. Every split quad uses its lowest-to-highest-corner diagonal, so cells always share whole faces | MSH 4.1 ASCII |
| `poly` | polyhedra (median dual) | One cell per vertex of the `tet` mesh; faces are the median-dual rings of face and tetrahedron centroids around each edge (Barth 1994), closed through edge midpoints and vertices on the boundary | Fluent ASCII |
| `agglomerated` | polyhedra, up to ~330 faces | The `poly` mesh with the cells of every other box of a lattice of edge ≈ 4/n merged: non-convex cells with 100+ faces among ordinary ones | Fluent ASCII |

**`ringleb`.** Not a cube: the HiOCFD1 C1.2 domain of Ringleb's flow, one layer of hexahedra mapped from the hodograph (streamlines k = 0.7 … 1.5, n cells across, and flow angles θ from asin(0.5/k) to π − asin(0.5/k), 2n cells along, so that V = k sin θ). The layer is 0.1 thick at every level. Zones `k-min`, `k-max`, `inflow`, `outflow` (the iso-speed V = 0.5 at y > 0 and y < 0), and `back`, `front` (the planes of the layer). Each level `ringleb-nNNN.msh` comes with `ringleb-nNNN.states.json`, the exact states on the four in-plane sides in the [face-state format](../contracts/face-states.md) for the `oracle-state` condition, taken from the oracle's analytic continuation where a face centroid lies on a chord just outside the curved side. Not perturbed.

**`vortex-hex`, `vortex-prism`.** Not a cube: the periodic square [−10, 10]² of the convected isentropic vortex (Spiegel, Huynh and DeBonis 2015), one layer thick in z. `vortex-hex` has n × n quadrilaterals; `vortex-prism` has triangles from an 8 × 8 lattice whose nodes move by a seeded jitter of up to a quarter of the lattice step (on the sides only along them, and alike on both sides of a pair) and whose squares split along a seeded diagonal, each finer level splitting every triangle into four at its edge midpoints: levels are n = 8 × 2^k and nested, a level being the next finer one with every other line removed. The layer is 1 thick at every level (one domain for the order study); the in-plane size h = √(area/N) is 20/n for quadrilaterals and 20/(√2 n) for triangles. Zones `xmin`, `xmax`, `ymin`, `ymax` form two periodic pairs, declared as `$Periodic` translations of 20 along x and y ([mesh.md](../contracts/mesh.md#periodic-pairs)); `zmin`, `zmax` bound the layer: a case sets them to symmetry and names no periodic zone in its boundary rules (E0424). Not perturbed.

The hybrid kind is built in Python rather than by Gmsh because Gmsh 4.15's tetrahedral optimiser duplicates tetrahedra next to the pyramids it inserts on quad faces, and without it leaves inverted slivers, depending on the seed.

**Dirty variant.** `--perturbation α` moves every node by a seeded random vector, uniform in a ball of radius α × its shortest incident edge, tangentially on the cube faces so the domain is unchanged. Nodes of a cell that would become inverted or non-star-shaped have their move halved until every cell is valid. The seed of each level derives from (seed, n, kind).

## Files and zones

Agreed with the core mesh readers (#1):

- MSH 4.1 ASCII, linear elements only (tet 4, hex 5, prism 6, pyramid 7; tri 2, quad 3 on boundaries). One 3D physical group `fluid`; six 2D physical groups `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax`; nothing else. The file is written through a discrete Gmsh model from the in-memory arrays, so it holds exactly the mesh that was checked.
- Fluent ASCII for polyhedra: sections 0, 2, 10 (nodes), 12 (cells, element type 7), 13 (faces, mixed type 0, each line starting with its node count), 45 (zone names); indices in hexadecimal, 1-based. The right-hand-rule normal of a face points toward c0; boundary faces have c1 = 0. Cell zone `fluid`; face zones `interior` (type 2), `ymin` as `wall` (type 3, the RANS-SA wall), the five others as `pressure-far-field` (type 9). Every MMS boundary is Dirichlet from the exact solution whatever its type.
- Coordinates are rounded to the 16 significant digits Gmsh writes, so file and memory agree bit for bit.

## NACA 0012 O-meshes

`autan-vv meshes naca0012 --levels 32,64,128,256,512 --out DIR` (source `vv/src/autan_vv/meshes/naca0012.py`) writes the grids of Vassberg and Jameson (2010) for the M2 NACA 0012 gates, `naca0012-nNNNN.msh` with NC × NC hexahedra in one layer.

- **Airfoil.** NACA 0012 equation extended to its root x_TE = 1.00893041137, so the trailing edge is sharp (included angle 16.1501°); the paper prints x_TE rounded to 1.0089304115. Reference chord 1.
- **Grid.** Karman–Trefftz map with singular points at the trailing edge and half the leading-edge radius inside the leading edge; surface nodes on the analytic airfoil at equal angles about the midpoint of the mapped chord, field nodes on circles of radius R₁ e^(2πj/NC) blended from the near circle (the paper's eqs. 9–14; eq. 14 is printed with r_i,1 (R_NC+1 − R_1), which ends on no circle and does not give back eq. 13 for a circle, so we use r_i,1 (R_NC+1 − R_j), which does both). Cells have aspect ratio 1 (median 1 + O(1/NC), at most 2.06 next to the trailing edge) and the far field lies 148–149 chords away. The trailing-edge cells have a 172° corner, the mapping's singularity, as in the paper.
- **Family.** R₁ comes from the paper's finest grid (NC = 4096), so level NC is the 4096² grid with lines removed: levels are nested bit for bit, refinement ratio exactly 2. The lower half mirrors the upper one exactly.
- **Slab.** The airfoil lies in the x–z plane (lift along +z, as case-schema.md orients α), extruded over y ∈ [0, 1]. Zones `airfoil`, `farfield`, `back` (y = 0), `front` (y = 1). The manifest's `h` is the in-plane size √(area / cells).

## Manifest and reproducibility

Each family directory has `manifest.json` (schema `autan.mesh-family/1`): kind, perturbation, seed, Gmsh version, and per level the file name, `file_sha256`, `mesh_sha256` (topology and coordinates, format-independent), counts per cell type, total volume, h = (V/N)^(1/3) and `min_subvolume_ratio`. The same seed, Gmsh version and platform give byte-identical files; a different seed gives different files (`test_families_regenerate_bit_for_bit`).

`min_subvolume_ratio` is the smallest (sub-tetrahedron / cell volume) of the decomposition used by the core (triangle fan around each face's mean point, apex at the cell's mean face point). It is slightly negative (≈ −0.005) on median-dual meshes because their faces are strongly non-planar; volumes and closure are still exact.

## Checks

Every generated mesh is closed to 10⁻¹² relative and has positive cell volumes (`V0301` otherwise). Tests (`vv/tests/test_meshes.py`) also check, for every kind with and without perturbation: total volume 1 to 10⁻¹³, each box face of area 1 with outward normals, interior normals from owner to neighbour; the MSH file read back through Gmsh's own reader (physical names, element types, coordinates bit for bit); the Fluent file read back by an independent parser in the test, then checked for closure, volumes, zones and the core rule that every edge of every cell is used exactly twice in opposite directions.

## Platform

Gmsh is GPL-2.0-or-later. It is a development-only dependency group of `vv/`, never imported by shipped code, approved by the lead for that use. Gmsh publishes no linux-aarch64 wheel: generate meshes on macOS or x86-64 CI and copy the files (with their manifest) to the DGX Spark machines, which only consume them. The generated meshes are data, not GPL code.

## References

- T. J. Barth, Aspects of unstructured grids and finite-volume solvers for the Euler and Navier-Stokes equations, VKI Lecture Series 1994-05, 1994 (median dual).
- C. Geuzaine and J.-F. Remacle, Gmsh: a 3-D finite element mesh generator with built-in pre- and post-processing facilities, *Int. J. Numer. Meth. Eng.* 79(11):1309–1331, 2009. doi:10.1002/nme.2579
- H. W. Kuhn, Some combinatorial lemmas in topology, *IBM J. Res. Dev.* 4(5):518–524, 1960 (cube triangulation).
- J. C. Vassberg and A. Jameson, In pursuit of grid convergence for two-dimensional Euler solutions, *J. Aircraft* 47(4):1152–1166, 2010. doi:10.2514/1.46737
