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

## ONERA M6 meshes

`autan-vv meshes onera-m6 --levels 12,24,48 --out DIR` (source `vv/src/autan_vv/meshes/onera_m6.py`) writes tetrahedral meshes `onera-m6-nNNN.msh` for the M2 ONERA M6 case.

- **Wing.** Public definition of Schmitt and Charpin (AGARD AR-138, chapter B1) as Destarac and Dumont restate it for the TMR: root leading edge (0, 0), root trailing edge (0.8059 m, 0), apex of the prolonged trapezoid (1.5787 m, 2.7345 m), semispan 1.1963 m (sweeps 30° and 15.8°, taper 0.5625); every section is the root section (table B1-1, 72 points) scaled about the apex, so the wing is the ruled surface between root and tip; the tip is the tip section revolved about its chord line (half body of revolution, reaching 1.2185 m). The trailing edge is made sharp by extending the last segment of each surface to the chord line, 0.57 % of the chord (the TMR model extends its CAD surfaces by 0.55 %). x/c keeps the original chord. Reference area 0.7532 m² and mean aerodynamic chord 0.64607 m are the trapezoid's.
- **Domain.** Half sphere y ≥ 0 of radius 20 m (25 root chords) centred at (0.6, 0, 0); zones `wing`, `symmetry` (y = 0 exactly), `farfield`. Axes: x downstream, y spanwise, z up, so the angle of attack turns the flow in the x–z plane.
- **Construction.** Gmsh OpenCASCADE: splines through the table (vertical tangent at the leading edge, the extended segment at the trailing edge), the upper surface ruled between root and tip, the lower surface its mirror image, the tip revolved; the shell is cut from the half ball. Boolean union of the wing and tip solids damages the surface of revolution, so the wing is assembled from its faces. The tip surface is meshed with MeshAdapt (Frontal-Delaunay cannot mesh its two poles).
- **Family.** Level n has a size c(y)/n on the wing, a third of it along the edges, growing as 6/n per metre away from the wing; every size scales as 1/n, so n → 2n multiplies the cells by about 7.3 (1.94–1.95 in N^(1/3)): 52 k, 366 k and 2.75 M tetrahedra for n = 12, 24, 48 (n = 48 takes 4 min and 12 GB on a laptop). Netgen's optimiser removes the flat tetrahedra Delaunay leaves: largest face non-orthogonality 53° at n = 24 instead of 70°. A faster growth (20) makes Gmsh 4.15 produce faces shared by three tetrahedra. Single-threaded Gmsh with a fixed seed regenerates a level bit for bit.

## Manifest and reproducibility

Each family directory has `manifest.json` (schema `autan.mesh-family/1`): kind, perturbation, seed, Gmsh version, and per level the file name, `file_sha256`, `mesh_sha256` (topology and coordinates, format-independent), counts per cell type, total volume, h = (V/N)^(1/3) and `min_subvolume_ratio`. The same seed, Gmsh version and platform give byte-identical files; a different seed gives different files (`test_families_regenerate_bit_for_bit`).

`min_subvolume_ratio` is the smallest (sub-tetrahedron / cell volume) of the decomposition used by the core (triangle fan around each face's mean point, apex at the cell's mean face point). It is slightly negative (≈ −0.005) on median-dual meshes because their faces are strongly non-planar; volumes and closure are still exact.

## Checks

Every generated mesh is closed to 10⁻¹² relative and has positive cell volumes (`V0301` otherwise). Tests (`vv/tests/test_meshes.py`) also check, for every kind with and without perturbation: total volume 1 to 10⁻¹³, each box face of area 1 with outward normals, interior normals from owner to neighbour; the MSH file read back through Gmsh's own reader (physical names, element types, coordinates bit for bit); the Fluent file read back by an independent parser in the test, then checked for closure, volumes, zones and the core rule that every edge of every cell is used exactly twice in opposite directions.

## Platform

Gmsh is GPL-2.0-or-later. It is a development-only dependency group of `vv/`, never imported by shipped code, approved by the lead for that use. Gmsh publishes no linux-aarch64 wheel: generate meshes on macOS or x86-64 CI and copy the files (with their manifest) to the DGX Spark machines, which only consume them. The generated meshes are data, not GPL code.

## References

- T. J. Barth, Aspects of unstructured grids and finite-volume solvers for the Euler and Navier-Stokes equations, VKI Lecture Series 1994-05, 1994 (median dual).
- D. Destarac and A. Dumont, ONERA M6 wing test-case, original and TMR, 2016 (TMR, `Onerawingnumerics_val/ONERA_M6_Test_Case_TMR.pdf`).
- C. Geuzaine and J.-F. Remacle, Gmsh: a 3-D finite element mesh generator with built-in pre- and post-processing facilities, *Int. J. Numer. Meth. Eng.* 79(11):1309–1331, 2009. doi:10.1002/nme.2579
- H. W. Kuhn, Some combinatorial lemmas in topology, *IBM J. Res. Dev.* 4(5):518–524, 1960 (cube triangulation).
- V. Schmitt and F. Charpin, Pressure distributions on the ONERA-M6-wing at transonic Mach numbers, AGARD AR-138, chapter B1, 1979.
- J. C. Vassberg and A. Jameson, In pursuit of grid convergence for two-dimensional Euler solutions, *J. Aircraft* 47(4):1152–1166, 2010. doi:10.2514/1.46737
