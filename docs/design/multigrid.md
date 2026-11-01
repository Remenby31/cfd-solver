# Agglomeration multigrid

The linear solver of the implicit step of [PLAN.md](../PLAN.md) M3: each pseudo-transient step of a steady run solves (V/Δt + ∂R/∂U) ΔU = −R ([euler.md](euler.md), [ns-sa.md](ns-sa.md#implicit-solver)) with an agglomeration multigrid whose cost per order of residual drop does not grow with the mesh. Code: `solvers/` (`autan::solvers::Multigrid`), called by the steady physics in `physics/src/euler_physics.cpp`. Issue: #148.

## Why

Block Gauss-Seidel damps the error components a few cells long. The viscous Jacobian is elliptic, so its smooth components need O(n²) sweeps on a mesh n cells wide; four sweeps per step left the linear error large and the number of steps grew with the mesh (laminar MMS: 216 steps on 24³, 706 on 48³). FUN3D, NSU3D and the Fluent coupled solver remove the smooth error on coarser agglomerated levels and solve the anisotropic wall layers along lines ([research §1.7](../research/state-of-the-art-2026-10.md)).

## Choices

| Question | Choice | Reason |
|---|---|---|
| Linear or nonlinear (FAS) multigrid | **Linear**, on the Jacobian of the implicit step | Mavriplis (2002, J. Comput. Phys. 175:302–325): with the Jacobian the outer iteration already uses, linear multigrid converges as fast per cycle as FAS and costs less, since no residual, geometry or boundary condition is evaluated on coarse levels. The outer loop (SER CFL, admissible step, first-order start) and its fixed point R(U) = 0 are unchanged, so the converged solution is the same discrete solution. |
| Coarse operators | **Galerkin** A_c = Pᵀ A P, P constant on each agglomerate | Exact for any agglomerate and any Jacobian (Euler, viscous, boundary blocks, the V/Δt shift) with no coarse geometry: block sums, in a fixed order. Rediscretisation would need the residual's kernels and boundary conditions on agglomerated polyhedra. Its weakness, a coarse diffusion operator about twice too stiff for a 2 × 2 × 2 agglomerate, is corrected by the Krylov acceleration below. |
| Cycle | **K-cycle** (Notay and Vassilevski 2008, Numer. Linear Algebra Appl. 15:123–145): each coarse solve is two steps of GCR preconditioned by the next level's cycle | V and W cycles with piecewise-constant transfers lose convergence as levels are added on diffusion; the Krylov steps restore it at the cost of two products per level. V and W remain options. |
| Outer solve | **GCR** preconditioned by one cycle per iteration, to a relative residual of 0.1 (at most 10 iterations), on the residual scaled by its largest entry | Inexact Newton: a tolerance of 0.01 left the steps unchanged (Euler MMS 48³: 59 instead of 60), the outer defect correction sets the rate. A solve that stops above 0.1 is logged (`linear` warning in `log.jsonl`); one that leaves more than 0.9 of the residual, or meets a singular block, has failed: the step is rejected and the next CFL number is half this one. Nothing falls back to Gauss-Seidel. |
| Start of the pseudo-time continuation | **CFL = 1000 / drop** (SER), instead of 10 / drop | With an exact-enough linear solve, the number of steps is set by the pseudo-time continuation: with the viscous time step ∝ h², a start at 10 made the laminar steps grow as the mesh (51 → 200 from 16³ to 48³). Gauss-Seidel gains nothing from a larger start (its solve is the limit). Checked on Ni's transonic bump from the freestream ([measurements](#measurements)). |
| Smoother | **Line-implicit block Gauss-Seidel**, one symmetric sweep before and after each coarse correction | Lines are solved exactly by block-tridiagonal elimination; the other cells are point block Gauss-Seidel, as before. Lines of one colour share no face, so their solves are independent; colours are visited forwards then backwards. |

## Lines

The coupling of a face is w_f = A_f / |x_N − x_O| (that of a boundary face uses the distance to the mirror cell, 2 |x_f − x_c|): the diffusion coefficient a viscous Jacobian has between the two cells. A cell is anisotropic when its strongest coupling is at least 4 times its weakest (aspect ratio ≥ 2). Seeds are the anisotropic cells, boundary cells first, then by decreasing anisotropy. A line grows from its seed, in both directions, through the strongest face whose coupling is at least 4 times the weakest coupling of both cells (Mavriplis 1998, J. Comput. Phys. 145:141–165), and starts at its boundary end when it has one. On wall layers, each wall cell starts a line normal to the wall that stops where the cells become isotropic. Lines of fewer than 4 cells (pairs of right prisms across their large face) are smoothed as lines but agglomerated as isotropic cells, whole.

## Agglomeration

Each level is a graph: nodes (cells, then agglomerates), edges (faces between them) with their summed couplings, the cell zone, the boundary zones touched and the sum of the boundary area vectors. A coarse node:

- never mixes cell zones;
- holds only boundary zones its seed touches, and boundary faces facing the same side (Σ S_b · Σ S_b' > 0), so that an agglomerate keeps its patches and never wraps around a thin wall.

Lines coarsen first, **in the wall plane**: a group of at most 4 lines grows from a seed line by the free line touching the most lines of the group at the most layers, the nearest among equals (2 × 2 blocks of wall lines), and layer k of the group becomes one coarse node, so a coarse level keeps lines of the same length and the line smoother keeps working on coarse levels. A neighbouring line may run the other way: its layers are matched in the orientation that touches the group at more layers. The other cells form compact isotropic agglomerates of at most 8 nodes, grown from a seed by adding the free neighbour with the most faces on the agglomerate, the nearest to the seed among equals (2 × 2 × 2 blocks of hexahedra); a cell brings the rest of its short line, so that no strong coupling joins two agglomerates. Seeds follow an advancing front from the boundary corners inwards (Mavriplis and Venkatakrishnan 1996, Comput. Fluids 25:361–378). A node left alone joins the neighbouring agglomerate it is most coupled to, beyond 8 if need be. Coarsening stops at 64 nodes, or when a level would keep more than 80 % of the nodes; the coarsest level is factored densely (LU with partial pivoting) when it has at most 1000 unknowns, and smoothed by 10 sweeps otherwise. The steady physics logs the hierarchy at the start of a run (`linear` info event: nodes per level, `lu` or `sweeps`).

The hierarchy is built once from the mesh and its geometry. A mesh with ghost cells is refused (`std::invalid_argument`): coarse levels across partitions are #259 D2. On several ranks, `linear_solver = "lines"` runs the fine level alone (`LineSmoother`): the lines come from the global mesh (`smoother_lines`) and the partition keeps them whole, lines and remaining cells are swept over the runtime's colouring of the lines by global id with a ghost exchange after each colour: the one-rank algorithm on any number of ranks. `LineSmoother` sums each cell's couplings by global face id when given the ids, which gives the one-rank bits (tested on in-process partitions); the physics passes them once the mesh carries them (#261), and until then agrees with one rank to round-off, as `"gauss-seidel"`.

## Determinism and precision

Every loop runs in a fixed order and every choice breaks ties by index, so a run is bitwise reproducible. Blocks, vectors and reductions are FP64 in both precision policies; the Jacobian is assembled in FP64 by the physics. Work vectors exist only on the levels that use them: about 5 FP64 vectors per fine cell beyond the Jacobian, plus 2 per GCR iteration.

## GPU

The multigrid runs on the host, as the whole implicit step today. The port (#163) follows the device assembly of the Jacobian: BSR products and fixed-order dot products from the kernel generator, Galerkin products, transfers and the line smoother written by hand in `solvers/` as gathers over precomputed lists, one line per thread within a colour.

## Verification

| Test | Oracle | Where |
|---|---|---|
| Agglomerates connected, one cell zone, coarse edges one per adjacent pair; lines join adjacent nodes and partition the nodes; reference meshes and wall layers | definition | `tests/solvers/test_multigrid.cpp` |
| Wall layers (hexahedra, prisms, first cell 10⁻⁴): one line per wall cell, kept on every coarse level | geometry of the fixture | same |
| One line sweep solves a block-tridiagonal system | exact solution, 1000 ε | same |
| Two-level cycle without smoothing = P (PᵀAP)⁻¹ Pᵀ b, A(o, n) ≠ A(n, o), coarse edges of both orientations | dense product and elimination | same |
| Layers whose lines run opposite ways (cell numbering reversed along every other column) coarsen in the plane | ≥ 2:1 at the first level | same |
| Iterations to a 10⁻⁶ drop on diffusion, isotropic hexahedra and prisms 8 → 32 wide | ≤ 9, growing by at most 2 (measured: hexahedra 7, 6, 7; prisms 6, 7, 8) | same |
| Wall layers of aspect ratio 10⁴ (first cell 10⁻⁵, growth 1.2), hexahedra and prisms, 8 and 16 wide, with and without lines | ≤ 9 with lines, at least twice as many without (measured: 6–8 with lines, 20–31 without) | same |
| Steps to ten orders on Euler and laminar Navier-Stokes MMS, hexahedra 16³ → 96³, against Gauss-Seidel alone | [measurements](#measurements) | `physics/mms/study.sh` |

## Measurements

Steps of `autan-run` to ten orders, `fp64`, cpu backend, from the freestream. Gauss-Seidel: four sweeps per step, CFL = 10 / drop (main at 971ac5c); multigrid: this design. Manufactured solutions on `autan-vv meshes hex`; equations whose residual stays at round-off (the span momentum of one-cell-thick meshes) are exempt, as in the runtime.

| Case | Solver | 16³ | 24³ | 32³ | 48³ | 96³ |
|---|---|---|---|---|---|---|
| MMS `euler` | Gauss-Seidel | 87 | 121 | 157 | 232 | 657 |
| | multigrid | 41 | 47 | 52 | 60 | 77 |

The converged solutions of both solvers differ by at most 5.1 × 10⁻⁹ of each field's range (16³ to 48³), the residual tolerance's share: the observed orders are those of the discretisation (`euler-mms-order` PASS, 1.95–2.18). Two runs of the same case give the same history and solution bit for bit.

Ni's transonic bump (circular arc 10 % thick in a channel, inlet Mach 0.675, peak Mach 1.39, one hexahedron thick) and the NACA 0012 O-grids of Vassberg and Jameson (#186), from the freestream:

| Case | Mesh | Gauss-Seidel | multigrid | steps shortened / rejected (multigrid) |
|---|---|---|---|---|
| bump | 96 × 32 | 2714 | 55 | 0 / 0 |
| | 192 × 64 | > 4000 | 93 | 0 / 0 |
| | 384 × 128 | > 4000 | 293 | 6 / 0, in the first 20 steps |
| NACA 0012, M 0.5, α 1.25° | 64² / 128² / 256² | | 43 / 47 / 69 | 0 / 0 |
| NACA 0012, M 0.8, α 0° | 64² / 128² / 256² | 128²: no drop in 3000 | 234 / 261 / 1334 | 1 / 0 |
| NACA 0012, M 0.8, α 1.25° | 64² / 128² / 256² | | 54 / 57 / 109 | 1 / 0 |

Time to ten orders on a quiet GB10 (spark-5359, one Grace core, the run alone under the machine's lock):

| Case | Gauss-Seidel | multigrid | Gain |
|---|---|---|---|
| `euler` 24³ | 4.0 s (121 steps) | 2.2 s (47) | 1.8 × |
| `euler` 48³ | 70.2 s (232) | 28.4 s (60) | 2.5 × |
| `euler` 96³ | | 358.2 s (77) | |

A multigrid step costs about 1.5 Gauss-Seidel steps (Euler 48³: 0.47 s against 0.30 s). The remaining growth of the Euler steps (41 → 77) is the outer defect correction's: a linear tolerance of 0.01 gives the same counts. On viscous cases at large CFL numbers the defect correction with the approximate Jacobian can stall where the multigrid solves every step to its tolerance (#185): the Newton-Krylov end phase is the remedy.
