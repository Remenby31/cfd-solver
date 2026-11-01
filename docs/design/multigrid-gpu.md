# Agglomeration multigrid on the GPU

The device port of the [agglomeration multigrid](multigrid.md), the linear solver of the implicit step, so that a steady run with `time.linear_solver = "multigrid"` and a device backend keeps its Jacobian, its hierarchy and its cycles on the GPU. Issues: #163, #219. It reuses the generated block kernels of E2 (#218, [kernel-generator.md](kernel-generator.md#jacobian-blocks)) and adds, by hand in `solvers/`, the transfers, the Galerkin products and the line smoother.

## What stays on the host

The hierarchy. `solvers::agglomerate(mesh, geometry, options)` builds the `std::vector<Level>` once per run, as today, and the device multigrid uploads it unchanged: one hierarchy, bit for bit the host's, owned by the host multigrid. The device side adds only the index lists derived from it (below). The coarsest level's factorisation also runs on the host, once per Jacobian ([Coarsest level](#coarsest-level)).

Everything that runs once per cycle or once per Jacobian on more than the coarsest level runs on the device.

## Reference practice

No published CFD code runs agglomeration multigrid for a block-implicit compressible solver on a GPU with bitwise reproducibility, so the design combines published pieces:

- **Method**: linear multigrid on the Jacobian with Galerkin coarse operators and piecewise-constant transfers (Mavriplis 2002, J. Comput. Phys. 175:302–325), lines in the wall layers (Mavriplis 1998, J. Comput. Phys. 145:141–165), K-cycle (Notay and Vassilevski 2008, Numer. Linear Algebra Appl. 15:123–145). Unchanged from the host ([multigrid.md](multigrid.md#choices)).
- **FUN3D on GPUs** runs multicolour point-implicit and line-implicit relaxation on the device; its manual (NASA/TM-20250002308, v14.2, introduction) lists multigrid as "under active development". Its kernels set our bandwidth targets: the 15-sweep multicolour block GS at 81 % of A100 peak (Nastac et al., AIAA 2023-1226). We take its precision split: off-diagonal blocks in FP32, diagonal blocks and vectors in FP64.
- **AMG on GPUs.** AmgX (Naumov et al. 2015, SIAM J. Sci. Comput. 37(5):S602–S626) and Cusp (Bell, Dalton and Olson 2012, SIAM J. Sci. Comput. 34(4):C123–C152) build aggregation hierarchies with block CSR operators and form A_c = R A P on the device with a sparse matrix product. With piecewise-constant P, R A P reduces to block sums over fixed lists, so we need no SpGEMM. AmgX (BSD-3) is the comparison point: an AmgX aggregation V-cycle on the same exported 5 × 5 matrix, time per cycle and convergence factor, reported next to ours, never a gate.

## Data per level

| Data | Level 0 (the mesh) | Coarse level l |
|---|---|---|
| Topology | the physics' `DeviceMesh` topology, sliced incidences (`incidence_slice = 32`, as E2) | the generator's graph topology of `Level` l (`graph_topology`, from the J5 pull request): cells = nodes, interior faces = edges (owner < neighbour), no boundary faces, each node's incidences in increasing edge order (the host `Operator::node_edges` order), sliced with stride 32, rows in the order J5's layout asks for |
| Diagonal blocks | `diag`, `JacAccum` (FP64 under `fp32+acc64`), from E2's `assemble` | `diag`, FP64 |
| Off-diagonal blocks | `off`, one per incidence, `Jac` (FP32 under `fp32+acc64`) | `off`, one per incidence, FP64 |
| Inverse diagonals, line factors | `diag_inv` (J3); `lower`, `gain` per line link, FP64 | the same, FP64 |
| Vectors | `Residual.Vec5`: FP64 under `fp64` and `fp32+acc64` | FP64 |

Coarse levels are FP64 under every supported policy, as [PLAN.md](../PLAN.md) decides for coarse levels and as the host does. They hold about a seventh of the fine level's nodes, so FP64 costs little, and the generated loops run on them through their `fp64` instantiation. Under `fp32` the device multigrid refuses to start with a coded error, never silently: its vectors would be FP32 and no host reference exists. FP32 vectors are part of the M5 precision study.

The device operator is E2's sliced block CSR: per cell, its diagonal block and one block per incidence, column `Topology::incidence_cell`. Boundary incidences carry a zero block and are skipped by every list below. The host operator's `upper` and `lower` of edge e are the `off` blocks at the owner's and the neighbour's incidence of e.

## Transfers

Precomputed on the host from `Level::coarse_node`, uploaded once:

- **Restriction** b_c = Pᵀ r: one thread per coarse node and component sums the residuals of its children, read from a CSR list of children in increasing fine index. This is the host's order (`cycle_level` adds fine nodes in increasing index), so the result is bitwise the host's.
- **Prolongation** e = P x_c: injection, `e[i] = x_c[coarse_node[i]]`, exact. It is fused with the correction x += α e when α is known on the device.

No atomics: each output is written by one thread. Every sum starts from +0.0 and adds its terms in list order, as the host does, so a list of −0.0 terms gives +0.0 on both sides.

## Galerkin coarse operator

A_c = Pᵀ A P is a set of block sums. Per coarse level, the host builds once, from the fine `Level`, two gather lists in the host's summation order (`Multigrid::set_operator`):

- **diagonal of coarse node I**: the diagonal blocks of its children in increasing fine index, then, for each fine edge inside I in increasing edge index, the owner's then the neighbour's `off` block;
- **off-diagonal of coarse incidence (I, E)**: for each fine edge e with `coarse_edge[e] = E`, in increasing e, the `off` block of e's endpoint inside I (`coarse_flipped` decides which endpoint is the coarse owner).

One thread per coarse entry and block component sums its list sequentially in FP64, from +0.0, straight into the coarse level's sliced `off` and `diag` (the host builds the lists against the coarse topology's incidence positions). Same terms, same order, additions only and FP64 on both sides: the coarse operator is bitwise the host's, under `fp64` and under `fp32+acc64` (FP32 fine blocks convert exactly). It runs once per Jacobian.

Bitwise equality holds only for these sums. Any product may be contracted into a fused multiply-add differently by nvcc and the host compiler, so every piece below that multiplies is compared within a tolerance.

## Smoother

The host's line-implicit block Gauss-Seidel, colour by colour, forwards then backwards ([multigrid.md](multigrid.md#choices)). A Gauss-Seidel sweep depends on the colouring, so the device uses the host's colouring of each level's lines, uploaded as lists of lines per colour; it does not colour on its own.

- **Lines of one node**: point block GS, J5's generated `relax(d, rows, n_rows)` (#218) on the nodes of that colour: x_i = D_i⁻¹ (b_i − Σ_k off_k x_col(k)) over the row's incidences in order, products in the compute type, sums and D⁻¹ r in the accumulation type.
- **Lines of several nodes**: by hand, one thread per line, block-tridiagonal elimination with the host's factors D'_k, A(k, k−1) and D'_k⁻¹ A(k, k+1). The factorisation runs once per Jacobian, one thread per line, in the host's operation order (`Smoother::factor`, Gauss-Jordan inverse with partial pivoting), in FP64 on every level. As on the host, the coupling to the previous or next node of a line sums every edge joining the two nodes, and the right-hand side of a line omits every edge to them (`Operator::coupling`, `off_residual`).

Within a colour, lines share no edge, so the two launches of a colour are independent and their order does not change the result. A zero or non-finite pivot leaves NaN in the factor; the solve reports it as the host's `std::domain_error` (singular block), so the step is rejected as on the host.

## Coarsest level

The host factors the coarsest level densely (LU with partial pivoting) when it has at most 1000 unknowns, and sweeps it `coarsest_sweeps` times otherwise; the device sweeps it the same way in the second case. In the dense case it changes the form: after the Galerkin product, the coarsest blocks (at most 1000 × 1000 FP64, 8 MB, usually 320 × 320) are copied to the host, factored by the host's LU, and inverted column by column with its triangular solves; the device then applies x = A⁻¹ b, one warp per row, a fixed butterfly per row.

Why not triangular solves on the device: they run m sequential steps in one block, about m synchronisations per solve, and the K-cycle visits the coarsest level up to 2^(L−1) times per cycle. A dense product is one launch of at most 8 MB of reads. Why not Grace: a host solve per visit costs a device-to-host round trip each time and cannot sit inside a captured graph. The cost is one host transfer and an O(m³) factorisation per Jacobian, as on the host today.

The two forms differ by more than κ∞ says: the host's LU carries its pivot growth ρ, the stored inverse does not, and a matrix with large ρ and small κ separates them. Rule, declared before measuring: the device's x = A_c⁻¹ b must lie within 10⁻¹⁰ ‖x_host‖∞ of the host's LU solve on every test case; κ∞(A_c) and ρ are measured and reported on the most stretched cases (flat plate 137 × 97, NACA 0012 256²); on a case that misses the gate, the coarsest level is solved by triangular solves on the device in the host's order instead, and their cost is reported.

## Cycle and reductions

V, W and K cycles and the outer GCR follow the host's algorithm line for line (`Multigrid::cycle_level`, `coarse_solve`, and `solvers::Gcr` once #282 extracts it): pre- and post-sweeps, the correction factor α of `MultigridOptions::scaling` and its `den > 0` test, the scaling of the outer residual by its largest entry (an exact device maximum), the K-cycle's α₁, β, α₂ and its test |r₁|² ≤ |b|²/16, GCR's coefficients and stopping test. Every application of a level's operator (the residual b − A x, the correction's image A e, GCR's images A z) is a generated reduction whose diagonal product and sums run in the accumulation type; J4's `multiply` rounds them to the compute type, which can turn a small residual or image into zero and break GCR's `ww > 0` test. The scalars stay on the device:

- dot products are `gpu::exact_sums` (#227): they depend only on the two vectors, not on the launch. The host's are sequential sums until CFD - Multigrille routes them through `ExactSum` (D2, also on one rank); from then on equal vectors give equal dots;
- each scalar is formed by a one-thread kernel from the rounded sums; the branches of the K-cycle and GCR's stopping test become flags on the device, read by the kernels that follow, which return at once when the flag says so, the exact sums included;
- the host reads one value per solve: the residual drop and the cycle count of the `Report`.

This needs two additions to `runtime/include/autan/runtime/exact_sum.cuh`, requested from CFD - Noyau maillage: the rounding of `ExactWords` to a double on the device (`exact_value`, the bits of `ExactSum::value()`), and an `exact_sums` that reads a device flag and does nothing when it is clear (today it always clears its output and traverses). Capturing a whole solve in one CUDA Graph per hierarchy comes after the gates, as a performance step.

## Determinism

No floating-point atomics anywhere. Every output is written by one thread from a list in a fixed order; dot products are exact. Two runs give the same bits. The hierarchy, the colouring and every list follow the host's numbering; one rank only, as the host multigrid (`std::invalid_argument` on ghost cells). Coarse levels across ranks are M5.

## Gates

Declared before any measurement. u is the unit round-off of the type that computes the operation (`P::compute` for the generated fine-level products, FP64 elsewhere); the reference is the host multigrid on the same fine blocks (FP32 blocks converted exactly), and where stated a dense product. Kernel gates compare one application from identical inputs, so errors do not compound across rows. Test meshes: the V&V `hex`, `tet`, `poly` and `hybrid` families, the NACA 0012 O-grid 256² (#186), fp64 and fp32+acc64.

| Piece | Gate | Oracle |
|---|---|---|
| Restriction, prolongation | bitwise equal | host `cycle_level` |
| Galerkin operator | bitwise equal on every level; within 2 k u Σ\|terms\| per entry of a dense Pᵀ A P, k its number of terms (both sums' rounding) | host `set_operator`; dense product, independent of the lists |
| Point rows, one colour from the same x | componentwise ≤ max(64, 4 (k + 5)) u ‖D_i⁻¹‖∞ (\|b\| + Σ \|A\| \|x\|)_i, k the row's incidences: J5's agreed 64 u on mesh rows, growing with the degree of coarse rows | host `relax_line` on lines of one node |
| Line factors | each block within 10⁻¹² of the largest entry of the host's | host `Smoother::factor` |
| Line solves, one colour from the same x and the same factors | each node within 10⁻¹² ℓ max_k ‖D'_k⁻¹‖∞ max_k (\|b\| + Σ \|A\| \|x\|)_k over its line, ℓ the line's length (scaled by the data, not by a solution that may cancel to zero) | host `relax_line` |
| Coarsest solve | within 10⁻¹⁰ ‖x_host‖∞, κ∞ and pivot growth reported ([rule](#coarsest-level)) | host `solve_dense` |
| One cycle, V and K | each component within 10⁻¹⁰ (fp64) or 10⁻³ (fp32+acc64) of \|x_host\|∞ | host `cycle` |
| GCR to a drop of 10⁻⁶, MMS hex and tet 48³, NACA 256² | each iteration's drop within 10⁻⁸ (fp64) or 10⁻² (fp32+acc64) relative of the host's; the same iteration count | host `solve` |
| Steady Euler to convergence (the run's rule, the fp32+acc64 floor of [case-schema.md](../contracts/case-schema.md) included; MMS hex and tet 16³–96³, NACA 0012 cases of [multigrid.md](multigrid.md#measurements)) | the host's step count | `autan-run`, cpu backend, same policy |
| Converged solutions | fp64: each field within 10⁻⁸ of its range of the host multigrid's run (measured between host solvers: 5.1 × 10⁻⁹) and `euler-mms-order` PASS; fp32+acc64: `autan-vv agree` with the fp64 run, as [acceptance.md](../validation/acceptance.md) requires | host run; the manufactured solution |
| Two runs | bitwise equal | itself |
| Restriction, prolongation, Galerkin | ≥ 75 % of measured GB10 peak bandwidth, bytes counted as each array read or written once | `bench/`, spark-5359 |
| Fine-level product and sweep | E2's gates, unchanged | J4, J5 |
| Whole solve | ns per fine cell per cycle, reported with the host's and AmgX's | `bench/` |

The cycle and GCR tolerances are declared, not derived: rounding differences pass through the cycle's conditioning and its branches, so no operation count bounds them. The fp32+acc64 ones allow for the FP32 products of the fine level (64 u₃₂ ≈ 4 × 10⁻⁶ per row), the fp64 ones for the same rows at u₆₄. An equal count of iterations or steps is required, with one exception declared now: when the host's deciding quantity (the drop that stops GCR, the residual that ends the run) lies within 10⁻³ relative of its threshold, a count differing by one is reported, not failed. Nothing else is: a gate that fails is fixed in the code, never relaxed. The linear solver changes only the path to R(U) = 0, not the discretisation, so conservation, a property of the face fluxes at R(U) = 0, is unchanged; the converged solutions carry the check.

## Interfaces

| From | What | Owner |
|---|---|---|
| host multigrid | `agglomerate` and `Level`, as they are | CFD - Multigrille |
| host multigrid | `line_colours(level)`, called, never copied (D2 moves it to a colouring by global id), and `factor_dense` / `solve_dense`, public functions of `solvers/` | CFD - Multigrille, agreed |
| generator | `kernels/blocks.py`: `assemble`, `invert_diagonal` (J3), `multiply` (J4), `relax(d, rows, n_rows)` (J5) on a given list of rows per colour, a residual b − A x in the accumulation type, `graph_topology` for coarse levels, their `fp64` instantiation | CFD - Générateur kernels, agreed |
| runtime | `gpu::exact_sums`; new: `exact_value` on the device, `exact_sums` predicated on a device flag | CFD - Noyau maillage |
| physics | the device `diag`, `off` and right-hand side of E2; the call of `solve` behind `time.linear_solver = "multigrid"` on a device backend | CFD - Euler |

The device class, `autan::solvers::gpu::Multigrid` (`solvers/include/autan/solvers/multigrid_gpu.hpp`, CUDA and HIP), mirrors the host's: built from the hierarchy and the fine `Topology`; `set_operator(diag, off)` builds the coarse operators and factors; `solve(b, x, max_cycles, tolerance)` returns the host's `Report`.

## Order of work

One pull request each, each closing its gates above:

1. Device hierarchy and transfers.
2. Galerkin products and the coarsest inverse.
3. Line factorisation and smoother.
4. Cycles and GCR on the device, against the host on the linear systems of the MMS and NACA cases.
5. Integration in the steady Euler physics on a device backend, steps to ten orders, then the bandwidth and time reports.
