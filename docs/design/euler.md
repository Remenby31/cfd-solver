# Compressible Euler discretisation

The M2 physics of [PLAN.md](../PLAN.md): cell-centred finite volumes on arbitrary polyhedra, written once in the kernel DSL (`kernels/euler.py`, [kernel-generator.md](kernel-generator.md)) and driven by `autan::physics::Euler<P>` (`physics/`). Gates: [acceptance.md §M2](../validation/acceptance.md#m2--compressible-euler).

## Residual

R_c = Σ_f F(W_L, W_R)·S_f − s_c V_c over the faces of cell c, in conserved variables U = (ρ, ρu, ρv, ρw, ρE) of an ideal gas.

| Step | Choice | Source |
|---|---|---|
| Variables reconstructed | primitive W = (ρ, u, v, w, p) | |
| Gradients | weighted least squares (1/d²) over the **face2 stencil**: face neighbours, their face neighbours, and the boundary faces of both | [Stencil](#gradient-stencil) |
| Limiter | Venkatakrishnan, bounds over the gradient stencil, ε_k² = (K/L)³ V q_k², K = 10, L the reference length, q_k the freestream scale of W_k | Venkatakrishnan (1995), eq. 5 |
| Face states | W_f = W_c + φ ∘ (∇W · (x_f − x_c)) | |
| Warped faces | flux of the linear reconstruction through the fan triangles: Σ_j dF_j[W; ∇W · M_{:,j}], M = Σ_t (c_t − x_f) ⊗ S_t | [Warped faces](#warped-faces) |
| Fluxes | Roe with Harten's fix on the acoustic waves only; HLLC with Einfeldt speeds; AUSM+-up with M_co the freestream Mach number | Roe (1981); Toro, Spruce, Speares (1994); Batten et al. (1997); Liou (2006) |

Harten's fix stays off the contact and shear eigenvalue so that every flux keeps a stationary contact exact (`tests/physics/test_euler.cpp`).

## Boundary conditions

Each boundary face carries a code and five values (`BoundaryKind`). The boundary state is computed from the reconstructed interior state at the face, then the numerical flux takes (W_f, W_b); slip faces take the pressure flux (0, p_f n, 0).

| Kind | Values | Boundary state |
|---|---|---|
| `state` | W | given (manufactured solutions) |
| `farfield` | freestream W, or W plus a [point vortex](#point-vortex) | Riemann invariants normal to the face; entropy and tangential velocity from the upwind side; supersonic faces take the upwind state |
| `slip` (slip wall, symmetry) | — | mirror state for the gradients; pressure flux |
| `inlet` | p₀, T₀ | flow normal to the face; R⁺ from the interior and c_p T₀ give c_b (Carlson 2011) |
| `outlet` | p | ρ and v from the outgoing characteristics; supersonic outflow extrapolates |

The gradient's boundary values are the boundary states of the first-order interior state (`boundary_states`, a frozen loop: its derivatives are dropped, so the generated adjoint of the gradient is approximate at boundary faces). They enter the least-squares fit only on `state` faces, where the condition gives the value at the face; on the other kinds the value is the adjacent cell's, which belongs at the cell centroid, and the fit uses the interior stencil alone (a cell whose interior stencil cannot determine a gradient keeps every boundary face). Offset components within 64 ulps of the coordinates count as zero: across a one-cell-thick slab the centroids differ only by round-off, and the fit would otherwise turn it into a gradient across the layer. The fit is determined when det A > 10⁻¹⁰ A₁₁A₂₂A₃₃, which is independent of the cells' aspect ratio. Where the linear reconstruction would make ρ or p non-positive, the face takes the cell value. The warped-face correction applies to interior faces and to the pressure of slip faces.

### Point vortex

A lifting 2D body leaves the circulation's vortex in the far field, so a uniform farfield state makes the lift depend on the outer radius. With `point_vortex`, each farfield face takes the freestream plus the compressible vortex of Thomas and Salas (1986, AIAA J. 24(7)) centred at `center`, in the x–z lift plane (steady cases, zero sideslip, subsonic freestream). The velocity is the Prandtl-Glauert vortex (β = √(1 − M∞²)):

u = u∞ + q sin θ, w = w∞ − q cos θ, q = Γβ / (2πr (1 − M∞² sin²(θ − α))).

Here r and θ are the face centroid's distance and polar angle from the vortex in x–z. The speed of sound follows from the freestream total enthalpy; density and pressure are isentropic (`physics/include/autan/physics/point_vortex.hpp`). The circulation is Kutta–Joukowski's Γ = L′/(ρ∞V∞), where L′ is the lift on the `surfaces` zones per unit of their extent along y. It is measured on the residual of each step and applied from the next, so a converged solution satisfies it exactly. The implicit Jacobian omits it, and restarts carry it (`VortexCirculation<i>` in `RestartState`).

## Gradient stencil

Least squares over the face neighbours alone make the unlimited second-order scheme **linearly unstable** on the tetrahedra of `autan-vv meshes tet`: the Jacobian of the residual has eigenvalues with negative real part. The same holds for the scalar advection of the generator spike, so the stencil, not the Euler flux, is at fault. Widening the stencil removes every unstable eigenvalue, as Diskin and Thomas (2011, AIAA J. 49(4), "inviscid fluxes") recommend for cell-centred schemes.

Spectrum of the dense Jacobian (finite differences) at the manufactured solution, tet n = 5 (762 cells), Roe, unlimited:

| Gradient stencil | Eigenvalues with Re < 0 | min Re (scaled) |
|---|---|---|
| face neighbours | 42 | −3.53 |
| face2 | 0 | +0.87 |
| vertex (cells sharing a node) | 0 | +0.91 |

Face neighbours stay stable on the hexahedral, prismatic, hybrid and polyhedral families (no eigenvalue with Re < 0 at n = 5–6). Cost on GB10 (tetrahedra n = 96, 3.9 M cells, `fp32`, measured by CFD - Générateur kernels): face neighbours 0.50 ns/cell (93 % of peak bandwidth), face2 1.82 ns/cell (58 %), vertex 20 ns/cell (25 %). face2 is the default: stable, second order, 11× cheaper than vertex. Both stencils give order 2 ([MMS](#manufactured-solution)). The limiter's bounds use the same stencil: bounds over the four face neighbours of a tetrahedron are narrower than the values the face2 gradient extrapolates, so the limiter clips smooth solutions and the limited scheme falls to first order on tetrahedra (measured: finest-pair order 0.58–1.20).

Next to `state` boundaries the face2 fit is one-sided: a cell whose stencil reaches a boundary face takes the exact value at the face centroid, half a cell away, and its gradient carries an O(h) error. The truncation R_h(u)/V − S at the exact MMS state, on the `hex` family 32–96 (Roe, unlimited), therefore falls in the band of cells reconstructing with those gradients. Excluding two rings of cells from the boundary, continuity converges at 1.86, 1.69 and 1.62 between levels (1.2 × 10⁻⁴ at 96³, about h^1.5 in L2 for an O(h) error one cell thick). Excluding three rings, it converges at 2.18, 1.97 and 1.91 (5.8 × 10⁻⁵), and deeper sets give the same level. The Euler MMS solution still converges at order 2 ([Manufactured solution](#manufactured-solution)).

## Warped faces

The faces of polyhedral duals are not planar. A one-point flux at the face centroid then commits an O(h) relative error per face, and the solution converges at first order (measured: finest-pair order 1.05–1.26 on the `poly` family). Integrating the linear reconstruction over the fan triangles through the face moment M restores order 2 (2.1–2.6).

## Precision

Cells store U − U_ref, the reference being the freestream, and the primitive pressure is the gauge pressure p − p_ref, computed from the differences only: p − p_ref = (γ − 1)(ΔE − (|m|²/ρ − |m_ref|²/ρ_ref)/2), the kinetic difference written with Δm and Δρ. Momentum fluxes take the gauge pressure (p_ref sums to zero over a closed cell), and the slip mirror keeps it, so that FP32 does not round a small gauge pressure away through p_ref; every other use of p adds p_ref back. FP32 storage resolves U to ε₃₂|U_ref| in absolute terms: a cell whose density falls below about 10⁻⁷ ρ_ref cannot be stored, its residual is not finite, and the run ends `diverged`. In `fp32+acc64` the residual evaluated in FP32 has a rounding floor: on the MMS (hex, n = 16, from the freestream) the residuals drop 5.9–6.0 orders with absolute storage and 6.3–6.4 orders with the reference state, then stay there. The rest of the floor is the FP32 rounding of the fluxes themselves, ε₃₂ times the face flux, so how many orders a case drops in `fp32+acc64` depends on how far its first residual stands above it (3.8 orders on a subsonic NACA 0012 slab from the freestream, where `fp64` drops 12 orders and the lift agrees to 3·10⁻⁶; the limiter plays no part). The physics' residual scale per equation is that flux, Φ_k = f_k √(Σ_i A_i² / N) over the history's N cells, A_i the surface of cell i and f_k the freestream's ρq, ρq² (each momentum) and ρHq, q = |v| + c. Measured plateaus sit at 0.01–0.04 ε₃₂Φ_k on the NACA 0012 (M 0.5, α 0° and 2°, limited or not), up to 0.07 on the MMS (hex, n = 16) and up to 0.12 on Ni's transonic bump (M 0.675, three meshes, rising with refinement; measured by CFD - Multigrille), so a steady `fp32+acc64` run converges at 0.5 ε₃₂Φ_k at the latest ([case-schema.md](../contracts/case-schema.md#time)). A ten-order drop is therefore measured in `fp64`; in `fp32+acc64` the gate is the integrated forces and the surface pressure against `fp64`, and the plateau is reported ([acceptance.md](../validation/acceptance.md#m2--compressible-euler)). The surface pressure must then agree with `fp64` to max |ΔCp| ≤ 10⁻³ over the wall faces, shock cells excepted (reported there): the plateau leaves state differences of order 10⁻⁶–10⁻⁵ relative, so 10⁻³ is a margin of 100 or more, and it is ten times finer than the 0.01 to which Cp distributions are compared with experiment.

## Time stepping

Steady runs march in pseudo-time with an implicit step, (V/Δt + J̃) ΔU = −R(U), Δt = CFL V / Σ_f (|v·n| + c) A local to each cell:

- J̃ is the approximate Jacobian: dF/dW of every face with the reconstruction's gradients and limiter held fixed, written by the face loops in the same pass as the residual (`*_jacobian` loops of `kernels/euler.py`), times dW/dU of each cell; it is assembled in 5 × 5 blocks (`BlockMatrix`) and the step's system is solved to a relative residual of 0.1 by the [agglomeration multigrid](multigrid.md) (`linear_solver = "multigrid"`, one rank), or by four symmetric block Gauss-Seidel sweeps over the runtime's colouring by global id (`"gauss-seidel"`), which runs the same algorithm on any number of ranks: ghost cells take ΔU after each colour, and dW/dU and the other cell data the coupling blocks read are filled into them first.
- The run starts at first order, where J̃ is the exact Jacobian, and switches to second order once the first-order residual has dropped three orders (at most 200 steps).
- CFL = 1000 / (largest residual relative to its value at the start of the current order) (switched evolution relaxation, Mulder and van Leer 1985), times a penalty halved whenever a step had to be shortened and recovering otherwise, between 1 and 10⁶.
- A step is shortened so that no cell loses more than half its density or pressure: each cell's admissible fraction (its density bound, then halved until its pressure bound holds) independently, the least over every rank. It is logged (`solver` warning with alpha and the CFL number); the next CFL number is then at most half the one used.

Steps to ten orders: [multigrid.md](multigrid.md#measurements). A Newton-Krylov variant (FGMRES on finite differences of the full residual, the limiter frozen, block Gauss-Seidel sweeps as preconditioner) was less robust far from the solution: on the MMS from the freestream (hex, n = 16) its steps were shortened at almost every iteration and the run stalled after 1.3 orders, where the defect correction converged ten orders in 177 steps (Gauss-Seidel, CFL = 10 / drop). It is left for M3 as an end phase, switched on once the residual has dropped.

Unsteady runs use SSP-RK3 (Shu and Osher 1988) with one global time step from `dt` or `cfl`.

Start: each cell takes its zone's `initial` state, or, with an initial-state file ([initial-state.md](../contracts/initial-state.md)), its row of the file; `autan_vv.initial_state.write_initial_state` writes one from an oracle (the isentropic vortex) at the cell centroids.

Restart: `restart_fields` holds the conserved variables bit for bit, the averaging sums, and the steady solver's state (penalty, order, reference residuals) as uniform fields, so that `continue` reproduces the uninterrupted run (`tests/physics/test_euler_run.cpp`).

## Partitions

On a partition (several ranks, or one with periodic image ghosts) the residual fills the ghost cells' primitives, gradients and limiter between its loops (`Euler<P>::set_ghosts`), and the physics updates, sums and reports its owned cells only; norms, residual scales and loads are exact sums over the ranks. The guarantee across rank counts is in [runtime.md](../contracts/runtime.md#ranks).

## Limiter and verification

Venkatakrishnan's threshold ε² ∝ h³ shrinks faster than the squared variations ∝ h² of a smooth solution, so the limiter clips smooth extrema more as the mesh is refined; K sets where that starts (Venkatakrishnan 1995; Michalak and Ollivier-Gooch 2008, AIAA 2008-776, discuss the same trade-off). On the MMS (tet, levels 12–32, Roe, host study), fitted / finest-pair orders of ρv with the limiter on: K = 5 gives 1.88/1.78, K = 10, 20 and 40 give 1.99/1.97, the unlimited values. K = 10 is the default: smooth flows keep the design order at these resolutions, and the limiter still acts at shocks (Sod gate, below). A manufactured solution is smooth by construction and measures the design order of the scheme; verification cases (`verification` in the solver input) run without the limiter, every other case with it.

## Manufactured solution

Steady `euler` solution of autan-vv (Roy et al. 2004 form), every boundary face at the exact state, cell sources at centroids, solved by `autan-run` and checked by `autan-vv verify --gate euler-mms-order` (Roe, unlimited, implicit steady solver; commit a6b557c, PR #134). Fitted / finest-pair L2 orders:

| Family, levels | Precision | ρ | ρu | ρv | ρw | ρE |
|---|---|---|---|---|---|---|
| tet, 12–32 | fp64, fp32+acc64 | 1.99/2.04 | 2.01/1.98 | 1.99/1.98 | 1.99/1.97 | 2.05/1.98 |
| hex, 12–32 | fp64, fp32+acc64 | 2.20/2.18 | 2.06/1.99 | 1.96/1.97 | 2.17/2.15 | 1.91/1.96 |
| poly, 12–32 | fp64, fp32+acc64 | 2.56/2.42 | 2.53/2.34 | 2.55/2.44 | 2.53/2.41 | 2.14/2.08 |
| hybrid, 16–48 | fp64, fp32+acc64 | 2.11/2.12 | 1.97/2.07 | 1.95/2.00 | 1.99/2.10 | 1.98/2.01 |

The hybrid family needs one level more: its pyramids, prisms and tetrahedra fill corner blocks that hold few cells on coarse levels, and the momentum orders between successive levels are 1.76–1.86 (12→16), 1.94–1.98 (16→24), 1.87–1.90 (24→32), 2.00–2.10 (32→48): they fluctuate by ±0.1, as on every non-nested family ([order.md](../validation/order.md)), and reach 2 on the finest pair. Polyhedral duals supraconverge (above 2.1: a warning of the gate, not a failure). The host study (`physics/mms/euler_mms.py`) gives the same orders for HLLC and AUSM+-up.
