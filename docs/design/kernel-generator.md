# Kernel generator

The kernel generator turns pointwise kernels written in a restricted Python subset (`kernels/*.py`) into C++20 for CPU, CUDA and HIP, with their tangents and adjoints. Design choices and their sources: [`research/kernel-generator.md`](../research/kernel-generator.md). Code: `codegen/` (Python package `autan-codegen`, runtime headers in `codegen/include/autan/kernels/`).

## Pipeline

1. `autan_codegen.model.load` imports a kernel module and collects every loop bound to a module-level name.
2. `autan_codegen.frontend` parses each body with `ast` into a scalar SSA IR (`autan_codegen.ir`): vectors and matrices are scalarised, `for i in range(N)` is unrolled, `if` becomes `select`, calls to other bodies are inlined. Value numbering and dead-code elimination follow.
3. `autan_codegen.derive` adds, for every loop, `<loop>_tangent` and `<loop>_adjoint`, built by `autan_codegen.ad` on the IR.
4. Emitters write, per module `m`: `m.hpp` (the `Data<P>` struct and every body), `m_cpu.hpp` (serial loops), `m_gpu.cuh` (CUDA/HIP kernels and launchers, one text for both), `m_gpu.cu` (explicit instantiations for the build's policies), and `m_metal.hpp` (Metal Shading Language source and host launchers, [metal-backend.md](metal-backend.md)).

The build runs the generator as a CMake custom command (`codegen/CMakeLists.txt`); it needs Python ≥ 3.12 and nothing outside the standard library. By hand: `PYTHONPATH=codegen/src python3 -m autan_codegen kernels/roe.py --out DIR --backends cpu,gpu,metal [--policies fp64,fp32+acc64,fp32] [--no-line]`.

| CMake option | Effect |
|---|---|
| default | CPU loops, `autan_kernels`, spike library `autan_spike` |
| `AUTAN_ENABLE_CUDA` (root) | GPU loops, `autan_kernels_gpu` (every loop × policy), `autan_spike_gpu`, `autan_loop_bench`, `autan_option_b` |
| `AUTAN_ENABLE_HIP` | the same generated GPU loops compiled with HIP into `autan_kernels_gpu`; exclusive with CUDA |
| `AUTAN_ENABLE_METAL` (macOS) | Metal loops, runtime `autan_metal`, driver `autan_spike_metal`, benchmarks `autan_metal_stream` and `autan_metal_loop_bench` |

The benchmarks record the commit they were built from, as every benchmark binary does ([bench.md](../infra/bench.md#benchmark-any-kernel)).

## DSL

```python
q = cells.field("q", State)                  # data: cells.field / faces.field, a role and a shape
res = cells.field("res", Residual)
velocity = param("velocity", Param.Vec3)

@pointwise
def upwind(qc: State, qn: State, n: Geom.Vec3, a: Param.Vec3) -> Residual:
    un = dot(a, n)
    return un * qc if un > 0.0 else un * qn

residual = face_loop(upwind, qc=q.left, qn=q.right, n=geometry.face_normal, a=velocity,
                     out=Inc(res, left=+1, right=-1))
```

**Types.** A role (`State`, `Grad`, `Residual`, `Geom`, `Param`) and a shape: `State`, `State.Vec3`, `State.Vec5`, `Geom.Mat3`, `Grad.Mat53` (5 × 3, row i the gradient of component i). The precision policy maps roles to storage types ([Precision](#precision)).

**Bodies** (`@pointwise`) use: arguments, locals, numeric module constants; `+ - * /`, `**` with an integer constant 0–8, `@` (matrix times vector or matrix); `< <= > >=`, `and`, `or`, `not`; `x if c else y` and `if`/`else` (a name assigned in one branch must be assigned in the other; `return` in both or neither); `for i in range(N)` with constant bounds; indexing with constants (`v[0]`, `m[i, j]`, `v.x`); `sqrt exp log sin cos abs min max dot cross norm outer transpose inverse identity Vec3 Vec5`; `rel(a, b)`, the difference of two `Geom` arguments in their storage type (FP64 geometry stays FP64 until the difference); calls to other `@pointwise` bodies. Anything else is rejected with a code and the `.py` line. `abs(x)` is `x if x >= 0 else -x`; `min(a, b)` is `a if a <= b else b`, `max(a, b)` is `a if a >= b else b`. `inverse(m)` of a square matrix is Gauss-Jordan elimination with partial pivoting in the operation order of the host `BlockMatrix` (`physics/src/block_matrix.cpp`), row swaps as selects; a zero or non-finite pivot makes every entry NaN.

**Data.** `cells.field`, `faces.field`, `param`; built-in FP64 geometry `geometry.face_area` (area vector), `face_normal`, `face_area_magnitude`, `face_centroid`, `face_moment` (Σ over the face's fan triangles of (c_t − x_f) ⊗ S_t, zero on planar faces), `face_r_left`, `face_r_right`, `cell_centroid`, `cell_volume`, `lsq_weight` (per incidence). Geometry is computed by `autan::kernels::pack_geometry` from the core's `Geometry`.

**Loops.**

| Declaration | Iterates | Bindings | Outputs |
|---|---|---|---|
| `face_loop` | interior faces that touch an owned cell (`[0, n_owned_interior_faces)`: halo faces between two ghosts are skipped, mesh contract invariant 6) | `d.left`, `d.right` (cells), `d` (faces), params | `Inc(d, left=s_L, right=s_R)` |
| `boundary_loop` | boundary faces of owned cells for `Inc`, all boundary faces for `Write` (states on the boundary faces of ghosts feed the stencils of owned cells) | `d.left`, `d` (faces), params | `Inc(d, left=s_L)`, or `Write(d)` of face datasets: no derivatives (face data are passive); reading active data needs `frozen=True` (E1012 otherwise), listed in the module's `frozen_loops` |
| `cell_loop` | owned cells | `d` (cells), params | `Write(d)` |
| `cell_reduce` | owned cells, then their faces in CSR order | `d.self_`, `d.other(boundary=f)` (`f=None`: zero on boundary faces), incidence data (`geometry.lsq_weight`; `geometry.incidence_side`, +1 where the cell is the face's left cell or the face is a boundary face, −1 where it is the right one; left and right in the face's global orientation under `Topology::face_reversed`, as `d.near` and `d.far` below), face data of the incidence's face (`d` on faces; `d.near` / `d.far`, the half [0] / [1] of a `(2, ...)` face dataset for the left cell and [1] / [0] for the right one, zero on boundary faces; `d.on_boundary`, zero on interior faces: loaded only where valid, so data the physics never writes there is not read), params | `Sum(d)`, `Min(d)`, `Max(d)`, optional `finalise` body; `entries=Write(e)`: the body's last outputs, one value per entry, into a `Jac` or `JacAccum` dataset of the iterated set (each entry written once, by its cell; no active input) |
| `cell_reduce(over=STENCIL)` | owned cells, then their stencil: cells, then boundary faces | `d.self_`, `d.other(boundary=f)` (stencil cell, or `f` of a boundary face), stencil data (`geometry.stencil_lsq_weight`), params | same |

**Stencils.** Least squares over face neighbours leaves the unlimited second-order scheme linearly unstable on tetrahedra (Diskin and Thomas, AIAA J. 49(4), 2011): on the V&V `tet` family at n = 5, −V⁻¹ ∂R/∂q of pure advection has five eigenvalues with positive real part (`tests/test_stencil.py`). `autan::kernels::cell_stencil(mesh, geometry, kind)` builds a wider stencil once per mesh, after renumbering: `vertex` holds the cells and boundary faces sharing a node with the cell (about 63 entries per tetrahedron), `face2` the face neighbours, theirs, and the boundary faces of both (about 13); either removes the unstable eigenvalues. It returns the CSR for `Topology::{stencil_offsets, stencil}`, the twins for `stencil_twin` and the LSQ weights for `geometry.stencil_lsq_weight`. Ghost cells get the stencil their known faces give, so the deterministic adjoint collects what owned cells send them; a partition needs a halo as deep as the stencil.

`Inc` targets have role `Residual` (accumulation type). `Min`/`Max` ties keep the first incidence. `Sum` accumulates in the accumulation type; `Min`, `Max` and `Follow` keep a body output in the compute type, where selecting is exact, which halves their registers when the two types differ (Euler's gradient: 95 → 72 registers, 34 % faster on GB10).

**Jacobian blocks.** Face, boundary and cell loops may add `Jacobian(d, wrt=("ul", "ur"))` to `out`: per face (or cell), the derivative of the body's outputs with respect to the named arguments, the others held fixed, written to `d` of role `Jac` or `JacAccum` and shape `(n_out, n_in)` for one argument or `(len(wrt), n_out, n_in)` for several, row-major — the blocks an implicit solver assembles. `ad.jacobian` derives it by forward mode, one direction per input component in a single function sharing the body's primal, structural zeros folded. Written once per face (by the owner in gather-recompute), so every strategy gives the same blocks; the loop's tangent and adjoint ignore it. `Jac` storage is the place to move to 16-bit formats with a per-block scale later (research §7). Block datasets (`Jac`, and `JacAccum` for blocks kept in the accumulation precision: diagonal blocks and their inverses, FP64 under `fp32+acc64` while off-diagonal blocks stay `Jac`) are also read by loops, as the passive data of a linear operator.

## Generated C++

Per module `m`, in namespace `autan::kernels::m`: `template <class P> struct Data` (a `Topology` and one pointer per dataset, structure of arrays: component `c` of entity `i` at `c * n + i`; parameters by value) and one function per body. Loops: `cpu::name(d)`; `gpu_loops::name(d, schedule)` for cell loops and reductions; `gpu_loops::name(d, strategy, schedule)` for face and boundary loops, with `gpu::Strategy` one of:

| Strategy | Mechanism | Deterministic |
|---|---|---|
| `atomic` | one thread per face, atomic increments | no |
| `warp_aggregated` | owner-sorted faces: segmented shuffle reduction per warp, one atomic per segment; neighbours atomic | no |
| `coloured` | one launch per face colour (greedy in face order), plain increments | yes |
| `store_gather` | face values to scratch, then a cell gather in CSR order | yes |
| `gather_recompute` | each cell evaluates its own faces | yes |

`gpu::Schedule` carries the block size, the colourings (`autan::kernels::colour_faces`), the store-gather and incidence scratch and the stream.

`gpu::DeviceMesh` (`autan/kernels/device.cuh`) builds, from a `Mesh`, its `PackedGeometry` and an optional `CellStencil`, everything a physics' generated loops read besides its own datasets: the device `Topology` (owned-face prefixes, incidence twins, stencil and stencil twins), the colourings of the owned face ranges, the scratch and the `Schedule`. `bind(d)` points every built-in geometry member of a generated `Data<P>` at a device copy in that member's own type (`P::geom` or `P::geom_value`), uploaded on first use, so one `DeviceMesh` serves every policy; `set_stencil_weights` replaces the stencil weights (a physics' own gradient weights). The options `stencil_slice` and `incidence_slice` store the stencil or the cell -> face incidences in slices of that many consecutive cells, entry j of a cell at its start + j × slice (`autan::kernels::sliced`, `sliced_incidences`), so a warp reading entry j of its cells reads contiguous memory; every per-entry array follows, and the sliced incidences carry the cell across each face (`Topology::incidence_cell`), so a reduction reads no owner or neighbour. Each cell keeps its entries in the same order: the same bits as CSR. The block solvers use `incidence_slice = 32`: their product reads 88–93 % of GB10 peak sliced, 31–68 % in CSR (tet n96, hex n128); CSR stays the default, the explicit Euler step being no faster sliced (+0.4 % tet, +1.1 % hex, one take). `gpu::min` and `gpu::first_index` are exact device reductions (global time step, first diverged cell): their result does not depend on the launch. Sums and norms use the runtime's exact device sum.

**Bit for bit across rank counts** (runtime.md, Ranks). A cell's result depends on the order it sums its faces and stencil entries and on how each face is evaluated, both of which follow the local numbering of a partition. `GlobalKeys` (a partition's cell and face global ids and `face_reversed`) removes that dependence: `global_order` lists each cell's faces by global face id (`Topology::cell_faces`, with twins), `pack_geometry` computes incidence LSQ weights in that order and face moments in each face's global orientation from a canonical fan start, `cell_stencil` orders entries by global id, and `Topology::face_reversed` makes every face loop evaluate a reversed face in its global orientation (left and right swapped, `face_area`, `face_normal`, `face_moment` negated, `face_r_left`/`face_r_right` swapped; Jacobian blocks refer to the global left and right). With the gather strategies, on the CPU (`cpu::name(d, strategy)`) or the GPU (`DeviceMesh` option `global`), every owned cell then holds the same bits on any number of ranks. Only the gathers (`store_gather`, `gather_recompute`) sum a cell in its global order; the scatter loop, `atomic`, `warp_aggregated` and `coloured` follow the local numbering, so a topology ordered this way (`Topology::rank_independent`) refuses them (E1109). The guarantee holds only if the run uses this mode on every rank count, one included: it is a run-level choice, not a multi-rank one. Its cost is the gather's: 1.5× the scatter loop on one CPU core (advection), 1.05–1.17× the fastest scatter strategy for the Euler residual on GB10 (hex n128, tet n96).

Team per cell (lanes of a warp striding one cell's stencil, then a fixed butterfly combine) was measured on GB10: it lifts narrow stencil reductions (face2 LSQ gradient 57 → 94 % of peak on tetrahedra, vertex stencil 18 → 88 %) but slows Euler's gradient, whose 25 FP64 accumulators cost more in shuffles and registers than the coalescing saves (3.86 → 4.89 ns/cell at 4 lanes). It is not merged; the branch `codegen/team-per-cell` keeps it for scalar reductions such as a turbulence variable's gradient.

The adjoint of a cell reduction sends values to neighbouring cells, so it takes a strategy too: with `atomic` and `warp_aggregated` it scatters with atomics; with the deterministic strategies it runs in two passes (research §6, store-then-gather on the transposed incidence list): each owned cell adds its own adjoints and stores, per incidence, what it sends across, then each cell, owned or ghost, sums what its owned neighbours stored for it, in the order of its incidences. The twin of an incidence (the same face in the other cell's list) is `Topology::incidence_twin`, from `autan::kernels::incidence_twins(mesh)`; over a stencil, the twin of an entry is the cell's entry in its neighbour's stencil (`stencil_twin`; stencils are symmetric).

`autan::kernels::gpu::sum_of_squares` (`norms.cuh`) computes Σ x² in FP64 on the device in chunks of 4096 values and a fixed tree, so the result depends on the data and its length only, never on the launch shape.

**Derivatives.** Active data are cell datasets; geometry, parameters, face datasets (boundary values) and block datasets (`Jac`, `JacAccum`) are passive. `<loop>_tangent` reads `<d>_tan` of each active input and increments or writes `<d>_tan` of the output. `<loop>_adjoint` transposes access modes (research §4.1): an `Inc` output's adjoint reads `<out>_adj` through the same map with the declared signs; each active gather increments `<d>_adj`; a `Write` output's adjoint is read, then zeroed; a reduction's adjoint recomputes the accumulators and the incidence each `Min`/`Max` selected, then sends adjoints to the incidences. Adjoint datasets have role `Residual`. The adjoint of a residual is the reverse sequence of loop adjoints (example: `advdiff_derivative` in `codegen/spike/capi.cpp`). Branches are linearised on the selected side.

## Precision

| Policy | state, grad | residual, accum, jac_accum | geom, param | compute | jac |
|---|---|---|---|---|---|
| `fp64` | double | double | double | double | double |
| `fp32_acc64` (`fp32+acc64`) | float | double | double | float | float |
| `fp32` | float | float | float | float | float |

A body declared `@pointwise(compute="accum")` computes in the accumulation type instead of `compute`: FP64 under `fp32+acc64`, for the inverse of the diagonal blocks (`kernels/blocks.py`). A cell loop's body or a reduction's body or finalise may compute so when every dataset it binds is a block dataset or a parameter, and its loop then reads no active data, so it has no derivatives (E1003, E1012), and Metal, whose accumulation type has no division, refuses it (E1013).

`complex_step` (host only) computes in `std::complex<double>` for complex-step tests.

Built-in geometry that no body of the module reads through `rel()` (directly or as a `boundary=` source) reaches bodies only converted to `compute`, so `Data<P>` stores it in `P::geom_value`, a real type of the compute precision (`float` in `fp32+acc64`, `double` in `fp64` and for the complex step): the same values (rounded once, when filled instead of when loaded) in fewer bytes. Geometry `rel()` reads (in the spike kernels, the centroids) and `Geom` datasets a kernel writes keep `P::geom`.

## Errors

| Code | Name | Raised when |
|---|---|---|
| E1001 | `unsupported_construct` | A body uses a statement, expression or call outside the subset. |
| E1002 | `not_a_dsl_type` | An argument has no annotation, or one that is not a role type. |
| E1003 | `invalid_binding` | Loop bindings do not match the body's arguments, their types, or the loop kind. |
| E1004 | `invalid_loop_output` | A loop's outputs do not fit its kind (`Inc` signs, `Write`, accumulators). |
| E1005 | `invalid_control_flow` | A name is assigned in one branch only, `return` is in one branch only, or code follows `return`. |
| E1006 | `type_mismatch` | Shapes or kinds of operands disagree. |
| E1007 | `rel_of_computed_value` | `rel()` receives something other than `Geom` arguments as loaded. |
| E1008 | `inc_target_not_accumulation` | An `Inc` target does not have role `Residual`. |
| E1009 | `invalid_kernel_module` | A kernel module cannot be imported or declares no loop. |
| E1010 | `unknown_backend` | `--backends` names an unknown backend. |
| E1011 | `unknown_policy` | `--policies` names an unknown policy. |
| E1012 | `ad_unsupported` | A loop with several output datasets is differentiated, or a boundary loop writes face data computed from active data without frozen=True. |
| E1013 | `policy_unsupported_by_backend` | A precision policy is requested on a backend that cannot run it (`fp64` on Metal, a body computing in the accumulation type on Metal). |
| E1100 | `runtime_failure` | A generated loop or the spike driver fails without a more specific code. |
| E1101 | `gpu_runtime_error` | A CUDA or HIP call or kernel launch fails. |
| E1102 | `unknown_precision_policy` | A driver receives an unknown policy name. |
| E1103 | `unknown_race_strategy` | A driver receives an unknown strategy name. |
| E1104 | `face_scratch_too_narrow` | Store-gather scratch has fewer components per face than the loop's outputs. |
| E1105 | `incidence_scratch_too_narrow` | The deterministic adjoint of a cell reduction has fewer scratch components per incidence than the values it sends to neighbours. |
| E1106 | `incidence_twins_missing` | The deterministic adjoint of a cell reduction runs on a topology without incidence twins. |
| E1107 | `metal_runtime_error` | A Metal call, library compilation or command buffer fails. |
| E1108 | `strategy_unavailable` | A face-loop strategy needs atomics the backend lacks for the policy's accumulation type. |
| E1109 | `strategy_not_rank_independent` | A face or boundary loop on a rank-independent topology uses a strategy whose order follows the local numbering (scatter, atomic, warp-aggregated, coloured). |

## Spike (research §8)

Results, verdict and open items: [PR #44](https://github.com/Remenby31/autan/pull/44) and [kernel-generator-spike.md](kernel-generator-spike.md).
