# Acceptance criteria

The numbers each milestone gate in [PLAN.md](../PLAN.md) must meet, where they come from, and the `autan-vv` tool that measures them. A threshold marked *ours* is a project decision; its rationale is in the row. Dataset details (files, conditions, licences, further cases) live in [validation-datasets.md](../research/validation-datasets.md); this page states the gate once.

Conventions for every gate (a gate identifier in a row means `autan-vv verify <runs> --against <oracle> --gate <id>` checks it, see [harness.md](harness.md); the `naca0012-*` force gates are checked by `autan-vv forces naca0012 <runs>` and `onera-m6-residual-drop` by `autan-vv sections <run>`, see [cases.md](cases.md#naca-0012-inviscid)):

- **Order** is the least-squares order of the L2 error over at least four levels and the order of the finest pair, both from `autan-vv order` ([order.md](order.md)). L∞ is reported, not gated: on non-nested unstructured families it fluctuates by ±1 (measured, [order.md](order.md#measured-orders)).
- **Code-to-code** gates compare our **continuum estimate** (Richardson from the three finest of at least four levels, with its GCI, `autan_vv.order.richardson`) with the published continuum estimate, never level by level.
- **Round-off** tolerances are multiples of the machine epsilon of the precision under test (AGENTS.md, Tests).
- **FP32** orders are taken on levels whose error is at least 100× the FP32 rounding floor ([kernel-generator.md](../research/kernel-generator.md) §8).
- Every reported run records mesh hash, case hash, commit and precision.

## M1 — generator spike

| Criterion | Threshold | Source |
|---|---|---|
| MMS `advection_diffusion`, second-order reconstruction | L2 order ≥ 1.9 (fit and finest pair) on tet, hex, poly (incl. `agglomerated`, 100+ faces) and hybrid families, clean and perturbed, in `fp64`, `fp32`, `fp32+acc64` | PLAN M1; kernel-generator.md §8. The reference solve reaches 1.95–2.55 on the same families ([order.md](order.md#measured-orders)), so the bar is attainable |
| Linear fields | LSQ and corrected Green-Gauss gradients exact to 64 ε; the reference solve reproduces a linear solution to ≤ 256 ε | Exactness of linear reconstruction; kernel-generator.md §4.3 |
| Conservation | Σ residual = Σ boundary fluxes to 64 ε × Σ|flux| | Discrete divergence theorem |
| Adjoint dot product | gap ≤ 1×10⁻¹³ (FP64), ≤ 1×10⁻⁵ (FP32, `fp32+acc64`) | kernel-generator.md §8 |
| Bandwidth | ≥ 70 % of measured GB10 peak on the gradient and flux loops | PLAN M1, AGENTS.md |

## M2 — compressible Euler

| Criterion | Threshold | Source |
|---|---|---|
| Freestream preservation, gate `freestream-preservation` | uniform freestream kept to L∞ ≤ 64 ε of the run precision, relative, in every field, on every cell family | Discrete conservation of a constant state (closed cells) |
| MMS `euler`, gate `euler-mms-order` | L2 order ≥ 1.9 (fit and finest pair) for each conserved variable, errors decreasing strictly from level to level, 4 cell families (levels n = 12, 16, 24, 32 of `autan-vv meshes`; 16, 24, 32, 48 for `hybrid`, pre-asymptotic below, [euler.md](../design/euler.md#manufactured-solution)), Roe, HLLC, AUSM+-up, FP32 and FP64; above 2.1 a warning, above 2.6 the PR explains it | Roy et al. 2004 (design order); [datasets §M2](../research/validation-datasets.md#m2--compressible-euler). *Ours* (lead, 2026-11-01): non-nested families supraconverge (polyhedral duals reach 2.5, as in M1), so only the lower bound blocks |
| Ringleb flow, gate `ringleb-order` | L2 order 2 ± 0.1 in density and entropy | HiOCFD1 C1.2; datasets §M2 |
| Isentropic vortex, one period (t = 40 at Mach 0.5), domain [−10, 10]², gate `vortex-order` | spatial order 2 ± 0.1 for each conserved variable; temporal order = RK order ± 0.1 | Spiegel, Huynh and DeBonis, AIAA 2015-2444 |
| Sod shock tube, gate `sod-order` | density L1 order between 0.6 and 1.1 | Banks, Aslam and Rider 2008 (between 2/3 and 1 for a contact); *ours*: 0.6 and 1.1 allow ±0.1 of fitting noise |
| NACA 0012 inviscid, M 0.8, α 0°, gate `naca0012-m080-a000-cd` | continuum CD within 1 count of 0.008342 in `fp64`; `fp32+acc64` as for ONERA M6 | Vassberg and Jameson 2010, Tables 10–11 (via datasets §M2) |
| NACA 0012 inviscid, M 0.8, α 1.25°, gates `naca0012-m080-a125-cl`, `naca0012-m080-a125-cd` | CL inside 0.3516–0.3562 widened by our GCI; CD within 1 count of 0.02268 in `fp64`; `fp32+acc64` as for ONERA M6 | idem, point-vortex farfield runs |
| NACA 0012 inviscid, M 0.5, α 1.25°, gate `naca0012-m050-a125-cd` | \|CD\| decreases monotonically (the paper's CD tends to zero from below on its fine grids); extrapolated \|CD\| < 0.5 count in `fp64`; `fp32+acc64` as for ONERA M6 | idem (subcritical, CD → 0) |
| ONERA M6 inviscid (run 2308 conditions), gate `onera-m6-residual-drop` | `fp64`: residual drop ≥ 10 orders. `fp32+acc64`: CL, CD, CM within 0.1 % of `fp64` and Cp within 10⁻³ of it on the same mesh ([euler.md](../design/euler.md#precision)); the residual plateau is reported, not gated; shock x/c within ±0.03 of experiment at η = 0.20–0.65 (report) | PLAN M2; Schmitt and Charpin 1979 (`onera-m6` case). *Ours* (lead, 2026-11-01): 10 orders prove the discretisation and the solver, so they are measured in `fp64`; in `fp32+acc64` each face flux carries FP32 rounding, which bounds the drop by an amount that depends on the case (3.8 orders on a NACA 0012, [euler.md](../design/euler.md#precision)) |

## M3 — Navier-Stokes and Spalart-Allmaras

| Criterion | Threshold | Source |
|---|---|---|
| MMS `navier_stokes`, gate `ns-mms-order` | L2 order ≥ 1.9 (fit and finest pair) for each conserved variable, errors decreasing strictly from level to level, 4 families + perturbed, FP32 and FP64, non-orthogonal correction on; above 2.1 a warning, above 2.6 the PR explains it | Roy et al. 2004. *Ours* (lead, 2026-11-01): the rule of `euler-mms-order` holds for every MMS order gate |
| MMS `rans_sa`, gate `rans-sa-mms-order` (the case schema's SA is SA-neg; `rans_sa_noft2` is for codes that offer SA-noft2) | L2 order ≥ 1.9 (fit and finest pair) for the mean flow and ν̂ with second-order turbulence advection, errors decreasing strictly from level to level; above 2.1 a warning, above 2.6 the PR explains it | Eça et al. 2007; Oliver et al. 2012; [mms.md](mms.md). *Ours* (lead, 2026-11-01): as `ns-mms-order` |
| Compressible Couette, gate `couette-velocity-exact` | velocity exact to 64 ε of the run precision, relative | Exact solution, linear velocity ([oracles.md](oracles.md)) |
| Compressible Couette, gate `couette-temperature-order` | T order 2 ± 0.1; wall heat flux order 2 ± 0.1 as well once results.md carries it (the gate checks T only until then) | Exact solution, quadratic temperature |
| TMR flat plate (`tmr-2dzp-flatplate`) | continuum cf(x = 0.97) within 0.5 % of 2.7056×10⁻³ (hex) and 1 % (tet, poly); continuum CD within 0.5 % of 2.8589×10⁻³ | CFL3D and FUN3D continuum estimates from `cf_convergence.dat` and `drag_convergence.dat` (three finest grids): cf 2.70524 / 2.70601×10⁻³ (p = 1.98 / 1.34), CD 2.85924 / 2.85861×10⁻³ (p = 1.75 / 0.80), the orders TMR publishes. *Ours*: 0.5 % is ten times the CFL3D–FUN3D cf spread (0.03 %) |
| TMR bump-in-channel (`tmr-2dbc-bump`) | continuum CD within 0.5 % of 3.5714×10⁻³ and continuum cf(0.75) within 0.5 % of 6.1536×10⁻³ | CFL3D continuum estimates from `force_convergence.dat` and `cf_convergence_sa.dat` (three finest grids, p = 2.37 and 1.43, GCI 0.03 % and 0.09 %); FUN3D's three finest CD values oscillate, so it has no continuum CD and its finest-grid 3.5611×10⁻³ is reported only. *Ours*: 0.5 % is about the CFL3D–FUN3D finest-grid CD spread (0.3 %) plus the GCI |
| TMR NACA 0012, α 10°, family II (`tmr-2dn00-naca0012-numerics`) | \|ΔCL\| ≤ 0.002 from 1.0912–1.0913; \|ΔCD\| ≤ 1 count from 0.01221–0.01222 | Diskin et al. 2015 continuum range; *ours*: tighter than the code-to-code SA-neg spread of HLPW-5 test case 1 (ΔCL ≈ 0.005, ΔCD ≈ 5 counts, datasets §M7) because this 2D case is better defined |
| NACA 0012 validation, α 0/10/15° (`tmr-2dn00-naca0012`) | report CL, CD, Cp, cf against CFL3D and Ladson, Gregory and O'Reilly; no gate on experiment | TMR 2DN00 page |
| Degraded meshes | residual drop ≥ 8 orders on the perturbed families (`--perturbation 0.3`) | PLAN M3 ("convergence on degraded meshes"); *ours*: 8 orders leaves forces converged to < 0.01 % |

## M5 — multi-GPU and precision

| Criterion | Threshold | Source |
|---|---|---|
| Precision invariance | CL and CD in `fp32+acc64` within 0.1 % of `fp64` on the M3 TMR cases and ONERA M6 | PLAN M5 |
| Scaling | parallel efficiency ≥ 80 % at 8 GPUs, measured to 64 | PLAN M5 |

## M6 — discrete adjoint

| Criterion | Threshold | Source |
|---|---|---|
| Dot product, every kernel and the composed residual | as M1 | kernel-generator.md §8 |
| Gradient vs central finite differences (FP64) | relative difference ≤ 10⁻⁶ at the step minimising truncation plus round-off | *ours*: FD truncation O(h²) with h ≈ ε^(1/3) gives ≈ 10⁻¹⁰–10⁻⁶ |
| RAE 2822 and ONERA M6 drag minimisation | each optimisation ends feasible (CL within 10⁻³ of its target, thickness or volume constraints met to 10⁻⁶ relative) and optimal (projected gradient reduced by 10⁻³ from the start), with CD lower than the baseline's continuum CD by more than the baseline's GCI | PLAN M6; *ours*: a reduction smaller than the baseline's discretisation uncertainty is not a result. Results are compared with the ADODG benchmark cases, to be added to the library with M6 |
| Baseline against experiment (report) | RAE 2822 case 9: CN 0.803, CM −0.099, CD 0.0168 at M 0.73, α 2.79° corrected, Re 6.5×10⁶ | Cook, McDonald and Firmin 1979, decoded from the 1981 tape (`rae2822` case) |

## M7 — SST, DDES, WMLES

| Criterion | Threshold | Source |
|---|---|---|
| Driver–Seegmiller backward-facing step | x_reatt/H within ±0.3 of 6.26 | Driver and Seegmiller 1985 (6.26 ± 0.10), via TMR; datasets §M3 |
| HLPW-5 test case 1, CRM-HL wing-body at α 11° (verification) | our continuum CL, CD, CM inside the envelope of the HLPW-5 SA-neg submissions, whose spread is ΔCL ≈ 0.005, ΔCD ≈ 5 counts, ΔCM ≈ 0.003; the absolute envelope is transcribed from the summary's figure 5 when M7 starts | HLPW-5 summary (NTRS 20240014255); datasets §M7 |
| HLPW-5 test case 2.4, near CLmax (validation) | report CL(α) and CLmax against the matching submissions and the experiment; no gate (RANS scatter near CLmax is about 0.5 in CL) | HLPW-5 summary; state-of-the-art.md §1.3 |

Milestones M4 and M8–M11 gate on functionality or on cases still to be added; their numbers will be set here when those cases enter the library.
