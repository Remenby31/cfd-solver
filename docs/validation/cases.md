# Reference cases

Source: `vv/src/autan_vv/cases/` and its catalogue `catalog.json` (schema `autan.case-library/1`). The wider survey of datasets and their licences is [docs/research/validation-datasets.md](../research/validation-datasets.md); this page covers what the code fetches.

```sh
uv run autan-vv cases list
uv run autan-vv cases fetch tmr-2dzp-flatplate      # into ~/.cache/autan-vv/<case>/
uv run pytest -m network                            # fetch, verify and parse every case
```

## How files are pinned

Every file has a URL, a size and a SHA-256. TMR data files use `raw.githubusercontent.com/TMBWG/turbmodels/<commit>/…` at commit `20a39f5`, which is immutable; the TMR site moved from turbmodels.larc.nasa.gov to [tmbwg.github.io/turbmodels](https://tmbwg.github.io/turbmodels/) and its old links now redirect to a NASA notice. TMR grids exist only inside the zip archives on nasa.gov (`wp-content/uploads/2026/02/…`); the catalogue pins each archive and each member's own SHA-256. Archives are downloaded once into `blobs/` by checksum. Every file is written under a unique temporary name, verified, then renamed, so concurrent fetches never see unverified bytes. A mismatch raises `V0601` (download) or `V0602` (file or member) and leaves nothing behind; there is no mirror fallback.

## Cases

| Case | Content | Parsers |
|---|---|---|
| `tmr-2dzp-flatplate` | 5 nested PLOT3D grids (35×25 … 545×385, 16 MB zip); CFL3D and FUN3D SA: cf along the plate, cf(0.97) and CD per grid, profiles at x = 0.97 | PLOT3D 2D, Tecplot |
| `tmr-2dbc-bump` | 5 grids (89×41 … 1409×641, 166 MB zip); CFL3D and FUN3D SA: cp, cf, forces per grid, profiles at x = 0.75 | PLOT3D 2D, Tecplot |
| `tmr-2dn00-naca0012` | 5 grids (113×33 … 1793×513, 110 MB zip) and the exact airfoil points; CFL3D SA cp, cf, CL, CD at α = 0, 10, 15°; experiments of Ladson, Gregory and O'Reilly, Abbott and von Doenhoff, McCroskey | PLOT3D 2D, Tecplot |
| `tmr-2dn00-naca0012-numerics` | CFL3D and FUN3D SA forces on grid families I–III at α = 10° (Diskin et al. 2015), cp and cf | Tecplot |
| `onera-m6` | Schmitt and Charpin pressure at 7 span stations, runs 2308 (M 0.8395, α 3.06°) and 2565 (M 0.8372, α 6.06°), from TMR and NPARC (NPARC `.ex` files store −Cp); section profiles | Tecplot, columns |
| `rae2822` | Geometry; case 6 Cp (NPARC); cases 1, 6, 7, 9, 12 conditions, forces, Cp, cf from the 1981 AFOSR-HTTM-Stanford conference tape (72 coded files decoded; the text file is skipped; see known defects) | columns, 1981 tape |

Pointers only (too large to fetch): HLPW-5 (CRM-HL, finest grids 2×10⁸–10⁹ cells, archive 1.75 GB at aiaa-hlpw.org) and DPW-7/DPW-8 (dpw.larc.nasa.gov; DPW-7 Vassberg grids 5.3–221 M points), in `pointers()`.

**Known defects of the sources.** Section 1124 of the 1981 tape (case 6 Cp) is damaged in the checksummed copy: records 23–29 are bit-shifted bytes. The catalogue declares it, the reader skips it and raises `V0603` on any undeclared malformed section; take case 6 Cp from NPARC. NPARC files store −Cp (`raetaf01_cp.exp.gen`, the ONERA M6 `cp*.ex`), as their catalogue descriptions say.

**RAE 2822 case 10** has no reachable digital source (the AGARD AR-138 PDF times out at sto.nato.int and DTIC refuses); it is not in the catalogue. The NWTF/Airbus RAE 2822 dataset is CC BY-NC 4.0, refused by our licence policy.

## Verification

Offline tests check the checksum and member errors through `file://` URLs and each parser on snippets written from its format description. Network tests (`-m network`) fetch every file of every case, verify it, parse it, check grid dimensions against the file names, compare the values transcribed in the catalogue from the TMR pages with the downloaded data files (flat plate cf and CD for CFL3D and FUN3D, NACA 0012 CL and CD), reproduce with `autan_vv.order.richardson` the orders TMR publishes for the flat plate (cf p = 1.98 and 1.34, CD p = 1.75 and 0.80), and decode RAE 2822 case 9 from the 1981 tape to its published conditions (M = 0.73, α = 3.19°, corrected 2.79°, Re = 6.5×10⁶).

## NACA 0012 inviscid

The three M2 conditions of Vassberg and Jameson (2010), on the [O-mesh family](meshes.md#naca-0012-o-meshes) (source `vv/src/autan_vv/cases/naca0012.py`); there is nothing to download.

```sh
autan-vv meshes naca0012 --levels 32,64,128,256,512 --out meshes/naca0012
autan-vv m2-cases naca0012 --family meshes/naca0012 --out cases/naca0012   # one case per condition and level
autan-vv forces naca0012 runs/<one run per level> --out study.json           # exit 1 if a gate fails
```

- **Cases** `m050-a125`, `m080-a000`, `m080-a125` (M, α in hundredths of a degree): Euler, FP64, slip `airfoil`, `farfield`, symmetry `back` and `front`, steady to 10 orders of residual drop; forces on the airfoil with reference chord 1 m, area chord × span and moments about the quarter chord (`cmy` is the nose-up pitching moment).
- **Study.** Each run must be the case of its condition up to mesh, precision and iteration budget, converging at least 10 orders (`V0801`); the runs must be one family (`V0802`: same case apart from mesh, commit, backend, precision, distinct meshes), at least four levels, all `converged` (`V0803`). `m2-cases` refuses a directory whose manifest is not a `naca0012` family (`V0805`). h = N^(−1/2) ∝ 1/NC. CL, CD and CM are extrapolated from the three finest levels with `autan_vv.order.richardson`, which for a ratio of 2 is the paper's eqs. 20–24; a coefficient that does not converge monotonically gets the code of its failure instead of an estimate, and its gates fail. The report (`autan.vv-force-study/1`) records each level's run id, case and mesh hashes, solver commit, backend and precision, and gives the paper's eq. 25 order, meaningful for the subcritical CD whose limit is zero. It is evidence only in `fp64` (where acceptance.md measures these gates) and when no run comes from the fake backend.
- **Gates** of [acceptance.md](acceptance.md#m2--compressible-euler): `naca0012-m080-a000-cd`, `naca0012-m080-a125-cl`, `naca0012-m080-a125-cd`, `naca0012-m050-a125-cd`.
- **Tests** (`vv/tests/test_naca0012.py`): the geometry to every digit of the paper's eqs. 4–8 and 11; the continuum and order of the paper's Tables 2, 3 and 9 from their three finest grids; the gates on FLO82's sequences, including its |CD| rise from 256² to 512² at M 0.5, which fails the monotonic test; a family of fake-backend runs read back through `autan.Results`.
- **Far field.** The CL band of the M 0.8, α 1.25° gate comes from point-vortex far-field runs; the Euler far field takes the uniform freestream. The correction is requested from the Euler lane in #186.

## ONERA M6 inviscid

Run 2308 of Schmitt and Charpin (M 0.8395, α 3.06°) in Euler, on the [ONERA M6 family](meshes.md#onera-m6-meshes) (source `vv/src/autan_vv/cases/onera_m6.py`).

```sh
autan-vv meshes onera-m6 --levels 12,24,48 --out meshes/onera-m6
autan-vv m2-cases onera-m6 --family meshes/onera-m6 --out cases/onera-m6
autan-vv cases fetch onera-m6                                              # case_2308.dat
autan-vv sections runs/<run> --experiment ~/.cache/autan-vv/onera-m6/case_2308.dat --out m6.json
```

- **Case.** Slip `wing`, `symmetry`, `farfield`, FP64, steady to 10 orders; forces on the wing with reference area 0.7532 m² and length 0.64607 m.
- **Run check.** The run must be this case on its mesh, up to precision, iteration budget and monitors (`V0801`): Mach, α, physics, gas and boundaries included.
- **Gate** `onera-m6-residual-drop`, in `fp64` only: the run ended `converged` under a criterion of at least 10 orders, as the runtime applies it (reference residuals of the original run, exempt zero equations, [case-schema.md](../contracts/case-schema.md#time)); the report also gives each residual's drop. Evidence only off the fake backend; exit 1 when it fails. In `fp32+acc64` there is no residual gate (acceptance.md compares forces and Cp with `fp64`).
- **Report.** The wall pressure (`WallSolution`) becomes Cp with the freestream's p and q; it is averaged to the nodes (area-weighted) and interpolated linearly where the wing's edges cross the plane y = η b of each of the seven stations; x/c uses the original local chord. The shock position is the rearmost rise of the upper-surface Cp through the critical Cp* (sonic), by linear interpolation, for the computation and for the taps alike; at η = 0.20, 0.44, 0.65 the report says whether the rear shocks lie within 0.03 of each other. The Cp RMS difference with the taps leaves out taps within 0.05 of a rise through Cp* on their surface, computed or measured.
- **Tests** (`vv/tests/test_onera_m6.py`): the section table against rows typed from the TMR document and, with `-m network`, against NPARC's `airfoil.txt`; planform against the published sweeps, taper, area, MAC and the TMR's trailing-edge extension; mesh nodes on the conical surface, symmetry plane, far-field sphere, closed boundary; regeneration bit for bit; Cp* against the isentropic relation; the cut exact for linear fields; a run written through `autan`'s writers read back and cut, and runs at another incidence refused; the TMR and NPARC transcriptions of run 2308 give the same shocks (`-m network`).

## Licences

TMR repository: CC0-1.0. nasa.gov archives and NPARC (grc.nasa.gov) pages: US Government works with no stated restriction. Nothing is vendored: files are fetched by hash at use time.

## References

- C. L. Rumsey, B. R. Smith and G. P. Huang, Description of a website resource for turbulence modeling verification and validation, AIAA 2010-4742. doi:10.2514/6.2010-4742
- B. Diskin, J. L. Thomas, C. L. Rumsey and A. Schwöppe, Grid convergence for turbulent flows, AIAA 2015-1746; *AIAA J.* 54(9):2563–2588, 2016. doi:10.2514/1.J054555
- V. Schmitt and F. Charpin, Pressure distributions on the ONERA-M6-wing at transonic Mach numbers, AGARD AR-138, 1979.
- P. H. Cook, M. A. McDonald and M. C. P. Firmin, Aerofoil RAE 2822 — pressure distributions, and boundary layer and wake measurements, AGARD AR-138, 1979.
- J. C. Vassberg and A. Jameson, In pursuit of grid convergence for two-dimensional Euler solutions, *J. Aircraft* 47(4):1152–1166, 2010. doi:10.2514/1.46737
- C. L. Ladson, NASA TM-4074, 1988; N. Gregory and C. L. O'Reilly, ARC R&M 3726, 1970; W. J. McCroskey, NASA TM-100019, 1987; I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover, 1959.
