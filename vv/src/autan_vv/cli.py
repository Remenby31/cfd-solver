"""`autan-vv` command line: manufactured solutions, mesh families, order studies, reference cases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from autan import CaseError

from autan_vv.errors import VVError


def _say(line: str) -> None:
    sys.stdout.write(line + "\n")


def _levels(text: str) -> tuple[int, ...]:
    return tuple(int(v) for v in text.split(","))


def _cmd_mms(args: argparse.Namespace) -> None:
    from autan_vv.mms import CATALOG, dump_json, to_cpp  # noqa: PLC0415

    names = sorted(CATALOG) if args.name == "all" else [args.name]
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        ms = CATALOG[name]()
        (args.out / f"{name}.hpp").write_text(to_cpp(ms))
        (args.out / f"{name}.json").write_text(dump_json(ms))
        _say(f"{name}: {args.out / name}.hpp, {args.out / name}.json")


def _cmd_meshes(args: argparse.Namespace) -> None:
    if args.kind == "naca0012":
        from autan_vv.meshes import naca0012  # noqa: PLC0415

        family, _ = naca0012.generate_family(args.levels, args.out)
    else:
        from autan_vv.meshes.families import generate_family  # noqa: PLC0415

        family, _ = generate_family(
            args.kind, args.levels, args.out, perturbation=args.perturbation, seed=args.seed
        )
    for lv in family.levels:
        _say(f"{lv.file}: {lv.n_cells} cells, h = {lv.h:.4g}, sha256 {lv.file_sha256[:12]}")


def _cmd_m2_cases(args: argparse.Namespace) -> None:
    from autan_vv.cases import naca0012  # noqa: PLC0415
    from autan_vv.meshes.families import MANIFEST_SCHEMA  # noqa: PLC0415

    manifest = json.loads((args.family / "manifest.json").read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("kind") != "naca0012":
        msg = f"{args.family} is not a naca0012 mesh family ({MANIFEST_SCHEMA})"
        raise VVError(msg, code="V0805")
    for condition in naca0012.CONDITIONS:
        out = args.out / condition
        out.mkdir(parents=True, exist_ok=True)
        for level in manifest["levels"]:
            mesh = (args.family / level["file"]).resolve()
            case = naca0012.case(
                condition,
                os.path.relpath(mesh, out),
                precision=args.precision,
                max_iterations=args.max_iterations,
            )
            path = out / f"{Path(level['file']).stem}.toml"
            path.write_text(case.to_toml())
            _say(str(path))


def _cmd_forces(args: argparse.Namespace) -> None:
    from autan_vv.cases import naca0012  # noqa: PLC0415

    report = naca0012.study(args.runs)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for level in report["levels"]:
        _say(
            f"{level['run']}: {level['n_cells']} cells, CL {level['cl']:.9f}, "
            f"CD {level['cd']:.9f}, CM {level['cmy']:.9f}"
        )
    for name, est in report["continuum"].items():
        if "error" in est:
            _say(f"{name}: no continuum estimate ({est['error']})")
        else:
            _say(
                f"{name}: continuum {est['estimate']:.9f}, p = {est['order']:.3f}, GCI {est['gci']:.3g}"
            )
    note = (
        ""
        if report["evidence"]
        else f" (not evidence: {'; '.join(report['not_evidence_because'])})"
    )
    for verdict in report["verdicts"]:
        _say(f"gate {verdict['gate']}: {'PASS' if verdict['pass'] else 'FAIL'}{note}")
    if not report["pass"]:
        raise SystemExit(1)


def _cmd_su2(args: argparse.Namespace) -> None:
    from autan_vv.su2 import naca0012 as su2  # noqa: PLC0415

    if args.action == "prepare":
        if args.family is None or args.out is None or args.runs:
            msg = "su2 prepare takes --family and --out, and no run directories"
            raise VVError(msg, code="V0903")
        for run in su2.prepare(
            args.family,
            args.out,
            dimension=args.dimension,
            max_iterations=args.max_iterations,
            stop_margin=args.stop_margin,
            scheme=args.scheme,
            published=args.published,
        ):
            _say(str(run.relative_to(args.out)))
        return
    if args.action == "history":
        from autan_vv.su2 import history  # noqa: PLC0415

        rows = {}
        for run in args.runs:
            conv = history.convergence(history.read_su2(run / "history.csv"), args.orders)
            rows[str(run)] = {**asdict(conv), **history.provenance(run)}
            state = "converged" if conv.converged else "NOT converged"
            _say(
                f"{run}: {state} in {conv.iterations} iterations, CL {conv.cl:.9f}, "
                f"CD {conv.cd:.9f}, CM {conv.cmy:.9f}, drop {min(conv.drop.values()):.2f}"
            )
        if args.out is not None:
            args.out.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        return
    if not args.runs:
        msg = "su2 study takes the run directories of one condition"
        raise VVError(msg, code="V0905")
    report = su2.study(args.runs)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for lv in report["levels"]:
        state = "converged" if lv["converged"] else "NOT converged"
        _say(
            f"{lv['run']}: {state} in {lv['iterations']} iterations, CL {lv['cl']:.9f}, "
            f"CD {lv['cd']:.9f}, CM {lv['cmy']:.9f}"
        )
    for name, est in (report["continuum"] or {}).items():
        if "error" in est:
            _say(f"{name}: no continuum estimate ({est['error']})")
        else:
            _say(
                f"{name}: continuum {est['estimate']:.9f}, p = {est['order']:.3f}, GCI {est['gci']:.3g}"
            )
    for verdict in report["verdicts"]:
        _say(f"gate {verdict['gate']}: {'PASS' if verdict['pass'] else 'FAIL'}")


def _cmd_order(args: argparse.Namespace) -> None:
    from autan_vv.study import order_study, write_report  # noqa: PLC0415

    data = json.loads(args.errors.read_text())
    report = order_study(data.get("label", args.errors.stem), int(data["dim"]), data["levels"])
    write_report(report, args.out, args.plot)
    for norm, fit in report["fits"].items():
        _say(f"{norm}: p = {fit['order']:.3f}, finest pair {fit['pairwise_orders'][-1]:.3f}")


def _cmd_reference(args: argparse.Namespace) -> None:
    from autan_vv.verify import advection_diffusion_study  # noqa: PLC0415

    report = advection_diffusion_study(
        args.kind,
        args.levels,
        args.out,
        perturbation=args.perturbation,
        seed=args.seed,
        precision=args.precision,
    )
    for norm, fit in report["fits"].items():
        _say(f"{norm}: p = {fit['order']:.3f}, finest pair {fit['pairwise_orders'][-1]:.3f}")


def _cmd_cases(args: argparse.Namespace) -> None:
    from autan_vv.cases import fetch, get_case, list_cases  # noqa: PLC0415

    if args.action == "list":
        for case in list_cases():
            _say(f"{case.name}: {case.title} ({len(case.files)} files)")
        return
    cases = list_cases() if args.case == "all" else [get_case(args.case)]
    for case in cases:
        for path in fetch(case, args.cache):
            _say(str(path))


def _cmd_oracles(_args: argparse.Namespace) -> None:
    from autan_vv.oracles.registry import ORACLES  # noqa: PLC0415

    for name in sorted(ORACLES):
        _say(f"{name}: {ORACLES[name]().description}")


def _cmd_verify(args: argparse.Namespace) -> None:
    import json  # noqa: PLC0415

    from autan_vv.harness import verify  # noqa: PLC0415

    report = verify(args.runs, args.against, args.gate)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for run in report["runs"]:
        relative = {
            name: e["L2"] / e["scale"] if e["L2"] is not None and e["scale"] else e["L2"]
            for name, e in run["errors"].items()
        }
        worst = max(relative, key=lambda n: float("inf") if relative[n] is None else relative[n])
        value = run["errors"][worst]["L2"]
        shown = "non-finite" if value is None else f"{value:.3g}"
        _say(
            f"{run['run_id']}: {run['n_cells']} cells, h = {run['h']:.4g}, worst L2 {worst} {shown}"
        )
    for name, fit in report["fits"].items():
        l2 = fit.get("L2", {})
        if "order" in l2:
            _say(f"{name}: L2 order {l2['order']:.3f}, finest pair {l2['pairwise_orders'][-1]:.3f}")
        else:
            _say(
                f"{name}: L2 order not measured ({l2.get('levels', 0)} levels above the rounding floor)"
            )
    verdict = report["verdict"]
    if verdict is not None:
        state = "PASS" if verdict["pass"] else "FAIL"
        if verdict["investigate"]:
            state += ", TO INVESTIGATE"
        for warning in verdict["warnings"]:
            _say(f"warning: {warning}")
        note = (
            ""
            if verdict["evidence"]
            else f" (not evidence: {'; '.join(verdict['not_evidence_because'])})"
        )
        _say(f"gate {verdict['gate']}: {state}{note}")
        if not verdict["pass"]:
            raise SystemExit(1)


def _cmd_agree(args: argparse.Namespace) -> None:
    import json  # noqa: PLC0415

    from autan_vv.agreement import agreement  # noqa: PLC0415

    report = agreement(args.fp64, args.fp32, args.k)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for pair in report["pairs"]:
        worst = max(pair["fields"], key=pair["fields"].get)
        _say(f"{pair['n_cells']} cells: worst {worst} {pair['fields'][worst]:.3g} x 2^-23")
    _say(f"fp32 agreement (k = {report['k']:g}): {'PASS' if report['pass'] else 'FAIL'}")
    if not report["pass"]:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    """Run the `autan-vv` CLI; returns the process exit code (2 on a coded V&V error)."""
    parser = argparse.ArgumentParser(prog="autan-vv", description=__doc__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("mms", help="export manufactured solutions to C++ and JSON")
    p.add_argument("name", help="solution name or 'all'")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_cmd_mms)

    p = sub.add_parser(
        "meshes",
        help="generate a seeded mesh family of the unit cube, of the Ringleb domain with its face"
        " states, of the periodic square of the isentropic vortex, or the NACA 0012 O-meshes",
    )
    p.add_argument(
        "kind",
        choices=(
            "tet",
            "hex",
            "prism",
            "hybrid",
            "poly",
            "agglomerated",
            "ringleb",
            "vortex-hex",
            "vortex-prism",
            "naca0012",
        ),
    )
    p.add_argument("--levels", type=_levels, default=(8, 16, 32, 64))
    p.add_argument("--perturbation", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_cmd_meshes)

    p = sub.add_parser("m2-cases", help="write the NACA 0012 cases of every condition and level")
    p.add_argument("case", choices=("naca0012",))
    p.add_argument("--family", type=Path, required=True, help="directory of the mesh manifest")
    p.add_argument("--precision", choices=("fp64", "fp32+acc64"), default="fp64")
    p.add_argument("--max-iterations", type=int, default=200000)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_cmd_m2_cases)

    p = sub.add_parser(
        "forces", help="continuum forces of a NACA 0012 family and its gates (exit 1 on failure)"
    )
    p.add_argument("case", choices=("naca0012",))
    p.add_argument("runs", type=Path, nargs="+", help="run directories of one condition")
    p.add_argument("--out", type=Path, help="write the report (autan.vv-force-study/1) as JSON")
    p.set_defaults(func=_cmd_forces)

    p = sub.add_parser(
        "su2",
        help="SU2 on the NACA 0012 family: prepare its cases, study its runs, or read any run's"
        " history against the convergence criterion",
    )
    p.add_argument("action", choices=("prepare", "study", "history"))
    p.add_argument("runs", type=Path, nargs="*", help="study: run directories of one condition")
    p.add_argument("--family", type=Path, help="prepare: directory of the mesh manifest")
    p.add_argument("--dimension", type=int, choices=(2, 3), default=2)
    p.add_argument("--scheme", choices=("roe", "jst"), default="roe", help="prepare: SU2 scheme")
    p.add_argument(
        "--published", type=Path, help="prepare: run this SU2-published configuration as it is"
    )
    p.add_argument("--max-iterations", type=int, default=20000)
    p.add_argument("--stop-margin", type=float, default=2.0, help="prepare: orders SU2 runs on")
    p.add_argument("--orders", type=float, default=10.0, help="history: residual drop criterion")
    p.add_argument("--out", type=Path, help="prepare: output root; study: report JSON")
    p.set_defaults(func=_cmd_su2)

    p = sub.add_parser(
        "order", help="observed order from errors on a mesh family (JSON in, JSON out)"
    )
    p.add_argument("errors", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plot", type=Path)
    p.set_defaults(func=_cmd_order)

    p = sub.add_parser(
        "reference", help="reference advection-diffusion solve and order study on a family"
    )
    p.add_argument("kind", choices=("tet", "hex", "prism", "hybrid", "poly", "agglomerated"))
    p.add_argument("--levels", type=_levels, default=(8, 16, 24, 32))
    p.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    p.add_argument("--perturbation", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_cmd_reference)

    p = sub.add_parser(
        "verify", help="verify solver runs against an oracle; with --gate, a verdict"
    )
    p.add_argument("runs", type=Path, nargs="+", help="run directories (autan.results/1)")
    p.add_argument(
        "--against", required=True, help="oracle name (autan-vv oracles) or 'freestream'"
    )
    p.add_argument("--gate", help="gate of docs/validation/acceptance.md (exit 1 if it fails)")
    p.add_argument("--out", type=Path, help="write the report (autan.vv-verdict/1) as JSON")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser(
        "agree",
        help="check fp32+acc64 runs against the fp64 runs of the same cases, level by level",
    )
    p.add_argument("--fp64", type=Path, nargs="+", required=True)
    p.add_argument("--fp32", type=Path, nargs="+", required=True)
    p.add_argument("--k", type=float, default=500.0)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=_cmd_agree)

    p = sub.add_parser("cases", help="list or download reference cases, verifying checksums")
    p.add_argument("action", choices=("list", "fetch"))
    p.add_argument("case", nargs="?", default="all")
    p.add_argument("--cache", type=Path, default=Path("~/.cache/autan-vv").expanduser())
    p.set_defaults(func=_cmd_cases)

    p = sub.add_parser(
        "oracles", help="list the exact and reference solutions runs are verified against"
    )
    p.set_defaults(func=_cmd_oracles)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (VVError, CaseError) as err:
        # CaseError: the SDK's coded diagnostics (unreadable run, contract violation).
        sys.stderr.write(f"{err}\n")
        return 2
    return 0
