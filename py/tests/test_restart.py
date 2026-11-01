"""Restarting from a previous run (solver-input.md#start, results.md#runjson).

The oracle of `continue` is the uninterrupted run: continuing a stopped run must give
the history the run would have had without the stop.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import autan
import autan.solver_input
import jsonschema
import pytest
from autan.cli import json_schema, main

from tests.conftest import CRM_HL, FIXTURES, identity_of


def history(result: autan.Results) -> list[str]:
    return (result.path / "history.jsonl").read_text().splitlines()


@pytest.fixture
def case(write_case: Callable[..., Path]) -> autan.Case:
    return autan.load(write_case(CRM_HL))


def test_continue_extends_the_run_as_if_it_never_stopped(
    case: autan.Case, crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    whole = case.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path / "whole")
    stopped = case.run(crm_mesh, backend=autan.FakeBackend(iterations=30), runs=tmp_path / "parts")
    assert stopped.run.status == "stopped"
    resumed = case.run(
        crm_mesh, backend=autan.FakeBackend(), runs=tmp_path / "parts", start=stopped
    )
    assert history(stopped) + history(resumed) == history(whole)
    assert (resumed.run.status, resumed.run.iterations) == (whole.run.status, 100)

    start = resumed.run.start
    assert start is not None
    solution = (stopped.path / "solution.cgns").read_bytes()
    assert (start.run_id, start.identity, start.mode, start.iterations) == (
        stopped.run.run_id,
        stopped.run.identity,
        "continue",
        30,
    )
    assert start.solution_sha256 == hashlib.sha256(solution).hexdigest()
    run = resumed.run
    assert run.start is not None
    assert (run.start.mode, run.start.iterations) == ("continue", 30)
    assert run.identity == identity_of(run.document())


def test_warm_start_counts_from_one(
    write_case: Callable[..., Path], crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    previous = autan.load(write_case(CRM_HL)).run(
        crm_mesh, backend=autan.FakeBackend(), runs=tmp_path
    )
    other = autan.load(write_case(CRM_HL.replace("mach=0.2,", "mach=0.25,"), name="other.py"))
    warm = other.run(
        crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=previous, mode="warm"
    )
    assert history(warm)[0].startswith('{"iteration": 1,')
    assert warm.run.start is not None
    assert warm.run.start.mode == "warm"


@pytest.mark.parametrize(
    ("change", "mode", "code"),
    [
        (lambda body: body.replace("mach=0.2,", "mach=0.25,"), "continue", "E0708"),
        (lambda body: body + "case.time = autan.Steady(max_iterations=30)\n", "continue", "E0708"),
        (lambda body: body.replace("autan.SA()", "autan.SST()"), "warm", "E0708"),
    ],
)
def test_a_start_must_match_the_case(
    change: Callable[[str], str],
    mode: str,
    code: str,
    write_case: Callable[..., Path],
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
) -> None:
    previous = autan.load(write_case(CRM_HL)).run(
        crm_mesh, backend=autan.FakeBackend(iterations=30), runs=tmp_path
    )
    changed = autan.load(write_case(change(CRM_HL), name="changed.py"))
    with pytest.raises(autan.CaseError) as error:
        changed.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=previous, mode=mode)  # type: ignore[arg-type]
    assert error.value.codes == [code]


def test_only_finished_runs_on_the_same_mesh_can_be_resumed(
    case: autan.Case,
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
    mesh_variant: Callable[..., autan.MeshInfo],
) -> None:
    diverged = case.run(crm_mesh, backend=autan.FakeBackend(diverge_at=10), runs=tmp_path)
    with pytest.raises(autan.CaseError) as error:
        case.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=diverged)
    assert error.value.codes == ["E0707"]

    stopped = case.run(crm_mesh, backend=autan.FakeBackend(iterations=5), runs=tmp_path)

    def other_mesh(document: dict[str, dict[str, str]]) -> None:
        document["source"]["sha256"] = "1" * 64

    with pytest.raises(autan.CaseError) as error:
        case.run(
            mesh_variant("crm-hl", other_mesh),
            backend=autan.FakeBackend(),
            runs=tmp_path,
            start=stopped,
        )
    assert "E0203" in error.value.codes


def test_unsteady_continue_keeps_time_and_averaging(
    write_case: Callable[..., Path], crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    body = CRM_HL.replace("autan.SA()", "autan.DDES(base=autan.SA())") + (
        "from autan.units import s\n"
        "case.time = autan.Unsteady(steps=20, dt=1e-3 * s, average_from=5)\n"
    )
    case = autan.load(write_case(body))
    stopped = case.run(crm_mesh, backend=autan.FakeBackend(iterations=8), runs=tmp_path)
    resumed = case.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=stopped)
    first = json.loads(history(resumed)[0])
    assert (first["iteration"], first["time"]) == (9, pytest.approx(9e-3))
    assert resumed.run.time == pytest.approx(20e-3)
    assert resumed.run.averaging is not None
    assert (resumed.run.averaging.from_step, resumed.run.averaging.to_step) == (6, 20)


def test_prepare_writes_the_start(
    case: autan.Case,
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    write_case: Callable[..., Path],
) -> None:
    stopped = case.run(crm_mesh, backend=autan.FakeBackend(iterations=30), runs=tmp_path)
    argv = [
        "prepare",
        str(write_case(CRM_HL)),
        "--mesh-info",
        str(FIXTURES / "crm-hl.meshinfo.json"),
        "--start",
        str(stopped.path),
    ]
    assert main(argv) == 0
    document = json.loads(capsys.readouterr().out)
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    assert document["start"]["run"] == str(stopped.path.absolute())
    assert (document["start"]["mode"], document["start"]["iterations"]) == ("continue", 30)
    assert main([*argv, "--warm"]) == 0
    assert json.loads(capsys.readouterr().out)["start"]["mode"] == "warm"


def test_continue_and_warm_from_the_same_run_have_different_identities(
    case: autan.Case, crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    stopped = case.run(crm_mesh, backend=autan.FakeBackend(iterations=30), runs=tmp_path)
    continued = case.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=stopped)
    warm = case.run(
        crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=stopped, mode="warm"
    )
    assert continued.run.identity != warm.run.identity


def test_a_chain_of_continuations_keeps_the_original_residual_reference(
    case: autan.Case, crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    first = case.run(crm_mesh, backend=autan.FakeBackend(iterations=10), runs=tmp_path)
    second = case.run(
        crm_mesh, backend=autan.FakeBackend(iterations=10), runs=tmp_path, start=first
    )
    third = case.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=second)
    original = json.loads(history(first)[0])["residuals"]
    assert third.run.start is not None
    assert third.run.start.residual_reference == original


def test_warm_start_with_no_iteration_keeps_the_source_state(
    write_case: Callable[..., Path], crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    source = autan.load(write_case(CRM_HL)).run(
        crm_mesh, backend=autan.FakeBackend(), runs=tmp_path
    )
    other = autan.load(write_case(CRM_HL.replace("mach=0.2,", "mach=0.25,"), name="b.py"))
    warm = other.run(
        crm_mesh, backend=autan.FakeBackend(iterations=0), runs=tmp_path, start=source, mode="warm"
    )
    assert warm.field("VelocityX").tobytes() == source.field("VelocityX").tobytes()


def test_the_source_must_be_a_sound_restartable_run(
    write_case: Callable[..., Path], crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    case = autan.load(write_case(CRM_HL))
    empty = case.run(crm_mesh, backend=autan.FakeBackend(iterations=0), runs=tmp_path)
    with pytest.raises(autan.CaseError) as error:
        case.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=empty)
    assert error.value.codes == ["E0707"]

    a = case.run(crm_mesh, backend=autan.FakeBackend(iterations=10), runs=tmp_path)
    b = autan.load(write_case(CRM_HL.replace("mach=0.2,", "mach=0.25,"), name="b.py")).run(
        crm_mesh, backend=autan.FakeBackend(iterations=10), runs=tmp_path
    )
    (a.path / "solution.cgns").write_bytes((b.path / "solution.cgns").read_bytes())
    with pytest.raises(autan.CaseError) as error:
        case.run(
            crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=autan.Results.open(a.path)
        )
    assert error.value.codes == ["E0707"]


def test_warm_start_needs_the_same_transported_variables(
    write_case: Callable[..., Path], crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    unsteady = "from autan.units import s\ncase.time = autan.Unsteady(steps=20, dt=1e-3 * s)\n"
    sa = autan.load(
        write_case(CRM_HL.replace("autan.SA()", "autan.DDES(base=autan.SA())") + unsteady)
    )
    sst = autan.load(
        write_case(
            CRM_HL.replace("autan.SA()", "autan.DDES(base=autan.SST())") + unsteady, name="sst.py"
        )
    )
    source = sa.run(crm_mesh, backend=autan.FakeBackend(iterations=5), runs=tmp_path)
    with pytest.raises(autan.CaseError) as error:
        sst.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=source, mode="warm")
    assert error.value.codes == ["E0708"]


@pytest.mark.parametrize("value", ["30", True])
def test_start_counts_are_strict(
    value: object,
    case: autan.Case,
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    write_case: Callable[..., Path],
) -> None:
    stopped = case.run(crm_mesh, backend=autan.FakeBackend(iterations=30), runs=tmp_path)
    assert (
        main(
            [
                "prepare",
                str(write_case(CRM_HL)),
                "--mesh-info",
                str(FIXTURES / "crm-hl.meshinfo.json"),
                "--start",
                str(stopped.path),
            ]
        )
        == 0
    )
    document = json.loads(capsys.readouterr().out)
    document["start"]["iterations"] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    with pytest.raises(ValueError, match="iterations"):
        autan.solver_input.SolverInput.model_validate(document)


def _zero_momentum_y(result: autan.Results, wake_at: int | None) -> None:
    """Make momentum-y identically zero at iteration 1, rising above its floor at ``wake_at``."""
    path = result.path / "history.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    for line in lines:
        woke = wake_at is not None and line["iteration"] == wake_at
        line["residuals"]["momentum-y"] = 1.0 if woke else 0.0
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


@pytest.mark.parametrize(("wake_at", "woken"), [(None, []), (10, ["momentum-y"])])
def test_a_continuation_carries_the_woken_equations(
    wake_at: int | None,
    woken: list[str],
    case: autan.Case,
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
) -> None:
    first = case.run(crm_mesh, backend=autan.FakeBackend(iterations=20), runs=tmp_path)
    _zero_momentum_y(first, wake_at)
    second = case.run(
        crm_mesh,
        backend=autan.FakeBackend(iterations=10),
        runs=tmp_path,
        start=autan.Results.open(first.path),
    )
    assert second.run.start is not None
    assert second.run.start.woken == woken
    _zero_momentum_y(second, None)  # it stays below its floor in the second part
    third = case.run(
        crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=autan.Results.open(second.path)
    )
    assert third.run.start is not None
    assert third.run.start.woken == woken  # carried along the chain
    warm = case.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=first, mode="warm")
    assert warm.run.start is not None
    assert warm.run.start.woken is None


@pytest.mark.parametrize(
    "woken", [None, ["bogus"], ["momentum-y", "momentum-y"], ["momentum-y", "density"]]
)
def test_woken_must_name_reference_equations_in_order(
    woken: list[str] | None,
    case: autan.Case,
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    write_case: Callable[..., Path],
) -> None:
    stopped = case.run(crm_mesh, backend=autan.FakeBackend(iterations=30), runs=tmp_path)
    argv = [
        "prepare",
        str(write_case(CRM_HL)),
        "--mesh-info",
        str(FIXTURES / "crm-hl.meshinfo.json"),
    ]
    assert main([*argv, "--start", str(stopped.path)]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["start"]["woken"] == []
    document["start"]["woken"] = woken
    with pytest.raises(ValueError, match="woken"):
        autan.solver_input.SolverInput.model_validate(document)
    if woken is None or woken[-1] != "density":  # order is the readers' check only
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)


def test_a_run_whose_start_lost_its_woken_equations_is_rejected(
    case: autan.Case, crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    first = case.run(crm_mesh, backend=autan.FakeBackend(iterations=10), runs=tmp_path)
    second = case.run(
        crm_mesh, backend=autan.FakeBackend(iterations=10), runs=tmp_path, start=first
    )
    document = json.loads((second.path / "run.json").read_text())
    document["start"]["woken"] = None
    (second.path / "run.json").write_text(json.dumps(document))
    with pytest.raises(autan.CaseError) as error:
        autan.Results.open(second.path)
    assert error.value.codes == ["E0701"]
