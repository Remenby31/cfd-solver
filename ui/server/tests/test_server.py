import hashlib
import http.client
import json
import math
import os
import shutil
import socket
import threading
import time
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import ANY

import autan
import autan.initial_state
import httpx
import numpy as np
import numpy.typing as npt
import pytest
import uvicorn
from autan_ui import create_app
from autan_ui.confine import Folder
from autan_ui.history import read_history
from autan_ui.registry import Registry
from autan_ui.summary import residual_orders
from autan_ui.surface import synthetic_wing
from fastapi import FastAPI

PY = Path(__file__).parents[3] / "py"


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    case = autan.load(PY / "examples" / "crm_hl.py")
    mesh_info = autan.MeshInfo.load(PY / "tests" / "fixtures" / "crm-hl.meshinfo.json")
    case.run(mesh_info, backend=autan.FakeBackend(iterations=30), runs=tmp_path / "runs")
    return tmp_path / "runs"


@pytest.fixture
def static(tmp_path: Path) -> Path:
    """A built app next to a file it must never serve, with a link pointing at that file."""
    (tmp_path / "secret.txt").write_text("SECRET")
    app = tmp_path / "static"
    (app / "assets").mkdir(parents=True)
    (app / "index.html").write_text("INDEX")
    (app / "assets" / "app.js").write_text("APP")
    (app / "leak").symlink_to(tmp_path / "secret.txt")
    return app


@contextmanager
def serving(app: FastAPI) -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    instance = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    while not instance.started:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        instance.should_exit = True
        thread.join()


@pytest.fixture
def server(runs: Path, static: Path) -> Iterator[str]:
    with serving(create_app(runs, static=static)) as url:
        yield url


def sse_events(response: httpx.Response) -> Iterator[tuple[str, Any]]:
    event = None
    for line in response.iter_lines():
        if line.startswith("event: "):
            event = line.removeprefix("event: ")
        elif line.startswith("data: ") and event:
            yield event, json.loads(line.removeprefix("data: "))
            event = None


def test_registry_lists_conforming_runs_and_rejects_others(runs: Path) -> None:
    source = next(runs.iterdir())
    (runs / "notes").mkdir()
    foreign = runs / "run-20261031T000000Z-00000000"
    foreign.mkdir()
    (foreign / "run.json").write_text('{"schema": "other/1"}')
    bad_time = shutil.copytree(source, runs / "run-20261031T000001Z-00000001")
    run = json.loads((bad_time / "run.json").read_text())
    run |= {"run_id": bad_time.name, "started": "not-a-timestamp"}
    (bad_time / "run.json").write_text(json.dumps(run))
    (runs / ".run-20261031T000002Z-00000002.tmp").mkdir()

    snapshot = Registry(runs).scan()

    assert [(r["case"]["name"], r["status"]) for r in snapshot.runs.values()] == [
        ("crm-hl", "stopped")
    ]
    assert sorted(snapshot.rejected) == sorted(["notes", foreign.name, bad_time.name])
    assert snapshot.rejected[foreign.name].startswith("E0701")


def test_the_registry_lists_the_runs_whose_solution_is_a_result(runs: Path) -> None:
    # results.md: an ended run's solution.cgns is its result, history or not; a running run's
    # is not yet.
    case = autan.load(PY / "examples" / "crm_hl.py")
    mesh_info = autan.MeshInfo.load(PY / "tests" / "fixtures" / "crm-hl.meshinfo.json")
    early = case.run(mesh_info, backend=autan.FakeBackend(iterations=0), runs=runs).path
    source = next(p for p in runs.iterdir() if p != early)
    assert (early / "solution.cgns").is_file()
    assert sorted(Registry(runs).scan().solutions) == sorted([source.name, early.name])

    run = json.loads((source / "run.json").read_text())
    (source / "run.json").write_text(
        json.dumps(run | {"status": "running", "finished": None, "monitors": None})
    )
    assert Registry(runs).scan().solutions == [early.name]


def test_a_run_without_history_has_a_summary_with_its_settings(runs: Path) -> None:
    # py/examples/crm_hl.py: M 0.2, 19.57 deg; no iteration recorded yet.
    case = autan.load(PY / "examples" / "crm_hl.py")
    mesh_info = autan.MeshInfo.load(PY / "tests" / "fixtures" / "crm-hl.meshinfo.json")
    early = case.run(mesh_info, backend=autan.FakeBackend(iterations=0), runs=runs).path

    summary = Registry(runs).scan().summaries[early.name]

    assert (summary["iteration"], summary["trend"]) == (None, [])
    assert (summary["settings"]["mach"], summary["settings"]["aoa"]) == (0.2, "19.57 deg")


def test_registry_rechecks_a_run_when_any_contract_file_changes(runs: Path) -> None:
    source = next(runs.iterdir())
    registry = Registry(runs)
    case = (source / "case.toml").read_text()

    (source / "case.toml").write_text("schema = 1\n")
    assert source.name in registry.scan().rejected
    (source / "case.toml").write_text(case)
    assert source.name in registry.scan().runs


def test_an_unreadable_run_is_rejected_not_fatal(runs: Path) -> None:
    source = next(runs.iterdir())
    (source / "history.jsonl").unlink()
    (source / "history.jsonl").mkdir()
    assert source.name in Registry(runs).scan().rejected


def test_history_holds_a_partial_line_only_while_live(runs: Path) -> None:
    directory = next(runs.iterdir())
    folder = Folder(directory)
    first = read_history(folder, 0, live=False)
    assert [r["iteration"] for r in first.records] == list(range(1, 31))

    with (directory / "history.jsonl").open("a") as history:
        history.write('{"iteration": 31, "time": null, "resid')
    assert read_history(folder, first.offset, live=True).records == []
    with pytest.raises(ValueError, match="not terminated"):
        read_history(folder, first.offset, live=False)

    with (directory / "history.jsonl").open("a") as history:
        history.write('uals": {"density": 1e-9}}\n')
    chunk = read_history(folder, first.offset, live=True)
    assert [r["iteration"] for r in chunk.records] == [31]
    folder.close()


def test_registry_stream_reports_runs_and_rejections_as_they_appear(
    runs: Path, server: str
) -> None:
    source = next(runs.iterdir())
    with httpx.stream("GET", f"{server}/api/runs/events", timeout=10) as response:
        events = (e for e in sse_events(response) if e[0] != "summaries")
        kind, snapshot = next(events)
        assert (kind, len(snapshot["runs"])) == ("runs", 1)

        staging = shutil.copytree(source, runs / ".staging")
        run = json.loads((staging / "run.json").read_text())
        run |= {"run_id": "run-20261031T235959Z-0123abcd", "started": "2026-10-31T23:59:59Z"}
        (staging / "run.json").write_text(json.dumps(run))
        copy = staging.rename(runs / run["run_id"])
        kind, data = next(events)
        assert (kind, data["run_id"]) == ("run", copy.name)
        assert next(events) == ("solutions", sorted([source.name, copy.name]))

        (copy / "run.json").write_text("{")
        assert [next(events), next(events), next(events)] == [
            ("removed", copy.name),
            ("solutions", [source.name]),
            ("rejected", [{"dir": copy.name, "reason": ANY}]),
        ]


def test_history_stream_delivers_appended_lines_and_status(runs: Path, server: str) -> None:
    directory = next(runs.iterdir())
    run_id = directory.name
    ended = (directory / "run.json").read_text()
    running = json.loads(ended) | {"status": "running", "finished": None, "monitors": None}
    (directory / "run.json").write_text(json.dumps(running))
    with httpx.stream("GET", f"{server}/api/runs/{run_id}/history/events", timeout=10) as response:
        events = (e for e in sse_events(response) if e[0] != "log")
        kind, data = next(events)
        assert (kind, data["status"]) == ("run", "running")
        kind, records = next(events)
        assert kind == "history"
        assert len(records) == 30

        with (directory / "history.jsonl").open("a") as history:
            history.write(json.dumps({"iteration": 31, "time": None, "residuals": {}}) + "\n")
        kind, records = next(events)
        assert (kind, [r["iteration"] for r in records]) == ("history", [31])

        (directory / ".run.json").write_text(ended)
        (directory / ".run.json").replace(directory / "run.json")
        kind, data = next(events)
        assert (kind, data["status"]) == ("run", "stopped")


def test_a_stream_ends_when_the_case_its_run_is_checked_against_changes(
    runs: Path, server: str
) -> None:
    directory = next(runs.iterdir())
    with httpx.stream(
        "GET", f"{server}/api/runs/{directory.name}/history/events", timeout=10
    ) as response:
        events = (e for e in sse_events(response) if e[0] not in ("log", "history"))
        assert next(events)[0] == "run"

        # results.md: run.json's case.sha256 is the stored case's hash; this case is another.
        case = (directory / "case.toml").read_text()
        (directory / "case.toml").write_text(case.replace('name = "crm-hl"', 'name = "other"'))
        kind, _ = next(events)
        assert kind == "invalid"


def test_a_replaced_run_json_is_checked_again_whatever_its_time_and_size(runs: Path) -> None:
    directory = next(runs.iterdir())
    registry = Registry(runs)
    assert directory.name in registry.scan().runs
    path = directory / "run.json"
    before = path.stat()
    text = path.read_text()
    identity = json.loads(text)["identity"]
    # Another identity of the same length: run.json no longer matches its definition.
    other = ("0" if identity[0] != "0" else "1") + identity[1:]
    (directory / ".run.json").write_text(text.replace(identity, other))
    (directory / ".run.json").replace(path)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (path.stat().st_mtime_ns, path.stat().st_size) == (before.st_mtime_ns, before.st_size)

    assert directory.name in registry.scan().rejected


def test_unknown_run_is_404_before_any_stream(server: str) -> None:
    unknown = "run-20261031T000000Z-deadbeef"
    assert httpx.get(f"{server}/api/runs/{unknown}").status_code == 404
    assert httpx.get(f"{server}/api/runs/{unknown}/history/events").status_code == 404
    assert httpx.get(f"{server}/api/runs/..%2F..%2Fetc").status_code == 404


def test_a_rejected_run_is_409_with_its_reason(runs: Path, server: str) -> None:
    source = next(runs.iterdir())
    (source / "mesh-info.json").write_text("{}")
    response = httpx.get(f"{server}/api/runs/{source.name}")
    assert (response.status_code, response.json()["detail"][:5]) == (409, "E0701")
    assert httpx.get(f"{server}/api/runs/{source.name}/history/events").status_code == 409


def test_foreign_host_header_is_refused(server: str) -> None:
    assert httpx.get(f"{server}/api/runs", headers={"Host": "evil.example"}).status_code == 400


def test_synthetic_surface_layout() -> None:
    message = synthetic_wing(10_000)
    length = int.from_bytes(message[:4], "little")
    header = json.loads(message[4 : 4 + length])
    body = memoryview(message)[4 + length :]
    buffers = {b["name"]: b for b in header["buffers"]}
    indices = np.frombuffer(
        body, np.uint32, buffers["indices"]["length"], buffers["indices"]["offset"]
    )

    assert (4 + length) % 4 == 0
    assert sum(zone["triangles"] for zone in header["zones"]) == header["triangles"]
    assert indices.size == 3 * header["triangles"]
    assert int(indices.max()) < header["nodes"]
    assert [zone["name"] for zone in header["zones"]] == ["wing.upper", "wing.lower", "wing.tip"]


def test_every_run_reports_its_iteration_and_residual_drop(runs: Path) -> None:
    source = next(runs.iterdir())
    run = json.loads((source / "run.json").read_text())
    (source / "run.json").write_text(
        json.dumps(run | {"status": "running", "finished": None, "monitors": None})
    )
    records = [json.loads(line) for line in (source / "history.jsonl").read_text().splitlines()]
    with (source / "history.jsonl").open("a") as history:
        history.write('{"iteration": 31, "time": null, "resid')
    registry = Registry(runs)

    summary = registry.scan().summaries[source.name]

    first, last = records[0]["residuals"], records[-1]["residuals"]
    slowest = min(math.log10(first[name] / last[name]) for name in first)
    assert summary["iteration"] == 30
    assert summary["orders"] == pytest.approx(slowest, abs=1e-3)
    assert len(summary["trend"]) == 30
    case = tomllib.loads((source / "case.toml").read_text())
    assert (summary["target_orders"], summary["max_iterations"]) == (
        case["time"]["residual_orders"],
        case["time"]["max_iterations"],
    )
    assert summary["settings"]["aoa"] == case["freestream"]["aoa"]
    assert summary["settings"]["turbulence"] == case["physics"]["turbulence"]["model"]

    # Lines appended after the directory was checked are not re-validated by the registry.
    with (source / "history.jsonl").open("a") as history:
        history.write('uals": {}, "padding": "' + "x" * 200_000 + '"}\n')
        history.write('{"iteration": 1e400}\n')
    snapshot = registry.scan()
    assert source.name in snapshot.runs
    assert snapshot.summaries[source.name]["iteration"] == 31


def test_run_details_and_plan(runs: Path, server: str) -> None:
    run_id = next(runs.iterdir()).name
    details = httpx.get(f"{server}/api/runs/{run_id}").json()
    assert {o["name"] for o in details["outputs"]} >= {
        "run.json",
        "case.toml",
        "mesh-info.json",
        "history.jsonl",
        "solution.cgns",
    }
    assert details["case_toml"] == (runs / run_id / "case.toml").read_text()
    mesh_info = json.loads((runs / run_id / "mesh-info.json").read_text())

    plan = httpx.get(f"{server}/api/runs/{run_id}/plan").json()

    assert plan["ok"] is True
    assert {b["zone"] for b in plan["boundaries"]} == {
        z["name"] for z in mesh_info["boundary_zones"]
    }


@pytest.mark.parametrize(
    "target",
    [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/assets/..%2f..%2fsecret.txt",
        "//etc/passwd",
        "/leak",
    ],
)
def test_static_files_never_leave_the_build(server: str, target: str) -> None:
    host, port = server.removeprefix("http://").split(":")
    connection = http.client.HTTPConnection(host, int(port))
    connection.putrequest("GET", target, skip_accept_encoding=True)
    connection.endheaders()
    body = connection.getresponse().read()
    connection.close()
    assert body == b"INDEX"
    assert httpx.get(f"{server}/assets/app.js").text == "APP"


def test_static_files_are_checked_when_served(server: str, static: Path, tmp_path: Path) -> None:
    (static / "added.js").write_text("ADDED")
    (static / "assets" / "app.js").unlink()
    (static / "assets" / "app.js").symlink_to(tmp_path / "secret.txt")

    assert httpx.get(f"{server}/added.js").text == "INDEX"
    assert httpx.get(f"{server}/assets/app.js").text == "INDEX"
    (static / "assets" / "app.js").unlink()
    assert httpx.get(f"{server}/assets/app.js").text == "INDEX"

    elsewhere = tmp_path / "elsewhere-assets"
    elsewhere.mkdir()
    (elsewhere / "app.js").write_text("SECRET")
    shutil.rmtree(static / "assets")
    (static / "assets").symlink_to(elsewhere, target_is_directory=True)
    assert httpx.get(f"{server}/assets/app.js").text == "INDEX"


def test_a_linked_index_disables_static_files(runs: Path, static: Path, tmp_path: Path) -> None:
    (static / "index.html").unlink()
    (static / "index.html").symlink_to(tmp_path / "secret.txt")
    with serving(create_app(runs, static=static)) as url:
        assert httpx.get(f"{url}/").status_code == 404
        assert httpx.get(f"{url}/api/runs").status_code == 200


@pytest.mark.parametrize("linked", ["directory", "run.json", "history.jsonl", "log.jsonl"])
def test_a_run_reached_through_a_link_is_refused(
    runs: Path, server: str, tmp_path: Path, linked: str
) -> None:
    source = next(runs.iterdir())
    outside = shutil.copytree(source, tmp_path / "elsewhere" / source.name)
    if linked == "directory":
        shutil.rmtree(source)
        source.symlink_to(outside, target_is_directory=True)
    else:
        (source / linked).unlink()
        (source / linked).symlink_to(outside / linked)

    assert httpx.get(f"{server}/api/runs/{source.name}").status_code == 404
    assert httpx.get(f"{server}/api/runs/{source.name}/history/events").status_code == 404
    assert Registry(runs).scan().rejected == {source.name: "is or contains a symbolic link"}


def test_a_stream_keeps_reading_the_run_it_opened(runs: Path, server: str, tmp_path: Path) -> None:
    source = next(runs.iterdir())
    outside = shutil.copytree(source, tmp_path / "elsewhere" / source.name)
    url = f"{server}/api/runs/{source.name}/history/events"
    with httpx.stream("GET", url, timeout=10) as response:
        events = sse_events(response)
        assert next(events)[0] == "run"
        assert next(events)[0] == "history"

        moved = source.rename(tmp_path / "moved")
        source.symlink_to(outside, target_is_directory=True)
        for directory, iteration in ((outside, 999), (moved, 31)):
            with (directory / "history.jsonl").open("a") as history:
                record: dict[str, Any] = {"iteration": iteration, "time": None, "residuals": {}}
                history.write(json.dumps(record) + "\n")

        _, records = next(e for e in events if e[0] == "history")
        assert [r["iteration"] for r in records] == [31]


def test_a_manufactured_solution_run_plans_against_its_archive(
    runs: Path, server: str, tmp_path: Path
) -> None:
    fixture = PY / "tests" / "fixtures" / "euler.mms.json"
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    case = tmp_path / "mms.py"
    case.write_text(
        "import autan\nfrom autan.units import K, Pa, kg, mol, s\n"
        f"gas = autan.IdealGas(name='mms', molar_mass={8.314462618 / 287.0!r} * kg / mol,"
        " gamma=1.4, prandtl=0.72, viscosity=autan.ConstantViscosity(mu=1e-3 * Pa * s))\n"
        "case = autan.Case('mms', mesh='crm-hl.cgns')\n"
        "case.physics = autan.Compressible(turbulence=autan.Inviscid(), gas=gas)\n"
        "case.freestream(mach=0.3, pressure=101325 * Pa, temperature=300 * K)\n"
        f"case.verification = autan.Manufactured(path='euler.mms.json', sha256={digest!r})\n"
    )
    shutil.copy(fixture, tmp_path / "euler.mms.json")
    mesh_info = autan.MeshInfo.load(PY / "tests" / "fixtures" / "crm-hl.meshinfo.json")
    result = autan.load(case).run(mesh_info, backend=autan.FakeBackend(iterations=2), runs=runs)

    plan = httpx.get(f"{server}/api/runs/{result.path.name}/plan").json()

    assert plan["ok"] is True, plan


INF = math.inf


@pytest.mark.parametrize(
    ("first", "now", "woken", "floors", "expected", "now_woken"),
    [
        ({"rho": 1e-300, "e": 1.0}, {"rho": 1e-310, "e": 1e-8}, set(), {}, 8.0, set()),
        ({"rho": 1.0, "e": 0.0}, {"rho": 1e-8, "e": 0.0}, set(), {}, 8.0, set()),
        ({"rho": 1.0, "e": 1.0}, {"rho": 0.0, "e": 1e-8}, set(), {}, 8.0, set()),
        ({"rho": 1.0, "e": 1.0}, {"rho": 0.0, "e": 0.0}, set(), {}, INF, set()),
        # Identically zero at iteration 1 (below 1e-12 of the largest): exempt at the floor...
        ({"rho": 1.0, "my": 0.0}, {"rho": 1e-8, "my": 1e-12}, set(), {}, 8.0, set()),
        ({"rho": 1.0, "my": 1e-30}, {"rho": 1e-8, "my": 1e-33}, set(), {}, 8.0, set()),
        # ...woken above it, for good: it must drop like the others, which it cannot from zero.
        ({"rho": 1.0, "my": 0.0}, {"rho": 1e-8, "my": 1e-11}, set(), {}, None, {"my"}),
        ({"rho": 1.0, "my": 0.0}, {"rho": 1e-8, "my": 1e-33}, {"my"}, {}, None, {"my"}),
        ({"rho": 1.0, "my": 1e-30}, {"rho": 1e-8, "my": 1e-33}, {"my"}, {}, 3.0, {"my"}),
        # Every equation of a line wakes, even after one that cannot be compared.
        (
            {"rho": 1.0, "a": 0.0, "b": 1e-30},
            {"rho": 1e-8, "a": 1.0, "b": 1.0},
            set(),
            {},
            None,
            {"a", "b"},
        ),
        # At or below its round-off floor (run.json convergence.floor), an equation has converged.
        ({"rho": 1.0, "e": 1.0}, {"rho": 1e-4, "e": 1e-6}, set(), {"rho": 1e-4}, 6.0, set()),
        (
            {"rho": 1.0, "e": 1.0},
            {"rho": 1e-4, "e": 1e-6},
            set(),
            {"rho": 1e-4, "e": 1e-6},
            INF,
            set(),
        ),
        (
            {"rho": 1.0, "e": 1.0},
            {"rho": 2e-4, "e": 1e-6},
            set(),
            {"rho": 1e-4, "e": 1e-6},
            4 - math.log10(2),
            set(),
        ),
    ],
)
def test_residual_drop_is_the_slowest_equation_not_converged(
    first: dict[str, float],
    now: dict[str, float],
    woken: set[str],
    floors: dict[str, float],
    expected: float | None,
    now_woken: set[str],
) -> None:
    orders = residual_orders(first, now, woken, floors)
    assert orders == (None if expected is None else pytest.approx(expected))
    assert woken == now_woken


def test_a_run_at_its_round_off_floors_reports_its_target(runs: Path) -> None:
    source = next(runs.iterdir())
    lines = (source / "history.jsonl").read_text().splitlines()
    last = json.loads(lines[-1])["residuals"]
    run = json.loads((source / "run.json").read_text())
    (source / "run.json").write_text(json.dumps(run | {"convergence": {"floor": last}}))

    summary = Registry(runs).scan().summaries[source.name]

    case = tomllib.loads((source / "case.toml").read_text())
    assert summary["orders"] == case["time"]["residual_orders"]
    assert summary["trend"][-1] == case["time"]["residual_orders"]


def test_floors_written_after_the_history_was_read_recompute_the_drop(runs: Path) -> None:
    source = next(runs.iterdir())
    registry = Registry(runs)
    before = registry.scan().summaries[source.name]["orders"]
    last = json.loads((source / "history.jsonl").read_text().splitlines()[-1])["residuals"]
    run = json.loads((source / "run.json").read_text())
    (source / "run.json").write_text(json.dumps(run | {"convergence": {"floor": last}}))

    after = registry.scan().summaries[source.name]["orders"]

    case = tomllib.loads((source / "case.toml").read_text())
    assert before < case["time"]["residual_orders"]
    assert after == case["time"]["residual_orders"]


def _archived_state_run(kind: str, runs: Path, tmp_path: Path) -> tuple[Path, Path]:
    # The case names its state file by absolute path: a plan that reads it is not confined.
    mesh = json.loads((PY / "tests" / "fixtures" / "crm-hl.meshinfo.json").read_text())
    case = (PY / "examples" / "crm_hl.py").read_text().replace("autan.SA()", "autan.Laminar()")
    source = tmp_path / "state.json"
    row = [1.2, 30, 0, 0, 1e5]
    if kind == "oracle-state":
        for zone in mesh["boundary_zones"]:
            if zone["name"] == "farfield":
                zone.update(faces=2, type=None, native_type=None)
        states = {
            "schema": "autan.face-states/1",
            "oracle": "test",
            "mesh_sha256": mesh["source"]["sha256"],
            "variables": ["density", "velocity-x", "velocity-y", "velocity-z", "pressure"],
            "zones": {
                "farfield": {"faces": 2, "centroids": [[0, 0, 0], [1, 0, 0]], "values": [row] * 2}
            },
        }
        source.write_text(json.dumps(states))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        case += f"case.boundary('farfield').oracle_state(path={str(source)!r}, sha256={digest!r})\n"
    else:
        mesh["counts"]["cells"] = 4
        mesh["counts"]["cells_by_type"] = dict.fromkeys(mesh["counts"]["cells_by_type"], 0)
        mesh["counts"]["cells_by_type"]["hexa"] = 4
        mesh["cell_zones"] = [{"name": "fluid", "type": None, "native_type": None, "cells": 4}]
        mesh["interfaces"] = []
        digest = autan.initial_state.write(
            source,
            oracle="test",
            mesh_sha256=mesh["source"]["sha256"],
            centroids=[[0.1 * i, 0, 0] for i in range(4)],
            values=[row] * 4,
        )
        case += f"case.initial_state = autan.StateFile(path={str(source)!r}, sha256={digest!r})\n"
    (tmp_path / "mesh.meshinfo.json").write_text(json.dumps(mesh))
    (tmp_path / "case.py").write_text(case)
    mesh_info = autan.MeshInfo.load(tmp_path / "mesh.meshinfo.json")
    backend = autan.FakeBackend(iterations=2)
    run = autan.load(tmp_path / "case.py").run(mesh_info, backend=backend, runs=runs).path
    return run, source


STATE_FILES = pytest.mark.parametrize(
    ("kind", "missing"), [("oracle-state", "E0420"), ("initial-state", "E0508")]
)


@pytest.mark.parametrize("kind", ["oracle-state", "initial-state"])
def test_a_run_plans_against_its_archived_state_file(
    kind: str, runs: Path, server: str, tmp_path: Path
) -> None:
    run, source = _archived_state_run(kind, runs, tmp_path)
    source.unlink()  # The plan reads the run's archive, not the source.

    plan = httpx.get(f"{server}/api/runs/{run.name}/plan").json()

    assert plan["ok"] is True, plan


@STATE_FILES
def test_a_missing_state_archive_is_never_replaced_by_its_source(
    kind: str, missing: str, runs: Path, server: str, tmp_path: Path
) -> None:
    run, _ = _archived_state_run(kind, runs, tmp_path)
    assert httpx.get(f"{server}/api/runs/{run.name}/plan").json()["ok"] is True
    for archive in (run / "states").iterdir():
        archive.unlink()  # After the run was accepted: its source is still at tmp_path.

    plan = httpx.get(f"{server}/api/runs/{run.name}/plan").json()

    assert plan["ok"] is False
    assert missing in {d["code"] for d in plan["diagnostics"]}


def test_several_histories_share_one_stream(runs: Path, server: str) -> None:
    first = next(runs.iterdir())
    second = shutil.copytree(first, runs / ".staging")
    run = json.loads((second / "run.json").read_text())
    run |= {"run_id": "run-20261031T235959Z-0123abcd", "started": "2026-10-31T23:59:59Z"}
    (second / "run.json").write_text(json.dumps(run))
    second = second.rename(runs / run["run_id"])
    url = f"{server}/api/runs/history/events?run={first.name}&run={second.name}"
    resume = {"Last-Event-ID": json.dumps({first.name: 0, second.name: None})}
    with httpx.stream("GET", url, headers=resume, timeout=10) as response:
        events = sse_events(response)
        seen = [next(events) for _ in range(6)]
    assert sorted((kind, data["run_id"]) for kind, data in seen) == sorted(
        [(k, r) for k in ("run", "history", "log") for r in (first.name, second.name)]
    )
    assert (
        httpx.get(f"{server}/api/runs/history/events?run=run-20261031T000000Z-deadbeef").status_code
        == 404
    )


def test_the_log_streams_with_a_documentation_link_per_code(runs: Path, server: str) -> None:
    case = autan.load(PY / "examples" / "crm_hl.py")
    mesh_info = autan.MeshInfo.load(PY / "tests" / "fixtures" / "crm-hl.meshinfo.json")
    diverged = case.run(mesh_info, backend=autan.FakeBackend(diverge_at=20), runs=runs).path
    written = [json.loads(line) for line in (diverged / "log.jsonl").read_text().splitlines()]
    url = f"{server}/api/runs/{diverged.name}/history/events"
    with httpx.stream("GET", url, timeout=10) as response:
        _, lines = next(e for e in sse_events(response) if e[0] == "log")

    assert [{k: v for k, v in line.items() if k != "doc"} for line in lines] == written
    coded = [line for line in lines if line["code"]]
    assert [line["code"] for line in coded if line["code"] == "D0107"] == ["D0107"]
    assert all(line["doc"].endswith(line["code"].lower()) for line in coded)
    assert all(line["doc"] is None for line in lines if not line["code"])


def test_removing_the_log_after_a_scan_rejects_the_run(runs: Path) -> None:
    source = next(runs.iterdir())
    registry = Registry(runs)
    assert source.name in registry.scan().runs
    (source / "log.jsonl").unlink()
    assert source.name in registry.scan().rejected


def test_stream_cursors_count_only_what_was_sent(runs: Path, server: str) -> None:
    directory = next(runs.iterdir())
    url = f"{server}/api/runs/{directory.name}/history/events"
    ids = []
    with httpx.stream("GET", url, timeout=10) as response:
        for line in response.iter_lines():
            if line.startswith("id: "):
                ids.append(json.loads(line.removeprefix("id: ")))
            if len(ids) == 2:
                break
    history_size = (directory / "history.jsonl").stat().st_size
    log_size = (directory / "log.jsonl").stat().st_size
    assert ids == [
        {"history.jsonl": history_size, "log.jsonl": 0},
        {"history.jsonl": history_size, "log.jsonl": log_size},
    ]


def test_a_log_line_that_breaks_the_contract_ends_the_stream(runs: Path, server: str) -> None:
    directory = next(runs.iterdir())
    run = json.loads((directory / "run.json").read_text())
    (directory / "run.json").write_text(
        json.dumps(run | {"status": "running", "finished": None, "monitors": None})
    )
    url = f"{server}/api/runs/{directory.name}/history/events"
    with httpx.stream("GET", url, timeout=10) as response:
        events = sse_events(response)
        assert [next(events)[0] for _ in range(3)] == ["run", "history", "log"]
        with (directory / "log.jsonl").open("a") as log:
            log.write('{"time": "yesterday", "iteration": null, "level": "info"}\n')
        kind, reason = next(e for e in events if e[0] != "history")
    assert kind == "invalid"
    assert reason.startswith("log.jsonl")


def test_a_run_json_its_reader_refuses_ends_the_stream(runs: Path, server: str) -> None:
    directory = next(runs.iterdir())
    run = json.loads((directory / "run.json").read_text())
    url = f"{server}/api/runs/{directory.name}/history/events"
    with httpx.stream("GET", url, timeout=10) as response:
        events = sse_events(response)
        assert next(events)[0] == "run"
        # Well-formed, but not what the run computes (results.md#runjson, identity), replaced
        # atomically so that no reader sees a half-written file.
        (directory / ".run.json").write_text(json.dumps(run | {"identity": "0" * 64}))
        (directory / ".run.json").replace(directory / "run.json")
        kind, reason = next(e for e in events if e[0] not in ("history", "log"))
    assert kind == "invalid"
    assert reason.startswith("run.json")
    # Refused with its reason: 409 once the registry has rejected the run, 422 before.
    for path in ("", "/plan", "/surface"):
        response = httpx.get(f"{server}/api/runs/{directory.name}{path}")
        assert response.status_code in (409, 422)
        assert response.json()["detail"]


def test_the_wall_surface_carries_cp_and_cf_from_the_wall_solution(runs: Path, server: str) -> None:
    import h5py  # noqa: PLC0415 -- only this test edits a CGNS file
    from autan.solver_input import freestream_state  # noqa: PLC0415

    directory = next(runs.iterdir())
    state = freestream_state(autan.Results.open(directory).case)
    q_inf = 0.5 * state.density * state.speed**2
    with h5py.File(directory / "solution.cgns", "r+") as tree:
        zone = tree["Base"]["Zone"]
        wall = zone["WallSolution"]
        wall["Pressure"][" data"][...] = state.pressure + q_inf
        wall["WallShearStressX"][" data"][...] = 0.6 * q_inf
        wall["WallShearStressY"][" data"][...] = 0.8 * q_inf
        wall["WallShearStressZ"][" data"][...] = 0.0
        start = zone["NGonElements"]["ElementStartOffset"][" data"][()]
        walls = {
            name
            for name, bc in zone["ZoneBC"].items()
            if bytes(bc[" data"][()]).decode().startswith("BCWall")
        }
        faces = {name: zone["ZoneBC"][name]["PointList"][" data"][()].ravel() - 1 for name in walls}

    message = httpx.get(f"{server}/api/runs/{directory.name}/surface").content
    length = int.from_bytes(message[:4], "little")
    header = json.loads(message[4 : 4 + length])
    body = memoryview(message)[4 + length :]
    field = {b["name"]: b for b in header["buffers"]}

    def values(name: str) -> npt.NDArray[np.float32]:
        b = field[name]
        return np.frombuffer(body, np.float32, b["length"], b["offset"])

    assert {z["name"] for z in header["zones"]} == walls
    for z in header["zones"]:
        counts = start[faces[z["name"]] + 1] - start[faces[z["name"]]]
        assert z["triangles"] == int((counts - 2).sum())
    np.testing.assert_allclose(values("field:Cp"), 1.0, rtol=1e-5)
    np.testing.assert_allclose(values("field:Cf"), 1.0, rtol=1e-5)


def test_an_inviscid_run_shows_no_skin_friction(tmp_path: Path) -> None:
    # Euler equations carry no wall shear: Cf is not a field of the run, not a field of zeros.
    source = (PY / "examples" / "crm_hl.py").read_text()
    (tmp_path / "case.py").write_text(
        source.replace("autan.SA()", "autan.Inviscid()").replace(".wall()", ".slip()")
    )
    mesh_info = autan.MeshInfo.load(PY / "tests" / "fixtures" / "crm-hl.meshinfo.json")
    run = autan.load(tmp_path / "case.py").run(
        mesh_info, backend=autan.FakeBackend(iterations=5), runs=tmp_path / "runs"
    )

    with serving(create_app(tmp_path / "runs", static=None)) as url:
        message = httpx.get(f"{url}/api/runs/{run.path.name}/surface").content

    header = json.loads(message[4 : 4 + int.from_bytes(message[:4], "little")])
    assert [f["name"] for f in header["fields"]] == ["Cp", "Pressure"]


def test_a_continued_run_measures_its_drop_from_the_original_first_line(runs: Path) -> None:
    case = autan.load(PY / "examples" / "crm_hl.py")
    mesh_info = autan.MeshInfo.load(PY / "tests" / "fixtures" / "crm-hl.meshinfo.json")
    first = autan.Results.open(next(runs.iterdir()))
    continued = case.run(
        mesh_info, backend=autan.FakeBackend(iterations=60), runs=runs, start=first
    )
    run = json.loads((continued.path / "run.json").read_text())
    reference = run["start"]["residual_reference"]
    last = json.loads((continued.path / "history.jsonl").read_text().splitlines()[-1])["residuals"]

    summary = Registry(runs).scan().summaries[continued.path.name]

    expected = min(math.log10(reference[k] / last[k]) for k in reference)
    assert summary["orders"] == pytest.approx(expected, abs=1e-3)
    assert summary["trend"][0] > 0  # the first line of the continuation is already below
