import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WRAPPER = ROOT / "lock" / "autan-gpu-lock"
# The default memory floor (24 GiB) is a Spark's; a test host may have less memory in all.
NO_FLOOR = ["-f", "0G"]
PRIORITY = ROOT / "ci" / "priority.sh"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="flock(1) and /proc are Linux's")


@pytest.fixture
def spark(tmp_path: Path) -> dict[str, str]:
    """A lock directory as deploy.sh prepares it, and the environment of an agent outside CI."""
    for name in ("gpu.lock", "ci.lock", "main.lock", "holders.log"):
        (tmp_path / name).touch()
    env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_ACTIONS", "GITHUB_REF")}
    env |= {"AUTAN_GPU_LOCK": str(tmp_path / "gpu.lock"), "RUNNER_TEMP": str(tmp_path)}
    return env


def agent(env: dict[str, str], line: str, order: Path, *args: str) -> subprocess.Popen:
    cmd = [
        str(WRAPPER),
        *NO_FLOOR,
        "-m",
        "agent",
        *args,
        "--",
        "sh",
        "-c",
        f"echo {line} >> {order}",
    ]
    return subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE)


def ci_step(env: dict[str, str], order: Path) -> subprocess.Popen:
    """A CI GPU step: it takes the GPU lock with flock and never yields."""
    lock = env["AUTAN_GPU_LOCK"]
    return subprocess.Popen(["flock", lock, "sh", "-c", f"echo ci >> {order}; sleep 0.3"])


def priority(env: dict[str, str], *args: str) -> None:
    subprocess.run([str(PRIORITY), *args], env=env, check=True, timeout=30)


def events(env: dict[str, str]) -> list[str]:
    log = Path(env["AUTAN_GPU_LOCK"]).parent / "holders.log"
    return [line.split("\t")[1] for line in log.read_text().splitlines()]


def lines(order: Path) -> list[str]:
    return order.read_text().split() if order.exists() else []


def test_an_agent_waiting_with_ci_goes_after_ci(spark, tmp_path):
    order = tmp_path / "order"
    # Someone holds the GPU; CI announces itself and queues; an agent queues after it.
    hold = ["--", "sh", "-c", f"echo first >> {order}; sleep 1"]
    first = subprocess.Popen([str(WRAPPER), *NO_FLOOR, "-m", "first", *hold], env=spark)
    time.sleep(0.3)
    priority(spark, "start", "5")
    ci = ci_step(spark, order)
    second = agent(spark, "second", order)
    assert ci.wait(timeout=20) == 0
    time.sleep(0.5)
    assert lines(order) == ["first", "ci"]
    assert second.poll() is None
    priority(spark, "stop")
    assert second.wait(timeout=20) == 0
    assert first.wait(timeout=20) == 0
    assert lines(order) == ["first", "ci", "second"]
    assert "yield" in events(spark)


def test_an_agent_never_waits_for_ci_beyond_its_limit(spark, tmp_path):
    priority(spark, "start", "5")
    try:
        start = time.monotonic()
        proc = agent(spark, "late", tmp_path / "order", "-w", "1")
        assert proc.wait(timeout=20) == 75
        assert time.monotonic() - start < 5
        assert events(spark)[-2:] == ["yield", "timeout"]
    finally:
        priority(spark, "stop")


def test_ci_itself_never_yields(spark, tmp_path):
    priority(spark, "start", "5")
    try:
        order = tmp_path / "order"
        proc = agent({**spark, "GITHUB_ACTIONS": "true"}, "ci", order, "-w", "5")
        assert proc.wait(timeout=20) == 0
        assert lines(order) == ["ci"]
    finally:
        priority(spark, "stop")


def test_a_self_locking_command_waits_for_ci_then_takes_the_lock_itself(spark, tmp_path):
    # -n: for autan-bench, which takes the lock itself; nesting it under a held lock would
    # deadlock. The wrapper only waits for CI and for a free lock.
    order = tmp_path / "order"
    lock = spark["AUTAN_GPU_LOCK"]
    priority(spark, "start", "5")
    take = f"flock -n {lock} sh -c 'echo agent >> {order}'"
    proc = subprocess.Popen([str(WRAPPER), *NO_FLOOR, "-n", "-m", "agent", "--", "sh", "-c", take],
                            env=spark)  # fmt: skip
    time.sleep(1)
    assert proc.poll() is None
    ci = ci_step(spark, order)
    assert ci.wait(timeout=20) == 0
    priority(spark, "stop")
    assert proc.wait(timeout=20) == 0
    assert lines(order) == ["ci", "agent"]
    assert "start" in events(spark)


GUARD = ROOT / "lock" / "autan-gpu-guard"


def test_the_guard_records_every_holder_and_never_announces_ci_for_others(spark):
    lock = spark["AUTAN_GPU_LOCK"]
    guard = subprocess.Popen([sys.executable, str(GUARD)], env=spark, stderr=subprocess.PIPE)
    try:
        time.sleep(1)
        subprocess.run(["flock", lock, "sleep", "1.5"], check=True)
        time.sleep(1.5)
        held = [e for e in events(spark) if not e.startswith("foreign")]
        assert held[:2] == ["held", "released"]
        # Not the runner container: ci.lock stays free.
        assert (
            subprocess.run(
                ["flock", "-n", "-x", str(Path(lock).parent / "ci.lock"), "true"], check=False
            ).returncode
            == 0
        )
    finally:
        guard.terminate()
        guard.wait(timeout=10)


def test_the_guard_reads_holders_and_waiters_from_proc_locks():
    loader = importlib.machinery.SourceFileLoader("guard", str(GUARD))
    spec = importlib.util.spec_from_loader("guard", loader)
    guard = importlib.util.module_from_spec(spec)
    loader.exec_module(guard)
    key = "103:02:6985512"
    text = (
        f"22: FLOCK  ADVISORY  WRITE 4242 {key} 0 EOF\n"
        f"22: -> FLOCK  ADVISORY  WRITE 4343 {key} 0 EOF\n"
        "23: POSIX  ADVISORY  WRITE 99 103:02:1 0 EOF\n"
    )
    assert guard.lock_users(key, text) == ({4242}, {4343})


def test_a_main_job_goes_before_a_pull_request_s(spark, tmp_path):
    # The main job's priority.sh holds main.lock; a pull request's CI job (here the wrapper with
    # GITHUB_ACTIONS and a pull request ref) waits for it, then runs.
    order = tmp_path / "order"
    main_env = {**spark, "GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/main"}
    pr_env = {**spark, "GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/pull/7/merge"}
    priority(main_env, "start", "5")
    pr = agent(pr_env, "pr", order)
    time.sleep(1)
    assert pr.poll() is None
    main = agent(main_env, "main", order)
    assert main.wait(timeout=20) == 0
    priority(main_env, "stop")
    assert pr.wait(timeout=20) == 0
    assert lines(order) == ["main", "pr"]


def test_the_guard_names_the_gpu_processes_outside_autan_containers(tmp_path, monkeypatch):
    loader = importlib.machinery.SourceFileLoader("guard", str(GUARD))
    spec = importlib.util.spec_from_loader("guard", loader)
    guard = importlib.util.module_from_spec(spec)
    loader.exec_module(guard)
    # nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
    # as it printed on spark-5512 (a process name may contain commas), and a hostile name.
    csv = (
        "3285079, VLLM::EngineCore, 22484\n3292695, /w/build/autan_halo_tests, 170\n7, a, b, 3\n"
        "9, ##[stop-commands]tok\t::error::x, 1\n"
    )
    assert guard.gpu_apps(csv)[7] == ("a, b", "3")
    # The external boundaries: nvidia-smi, and the cgroups that place each process.
    fake = tmp_path / "nvidia-smi"
    fake.write_text(f"#!/bin/sh\nprintf '{csv}'\n")
    fake.chmod(0o755)
    monkeypatch.setattr(guard, "GPU_APPS", [str(fake)])
    monkeypatch.setattr(guard, "FOREIGN", tmp_path / "foreign-gpu")
    ours, theirs = "a" * 64, "b" * 64
    places = {3285079: theirs, 3292695: ours, 7: None, 9: theirs}
    monkeypatch.setattr(guard, "container_of", places.get)
    containers = guard.Containers()
    containers.names = {ours: "autan-runner", theirs: "mcpbench-vllm"}
    foreign = guard.foreign_apps(containers)
    assert {pid: f["where"] for pid, f in foreign.items()} == {
        3285079: "mcpbench-vllm",
        7: "host",
        9: "mcpbench-vllm",
    }
    env = {**os.environ, "AUTAN_GPU_LOCK": str(tmp_path / "gpu.lock")}

    def report() -> list[str]:
        return subprocess.run([str(ROOT / "ci" / "gpu-tenants.sh")], env=env, capture_output=True,
                              text=True, check=True).stdout.splitlines()  # fmt: skip

    guard.publish(foreign)
    out = report()
    (error,) = [line for line in out if line.startswith("::error::")]
    assert "VLLM::EngineCore (mcpbench-vllm, pid 3285079, 22484 MiB)" in error
    assert "autan_halo_tests" not in error
    assert not [line for line in out if line.startswith(("::", "##")) and line != error]
    assert not any("#" in line or "[" in line for line in out[1:])
    guard.publish({})
    assert not [line for line in report() if line.startswith("::")]
    guard.publish(None)
    assert [line.split("::")[1] for line in report() if line.startswith("::")] == ["warning"]
    (tmp_path / "foreign-gpu").unlink()
    assert [line.split("::")[1] for line in report() if line.startswith("::")] == ["warning"]
