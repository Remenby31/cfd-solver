"""Generate the C++ files of one kernel module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autan_codegen import model
from autan_codegen.derive import derive
from autan_codegen.emit_cpp import emit_common_header
from autan_codegen.emit_cpu import emit_cpu_header
from autan_codegen.emit_gpu import POLICIES, emit_gpu_header, emit_gpu_instances
from autan_codegen.emit_metal import POLICIES as METAL_POLICIES
from autan_codegen.emit_metal import emit_metal_header
from autan_codegen.errors import Code, CodegenError

if TYPE_CHECKING:
    from pathlib import Path

BACKENDS = ("cpu", "gpu", "metal")


def generate(
    path: Path,
    out: Path,
    backends: list[str],
    line_directives: bool = True,
    policies: list[str] | None = None,
) -> list[Path]:
    """Write `<module>.hpp` and the files of each backend into `out`; return the paths written.

    For the GPU backend, `<module>_gpu.cu` instantiates every loop for `policies` (default: all);
    the Metal backend instantiates the policies Metal has among them, and refuses an explicit
    `fp64`, which Metal cannot run.
    """
    for b in backends:
        if b not in BACKENDS:
            raise CodegenError(Code.UNKNOWN_BACKEND, f"unknown backend {b!r}")
    explicit = policies is not None
    policies = list(POLICIES) if policies is None else policies
    for pol in policies:
        if pol not in POLICIES:
            raise CodegenError(Code.UNKNOWN_POLICY, f"unknown precision policy {pol!r}")
        if explicit and "metal" in backends and pol not in METAL_POLICIES:
            raise CodegenError(
                Code.POLICY_UNSUPPORTED_BY_BACKEND,
                f"Metal has no FP64: policy {pol!r} is not available on the metal backend",
            )
    m = model.load(path.resolve())
    derive(m)
    files = {f"{m.name}.hpp": emit_common_header(m, line_directives)}
    if "cpu" in backends:
        files[f"{m.name}_cpu.hpp"] = emit_cpu_header(m)
    if "gpu" in backends:
        files[f"{m.name}_gpu.cuh"] = emit_gpu_header(m)
        files[f"{m.name}_gpu.cu"] = emit_gpu_instances(m, policies)
    if "metal" in backends:
        files[f"{m.name}_metal.hpp"] = emit_metal_header(
            m, [p for p in policies if p in METAL_POLICIES], line_directives
        )
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in files.items():
        p = out / name
        # Unchanged files keep their timestamp so that the build does not recompile them.
        if not p.exists() or p.read_text() != text:
            p.write_text(text)
        written.append(p)
    return written
