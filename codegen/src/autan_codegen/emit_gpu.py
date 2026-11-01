"""GPU loops for CUDA and HIP from one text: `<module>_gpu.cuh`.

Face and boundary loops are emitted in every race strategy of research section 6 that the spike
implements (atomic, warp-aggregated, coloured, store-gather, gather-recompute); cell loops and cell
reductions run one thread per cell and need none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from autan_codegen import dsl
from autan_codegen.derive import ReduceAdjoint
from autan_codegen.emit_cpp import Emitter, at, call_args, incidence_loop
from autan_codegen.emit_cpu import (
    boundary_write_body,
    cell_stores,
    face_ends,
    global_left,
    jacobian_lines,
    n_out,
    offsets,
    other_targets,
    reduce_adjoint_body,
    reduce_body,
    twin_gather_body,
)

if TYPE_CHECKING:
    from autan_codegen.model import LoopModel, Module

STRATEGIES = ("atomic", "warp_aggregated", "coloured", "store_gather", "gather_recompute")


def _range(lm: LoopModel) -> tuple[str, str]:
    if lm.kind == "face":
        return "0", "d.topo.n_owned_interior_faces"
    return "d.topo.n_interior_faces", "d.topo.n_owned_faces"


def _face_eval(lm: LoopModel, indent: str, jac: str | None = "true") -> list[str]:
    """Loads l (and r), evaluates the body of face f into `out`, and writes the face's Jacobian
    outputs when `jac` (a C++ condition; None: never) holds."""
    lines = face_ends(lm.kind, indent)
    args = call_args(
        lm.sig, lm.loop.bindings, {"left": "l", "right": "r", "direct": "f"},
        reversible=lm.kind == "face",
    )  # fmt: skip
    lines.append(f"{indent}{lm.fn.name}<P>({', '.join([*args, 'out'])});")
    if lm.jac_fns and jac is not None:
        if jac == "true":
            lines += jacobian_lines(lm, args, "f", indent)
        else:
            lines.append(f"{indent}if ({jac}) {{")
            lines += jacobian_lines(lm, args, "f", indent + "  ")
            lines.append(f"{indent}}}")
    return lines


def _incs(lm: LoopModel, side: str, index: str, indent: str, mode: str) -> list[str]:
    """Increments of the `side` cell; mode: atomic | plain."""
    lines = []
    for o, off in zip(lm.loop.outputs, offsets(lm.loop.outputs), strict=True):
        assert isinstance(o, dsl.Inc)
        sign = o.left if side == "left" else o.right
        if not sign:
            continue
        for c in range(o.dataset.type.size):
            v = _signed(sign, f"A(out[{off + c}])")
            target = at(o.dataset, index, c)
            if mode == "atomic":
                lines.append(f"{indent}atomicAdd(&{target}, {v});")
            else:
                lines.append(f"{indent}{target} += {v};")
    return lines


def _signed(sign: int | None, v: str) -> str:
    if not sign:
        return "A(0)"
    if sign == 1:
        return v
    if sign == -1:
        return f"-{v}"
    return f"A({sign}) * {v}"


def _kernel_head(e: Emitter, name: str, extra: str = "") -> None:
    e.w("template <class P>")
    e.w(f"__global__ void {name}(const Data<P> d{extra}) {{")
    e.w("  using A [[maybe_unused]] = typename P::accum;")


def emit_face_kernels(e: Emitter, lm: LoopModel) -> None:
    n = n_out(lm)
    tid = "static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x)"
    # atomic
    _kernel_head(e, f"{lm.name}_atomic", ", const index_t lo, const index_t hi")
    e.w(f"  const index_t f = lo + {tid};")
    e.w("  if (f >= hi) {")
    e.w("    return;")
    e.w("  }")
    e.w(f"  typename P::compute out[{n}];")
    for s in _face_eval(lm, "  "):
        e.w(s)
    for s in _incs(lm, "left", "l", "  ", "atomic"):
        e.w(s)
    if lm.kind == "face":
        for s in _incs(lm, "right", "r", "  ", "atomic"):
            e.w(s)
    e.w("}")
    e.w()
    # warp-aggregated: owner segments of the warp reduced by shuffles, one atomic per segment.
    _kernel_head(e, f"{lm.name}_warp_aggregated", ", const index_t lo, const index_t hi")
    e.w(f"  const index_t f = lo + {tid};")
    e.w("  const bool active = f < hi;")
    e.w(f"  typename P::compute out[{n}] = {{}};")
    e.w("  index_t key = -1;")
    e.w("  if (active) {")
    for s in _face_eval(lm, "    "):
        e.w(s)
    e.w("    key = l;")
    if lm.kind == "face":
        for s in _incs(lm, "right", "r", "    ", "atomic"):
            e.w(s)
    e.w("  }")
    e.w("  const int lane = static_cast<int>(threadIdx.x) % gpu::warp_size;")
    e.w("  const index_t prev = gpu::shfl_up(key, 1);")
    e.w("  const bool head = lane == 0 || prev != key;")
    e.w("  const int seg = gpu::segment_id(head);")
    for o, off in zip(lm.loop.outputs, offsets(lm.loop.outputs), strict=True):
        assert isinstance(o, dsl.Inc)
        if not o.left:
            continue
        for c in range(o.dataset.type.size):
            k = off + c
            e.w(f"  A v{k} = {_signed(o.left, f'A(out[{k}])')};")
            e.w("  for (int o = 1; o < gpu::warp_size; o <<= 1) {")
            e.w(f"    const A w = gpu::shfl_down(v{k}, o);")
            e.w("    const int s = gpu::shfl_down(seg, o);")
            e.w(f"    v{k} += (lane + o < gpu::warp_size && s == seg) ? w : A(0);")
            e.w("  }")
            e.w(f"  if (head && key >= 0) {{ atomicAdd(&{at(o.dataset, 'key', c)}, v{k}); }}")
    e.w("}")
    e.w()
    # coloured: faces of one colour touch distinct cells.
    _kernel_head(
        e, f"{lm.name}_coloured", ", const index_t* __restrict__ faces, const index_t count"
    )
    e.w(f"  const index_t i = {tid};")
    e.w("  if (i >= count) {")
    e.w("    return;")
    e.w("  }")
    e.w("  const index_t f = faces[i];")
    e.w(f"  typename P::compute out[{n}];")
    for s in _face_eval(lm, "  "):
        e.w(s)
    for s in _incs(lm, "left", "l", "  ", "plain"):
        e.w(s)
    if lm.kind == "face":
        for s in _incs(lm, "right", "r", "  ", "plain"):
            e.w(s)
    e.w("}")
    e.w()
    # store-gather, pass 1: face values to scratch.
    _kernel_head(
        e,
        f"{lm.name}_store",
        ", const index_t lo, const index_t hi, typename P::compute* __restrict__ scratch",
    )
    e.w(f"  const index_t f = lo + {tid};")
    e.w("  if (f >= hi) {")
    e.w("    return;")
    e.w("  }")
    e.w(f"  typename P::compute out[{n}];")
    for s in _face_eval(lm, "  "):
        e.w(s)
    for k in range(n):
        e.w(
            f"  scratch[offset_t({k}) * d.topo.n_faces + f] = out[{k}];"
            if k
            else "  scratch[f] = out[0];"
        )
    e.w("}")
    e.w()
    # store-gather pass 2 and gather-recompute: one thread per cell, incidence order. Ghost cells
    # too: their sums are the partial contributions a reverse halo exchange sends to their owner.
    for variant in ("gather", "gather_recompute"):
        extra = ", const typename P::compute* __restrict__ scratch" if variant == "gather" else ""
        _kernel_head(e, f"{lm.name}_{variant}", extra)
        e.w(f"  const index_t c = {tid};")
        e.w("  if (c >= d.topo.n_cells) {")
        e.w("    return;")
        e.w("  }")
        e.w(f"  A acc[{n}] = {{}};")
        e.w(incidence_loop("  "))
        e.w("    const index_t f = d.topo.cell_faces[k];")
        # Faces outside the loop's range (halo faces, other kind) were not evaluated.
        cond = (
            "f >= d.topo.n_owned_interior_faces"
            if lm.kind == "face"
            else "f < d.topo.n_interior_faces || f >= d.topo.n_owned_faces"
        )
        e.w(f"    if ({cond}) {{")
        e.w("      continue;")
        e.w("    }")
        if variant == "gather":
            e.w(f"    A out[{n}];")
            for k in range(n):
                e.w(
                    f"    out[{k}] = scratch[offset_t({k}) * d.topo.n_faces + f];"
                    if k
                    else "    out[0] = scratch[f];"
                )
            if lm.kind == "face":
                e.w(f"    const bool left = {global_left()} == c;")
        else:
            e.w(f"    typename P::compute out[{n}];")
            for s in _face_eval(lm, "    ", "l == c"):
                e.w(s)
            if lm.kind == "face":
                e.w("    const bool left = l == c;")
        for o, off in zip(lm.loop.outputs, offsets(lm.loop.outputs), strict=True):
            assert isinstance(o, dsl.Inc)
            for c in range(o.dataset.type.size):
                k = off + c
                lv = _signed(o.left, f"A(out[{k}])")
                if lm.kind == "face":
                    rv = _signed(o.right, f"A(out[{k}])")
                    e.w(f"    acc[{k}] += left ? {lv} : {rv};")
                else:
                    e.w(f"    acc[{k}] += {lv};")
        e.w("  }")
        for o, off in zip(lm.loop.outputs, offsets(lm.loop.outputs), strict=True):
            for c in range(o.dataset.type.size):
                e.w(f"  {at(o.dataset, 'c', c)} += acc[{off + c}];")
        e.w("}")
        e.w()


def emit_face_launcher(e: Emitter, lm: LoopModel) -> None:
    lo, hi = _range(lm)
    col = "interior" if lm.kind == "face" else "boundary"
    e.w("template <class P>")
    e.w(
        f"void {lm.name}(const Data<P>& d, const gpu::Strategy strategy, const gpu::Schedule& s) {{"
    )
    e.w("  if (d.topo.rank_independent && strategy != ::autan::kernels::Strategy::store_gather &&")
    e.w("      strategy != ::autan::kernels::Strategy::gather_recompute) {")
    e.w(
        '    throw std::invalid_argument("E1109: a rank-independent topology needs a gather '
        'strategy (store-gather or gather-recompute)");'
    )
    e.w("  }")
    e.w(f"  const index_t lo = {lo};")
    e.w(f"  const index_t hi = {hi};")
    e.w(
        "  const auto grid = [&](index_t n) { return static_cast<unsigned>((n + s.block - 1) / s.block); };"
    )
    e.w("  const auto block = static_cast<unsigned>(s.block);")
    e.w("  switch (strategy) {")
    e.w("    case gpu::Strategy::atomic:")
    e.w("      if (hi > lo) {")
    e.w(f"        {lm.name}_atomic<P><<<grid(hi - lo), block, 0, s.stream>>>(d, lo, hi);")
    e.w("      }")
    e.w("      break;")
    e.w("    case gpu::Strategy::warp_aggregated:")
    e.w("      if (hi > lo) {")
    e.w(f"        {lm.name}_warp_aggregated<P><<<grid(hi - lo), block, 0, s.stream>>>(d, lo, hi);")
    e.w("      }")
    e.w("      break;")
    e.w("    case gpu::Strategy::coloured:")
    e.w(f"      for (std::size_t k = 0; k + 1 < s.{col}.offsets.size(); ++k) {{")
    e.w(f"        const index_t first = s.{col}.offsets[k];")
    e.w(f"        const index_t count = s.{col}.offsets[k + 1] - first;")
    e.w(
        f"        {lm.name}_coloured<P><<<grid(count), block, 0, s.stream>>>(d, s.{col}.faces + first, count);"
    )
    e.w("      }")
    e.w("      break;")
    e.w("    case gpu::Strategy::store_gather: {")
    e.w(f"      if (s.face_scratch_width < {n_out(lm)}) {{")
    e.w(
        '        throw std::invalid_argument("E1104: face scratch narrower than the loop outputs");'
    )
    e.w("      }")
    e.w("      auto* scratch = static_cast<typename P::compute*>(s.face_scratch);")
    e.w("      if (hi > lo) {")
    e.w(f"        {lm.name}_store<P><<<grid(hi - lo), block, 0, s.stream>>>(d, lo, hi, scratch);")
    e.w("      }")
    e.w(f"      {lm.name}_gather<P><<<grid(d.topo.n_cells), block, 0, s.stream>>>(d, scratch);")
    e.w("      break;")
    e.w("    }")
    e.w("    case gpu::Strategy::gather_recompute:")
    e.w(f"      {lm.name}_gather_recompute<P><<<grid(d.topo.n_cells), block, 0, s.stream>>>(d);")
    e.w("      break;")
    e.w("  }")
    e.w(f'  gpu::check(gpu::last_error(), "{lm.name}");')
    e.w("}")
    e.w()


def emit_cell_kernel(e: Emitter, lm: LoopModel) -> None:
    tid = "static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x)"
    if lm.loop.listed:
        _emit_listed_kernel(e, lm, tid)
        return
    _kernel_head(e, f"{lm.name}_kernel")
    if lm.kind == "boundary_write":
        e.w(f"  const index_t f = d.topo.n_interior_faces + {tid};")
        e.w("  if (f >= d.topo.n_faces) {")
        e.w("    return;")
        e.w("  }")
        for s in boundary_write_body(lm, "  "):
            e.w(s)
        e.w("}")
        e.w()
        e.w("template <class P>")
        e.w(f"void {lm.name}(const Data<P>& d, const gpu::Schedule& s) {{")
        e.w("  const index_t n = d.topo.n_faces - d.topo.n_interior_faces;")
        e.w("  if (n > 0) {")
        e.w("    const auto grid = static_cast<unsigned>((n + s.block - 1) / s.block);")
        e.w(f"    {lm.name}_kernel<P><<<grid, static_cast<unsigned>(s.block), 0, s.stream>>>(d);")
        e.w("  }")
        e.w(f'  gpu::check(gpu::last_error(), "{lm.name}");')
        e.w("}")
        e.w()
        return
    e.w(f"  const index_t c = {tid};")
    e.w("  if (c >= d.topo.n_owned_cells) {")
    e.w("    return;")
    e.w("  }")
    if lm.kind == "cell":
        e.w(f"  typename P::{lm.fn.compute} out[{n_out(lm)}];")
        args = call_args(lm.sig, lm.loop.bindings, {"direct": "c"})
        e.w(f"  {lm.fn.name}<P>({', '.join([*args, 'out'])});")
        for s in jacobian_lines(lm, args, "c", "  "):
            e.w(s)
        for s in cell_stores(lm, "  "):
            e.w(s)
    elif lm.kind == "reduce_adjoint":
        for s in reduce_adjoint_body(lm, "  ", "atomic"):
            e.w(s)
    else:
        for s in reduce_body(lm, "  "):
            e.w(s)
    e.w("}")
    e.w()
    if lm.kind == "reduce_adjoint":
        emit_reduce_adjoint_deterministic(e, lm)
        return
    e.w("template <class P>")
    e.w(f"void {lm.name}(const Data<P>& d, const gpu::Schedule& s) {{")
    e.w("  if (d.topo.n_owned_cells > 0) {")
    e.w(
        "    const auto grid = static_cast<unsigned>((d.topo.n_owned_cells + s.block - 1) / s.block);"
    )
    e.w(f"    {lm.name}_kernel<P><<<grid, static_cast<unsigned>(s.block), 0, s.stream>>>(d);")
    e.w("  }")
    e.w(f'  gpu::check(gpu::last_error(), "{lm.name}");')
    e.w("}")
    e.w()


def _emit_listed_kernel(e: Emitter, lm: LoopModel, tid: str) -> None:
    """A reduction over the cells listed at launch (one thread each)."""
    _kernel_head(e, f"{lm.name}_kernel", ", const index_t* __restrict__ rows, const index_t n_rows")
    e.w(f"  const index_t i = {tid};")
    e.w("  if (i >= n_rows) {")
    e.w("    return;")
    e.w("  }")
    e.w("  const index_t c = rows[i];")
    for s in reduce_body(lm, "  "):
        e.w(s)
    e.w("}")
    e.w()
    e.w("template <class P>")
    e.w(
        f"void {lm.name}(const Data<P>& d, const index_t* rows, const index_t n_rows, "
        "const gpu::Schedule& s) {"
    )
    e.w("  if (n_rows > 0) {")
    e.w("    const auto grid = static_cast<unsigned>((n_rows + s.block - 1) / s.block);")
    e.w(
        f"    {lm.name}_kernel<P><<<grid, static_cast<unsigned>(s.block), 0, s.stream>>>"
        "(d, rows, n_rows);"
    )
    e.w("  }")
    e.w(f'  gpu::check(gpu::last_error(), "{lm.name}");')
    e.w("}")
    e.w()


def emit_reduce_adjoint_deterministic(e: Emitter, lm: LoopModel) -> None:
    """Two-pass adjoint of a cell reduction for the deterministic strategies (research section 6,
    store-then-gather with a transposed incidence list), and the launcher choosing between it and
    the atomic kernel."""
    tid = "static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x)"
    width = len(other_targets(lm))
    _kernel_head(e, f"{lm.name}_store", ", typename P::accum* __restrict__ scratch")
    e.w(f"  const index_t c = {tid};")
    e.w("  if (c >= d.topo.n_owned_cells) {")
    e.w("    return;")
    e.w("  }")
    for s in reduce_adjoint_body(lm, "  ", "store"):
        e.w(s)
    e.w("}")
    e.w()
    _kernel_head(e, f"{lm.name}_twin_gather", ", const typename P::accum* __restrict__ scratch")
    e.w(f"  const index_t c = {tid};")
    e.w("  if (c >= d.topo.n_cells) {")
    e.w("    return;")
    e.w("  }")
    for s in twin_gather_body(lm, "  "):
        e.w(s)
    e.w("}")
    e.w()
    e.w("template <class P>")
    e.w(
        f"void {lm.name}(const Data<P>& d, const gpu::Strategy strategy, const gpu::Schedule& s) {{"
    )
    e.w(
        "  const auto grid = [&](index_t n) { return static_cast<unsigned>((n + s.block - 1) / s.block); };"
    )
    e.w("  const auto block = static_cast<unsigned>(s.block);")
    e.w("  if (d.topo.n_owned_cells == 0) {")
    e.w("    return;")
    e.w("  }")
    e.w("  if (!gpu::deterministic(strategy)) {")
    e.w(f"    {lm.name}_kernel<P><<<grid(d.topo.n_owned_cells), block, 0, s.stream>>>(d);")
    e.w("  } else {")
    if width:
        e.w(f"    if (s.incidence_scratch_width < {width}) {{")
        e.w(
            '      throw std::invalid_argument("E1105: incidence scratch narrower than the adjoint");'
        )
        e.w("    }")
        assert isinstance(lm.radj, ReduceAdjoint)
        over = lm.radj.primal.loop.over
        twin = "stencil_twin" if over == dsl.STENCIL else "incidence_twin"
        e.w(f"    if (d.topo.{twin} == nullptr) {{")
        e.w(
            '      throw std::invalid_argument("E1106: deterministic reduction adjoint needs incidence twins");'
        )
        e.w("    }")
    e.w("    auto* scratch = static_cast<typename P::accum*>(s.incidence_scratch);")
    e.w(f"    {lm.name}_store<P><<<grid(d.topo.n_owned_cells), block, 0, s.stream>>>(d, scratch);")
    if width:
        e.w(
            f"    {lm.name}_twin_gather<P><<<grid(d.topo.n_cells), block, 0, s.stream>>>(d, scratch);"
        )
    e.w("  }")
    e.w(f'  gpu::check(gpu::last_error(), "{lm.name}");')
    e.w("}")
    e.w()


def emit_gpu_header(m: Module) -> str:
    e = Emitter(f"{m.name}_gpu.cuh", line_directives=False)
    e.w(f"// Generated by autan_codegen from {m.path.name}. Do not edit.")
    e.w("#pragma once")
    e.w("#include <cstddef>")
    e.w()
    e.w('#include "autan/kernels/gpu.hpp"')
    e.w(f'#include "{m.name}.hpp"')
    e.w()
    e.w(f"namespace autan::kernels::{m.name}::gpu_loops {{")
    e.w("namespace gpu = ::autan::kernels::gpu;")
    e.w("using ::autan::kernels::index_t;")
    e.w("using ::autan::kernels::offset_t;")
    e.w()
    for lm in m.loops:
        if lm.kind in ("face", "boundary"):
            emit_face_kernels(e, lm)
            emit_face_launcher(e, lm)
        else:
            emit_cell_kernel(e, lm)
    e.w(f"}}  // namespace autan::kernels::{m.name}::gpu_loops")
    return e.text()


POLICIES = {"fp64": "fp64", "fp32+acc64": "fp32_acc64", "fp32": "fp32"}


def emit_gpu_instances(m: Module, policies: list[str]) -> str:
    """`<module>_gpu.cu`: explicit instantiations of every loop for the policies a build lists."""
    e = Emitter(f"{m.name}_gpu.cu", line_directives=False)
    e.w(f"// Generated by autan_codegen from {m.path.name}. Do not edit.")
    e.w(f'#include "{m.name}_gpu.cuh"')
    e.w()
    e.w(f"namespace autan::kernels::{m.name}::gpu_loops {{")
    for p in policies:
        pol = f"::autan::kernels::{POLICIES[p]}"
        for lm in m.loops:
            if lm.kind in ("face", "boundary", "reduce_adjoint"):
                e.w(
                    f"template void {lm.name}<{pol}>(const Data<{pol}>&, gpu::Strategy, const gpu::Schedule&);"
                )
            elif lm.loop.listed:
                e.w(
                    f"template void {lm.name}<{pol}>(const Data<{pol}>&, const index_t*, index_t, const gpu::Schedule&);"
                )
            else:
                e.w(f"template void {lm.name}<{pol}>(const Data<{pol}>&, const gpu::Schedule&);")
    e.w(f"}}  // namespace autan::kernels::{m.name}::gpu_loops")
    return e.text()
