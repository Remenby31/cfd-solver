"""Metal loops for Apple GPUs: `<module>_metal.hpp`.

The header holds the module's Metal Shading Language source (prelude, data struct, bodies,
kernels, one instantiation per kernel and policy) and the host launchers that dispatch it through
`autan/kernels/metal.hpp`. Kernels read `Data<P>` from a constant buffer whose pointers are GPU
addresses, so loop bodies are the same text as the CUDA and HIP loops. Metal has no FP64: the
`fp32+acc64` policy accumulates in double-float, `fp64` is refused, and the strategies that need
floating-point atomics on the accumulation type (atomic, warp-aggregated, the atomic scatter of a
cell reduction adjoint) exist in `fp32` only; the deterministic strategies run in both policies.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from autan_codegen import dsl
from autan_codegen.derive import ReduceAdjoint
from autan_codegen.emit_cpp import (
    Emitter,
    arg_name,
    at,
    body_lines,
    call_args,
    ctype,
    incidence_loop,
    rel_datasets,
    storage_type,
    unique_bodies,
)
from autan_codegen.emit_cpu import (
    boundary_write_body,
    cell_stores,
    global_left,
    n_out,
    offsets,
    other_targets,
    reduce_adjoint_body,
    reduce_body,
    twin_gather_body,
)
from autan_codegen.emit_gpu import _face_eval, _incs, _signed, jacobian_lines
from autan_codegen.errors import Code, CodegenError

if TYPE_CHECKING:
    from autan_codegen.ir import Function
    from autan_codegen.model import LoopModel, Module

# Policy name -> (MSL and host policy type, has float atomics on its accumulation type).
POLICIES = {"fp32": ("fp32", True), "fp32+acc64": ("fp32_acc64", False)}
PRELUDE = Path(__file__).with_name("metal_prelude.metal")
DELIMITER = "autan_msl"
CHUNK = 60000


def _msl_function(e: Emitter, fn: Function, sig_types: list[dsl.Type]) -> None:
    params = [
        f"[[maybe_unused]] const {ctype(t.role)} {arg_name(k, c)}"
        for k, t in enumerate(sig_types)
        for c in range(t.size)
    ]
    params.append("thread typename P::compute* out")
    e.w("template <class P>")
    e.w(f"inline void {fn.name}({', '.join(params)}) {{")
    e.w("  using C = typename P::compute;")
    body_lines(e, fn, indent="  ")
    for k, o in enumerate(fn.outputs):
        e.w(f"  out[{k}] = v{o};")
    e.restore_line()
    e.w("}")
    e.w()


def _msl_data_struct(e: Emitter, m: Module) -> None:
    e.w("template <class P>")
    e.w("struct Data {")
    e.w("  Topology topo;")
    rel_read = rel_datasets(m)
    for d in m.datasets:
        const = "const " if d.geometry else ""
        e.w(f"  device {const}{storage_type(d, rel_read)}* {d.name};")
    for p in m.params:
        e.w(f"  {ctype(p.type.role)} {p.name}[{p.type.size}];")
    e.w("};")
    e.w()


class _Kernels:
    """Kernel definitions and, per policy, their explicit instantiations."""

    def __init__(self, e: Emitter) -> None:
        self.e = e
        self.instances: list[tuple[str, list[str], bool]] = []

    def head(self, name: str, args: str | None, lane: bool = False, atomic: bool = False) -> None:
        params = ["constant Data<P>& d [[buffer(0)]]"]
        types = ["constant Data<P>&"]
        if args is not None:
            params.append(f"constant {args}& a [[buffer(1)]]")
            types.append(f"constant {args}&")
        params.append("uint gid [[thread_position_in_grid]]")
        types.append("uint")
        if lane:
            params.append("uint lane_id [[thread_index_in_simdgroup]]")
            types.append("uint")
        self.instances.append((name, types, atomic))
        self.e.w("template <class P>")
        self.e.w(f"kernel void {name}({', '.join(params)}) {{")
        self.e.w("  using A [[maybe_unused]] = typename P::accum;")

    def instantiate(self, policies: list[str]) -> None:
        for p in policies:
            pol, atomics = POLICIES[p]
            for name, types, atomic in self.instances:
                if atomic and not atomics:
                    continue
                ts = ", ".join(
                    t.replace("<P>", f"<{pol}>").replace("P::", f"{pol}::") for t in types
                )
                self.e.w(
                    f'template [[host_name("{name}_{pol}")]] kernel void {name}<{pol}>({ts.replace("typename ", "")});'
                )


def _face_kernels(k: _Kernels, lm: LoopModel) -> None:
    e = k.e
    n = n_out(lm)
    # atomic
    k.head(f"{lm.name}_atomic", "RangeArgs", atomic=True)
    e.w("  const index_t f = a.lo + index_t(gid);")
    e.w("  if (f >= a.hi) {")
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
    # warp-aggregated: owner segments of a SIMD group reduced by shuffles, one atomic each.
    k.head(f"{lm.name}_warp_aggregated", "RangeArgs", lane=True, atomic=True)
    e.w("  const index_t f = a.lo + index_t(gid);")
    e.w("  const bool active = f < a.hi;")
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
    e.w("  const int lane = int(lane_id);")
    e.w("  const index_t prev = metal::simd_shuffle_up(key, 1);")
    e.w("  const bool head = lane == 0 || prev != key;")
    e.w("  const int seg = segment_id(head, lane);")
    for o, off in zip(lm.loop.outputs, offsets(lm.loop.outputs), strict=True):
        assert isinstance(o, dsl.Inc)
        if not o.left:
            continue
        for c in range(o.dataset.type.size):
            j = off + c
            e.w(f"  A v{j} = {_signed(o.left, f'A(out[{j}])')};")
            e.w("  for (ushort o = 1; o < warp_size; o <<= 1) {")
            e.w(f"    const A w = metal::simd_shuffle_down(v{j}, o);")
            e.w("    const int s = metal::simd_shuffle_down(seg, o);")
            e.w(f"    v{j} += (lane + o < warp_size && s == seg) ? w : A(0);")
            e.w("  }")
            e.w(f"  if (head && key >= 0) {{ atomicAdd(&{at(o.dataset, 'key', c)}, v{j}); }}")
    e.w("}")
    e.w()
    # coloured: faces of one colour touch distinct cells.
    k.head(f"{lm.name}_coloured", "ColourArgs")
    e.w("  const index_t i = index_t(gid);")
    e.w("  if (i >= a.count) {")
    e.w("    return;")
    e.w("  }")
    e.w("  const index_t f = a.faces[i];")
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
    # store-gather, pass 1: face values to scratch, in the compute type they are evaluated in
    # (exact: accumulating them later in double-float gives the same sums, at half the traffic).
    k.head(f"{lm.name}_store", "StoreArgs<typename P::compute>")
    e.w("  const index_t f = a.lo + index_t(gid);")
    e.w("  if (f >= a.hi) {")
    e.w("    return;")
    e.w("  }")
    e.w(f"  typename P::compute out[{n}];")
    for s in _face_eval(lm, "  "):
        e.w(s)
    for j in range(n):
        e.w(f"  a.scratch[offset_t({j}) * d.topo.n_faces + f] = out[{j}];")
    e.w("}")
    e.w()
    # store-gather pass 2 and gather-recompute: one thread per cell (ghosts too), CSR order.
    for variant in ("gather", "gather_recompute"):
        k.head(
            f"{lm.name}_{variant}",
            "GatherArgs<typename P::compute>" if variant == "gather" else None,
        )
        e.w("  const index_t c = index_t(gid);")
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
            for j in range(n):
                e.w(f"    out[{j}] = A(a.scratch[offset_t({j}) * d.topo.n_faces + f]);")
            if lm.kind == "face":
                e.w(f"    const bool left = {global_left()} == c;")
        else:
            e.w(f"    typename P::compute out[{n}];")
            for s in _face_eval(lm, "    "):
                e.w(s)
            if lm.kind == "face":
                e.w("    const bool left = l == c;")
        for o, off in zip(lm.loop.outputs, offsets(lm.loop.outputs), strict=True):
            assert isinstance(o, dsl.Inc)
            for c in range(o.dataset.type.size):
                j = off + c
                lv = _signed(o.left, f"A(out[{j}])")
                if lm.kind == "face":
                    rv = _signed(o.right, f"A(out[{j}])")
                    e.w(f"    acc[{j}] += left ? {lv} : {rv};")
                else:
                    e.w(f"    acc[{j}] += {lv};")
        e.w("  }")
        for o, off in zip(lm.loop.outputs, offsets(lm.loop.outputs), strict=True):
            for c in range(o.dataset.type.size):
                e.w(f"  {at(o.dataset, 'c', c)} += acc[{off + c}];")
        e.w("}")
        e.w()


def _cell_kernel(k: _Kernels, lm: LoopModel) -> None:
    e = k.e
    k.head(f"{lm.name}_kernel", None, atomic=lm.kind == "reduce_adjoint")
    if lm.kind == "boundary_write":
        e.w("  const index_t f = d.topo.n_interior_faces + index_t(gid);")
        e.w("  if (f >= d.topo.n_faces) {")
        e.w("    return;")
        e.w("  }")
        for s in boundary_write_body(lm, "  "):
            e.w(s)
        e.w("}")
        e.w()
        return
    e.w("  const index_t c = index_t(gid);")
    e.w("  if (c >= d.topo.n_owned_cells) {")
    e.w("    return;")
    e.w("  }")
    if lm.kind == "cell":
        e.w(f"  typename P::compute out[{n_out(lm)}];")
        args = call_args(lm.sig, lm.loop.bindings, {"direct": "c"})
        e.w(f"  {lm.fn.name}<P>({', '.join([*args, 'out'])});")
        lines = [*jacobian_lines(lm, args, "c", "  "), *cell_stores(lm, "  ")]
    elif lm.kind == "reduce_adjoint":
        lines = reduce_adjoint_body(lm, "  ", "atomic")
    else:
        lines = reduce_body(lm, "  ")
    for s in lines:
        e.w(s)
    e.w("}")
    e.w()
    if lm.kind == "reduce_adjoint":
        # Deterministic two passes: the store pass writes what each cell sends to its neighbours
        # in the per-incidence scratch, the gather sums it in each cell's incidence order.
        k.head(f"{lm.name}_store", "ScratchArgs<typename P::accum>")
        e.w("  const index_t c = index_t(gid);")
        e.w("  if (c >= d.topo.n_owned_cells) {")
        e.w("    return;")
        e.w("  }")
        e.w("  [[maybe_unused]] device A* scratch = a.scratch;")
        for s in reduce_adjoint_body(lm, "  ", "store"):
            e.w(s)
        e.w("}")
        e.w()
        k.head(f"{lm.name}_twin_gather", "GatherArgs<typename P::accum>")
        e.w("  const index_t c = index_t(gid);")
        e.w("  if (c >= d.topo.n_cells) {")
        e.w("    return;")
        e.w("  }")
        e.w("  device const A* scratch = a.scratch;")
        for s in twin_gather_body(lm, "  "):
            e.w(s)
        e.w("}")
        e.w()


def emit_msl(m: Module, policies: list[str], line_directives: bool = True) -> str:
    """The Metal Shading Language source of a module, instantiated for `policies`."""
    accum = [
        lm.name
        for lm in m.loops
        if "accum" in (lm.fn.compute, lm.fin_fn.compute if lm.fin_fn else None)
    ]
    if accum:
        raise CodegenError(
            Code.POLICY_UNSUPPORTED_BY_BACKEND,
            f"Metal has no FP64 arithmetic for loops computing in the accumulation type: {accum}",
        )
    listed = [lm.name for lm in m.loops if lm.loop.listed]
    if listed:
        raise CodegenError(
            Code.UNSUPPORTED_CONSTRUCT, f"Metal has no reductions over listed cells: {listed}"
        )
    e = Emitter(f"{m.name}.metal", line_directives)
    e.w(f"// Generated by autan_codegen from {m.path.name}. Do not edit.")
    e.w(PRELUDE.read_text().rstrip("\n"))
    e.w()
    e.w("namespace autan {")
    e.w("namespace kernels {")
    e.w(f"namespace {m.name} {{")
    e.w()
    _msl_data_struct(e, m)
    for fn, sig in unique_bodies(m):
        _msl_function(e, fn, sig.types)
    for fn, types in m.extra_bodies:
        _msl_function(e, fn, types)
    k = _Kernels(e)
    for lm in m.loops:
        if lm.kind in ("face", "boundary"):
            _face_kernels(k, lm)
        else:
            _cell_kernel(k, lm)
    k.instantiate(policies)
    e.w()
    e.w(f"}}  // namespace {m.name}")
    e.w("}  // namespace kernels")
    e.w("}  // namespace autan")
    return e.text()


# ------------------------------------------------------------------------------- host side


def _face_launcher(e: Emitter, lm: LoopModel) -> None:
    lo, hi = (
        ("0", "d.topo.n_owned_interior_faces")
        if lm.kind == "face"
        else ("d.topo.n_interior_faces", "d.topo.n_owned_faces")
    )
    col = "interior" if lm.kind == "face" else "boundary"
    nm = lm.name
    e.w("template <class P>")
    e.w(
        f"void {nm}(mtl::Context& ctx, const Data<P>& d, const Strategy strategy, const mtl::Schedule& s) {{"
    )
    e.w(f"  const index_t lo = {lo};")
    e.w(f"  const index_t hi = {hi};")
    e.w("  switch (strategy) {")
    for strat, kern in (("atomic", "atomic"), ("warp_aggregated", "warp_aggregated")):
        e.w(f"    case Strategy::{strat}:")
        e.w("      if constexpr (!P::float_atomics) {")
        e.w(
            f'        throw std::invalid_argument("E1108: strategy {strat} needs atomics on the accumulation type, which Metal lacks in " + std::string(P::name));'
        )
        e.w("      } else if (hi > lo) {")
        e.w(
            f'        ctx.dispatch(library, "{nm}_{kern}", P::tag, d, mtl::RangeArgs{{lo, hi}}, hi - lo, s.block);'
        )
        e.w("      }")
        e.w("      break;")
    e.w("    case Strategy::coloured:")
    e.w(f"      for (std::size_t k = 0; k + 1 < s.{col}.offsets.size(); ++k) {{")
    e.w(f"        const index_t first = s.{col}.offsets[k];")
    e.w(f"        const index_t count = s.{col}.offsets[k + 1] - first;")
    e.w(
        f'        ctx.dispatch(library, "{nm}_coloured", P::tag, d, mtl::ColourArgs{{s.{col}.faces + first, count}}, count, s.block);'
    )
    e.w("      }")
    e.w("      break;")
    e.w("    case Strategy::store_gather: {")
    e.w(f"      if (s.face_scratch_width < {n_out(lm)}) {{")
    e.w(
        '        throw std::invalid_argument("E1104: face scratch narrower than the loop outputs");'
    )
    e.w("      }")
    e.w("      auto* scratch = static_cast<typename P::compute*>(s.face_scratch);")
    e.w("      if (hi > lo) {")
    e.w(
        f'        ctx.dispatch(library, "{nm}_store", P::tag, d, mtl::StoreArgs<typename P::compute>{{lo, hi, scratch}}, hi - lo, s.block);'
    )
    e.w("      }")
    e.w(
        f'      ctx.dispatch(library, "{nm}_gather", P::tag, d, mtl::GatherArgs<typename P::compute>{{scratch}}, d.topo.n_cells, s.block);'
    )
    e.w("      break;")
    e.w("    }")
    e.w("    case Strategy::gather_recompute:")
    e.w(
        f'      ctx.dispatch(library, "{nm}_gather_recompute", P::tag, d, d.topo.n_cells, s.block);'
    )
    e.w("      break;")
    e.w("  }")
    e.w("}")
    e.w()


def _cell_launcher(e: Emitter, lm: LoopModel) -> None:
    if lm.kind != "reduce_adjoint":
        n = (
            "d.topo.n_faces - d.topo.n_interior_faces"
            if lm.kind == "boundary_write"
            else "d.topo.n_owned_cells"
        )
        e.w("template <class P>")
        e.w(f"void {lm.name}(mtl::Context& ctx, const Data<P>& d, const mtl::Schedule& s) {{")
        e.w(f'  ctx.dispatch(library, "{lm.name}_kernel", P::tag, d, {n}, s.block);')
        e.w("}")
        e.w()
        return
    width = len(other_targets(lm))
    e.w("template <class P>")
    e.w(
        f"void {lm.name}(mtl::Context& ctx, const Data<P>& d, const Strategy strategy, const mtl::Schedule& s) {{"
    )
    e.w("  using A = typename P::accum;")
    e.w("  if (!::autan::kernels::deterministic(strategy)) {")
    e.w("    if constexpr (!P::float_atomics) {")
    e.w(
        f'      throw std::invalid_argument("E1108: {lm.name} scatters with atomics on the accumulation type, which Metal lacks in " + std::string(P::name));'
    )
    e.w("    } else {")
    e.w(
        f'      ctx.dispatch(library, "{lm.name}_kernel", P::tag, d, d.topo.n_owned_cells, s.block);'
    )
    e.w("    }")
    e.w("    return;")
    e.w("  }")
    if width:
        e.w(f"  if (s.incidence_scratch_width < {width}) {{")
        e.w(
            '    throw std::invalid_argument("E1105: incidence scratch narrower than the adjoint");'
        )
        e.w("  }")
        assert isinstance(lm.radj, ReduceAdjoint)
        twin = "stencil_twin" if lm.radj.primal.loop.over == dsl.STENCIL else "incidence_twin"
        e.w(f"  if (d.topo.{twin} == nullptr) {{")
        e.w(
            '    throw std::invalid_argument("E1106: deterministic reduction adjoint needs incidence twins");'
        )
        e.w("  }")
    e.w("  auto* scratch = static_cast<A*>(s.incidence_scratch);")
    e.w(
        f'  ctx.dispatch(library, "{lm.name}_store", P::tag, d, mtl::ScratchArgs<A>{{scratch}}, d.topo.n_owned_cells, s.block);'
    )
    if width:
        e.w(
            f'  ctx.dispatch(library, "{lm.name}_twin_gather", P::tag, d, mtl::GatherArgs<A>{{scratch}}, d.topo.n_cells, s.block);'
        )
    e.w("}")
    e.w()


def emit_metal_header(m: Module, policies: list[str], line_directives: bool = True) -> str:
    """`<module>_metal.hpp`: the MSL source as a string and the host launcher of every loop."""
    source = emit_msl(m, policies, line_directives)
    assert f'){DELIMITER}"' not in source
    e = Emitter(f"{m.name}_metal.hpp", line_directives=False)
    e.w(f"// Generated by autan_codegen from {m.path.name}. Do not edit.")
    e.w("#pragma once")
    e.w("#include <cstddef>")
    e.w("#include <stdexcept>")
    e.w("#include <string>")
    e.w()
    e.w('#include "autan/kernels/metal.hpp"')
    e.w(f'#include "{m.name}.hpp"')
    e.w()
    e.w(f"namespace autan::kernels::{m.name}::metal_loops {{")
    e.w("namespace mtl = ::autan::kernels::metal;")
    e.w("using ::autan::kernels::index_t;")
    e.w("using ::autan::kernels::Strategy;")
    e.w()
    # Pieces below 64 KiB: the longest string literal compilers must accept.
    e.w("/// Metal Shading Language source of the module, compiled by mtl::Context on first use.")
    e.w("inline constexpr const char* source[] = {")
    piece: list[str] = []
    size = 0
    for line in [*source.rstrip("\n").split("\n"), None]:
        if line is None or (piece and size + len(line) + 1 > CHUNK):
            e.w(f'R"{DELIMITER}(')
            e.lines.extend(piece)
            e.w(f'){DELIMITER}",')
            piece, size = [], 0
        if line is not None:
            piece.append(line)
            size += len(line) + 1
    e.w("    nullptr};")
    e.w(f'inline constexpr mtl::Library library{{"{m.name}", source}};')
    e.w()
    for lm in m.loops:
        if lm.kind in ("face", "boundary"):
            _face_launcher(e, lm)
        else:
            _cell_launcher(e, lm)
    e.w(f"}}  // namespace autan::kernels::{m.name}::metal_loops")
    return e.text()
