"""C++20 emission shared by every backend: bodies, the data struct, loads and stores.

Bodies are `template <class P> AUTAN_HD` functions of the precision policy P. Arguments arrive in
their storage type, are converted to `P::compute` on first use, and `rel` subtracts raw Geom values
in their storage type before converting. Backends emit loops around these bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from autan_codegen import dsl
from autan_codegen.model import ROLE_TYPE, Module

if TYPE_CHECKING:
    from autan_codegen.frontend import Signature
    from autan_codegen.ir import Function

INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/", "and": "&&", "or": "||"}
CALL = {"sqrt": "sqrt", "exp": "exp", "log": "log", "sin": "sin", "cos": "cos"}
CMP = {"lt", "le", "gt", "ge"}


def ctype(role: dsl.Role, policy: str = "P") -> str:
    return f"typename {policy}::{ROLE_TYPE[role]}"


def literal(x: float) -> str:
    r = repr(float(x))
    if r in ("inf", "-inf", "nan"):
        raise ValueError(r)
    return r if ("." in r or "e" in r) else r + ".0"


@dataclass
class Emitter:
    """Collects lines; `line_directives` maps body statements to the DSL source with #line."""

    header_name: str
    line_directives: bool = True

    def __post_init__(self) -> None:
        self.lines: list[str] = []

    def w(self, s: str = "") -> None:
        self.lines.extend(s.split("\n"))

    def source_line(self, line: int, path: str) -> None:
        if self.line_directives and line > 0:
            self.lines.append(f'#line {line} "{path}"')

    def restore_line(self) -> None:
        if self.line_directives:
            self.lines.append(f'#line {len(self.lines) + 2} "{self.header_name}"')

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def arg_name(k: int, c: int) -> str:
    return f"a{k}_{c}"


def emit_function(
    e: Emitter, fn: Function, sig_types: list[dsl.Type], n_out: int, name: str | None = None
) -> None:
    """Emit `fn` as `template <class P> AUTAN_HD inline void name(args..., C* out)`."""
    params = [
        f"[[maybe_unused]] const {ctype(t.role)} {arg_name(k, c)}"
        for k, t in enumerate(sig_types)
        for c in range(t.size)
    ]
    ct = f"typename P::{fn.compute}"
    params.append(f"{ct}* __restrict__ out" if n_out else f"{ct}*")
    e.w("template <class P>")
    e.w(f"AUTAN_HD inline void {name or fn.name}({', '.join(params)}) {{")
    e.w(f"  using C = {ct};")
    body_lines(e, fn, indent="  ")
    for k, o in enumerate(fn.outputs):
        e.w(f"  out[{k}] = v{o};")
    e.restore_line()
    e.w("}")
    e.w()


def body_lines(e: Emitter, fn: Function, indent: str) -> None:
    users_non_rel: set[int] = set()
    for ins in fn.instrs:
        if ins.op != "rel":
            users_non_rel.update(ins.args)
    users_non_rel.update(fn.outputs)
    last_line = -1
    for v, ins in enumerate(fn.instrs):
        if ins.line != last_line:
            e.source_line(ins.line, fn.source)
            last_line = ins.line
        a = [f"v{x}" for x in ins.args]
        if ins.op == "input":
            if v not in users_non_rel:
                continue
            e.w(f"{indent}const C v{v} = C({arg_name(*ins.attr)});")  # type: ignore[misc]
            continue
        if ins.op == "const":
            e.w(f"{indent}const C v{v} = C({literal(ins.attr)});")  # type: ignore[arg-type]
            continue
        if ins.op == "rel":
            x, y = (fn.instrs[i].attr for i in ins.args)
            e.w(f"{indent}const C v{v} = C({arg_name(*x)} - {arg_name(*y)});")  # type: ignore[misc]
            continue
        if ins.op in CMP:
            e.w(f"{indent}const bool v{v} = ::autan::kernels::{ins.op}({a[0]}, {a[1]});")
            continue
        if ins.op in ("and", "or"):
            e.w(f"{indent}const bool v{v} = {a[0]} {INFIX[ins.op]} {a[1]};")
            continue
        if ins.op == "not":
            e.w(f"{indent}const bool v{v} = !{a[0]};")
            continue
        if ins.op in INFIX:
            e.w(f"{indent}const C v{v} = {a[0]} {INFIX[ins.op]} {a[1]};")
        elif ins.op == "neg":
            e.w(f"{indent}const C v{v} = -{a[0]};")
        elif ins.op in CALL:
            e.w(f"{indent}const C v{v} = {CALL[ins.op]}({a[0]});")
        elif ins.op == "select":
            e.w(f"{indent}const C v{v} = {a[0]} ? {a[1]} : {a[2]};")
        else:
            raise AssertionError(ins.op)


# ---------------------------------------------------------------------------- data access


def stride(d: dsl.Dataset) -> str:
    return {
        dsl.MeshSet.CELLS: "d.topo.n_cells",
        dsl.MeshSet.FACES: "d.topo.n_faces",
        dsl.MeshSet.INCIDENCES: "d.topo.n_incidences",
        dsl.MeshSet.STENCIL: "d.topo.n_stencil",
    }[d.set]


def at(d: dsl.Dataset | dsl.Parameter, index: str, c: int, layout: str = "soa") -> str:
    if isinstance(d, dsl.Parameter):
        return f"d.{d.name}[{c}]"
    if d.type.size == 1:
        return f"d.{d.name}[{index}]"
    if layout == "aos":
        return f"d.{d.name}[offset_t({d.type.size}) * ({index}) + {c}]"
    return f"d.{d.name}[offset_t({c}) * {stride(d)} + ({index})]" if c else f"d.{d.name}[{index}]"


def incidence_loop(indent: str) -> str:
    """Opening of the loop over the incidences k of cell c, in CSR or sliced layout
    (Topology::incidence_stride): the same incidences in the same order."""
    return (
        f"{indent}for (offset_t k = d.topo.cell_face_offsets[c], kn = d.topo.incidence_stride == 1 "
        "? d.topo.cell_face_offsets[c + 1] : d.topo.cell_face_offsets[c] + "
        "offset_t(d.topo.incidence_length[c]) * d.topo.incidence_stride; "
        "k < kn; k += d.topo.incidence_stride) {"
    )


def at_dynamic(d: dsl.Dataset, index: str, comp: str, layout: str = "soa") -> str:
    """`at` with a component index computed at run time."""
    if layout == "aos":
        return f"d.{d.name}[offset_t({d.type.size}) * ({index}) + {comp}]"
    return f"d.{d.name}[offset_t{comp} * {stride(d)} + ({index})]"


def rel_datasets(m: Module) -> set[int]:
    """Ids of the datasets some body reads through rel(), in any loop of the module."""
    out: set[int] = set()
    for lm in m.loops:
        pairs = [(lm.fn, lm.loop.bindings)]
        if lm.fin_fn is not None:
            pairs.append((lm.fin_fn, lm.loop.finalise_bindings))
        for fn, bindings in pairs:
            for ins in fn.instrs:
                if ins.op != "rel":
                    continue
                for a in ins.args:
                    k = fn.instrs[a].attr[0]  # type: ignore[index]
                    b = bindings.get(fn.arg_names[k])
                    if b is not None:
                        out.add(id(b.dataset))
                        if b.boundary is not None:
                            out.add(id(b.boundary))
    # Members of a swapped pair (a reversed face reads one for the other) share their storage.
    for name, other in SWAPPED.items():
        if id(getattr(dsl.geometry, name)) in out:
            out.add(id(getattr(dsl.geometry, other)))
    return out


def storage_type(d: dsl.Dataset, rel_read: set[int]) -> str:
    """Storage of a dataset in Data<P>: its role's type, except built-in geometry no body
    reads through rel(): it only reaches bodies converted to P::compute and is stored in
    P::geom_value (the compute type's precision, as a real type): same values, fewer bytes.
    Datasets kernels write keep their role's type."""
    if d.geometry and d.type.role is dsl.Role.GEOM and id(d) not in rel_read:
        return "typename P::geom_value"
    return ctype(d.type.role)


def emit_data_struct(e: Emitter, m: Module) -> None:
    e.w(
        "/// Pointers to every dataset the module's loops use (host or device), and parameter values."
    )
    e.w("/// Datasets are structure-of-arrays: component c of entity i is at c * n + i.")
    e.w("template <class P>")
    e.w("struct Data {")
    e.w("  ::autan::kernels::Topology topo;")
    rel_read = rel_datasets(m)
    for d in m.datasets:
        const = "const " if d.geometry else ""
        e.w(
            f"  {const}{storage_type(d, rel_read)}* {d.name};  // {d.set.value}, {d.type.size} component(s)"
        )
    for p in m.params:
        e.w(f"  {ctype(p.type.role)} {p.name}[{p.type.size}];")
    e.w("};")
    e.w()


# Built-in face geometry that changes with the face's orientation: negated, or swapped in pairs.
ORIENTED = {"face_area", "face_normal", "face_moment"}
SWAPPED = {"face_r_left": "face_r_right", "face_r_right": "face_r_left"}


def call_args(
    sig: Signature,
    bindings: dict[str, dsl.Binding],
    index_of: dict[str, str],
    skip: int = 0,
    layout: str = "soa",
    reversible: bool = False,
) -> list[str]:
    """Loads of each scalar argument; `index_of[via]` is the index expression of that binding.
    With `reversible`, face geometry is read in the face's global orientation when the C++
    boolean `rev` holds (Topology::face_reversed)."""
    out = []
    for name, t in zip(sig.names[skip:], sig.types[skip:], strict=True):
        b = bindings[name]
        for c in range(t.size):
            if b.via in ("near", "far", "on_boundary"):
                zero = f"{ctype(t.role)}(0)"
                if b.via == "on_boundary":
                    out.append(f"(across ? {zero} : {at(b.dataset, 'f', c, layout)})")
                    continue
                # Half 0 of the face's (2, ...) data for its left cell, half 1 for its right one,
                # in the face's global orientation (Topology::face_reversed).
                left = (
                    "(d.topo.owner[f] == c) != "
                    "(d.topo.face_reversed != nullptr && d.topo.face_reversed[f] != 0)"
                )
                first = left if b.via == "near" else f"!({left})"
                comp = f"(({first}) ? {c} : {t.size + c})"
                out.append(f"(across ? {at_dynamic(b.dataset, 'f', comp, layout)} : {zero})")
                continue
            if b.via == "other":
                bnd = (
                    at(b.boundary, "f", c, layout)
                    if b.boundary is not None
                    else f"{ctype(t.role)}(0)"
                )
                out.append(f"(across ? {at(b.dataset, 'nb', c, layout)} : {bnd})")
            else:
                via = b.via
                if (
                    via == "direct"
                    and isinstance(b.dataset, dsl.Dataset)
                    and b.dataset.set is dsl.MeshSet.FACES
                    and "face" in index_of
                ):
                    via = "face"
                load = at(b.dataset, index_of[via], c, layout)
                ds = b.dataset
                if reversible and isinstance(ds, dsl.Dataset) and ds.geometry:
                    if ds.name in ORIENTED:
                        load = f"(rev ? -{load} : {load})"
                    elif ds.name in SWAPPED:
                        other = getattr(dsl.geometry, SWAPPED[ds.name])
                        load = f"(rev ? {at(other, index_of[via], c, layout)} : {load})"
                out.append(load)
    return out


def unique_bodies(m: Module) -> list[tuple[Function, Signature]]:
    seen: set[str] = set()
    out = []
    for fn, sig in m.bodies:
        if fn.name not in seen:
            seen.add(fn.name)
            out.append((fn, sig))
    return out


def emit_common_header(m: Module, line_directives: bool = True) -> str:
    """`<module>.hpp`: the Data struct and every body of the module."""
    e = Emitter(f"{m.name}.hpp", line_directives)
    e.w(f"// Generated by autan_codegen from {m.path.name}. Do not edit.")
    e.w("#pragma once")
    e.w('#include "autan/kernels/runtime.hpp"')
    e.w()
    e.w(f"namespace autan::kernels::{m.name} {{")
    e.w("using std::cos;")
    e.w("using std::exp;")
    e.w("using std::log;")
    e.w("using std::sin;")
    e.w("using std::sqrt;")
    e.w()
    emit_data_struct(e, m)
    for fn, sig in unique_bodies(m):
        emit_function(e, fn, sig.types, len(fn.outputs))
    for fn, types in m.extra_bodies:
        emit_function(e, fn, types, len(fn.outputs))
    frozen = [lm.name for lm in m.loops if lm.loop.frozen]
    e.w("/// Loops whose outputs are treated as passive although they depend on active data: the")
    e.w("/// derivatives of a residual that uses them are approximate (frozen boundary states).")
    e.w(
        f"inline constexpr const char* frozen_loops[] = {{{', '.join(f'{n!r}'.replace(chr(39), chr(34)) for n in frozen) or 'nullptr'}}};"
    )
    e.w(f"inline constexpr int n_frozen_loops = {len(frozen)};")
    e.w()
    e.w(f"}}  // namespace autan::kernels::{m.name}")
    return e.text()
