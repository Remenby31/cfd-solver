"""Load a kernel module and check each loop's bindings against its body: the backend-neutral model."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from autan_codegen import dsl
from autan_codegen.ad import jacobian
from autan_codegen.errors import Code, CodegenError
from autan_codegen.frontend import Signature, parse

if TYPE_CHECKING:
    from pathlib import Path

    from autan_codegen.ir import Function

ROLE_TYPE = {
    dsl.Role.STATE: "state",
    dsl.Role.GRAD: "grad",
    dsl.Role.RESIDUAL: "residual",
    dsl.Role.GEOM: "geom",
    dsl.Role.PARAM: "param",
    dsl.Role.JAC: "jac",
    dsl.Role.JAC_ACCUM: "accum",
}


@dataclass
class LoopModel:
    name: str
    loop: dsl.Loop
    fn: Function
    sig: Signature
    fin_fn: Function | None
    fin_sig: Signature | None
    # Finalise call order: ("acc", k) or ("bind", name); default: every accumulator, then bindings.
    fin_args: list[tuple[str, int | str]] | None = None
    # Adjoint of a cell reduction (kind "reduce_adjoint"): see derive.reduce_adjoint.
    radj: object = None
    # Jacobian outputs: each with its forward-derivative function (ad.jacobian).
    jac_fns: list[tuple[dsl.Jacobian, Function]] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return self.loop.kind

    def reads(self) -> list[dsl.Dataset | dsl.Parameter]:
        out: list[dsl.Dataset | dsl.Parameter] = []
        for b in [*self.loop.bindings.values(), *self.loop.finalise_bindings.values()]:
            out.append(b.dataset)
            if b.boundary is not None:
                out.append(b.boundary)
        return out

    def writes(self) -> list[dsl.Dataset]:
        return (
            [o.dataset for o in self.loop.outputs]
            + [j.dataset for j in self.loop.jacobians]
            + [w.dataset for w in self.loop.entries]
        )


@dataclass
class Module:
    name: str
    path: Path
    loops: list[LoopModel]
    datasets: list[dsl.Dataset]
    params: list[dsl.Parameter]
    bodies: list[tuple[Function, Signature]]
    extra_bodies: list[tuple[Function, list[dsl.Type]]] = field(default_factory=list)


ALLOWED_VIA = {
    "face": {
        dsl.MeshSet.CELLS: {"left", "right"},
        dsl.MeshSet.FACES: {"direct"},
    },
    "reduce_stencil": {
        dsl.MeshSet.CELLS: {"self", "other"},
        dsl.MeshSet.STENCIL: {"direct"},
    },
    "boundary": {
        dsl.MeshSet.CELLS: {"left"},
        dsl.MeshSet.FACES: {"direct"},
    },
    "boundary_write": {
        dsl.MeshSet.CELLS: {"left"},
        dsl.MeshSet.FACES: {"direct"},
    },
    "cell": {dsl.MeshSet.CELLS: {"direct"}},
    "reduce": {
        dsl.MeshSet.CELLS: {"self", "other"},
        dsl.MeshSet.INCIDENCES: {"direct"},
        # The face of the incidence; near, far and on_boundary load only where they are valid.
        dsl.MeshSet.FACES: {"direct", "near", "far", "on_boundary"},
    },
}


def _check_bindings(
    where: str, kind: str, sig: Signature, bindings: dict[str, dsl.Binding], skip: int = 0
) -> None:
    names = sig.names[skip:]
    if set(names) != set(bindings):
        missing = sorted(set(names) - set(bindings))
        extra = sorted(set(bindings) - set(names))
        raise CodegenError(
            Code.INVALID_BINDING,
            f"bindings do not match arguments: missing {missing}, extra {extra}",
            where,
        )
    for name, t in zip(sig.names[skip:], sig.types[skip:], strict=True):
        b = bindings[name]
        ds = b.dataset
        halves = b.via in ("near", "far")
        shape = ds.type.shape[1:] if halves and ds.type.shape[:1] == (2,) else ds.type.shape
        if halves and ds.type.shape[:1] != (2,):
            raise CodegenError(
                Code.INVALID_BINDING,
                f"{name}: near and far read a dataset of shape (2, ...)",
                where,
            )
        if shape != t.shape or ds.type.role is not t.role:
            raise CodegenError(
                Code.INVALID_BINDING,
                f"argument {name} is {t.role.value}{t.shape}, data {ds.name} is {ds.type.role.value}{ds.type.shape}",
                where,
            )
        if isinstance(ds, dsl.Parameter):
            if b.via != "direct":
                raise CodegenError(
                    Code.INVALID_BINDING, f"parameter {ds.name} is read directly", where
                )
            continue
        allowed = ALLOWED_VIA[kind].get(ds.set, set())
        if b.via not in allowed:
            raise CodegenError(
                Code.INVALID_BINDING,
                f"{ds.set.value} data {ds.name} cannot be read as {b.via} in a {kind} loop",
                where,
            )
        if b.via == "other":
            bd = b.boundary
            if bd is not None and (bd.set is not dsl.MeshSet.FACES or bd.type != ds.type):
                raise CodegenError(
                    Code.INVALID_BINDING,
                    f"{name}: other() takes None or a face dataset of the same type",
                    where,
                )


def _check_outputs(where: str, lm: LoopModel) -> None:
    loop = lm.loop
    sig = lm.fin_sig if lm.fin_sig is not None else lm.sig
    rets, body_rets = sig.returns, lm.sig.returns
    if rets is None or body_rets is None:
        raise CodegenError(Code.INVALID_BINDING, "loop bodies annotate their return types", where)
    # Entries take the body's last outputs; the accumulators (or the finalise) the others.
    n_entries = len(loop.entries)
    entry_rets = body_rets[len(body_rets) - n_entries :] if n_entries else []
    if lm.fin_sig is None:
        rets = rets[: len(rets) - n_entries]
    if len(rets) != len(loop.outputs) or len(body_rets) != len(loop.outputs) + n_entries:
        raise CodegenError(
            Code.INVALID_BINDING,
            f"body returns {len(body_rets)} values, loop has {len(loop.outputs)} outputs "
            f"and {n_entries} entries",
            where,
        )
    for t, w in zip(entry_rets, loop.entries, strict=True):
        ds = w.dataset
        if ds.type.shape != t.shape or ds.type.role not in dsl.PASSIVE_ROLES or ds.geometry:
            raise CodegenError(
                Code.INVALID_BINDING,
                f"entry output {ds.name} must be a Jac or JacAccum dataset of shape {t.shape}",
                where,
            )
    for t, o in zip(rets, loop.outputs, strict=True):
        ds = o.dataset
        wanted = dsl.MeshSet.FACES if loop.kind == "boundary_write" else dsl.MeshSet.CELLS
        if ds.set is not wanted or ds.geometry:
            raise CodegenError(
                Code.INVALID_BINDING, f"output {ds.name} must be a {wanted.value} dataset", where
            )
        if ds.type.shape != t.shape:
            raise CodegenError(
                Code.INVALID_BINDING,
                f"output {ds.name} has shape {ds.type.shape}, body gives {t.shape}",
                where,
            )
        if isinstance(o, dsl.Inc) and ds.type.role is not dsl.Role.RESIDUAL:
            # Inc targets hold the accumulation type of the policy (section 6, generator rules).
            raise CodegenError(
                Code.INC_TARGET_NOT_ACCUMULATION,
                f"Inc target {ds.name} must have role Residual",
                where,
            )
    if lm.fin_sig is not None:
        rets_body = (lm.sig.returns or [])[: len(loop.outputs)]
        for k, t in enumerate(rets_body):
            ft = lm.fin_sig.types[k]
            if ft.shape != t.shape or ft.role is not dsl.Role.RESIDUAL:
                raise CodegenError(
                    Code.INVALID_BINDING,
                    f"finalise argument {lm.fin_sig.names[k]} must be Residual{t.shape}",
                    where,
                )


def _jacobian_fn(where: str, lm: LoopModel, j: dsl.Jacobian) -> Function:
    """Check a Jacobian output against its loop and body; return its derivative function."""
    loop = lm.loop
    if loop.kind not in ("face", "boundary", "cell"):
        raise CodegenError(
            Code.INVALID_LOOP_OUTPUT,
            "Jacobian outputs belong to face, boundary or cell loops",
            where,
        )
    idx = []
    for name in j.wrt:
        if name not in lm.sig.names:
            raise CodegenError(
                Code.INVALID_BINDING, f"Jacobian wrt {name!r}: not an argument", where
            )
        b = loop.bindings[name]
        if not (
            isinstance(b.dataset, dsl.Dataset)
            and b.dataset.set is dsl.MeshSet.CELLS
            and not b.dataset.geometry
            and b.dataset.type.role not in dsl.PASSIVE_ROLES
        ):
            raise CodegenError(
                Code.INVALID_BINDING, f"Jacobian wrt {name!r}: not an active cell dataset", where
            )
        idx.append(lm.sig.names.index(name))
    sizes = {lm.sig.types[k].size for k in idx}
    if len(sizes) != 1:
        raise CodegenError(Code.TYPE_MISMATCH, "Jacobian wrt arguments of different sizes", where)
    n_in = sizes.pop()
    n_out = len(lm.fn.outputs)
    shape = (n_out, n_in) if len(idx) == 1 else (len(idx), n_out, n_in)
    ds = j.dataset
    wanted = dsl.MeshSet.CELLS if loop.kind == "cell" else dsl.MeshSet.FACES
    if ds.set is not wanted or ds.type.role not in dsl.PASSIVE_ROLES or ds.type.shape != shape:
        raise CodegenError(
            Code.INVALID_LOOP_OUTPUT,
            f"Jacobian dataset {ds.name} must be Jac{shape} on {wanted.value}, is "
            f"{ds.type.role.value}{ds.type.shape} on {ds.set.value}",
            where,
        )
    jfn = jacobian(lm.fn, idx)
    jfn.name = f"{lm.name}_jacobian_{ds.name}"
    return jfn


def build_loop(name: str, loop: dsl.Loop) -> LoopModel:
    where = f"{loop.body.source_file}: loop {name}"
    fn, sig = parse(loop.body)
    fin_fn = fin_sig = None
    kind = "reduce_stencil" if loop.kind == "reduce" and loop.over == dsl.STENCIL else loop.kind
    _check_bindings(where, kind, sig, loop.bindings)
    if loop.finalise is not None:
        fin_fn, fin_sig = parse(loop.finalise)
        n_acc = len(loop.outputs)
        _check_bindings(where, "cell", fin_sig, loop.finalise_bindings, skip=n_acc)
    accum = [(fn, loop.bindings)] if fn.compute == "accum" else []
    if fin_fn is not None and fin_fn.compute == "accum":
        accum.append((fin_fn, loop.finalise_bindings))
    for _, bound in accum:
        # Block datasets and parameters only: no derivatives, no geometry stored in geom_value.
        others = [
            n
            for n, b in bound.items()
            if not (
                isinstance(b.dataset, dsl.Parameter) or b.dataset.type.role in dsl.PASSIVE_ROLES
            )
        ]
        if loop.kind not in ("cell", "reduce") or others:
            raise CodegenError(
                Code.INVALID_BINDING,
                "compute='accum' bodies run in cell loops and reductions on blocks and "
                f"parameters, not {others}",
                where,
            )
    lm = LoopModel(name, loop, fn, sig, fin_fn, fin_sig)
    targets = [j.dataset for j in loop.jacobians]
    if len({id(t) for t in targets}) != len(targets):
        raise CodegenError(
            Code.INVALID_LOOP_OUTPUT, "two Jacobian outputs write the same dataset", where
        )
    lm.jac_fns = [(j, _jacobian_fn(where, lm, j)) for j in loop.jacobians]
    _check_outputs(where, lm)
    if loop.kind == "boundary_write" and not loop.frozen:
        active = [
            b.dataset.name
            for b in loop.bindings.values()
            if isinstance(b.dataset, dsl.Dataset)
            and not b.dataset.geometry
            and b.dataset.set is dsl.MeshSet.CELLS
            and b.dataset.type.role not in dsl.PASSIVE_ROLES
        ]
        if active:
            raise CodegenError(
                Code.AD_UNSUPPORTED,
                f"boundary loop writes face data from active {active}: derivatives would be "
                "dropped; declare frozen=True to accept that",
                where,
            )
    read = {id(d) for d in lm.reads()}
    for d in lm.writes():
        if id(d) in read:
            # In place updates would race between strategies and lose the adjoint of the input.
            raise CodegenError(Code.INVALID_BINDING, f"{d.name} is both read and written", where)
    return lm


def load(path: Path) -> Module:
    """Import a kernel module and build the model of every loop it binds to a global name."""
    name = path.stem
    spec = importlib.util.spec_from_file_location(f"autan_kernel_{name}", path)
    if spec is None or spec.loader is None:
        raise CodegenError(Code.INVALID_KERNEL_MODULE, f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    loops = []
    for gname, obj in vars(mod).items():
        if isinstance(obj, dsl.Loop):
            obj.name = gname
            loops.append(build_loop(gname, obj))
    if not loops:
        raise CodegenError(Code.INVALID_KERNEL_MODULE, f"{path} declares no loop")
    datasets: list[dsl.Dataset] = []
    params: list[dsl.Parameter] = []
    for lm in loops:
        for d in [*lm.reads(), *lm.writes()]:
            target = params if isinstance(d, dsl.Parameter) else datasets
            if all(d is not x for x in target):
                if any(d.name == x.name for x in target):
                    raise CodegenError(
                        Code.INVALID_BINDING, f"two different data named {d.name}", str(path)
                    )
                target.append(d)  # type: ignore[arg-type]
    # A reversed face reads face_r_left from face_r_right and back (Topology::face_reversed).
    pairs = {"face_r_left": dsl.geometry.face_r_right, "face_r_right": dsl.geometry.face_r_left}
    for d in list(datasets):
        partner = pairs.get(d.name) if d.geometry else None
        if partner is not None and all(partner is not x for x in datasets):
            datasets.append(partner)
    bodies: list[tuple[Function, Signature]] = []
    for lm in loops:
        bodies.append((lm.fn, lm.sig))
        bodies += [(jfn, lm.sig) for _, jfn in lm.jac_fns]
        if lm.fin_fn is not None and lm.fin_sig is not None:
            bodies.append((lm.fin_fn, lm.fin_sig))
    return Module(name, path, loops, datasets, params, bodies)
