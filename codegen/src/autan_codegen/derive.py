"""Tangent and adjoint loops of a module's loops (research section 4.1).

Active data are cell datasets; geometry, parameters, face datasets (boundary values) and matrix
blocks (Jac, JacAccum) are passive. The tangent of a loop reads the tangents of its active inputs and increments or writes
the tangents of its outputs. The adjoint transposes access modes: an increment's adjoint reads the
target's adjoint through the same map with the declared signs, a gather's adjoint increments the
source's adjoint, and a write's adjoint reads the target's adjoint and then zeroes it.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field

from autan_codegen import dsl
from autan_codegen.ad import adjoint, tangent
from autan_codegen.errors import Code, CodegenError
from autan_codegen.frontend import Signature
from autan_codegen.ir import Function, Instr, dce
from autan_codegen.model import LoopModel, Module

# Keyed by the dataset itself, weakly: an id() key outlives its dataset and can be reused by
# another one when one process generates several modules.
_TAN: weakref.WeakKeyDictionary[dsl.Dataset, dsl.Dataset] = weakref.WeakKeyDictionary()
_ADJ: weakref.WeakKeyDictionary[dsl.Dataset, dsl.Dataset] = weakref.WeakKeyDictionary()


def tan_ds(d: dsl.Dataset) -> dsl.Dataset:
    if d not in _TAN:
        _TAN[d] = dsl.Dataset(f"{d.name}_tan", d.set, d.type)
    return _TAN[d]


def adj_ds(d: dsl.Dataset) -> dsl.Dataset:
    """Adjoints accumulate in the accumulation type (research section 7, adjoint rows)."""
    if d not in _ADJ:
        _ADJ[d] = dsl.Dataset(f"{d.name}_adj", d.set, dsl.Type(dsl.Role.RESIDUAL, d.type.shape))
    return _ADJ[d]


def _is_active(b: dsl.Binding) -> bool:
    d = b.dataset
    return (
        isinstance(d, dsl.Dataset)
        and not d.geometry
        and d.set is dsl.MeshSet.CELLS
        and d.type.role not in dsl.PASSIVE_ROLES
    )


def _active(sig: Signature, bindings: dict[str, dsl.Binding], skip: int = 0) -> list[int]:
    return [k for k, n in enumerate(sig.names) if k >= skip and _is_active(bindings[n])]


def _keep_outputs(fn: Function, start: int) -> Function:
    out = Function(
        fn.name, fn.arg_names, fn.arg_sizes, list(fn.instrs), fn.outputs[start:], fn.source
    )
    return dce(out)


def _single_output(lm: LoopModel) -> dsl.Dataset:
    if len(lm.loop.outputs) != 1:
        raise CodegenError(
            Code.AD_UNSUPPORTED, f"loop {lm.name}: AD needs exactly one output dataset"
        )
    return lm.loop.outputs[0].dataset


def _types(sig: Signature) -> list[dsl.Type]:
    return list(sig.types)


@dataclass
class ReduceAdjoint:
    """What the emitters need for the adjoint of a cell reduction."""

    primal: LoopModel
    body_adj: Function
    body_active: list[int]
    fin_adj: Function | None
    fin_active: list[int]  # indices into the finalise arguments (accumulators first)
    datasets: list[dsl.Dataset] = field(default_factory=list)


def _derived(
    name: str,
    lm: LoopModel,
    fn: Function,
    names: list[str],
    types: list[dsl.Type],
    bindings: dict[str, dsl.Binding],
    outputs: list[object],
    kind: str | None = None,
) -> LoopModel:
    loop = dsl.Loop(kind or lm.kind, lm.loop.body, bindings, outputs, name=name, over=lm.loop.over)  # type: ignore[arg-type]
    fn.name = name + "_body"
    return LoopModel(name, loop, fn, Signature(names, types, None), None, None)


def tangent_loop(lm: LoopModel, m: Module) -> LoopModel:
    name = f"{lm.name}_tangent"
    active = _active(lm.sig, lm.loop.bindings)
    fn_t = tangent(lm.fn, active)
    names = list(fn_t.arg_names)
    types = [*_types(lm.sig), *(lm.sig.types[k] for k in active)]
    bindings = dict(lm.loop.bindings)
    for k in active:
        b = lm.loop.bindings[lm.sig.names[k]]
        assert isinstance(b.dataset, dsl.Dataset)
        bindings[names[len(lm.sig.names) + active.index(k)]] = dsl.Binding(
            tan_ds(b.dataset), b.via, None
        )
    n_primal = len(lm.fn.outputs)
    if lm.kind in ("face", "boundary", "cell"):
        out = _single_output(lm)
        fn_t = _keep_outputs(fn_t, n_primal)
        o = lm.loop.outputs[0]
        outputs: list[object] = [
            dsl.Inc(tan_ds(out), o.left, o.right)
            if isinstance(o, dsl.Inc)
            else dsl.Write(tan_ds(out))
        ]
        res = _derived(name, lm, fn_t, names, types, bindings, outputs)
        m.extra_bodies.append((fn_t, types))
        return res
    # Cell reduction: accumulate primal values (not written) and their tangents.
    accs: list[object] = [type(o)(o.dataset, write=False) for o in lm.loop.outputs]  # type: ignore[call-arg]
    for k, o in enumerate(lm.loop.outputs):
        t = tan_ds(o.dataset)
        accs.append(dsl.Sum(t) if isinstance(o, dsl.Sum) else dsl.Follow(t, leader=k))
    res = _derived(name, lm, fn_t, names, types, bindings, accs)
    m.extra_bodies.append((fn_t, types))
    if lm.fin_fn is not None and lm.fin_sig is not None:
        n_acc = len(lm.loop.outputs)
        fin_active = list(range(n_acc)) + _active(lm.fin_sig, lm.loop.finalise_bindings, skip=n_acc)
        fin_t = _keep_outputs(tangent(lm.fin_fn, fin_active), len(lm.fin_fn.outputs))
        fin_t.name = f"{name}_finalise"
        fin_names = list(fin_t.arg_names)
        fin_types = [*lm.fin_sig.types, *(lm.fin_sig.types[k] for k in fin_active)]
        fin_bind = dict(lm.loop.finalise_bindings)
        order: list[tuple[str, int | str]] = [("acc", k) for k in range(n_acc)]
        order += [("bind", n) for n in lm.fin_sig.names[n_acc:]]
        for j, k in enumerate(fin_active):
            if k < n_acc:
                order.append(("acc", n_acc + k))
            else:
                b = lm.loop.finalise_bindings[lm.fin_sig.names[k]]
                tname = fin_names[len(lm.fin_sig.names) + j]
                fin_bind[tname] = dsl.Binding(tan_ds(b.dataset), b.via, None)  # type: ignore[arg-type]
                order.append(("bind", tname))
        res.fin_fn = fin_t
        res.fin_sig = Signature(fin_names, fin_types, None)
        res.loop.finalise = lm.loop.finalise
        res.loop.finalise_bindings = fin_bind
        res.fin_args = order
        m.extra_bodies.append((fin_t, fin_types))
    return res


def adjoint_loop(lm: LoopModel, m: Module) -> LoopModel:
    name = f"{lm.name}_adjoint"
    active = _active(lm.sig, lm.loop.bindings)
    n_args = len(lm.sig.names)
    if lm.kind in ("face", "boundary", "cell"):
        out = _single_output(lm)
        o = lm.loop.outputs[0]
        n = len(lm.fn.outputs)
        seed_t = dsl.Type(dsl.Role.RESIDUAL, out.type.shape)
        if lm.kind == "face":
            assert isinstance(o, dsl.Inc)
            assert o.right is not None
            seeds = [[float(o.left)] * n, [float(o.right)] * n]
            seed_b = [adj_ds(out).left, adj_ds(out).right]
        elif lm.kind == "boundary":
            assert isinstance(o, dsl.Inc)
            seeds = [[float(o.left)] * n]
            seed_b = [adj_ds(out).left]
        else:
            seeds = [[1.0] * n]
            seed_b = [dsl.Binding(adj_ds(out))]
        fn_a = adjoint(lm.fn, active, seeds)
        names = list(fn_a.arg_names)
        types = [*_types(lm.sig), *(seed_t for _ in seeds)]
        bindings = dict(lm.loop.bindings)
        for i, b in enumerate(seed_b):
            bindings[names[n_args + i]] = b
        outputs: list[object] = []
        for k in active:
            b = lm.loop.bindings[lm.sig.names[k]]
            a = adj_ds(b.dataset)  # type: ignore[arg-type]
            if b.via == "right":
                outputs.append(dsl.Inc(a, left=0, right=1))
            else:
                outputs.append(dsl.Inc(a, left=1, right=0 if lm.kind == "face" else None))
        if lm.kind == "cell":
            # The overwritten target had no influence: its adjoint is read, then zeroed.
            fn_a.outputs = [*fn_a.outputs, *([_zero(fn_a)] * out.type.size)]
            outputs.append(dsl.Write(adj_ds(out)))
        res = _derived(name, lm, fn_a, names, types, bindings, outputs)
        m.extra_bodies.append((fn_a, types))
        return res
    # Cell reduction.
    n_out = len(lm.fn.outputs)
    body_adj = adjoint(lm.fn, active, [[1.0] * n_out])
    body_adj.name = f"{name}_body"
    seed_t = dsl.Type(dsl.Role.RESIDUAL, (n_out,))
    m.extra_bodies.append((body_adj, [*_types(lm.sig), seed_t]))
    fin_adj = None
    fin_active: list[int] = []
    datasets = [adj_ds(o.dataset) for o in lm.loop.outputs]
    datasets += [adj_ds(lm.loop.bindings[lm.sig.names[k]].dataset) for k in active]  # type: ignore[arg-type]
    if lm.fin_fn is not None and lm.fin_sig is not None:
        n_acc = len(lm.loop.outputs)
        fin_active = list(range(n_acc)) + _active(lm.fin_sig, lm.loop.finalise_bindings, skip=n_acc)
        n_fin = len(lm.fin_fn.outputs)
        fin_adj = adjoint(lm.fin_fn, fin_active, [[1.0] * n_fin])
        fin_adj.name = f"{name}_finalise"
        m.extra_bodies.append((fin_adj, [*lm.fin_sig.types, dsl.Type(dsl.Role.RESIDUAL, (n_fin,))]))
        datasets += [
            adj_ds(lm.loop.finalise_bindings[lm.fin_sig.names[k]].dataset)  # type: ignore[arg-type]
            for k in fin_active[n_acc:]
        ]
    res = LoopModel(
        name,
        dsl.Loop("reduce_adjoint", lm.loop.body, lm.loop.bindings, [], name=name),
        lm.fn,
        lm.sig,
        None,
        None,
    )
    res.loop.finalise_bindings = lm.loop.finalise_bindings
    res.radj = ReduceAdjoint(lm, body_adj, active, fin_adj, fin_active, datasets)
    return res


def _zero(fn: Function) -> int:
    return fn.add(Instr("const", (), 0.0))


def derive(m: Module) -> None:
    """Append the tangent and adjoint loops of every loop of `m`, and the datasets they use."""
    derived: list[LoopModel] = []
    for lm in m.loops:
        has_active = bool(_active(lm.sig, lm.loop.bindings)) or (
            lm.fin_sig is not None
            and bool(_active(lm.fin_sig, lm.loop.finalise_bindings, skip=len(lm.loop.outputs)))
        )
        accum = lm.fn.compute == "accum" or (lm.fin_fn is not None and lm.fin_fn.compute == "accum")
        if has_active and (lm.loop.entries or accum):
            raise CodegenError(
                Code.AD_UNSUPPORTED,
                f"{lm.name}: per-entry outputs and compute='accum' bodies take no active data",
            )
        if not has_active or lm.kind == "boundary_write":
            # Nothing active flows in: the tangent output is zero and there is no adjoint to send.
            continue
        derived.append(tangent_loop(lm, m))
        derived.append(adjoint_loop(lm, m))
    for lm in derived:
        extra = lm.radj.datasets if isinstance(lm.radj, ReduceAdjoint) else []
        for d in [*lm.reads(), *lm.writes(), *extra]:
            if isinstance(d, dsl.Dataset) and all(d is not x for x in m.datasets):
                m.datasets.append(d)
    m.loops.extend(derived)
