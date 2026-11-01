"""Kernel DSL: precision roles, value types, mesh data, pointwise bodies and loop declarations.

A kernel module declares data (`cells.field`, `faces.field`, `param`), pointwise bodies
(`@pointwise`) and loops (`face_loop`, `cell_loop`, `cell_reduce`). The generator reads every loop
bound to a module-level name and emits one C++ function per loop, backend and race strategy.
"""

from __future__ import annotations

import enum
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from autan_codegen.errors import Code, CodegenError

if TYPE_CHECKING:
    from collections.abc import Callable


class Role(enum.Enum):
    """Precision role of a value. A policy maps each role to a storage type."""

    STATE = "state"
    GRAD = "grad"
    RESIDUAL = "residual"
    GEOM = "geom"
    PARAM = "param"
    JAC = "jac"  # blocks of Jacobians, written for preconditioners (not differentiated)
    # Blocks kept in the accumulation precision (diagonal blocks and their inverses, FP64 in
    # fp32+acc64), off-diagonal blocks staying in `jac` (mixed-precision preconditioners).
    JAC_ACCUM = "jac_accum"


# Matrix blocks are data of a linear solve, never differentiated.
PASSIVE_ROLES = frozenset({Role.JAC, Role.JAC_ACCUM})


@dataclass(frozen=True)
class Type:
    """A role and a shape: () scalar, (n,) vector, (3, 3) matrix."""

    role: Role
    shape: tuple[int, ...] = ()

    @property
    def size(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


class RoleAnnotation:
    """Annotation object: `State` is a scalar, `State.Vec3` a 3-vector, `Geom.Mat3` a 3x3 matrix."""

    def __init__(self, role: Role) -> None:
        self.role = role
        self.scalar = Type(role)
        self.Vec3 = Type(role, (3,))
        self.Vec5 = Type(role, (5,))
        self.Mat3 = Type(role, (3, 3))
        self.Mat5 = Type(role, (5, 5))
        self.Mat53 = Type(role, (5, 3))  # gradient of a 5-component state: row i is grad of u_i


State = RoleAnnotation(Role.STATE)
Grad = RoleAnnotation(Role.GRAD)
Residual = RoleAnnotation(Role.RESIDUAL)
Geom = RoleAnnotation(Role.GEOM)
Param = RoleAnnotation(Role.PARAM)


class _JacAnnotation(RoleAnnotation):
    def shape(self, *dims: int) -> Type:
        """`Jac.shape(2, 5, 5)`: a block shape for Jacobian datasets."""
        return Type(self.role, tuple(dims))


Jac = _JacAnnotation(Role.JAC)
JacAccum = _JacAnnotation(Role.JAC_ACCUM)


def as_type(annotation: object) -> Type:
    if isinstance(annotation, RoleAnnotation):
        return annotation.scalar
    if isinstance(annotation, Type):
        return annotation
    raise CodegenError(Code.NOT_A_DSL_TYPE, f"not a DSL type: {annotation!r}")


# ---------------------------------------------------------------------------- mesh data


class MeshSet(enum.Enum):
    CELLS = "cells"
    FACES = "faces"
    INCIDENCES = "incidences"
    STENCIL = "stencil"  # entries of the cell -> cell stencil (Topology::stencil)


@dataclass(eq=False)
class Dataset:
    """An array over a mesh set. Geometry datasets are filled by the runtime and are passive."""

    name: str
    set: MeshSet
    type: Type
    geometry: bool = False

    @property
    def left(self) -> Binding:
        return Binding(self, "left")

    @property
    def right(self) -> Binding:
        return Binding(self, "right")

    @property
    def self_(self) -> Binding:
        return Binding(self, "self")

    @property
    def near(self) -> Binding:
        """In an incidence reduction, half [0] of a face dataset of shape (2, ...) if the cell is
        the face's left cell, half [1] if it is the right one, in the face's global orientation
        (Topology::face_reversed); zero on boundary faces."""
        return Binding(self, "near")

    @property
    def far(self) -> Binding:
        """The half `near` does not take: [1] for the left cell, [0] for the right one; zero on
        boundary faces."""
        return Binding(self, "far")

    @property
    def on_boundary(self) -> Binding:
        """In an incidence reduction, the face's value on a boundary face, zero on an interior
        face."""
        return Binding(self, "on_boundary")

    def other(self, boundary: Dataset | None) -> Binding:
        """Value across each incidence of a cell; `boundary` (a face dataset) on boundary faces,
        zero when None."""
        return Binding(self, "other", boundary)


@dataclass(eq=False)
class Parameter:
    name: str
    type: Type


@dataclass(frozen=True, eq=False)
class Binding:
    """How a body argument is loaded: directly, or through the face/incidence -> cell maps."""

    dataset: Dataset | Parameter
    via: str = "direct"  # direct | left | right | self | other | near | far | on_boundary
    boundary: Dataset | None = None


class _SetDecl:
    def __init__(self, s: MeshSet) -> None:
        self._set = s

    def field(self, name: str, annotation: object) -> Dataset:
        return Dataset(name, self._set, as_type(annotation))


cells = _SetDecl(MeshSet.CELLS)
faces = _SetDecl(MeshSet.FACES)
incidences = _SetDecl(MeshSet.INCIDENCES)


def param(name: str, annotation: object) -> Parameter:
    t = as_type(annotation)
    if t.role is not Role.PARAM:
        raise CodegenError(Code.NOT_A_DSL_TYPE, f"parameter {name} must have role Param")
    return Parameter(name, t)


class geometry:
    """FP64 geometry datasets, computed by the runtime from the core's Geometry."""

    face_area = Dataset("face_area", MeshSet.FACES, Geom.Vec3, geometry=True)
    face_centroid = Dataset("face_centroid", MeshSet.FACES, Geom.Vec3, geometry=True)
    # Sum over the face's fan triangles of (triangle centroid - face centroid) (x) area vector:
    # zero on planar faces, makes the flux of a linear field through a warped face exact.
    face_moment = Dataset("face_moment", MeshSet.FACES, Geom.Mat3, geometry=True)
    face_normal = Dataset("face_normal", MeshSet.FACES, Geom.Vec3, geometry=True)
    face_area_magnitude = Dataset("face_area_magnitude", MeshSet.FACES, Geom.scalar, geometry=True)
    # Face centroid minus owner (left) and neighbour (right) centroid; right is zero on boundaries.
    face_r_left = Dataset("face_r_left", MeshSet.FACES, Geom.Vec3, geometry=True)
    face_r_right = Dataset("face_r_right", MeshSet.FACES, Geom.Vec3, geometry=True)
    cell_centroid = Dataset("cell_centroid", MeshSet.CELLS, Geom.Vec3, geometry=True)
    cell_volume = Dataset("cell_volume", MeshSet.CELLS, Geom.scalar, geometry=True)
    lsq_weight = Dataset("lsq_weight", MeshSet.INCIDENCES, Geom.Vec3, geometry=True)
    # +1 where the cell is its face's left (owner) cell, and on boundary faces; -1 where it is
    # the right one: the sign of a face's contribution to the cell in a gather over incidences.
    incidence_side = Dataset("incidence_side", MeshSet.INCIDENCES, Geom.scalar, geometry=True)
    # The same least-squares weights over the entries of the cell stencil.
    stencil_lsq_weight = Dataset("stencil_lsq_weight", MeshSet.STENCIL, Geom.Vec3, geometry=True)


# ---------------------------------------------------------------------------- bodies


@dataclass(eq=False)
class Body:
    """A pointwise function in the restricted subset; parsed lazily into the IR."""

    fn: Callable[..., Any]
    compute: str = "compute"

    @property
    def name(self) -> str:
        return self.fn.__name__

    @property
    def source_file(self) -> str:
        return inspect.getsourcefile(self.fn) or "<unknown>"

    def __call__(self, *_args: object) -> object:
        raise CodegenError(
            Code.UNSUPPORTED_CONSTRUCT,
            f"{self.name} is a DSL body; it can only be called from another",
        )


def pointwise(fn: Callable[..., Any] | None = None, *, compute: str = "compute") -> Any:
    """Declare a pointwise body. Its source is parsed into the generator's IR.

    `@pointwise(compute="accum")` computes the body in the accumulation type of the policy
    instead of the compute type (FP64 under fp32+acc64), for bodies whose arguments are all
    passive (blocks, geometry, parameters) in cell loops.
    """
    if compute not in ("compute", "accum"):
        raise CodegenError(Code.TYPE_MISMATCH, f"compute={compute!r}: 'compute' or 'accum'")
    if fn is None:
        return lambda f: Body(f, compute)
    return Body(fn, compute)


# ---------------------------------------------------------------------------- loops


@dataclass(frozen=True, eq=False)
class Inc:
    """Increment a cell dataset through the face's owner (`left`) and neighbour (`right`)."""

    dataset: Dataset
    left: int = 1
    right: int | None = None


@dataclass(frozen=True, eq=False)
class Write:
    dataset: Dataset


@dataclass(frozen=True, eq=False)
class Jacobian:
    """Write, per face (face and boundary loops) or per cell (cell loops), the derivative of
    the body's outputs with respect to the arguments `wrt`, the other arguments held fixed:
    shape (n_outputs, n_inputs) for one argument, (len(wrt), n_outputs, n_inputs) for several,
    row-major. Computed by a forward derivative of the body alongside it; not differentiated."""

    dataset: Dataset
    wrt: tuple[str, ...]

    def __init__(self, dataset: Dataset, wrt: str | tuple[str, ...]) -> None:
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "wrt", (wrt,) if isinstance(wrt, str) else tuple(wrt))


@dataclass(frozen=True, eq=False)
class Sum:
    dataset: Dataset
    write: bool = True


@dataclass(frozen=True, eq=False)
class Min:
    dataset: Dataset
    write: bool = True


@dataclass(frozen=True, eq=False)
class Max:
    dataset: Dataset
    write: bool = True


@dataclass(frozen=True, eq=False)
class Follow:
    """Takes the incidence value of accumulator `leader` (a Min or Max) whenever it selects one:
    the tangent of a Min or Max reduction. Generated by the AD, not written by kernel authors."""

    dataset: Dataset
    leader: int
    write: bool = True


Accumulator = Sum | Min | Max | Follow


@dataclass(eq=False)
class Loop:
    kind: str  # face | boundary | cell | reduce
    body: Body
    bindings: dict[str, Binding]
    outputs: list[Inc | Write | Accumulator]
    finalise: Body | None = None
    finalise_bindings: dict[str, Binding] = field(default_factory=dict)
    name: str = ""
    over: str = "incidences"  # reductions: "incidences" (faces of the cell) or "stencil"
    jacobians: list[Jacobian] = field(default_factory=list)
    frozen: bool = False  # boundary_write reading active data: derivatives through it dropped
    entries: list[Write] = field(default_factory=list)  # reductions: one value per entry
    derivatives: bool = True  # False: no tangent or adjoint (linear-algebra loops)
    listed: bool = False  # reductions over the cells listed at launch, in place


def _bindings(raw: dict[str, object]) -> dict[str, Binding]:
    out: dict[str, Binding] = {}
    for k, v in raw.items():
        if isinstance(v, Binding):
            out[k] = v
        elif isinstance(v, Dataset | Parameter):
            out[k] = Binding(v)
        else:
            raise CodegenError(
                Code.INVALID_BINDING, f"argument {k}: not a dataset, binding or parameter: {v!r}"
            )
    return out


def _outs(out: object) -> list[Any]:
    return list(out) if isinstance(out, list | tuple) else [out]


def _split(out: object) -> tuple[list[Any], list[Jacobian]]:
    outs = _outs(out)
    return [o for o in outs if not isinstance(o, Jacobian)], [
        o for o in outs if isinstance(o, Jacobian)
    ]


def face_loop(body: Body, *, out: object, **args: object) -> Loop:
    """Loop over interior faces; `out` increments owner and neighbour cells, and may add
    Jacobian blocks per face."""
    outs, jacs = _split(out)
    for o in outs:
        if not isinstance(o, Inc) or o.right is None:
            raise CodegenError(
                Code.INVALID_LOOP_OUTPUT, "interior face loop outputs are Inc with both signs"
            )
    return Loop("face", body, _bindings(args), outs, jacobians=jacs)


def boundary_loop(body: Body, *, out: object, frozen: bool = False, **args: object) -> Loop:
    """Loop over boundary faces; `out` increments the owner cell (`Inc`) or writes face datasets
    (`Write`, for example boundary states). Face datasets are passive, so a writing loop has no
    tangent or adjoint: if it reads active cell data, declare `frozen=True` to accept that the
    derivatives of what it writes are dropped (the generated module lists it in
    `frozen_loops`, so runs can label their adjoints approximate)."""
    outs, jacs = _split(out)
    if outs and all(isinstance(o, Write) and o.dataset.set is MeshSet.FACES for o in outs):
        if jacs:
            raise CodegenError(
                Code.INVALID_LOOP_OUTPUT,
                "a boundary loop writing face data has no Jacobian outputs",
            )
        return Loop("boundary_write", body, _bindings(args), outs, frozen=frozen)
    for o in outs:
        if not isinstance(o, Inc) or o.right is not None:
            raise CodegenError(
                Code.INVALID_LOOP_OUTPUT,
                "boundary loop outputs are Inc of the owner only, or Write of face datasets only",
            )
    return Loop("boundary", body, _bindings(args), outs, jacobians=jacs)


def cell_loop(body: Body, *, out: object, **args: object) -> Loop:
    """Loop over owned cells; `out` writes cell datasets, and may add Jacobian blocks per cell."""
    outs, jacs = _split(out)
    for o in outs:
        if not isinstance(o, Write):
            raise CodegenError(Code.INVALID_LOOP_OUTPUT, "cell loop outputs are Write")
    return Loop("cell", body, _bindings(args), outs, jacobians=jacs)


INCIDENCES = "incidences"
STENCIL = "stencil"


def cell_reduce(
    body: Body,
    *,
    acc: object,
    finalise: Body | None = None,
    finalise_args: dict[str, object] | None = None,
    over: str = INCIDENCES,
    entries: object = (),
    derivatives: bool = True,
    listed: bool = False,
    **args: object,
) -> Loop:
    """Reduce `body` over the entries of each owned cell into accumulators, then write them.

    `over=INCIDENCES` iterates the faces of the cell: `d.other(boundary=fb)` is the cell across
    the face, or `fb` on a boundary face, and face data are those of that face. `over=STENCIL`
    iterates the cell stencil (Topology::stencil, built by autan::kernels::cell_stencil): its
    cells, then boundary faces; `d.other(boundary=fb)` is the stencil cell or `fb` of the face, and
    per-entry data live on `MeshSet.STENCIL` (geometry.stencil_lsq_weight).
    The body's outputs feed `acc` in order (Sum, Min or Max, each naming its target cell dataset).
    With `finalise`, the accumulated values are passed to it as its first arguments, followed by
    `finalise_args`, and its outputs are written to the accumulators' datasets instead.
    `entries` (Write outputs to datasets of the iterated set, of a passive role) take the body's
    outputs after the accumulators' ones, one value per entry, stored at the entry: each entry
    is written once, by its cell. A loop with entries has no active input (no derivatives).
    `derivatives=False` generates no tangent or adjoint (linear-algebra loops); its bodies may
    then compute in the accumulation type on Residual data too. `listed=True` runs over the
    cells listed at launch (`name(d, rows, n_rows)`), and may write what it reads across faces:
    the caller lists cells no two of which are neighbours (one colour of a colouring); it has
    no derivatives.
    """
    if over not in (INCIDENCES, STENCIL):
        raise CodegenError(
            Code.INVALID_BINDING, f"cell_reduce over {over!r}: incidences or stencil"
        )
    accs = _outs(acc)
    for a in accs:
        if not isinstance(a, Sum | Min | Max):
            raise CodegenError(
                Code.INVALID_LOOP_OUTPUT, "cell_reduce accumulators are Sum, Min or Max"
            )
    if listed and (over != INCIDENCES or derivatives):
        raise CodegenError(
            Code.INVALID_LOOP_OUTPUT,
            "listed reductions run over incidences and have no derivatives (derivatives=False)",
        )
    per_entry = _outs(entries) if entries != () else []
    for w in per_entry:
        if not isinstance(w, Write) or w.dataset.set.value != over:
            raise CodegenError(
                Code.INVALID_LOOP_OUTPUT,
                f"cell_reduce entries are Write outputs to {over} datasets",
            )
    return Loop(
        "reduce",
        body,
        _bindings(args),
        accs,
        finalise,
        _bindings(dict(finalise_args or {})),
        over=over,
        entries=per_entry,
        derivatives=derivatives,
        listed=listed,
    )


# ---------------------------------------------------------------------------- functions


def _intrinsic(*_: object) -> object:
    raise CodegenError(Code.UNSUPPORTED_CONSTRUCT, "DSL intrinsics only exist inside parsed bodies")


dot = cross = norm = sqrt = exp = log = sin = cos = rel = _intrinsic
transpose = inverse = identity = outer = Vec3 = Vec5 = _intrinsic
