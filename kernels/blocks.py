"""Block-sparse linear algebra of the implicit solvers: 5 x 5 blocks per cell and per incidence.

The physics gives, per interior face, the derivatives of the face flux with respect to its
owner's and its neighbour's states (`face_jac`, [0] and [1]) and, per boundary face, with respect
to its owner's (`bnd_jac`). `assemble` turns them into the rows of the matrix: per cell, the
diagonal block (owner +face_jac[0], neighbour -face_jac[1], boundary bnd_jac, in the order of
the cell's faces, then the pseudo-time shift) and one off-diagonal block per incidence (`off`:
owner +face_jac[1], neighbour -face_jac[0], zero on a boundary face), the coupling to the cell
across that face, physics BlockMatrix upper and lower. With the sliced incidence layout
(Topology::incidence_stride) the rows are a sliced block CSR whose column index is
Topology::incidence_cell. The diagonal blocks are inverted in the accumulation type (FP64 under
fp32+acc64), with the operation order of the host BlockMatrix (physics/src/block_matrix.cpp).
"""

from autan_codegen.dsl import (
    Geom,
    Jac,
    JacAccum,
    Residual,
    Sum,
    Write,
    cell_loop,
    cell_reduce,
    cells,
    faces,
    geometry,
    identity,
    incidences,
    inverse,
    pointwise,
)

face_jac = faces.field("face_jac", Jac.shape(2, 5, 5))
bnd_jac = faces.field("bnd_jac", Jac.Mat5)
shift = cells.field("shift", JacAccum)
diag = cells.field("diag", JacAccum.Mat5)
diag_inv = cells.field("diag_inv", JacAccum.Mat5)
off = incidences.field("off", Jac.Mat5)
x = cells.field("x", Residual.Vec5)
y = cells.field("y", Residual.Vec5)


@pointwise
def blocks_of_face(
    jn: Jac.Mat5, jf: Jac.Mat5, jb: Jac.Mat5, s: Geom
) -> tuple[JacAccum.Mat5, Jac.Mat5]:
    # Exact: products by +-1 and sums with zero.
    return s * jn + jb, s * jf


@pointwise(compute="accum")
def with_shift(d: Residual.Mat5, sh: JacAccum) -> JacAccum.Mat5:
    return d + sh * identity(5)


assemble = cell_reduce(
    blocks_of_face,
    jn=face_jac.near,
    jf=face_jac.far,
    jb=bnd_jac.on_boundary,
    s=geometry.incidence_side,
    acc=Sum(diag),
    entries=Write(off),
    finalise=with_shift,
    finalise_args={"sh": shift},
)


@pointwise(compute="accum")
def invert(a: JacAccum.Mat5) -> JacAccum.Mat5:
    return inverse(a)


invert_diagonal = cell_loop(invert, a=diag, out=Write(diag_inv))


@pointwise
def coupling(b: Jac.Mat5, xo: Residual.Vec5) -> Residual.Vec5:
    return b @ xo


@pointwise
def with_diagonal(acc: Residual.Vec5, a: JacAccum.Mat5, xc: Residual.Vec5) -> Residual.Vec5:
    return acc + a @ xc


# y = A x: each cell's off-diagonal blocks in incidence order, then its diagonal block.
multiply = cell_reduce(
    coupling,
    b=off,
    xo=x.other(None),
    acc=Sum(y),
    finalise=with_diagonal,
    finalise_args={"a": diag, "xc": x},
)
