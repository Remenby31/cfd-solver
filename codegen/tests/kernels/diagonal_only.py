"""kernels/blocks.py `assemble` without its per-incidence blocks: the diagonal of the same body,
for the benchmark that measures what writing the off-diagonal blocks costs the assembly."""

from autan_codegen.dsl import (
    Geom,
    Jac,
    JacAccum,
    Residual,
    Sum,
    cell_reduce,
    cells,
    faces,
    geometry,
    identity,
    pointwise,
)

face_jac = faces.field("face_jac", Jac.shape(2, 5, 5))
bnd_jac = faces.field("bnd_jac", Jac.Mat5)
shift = cells.field("shift", JacAccum)
diag = cells.field("diag", JacAccum.Mat5)


@pointwise
def diagonal_of_face(jn: Jac.Mat5, jb: Jac.Mat5, s: Geom) -> JacAccum.Mat5:
    return s * jn + jb


@pointwise(compute="accum")
def with_shift(d: Residual.Mat5, sh: JacAccum) -> JacAccum.Mat5:
    return d + sh * identity(5)


assemble = cell_reduce(
    diagonal_of_face,
    jn=face_jac.near,
    jb=bnd_jac.on_boundary,
    s=geometry.incidence_side,
    acc=Sum(diag),
    finalise=with_shift,
    finalise_args={"sh": shift},
)
