#!/bin/sh
# Writes run directories of the real runtime with the Euler physics, for the UI: a manufactured
# solution on the hybrid mesh, a two-zone Sod shock tube, and a NACA 0012 slab at four angles of
# attack (a polar for Compare). Meant for a DGX Spark through docker/spark.sh after
#   cmake --build build --target autan-run autan-mesh-info
# with the NACA mesh written by ui/scripts/naca0012_mesh.py, at build/naca0012.msh by default.
set -eu
runs=${1:?usage: euler-runs.sh RUNS [NACA_MESH]}
repo=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
hybrid="$repo/tests/data/mesh/hybrid/polyMesh"
mesh=${2:-$repo/build/naca0012.msh}
folder=$(CDPATH='' cd -- "$(dirname -- "$mesh")" && pwd)
export NACA_MESH="$folder/$(basename -- "$mesh")"
export AUTAN_RUN="$repo/build/runtime/autan-run"
info() { "$repo/build/io/autan-mesh-info" --json "$1" > "$2"; }
autan() { uv run --locked --project "$repo/ui/server" autan "$@"; }
info "$hybrid" "$work/hybrid.meshinfo.json"
info "$NACA_MESH" "$work/naca.meshinfo.json"
cp "$repo/py/tests/fixtures/euler.mms.json" "$work/"
digest=$(sha256sum "$work/euler.mms.json" | cut -d' ' -f1)

cat > "$work/mms.py" <<CASE
import autan
from autan.units import K, Pa, kg, mol, s
gas = autan.IdealGas(name="mms", molar_mass=8.314462618 / 287.0 * kg / mol, gamma=1.4,
                     prandtl=0.72, viscosity=autan.ConstantViscosity(mu=1e-3 * Pa * s))
case = autan.Case("euler-mms", mesh="$hybrid")
case.physics = autan.Compressible(turbulence=autan.Inviscid(), gas=gas)
case.freestream(mach=0.3, pressure=101325 * Pa, temperature=300 * K)
case.verification = autan.Manufactured(path="euler.mms.json", sha256="$digest")
CASE

cat > "$work/sod.py" <<CASE
import autan
from autan.units import K, Pa, m, s
case = autan.Case("sod", mesh="$hybrid")
case.physics = autan.Compressible(turbulence=autan.Inviscid())
case.freestream(mach=0.01, pressure=1e4 * Pa, temperature=278.7 * K)
case.time = autan.Unsteady(steps=400, cfl=0.5)
still = (0 * m / s, 0 * m / s, 0 * m / s)
case.cell_zone("lower").fluid(autan.gases.AIR, initial=autan.State(pressure=1e5 * Pa, temperature=348.4 * K, velocity=still))
case.cell_zone("upper").fluid(autan.gases.AIR, initial=autan.State(pressure=1e4 * Pa, temperature=278.7 * K, velocity=still))
case.boundary("*").slip()
CASE

for aoa in 0 2 4 6; do
cat > "$work/naca_$aoa.py" <<CASE
import os
import autan
from autan.units import K, Pa, deg, m
case = autan.Case("naca0012", mesh=os.environ["NACA_MESH"])
case.physics = autan.Compressible(turbulence=autan.Inviscid())
case.freestream(mach=0.5, aoa=$aoa * deg, pressure=101325 * Pa, temperature=288.15 * K, ref_length=1.0 * m)
case.time = autan.Steady()
case.boundary("airfoil").slip()
case.boundary("farfield").farfield()
case.boundary("symmetry").symmetry()
case.monitor(autan.forces("airfoil", ref_area=0.1 * m**2), every=10)
CASE
done

run() { autan run "$work/$1.py" --mesh-info "$work/$2.meshinfo.json" --backend cpu --runs "$runs"; }
run mms hybrid
run sod hybrid
for aoa in 0 2 4 6; do run "naca_$aoa" naca; done
rm -r "$work"
