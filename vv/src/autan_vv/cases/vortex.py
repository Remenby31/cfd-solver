"""The convected isentropic vortex of the `vortex-order` gate (acceptance.md, M2): one period of the
periodic square [-10, 10]^2, t = 40 at Mach 0.5 (Spiegel, Huynh and DeBonis, AIAA 2015-2444).

`case` builds the autan.case/1 case of one level of a `vortex-hex` or `vortex-prism` mesh family
(`autan_vv.meshes.vortex`), started from the oracle at t = 0 written by
`autan_vv.initial_state.write_initial_state`. The oracle's gas has R = 1/gamma, so that the
free-stream speed of sound is 1 at T = 1 K, density 1 and pressure 1/gamma.
"""

from __future__ import annotations

from typing import Literal

import autan
from autan.units import K, Pa, kg, mol, s

from autan_vv.harness import UNIVERSAL_GAS_CONSTANT
from autan_vv.oracles.inviscid import GAMMA

MACH = 0.5
PERIOD = 40.0  # the domain's 20 crossed at the free-stream speed 0.5
# Time steps per in-plane cell along x of the finest level over a period: dt = 2/n, a CFL number
# of about 0.4 for the fastest wave (free stream plus the vortex's peak speed) on quadrilaterals.
# Every level takes the finest level's step, so the levels solve one case and differ by their mesh.
STEPS_PER_CELL = 20


def case(
    mesh: str,
    n: int,
    initial_state: autan.StateFile,
    *,
    finest: int,
    precision: Literal["fp64", "fp32+acc64"] = "fp64",
) -> autan.Case:
    """The case of level `n` (n cells across the square) whose mesh is `mesh`, in a family whose
    finest level is `finest`: one period of the vortex with the finest level's fixed step, so that
    every level ends at t = 40 exactly."""
    result = autan.Case(f"isentropic-vortex-n{n:03d}", mesh=mesh)
    gas = autan.IdealGas(
        name="vortex",
        molar_mass=UNIVERSAL_GAS_CONSTANT * GAMMA * kg / mol,
        gamma=GAMMA,
        prandtl=0.72,
        viscosity=autan.ConstantViscosity(mu=1e-5 * Pa * s),
    )
    result.physics = autan.Compressible(turbulence=autan.Inviscid(), precision=precision, gas=gas)
    result.freestream(mach=MACH, pressure=1.0 / GAMMA * Pa, temperature=1.0 * K)
    steps = STEPS_PER_CELL * finest
    result.time = autan.Unsteady(steps=steps, dt=PERIOD / steps * s)
    result.boundary("zmin").symmetry()
    result.boundary("zmax").symmetry()
    result.initial_state = initial_state
    return result
