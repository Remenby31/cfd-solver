import ast
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import autan
import pytest
from autan.units import Quantity as Q

from tests.conftest import EXAMPLES

SHA = "0" * 64

# Canonical code of one case per feature. Each is written by hand from the rules of the
# canonical form: statement order, only non-default arguments, units as given, library
# materials by name, the shortest repr of each number.
CASES = {
    "conditions": f"""\
import autan
from autan.units import K, Pa, kPa, m

case = autan.Case("conditions", mesh="crm-hl.cgns")
case.physics = autan.Compressible(turbulence=autan.Inviscid(), precision="fp64")
case.freestream(mach=0.5, temperature=250.0 * K, pressure=50.0 * kPa)
case.boundary("hot").wall(temperature=400.0 * K)
case.boundary("wing").wall()
case.boundary("flap").slip()
case.boundary("sym").symmetry()
case.boundary("far").farfield(
    point_vortex=autan.PointVortex(surfaces="wing*", center=(0.25 * m, 0.0 * m, 0.0 * m))
)
case.boundary("far2").farfield()
case.boundary("in").inlet(total_pressure=60.0 * kPa, total_temperature=260.0 * K)
case.boundary("out").outlet(static_pressure=50000.0 * Pa)
case.boundary("box").oracle_state(
    path="box.face-states.json",
    sha256="{SHA}",
)
""",
    "incompressible": """\
import autan
from autan.units import Pa, kg, m, s

case = autan.Case("incompressible", mesh="crm-hl.cgns")
case.physics = autan.Incompressible(
    turbulence=autan.SST(intensity=0.01),
    fluid=autan.ConstantDensityFluid(
        name="water", density=998.0 * kg / m**3, viscosity=0.001 * Pa * s
    ),
)
case.freestream(velocity=2.0 * m / s)
case.boundary("in").velocity_inlet(velocity=2.0 * m / s)
case.boundary("out").outlet(static_pressure=101325.0 * Pa)
""",
    "materials": """\
import autan
from autan.units import J, K, Pa, W, deg, kg, m, mol, s

case = autan.Case("materials", mesh="heat-exchanger.msh")
case.physics = autan.Compressible(
    turbulence=autan.DDES(base=autan.SA(nu_tilde_ratio=5.0)),
    gas=autan.IdealGas(
        name="argon",
        molar_mass=0.039948 * kg / mol,
        gamma=1.667,
        prandtl=0.67,
        viscosity=autan.ConstantViscosity(mu=2.2e-05 * Pa * s),
    ),
)
case.freestream(mach=0.3, aoa=2.5 * deg, sideslip=-1.0 * deg, re=1000000.0, ref_length=0.5 * m)
case.time = autan.Unsteady(steps=200, dt=0.0001 * s, average_from=100)
case.cell_zone("air").fluid(
    autan.gases.AIR,
    initial=autan.State(
        pressure=101325.0 * Pa,
        temperature=300.0 * K,
        velocity=(10.0 * m / s, 0.0 * m / s, 0.0 * m / s),
    ),
)
case.cell_zone("hot").fluid(
    autan.IdealGas(
        name="helium",
        molar_mass=0.004003 * kg / mol,
        gamma=1.667,
        prandtl=0.68,
        viscosity=autan.Sutherland(mu_ref=1.87e-05 * Pa * s, t_ref=273.0 * K, s=79.4 * K),
    )
)
case.cell_zone("plate").solid(autan.materials.ALUMINIUM)
case.cell_zone("wall").solid(
    autan.Solid(
        name="steel",
        density=7850.0 * kg / m**3,
        specific_heat=490.0 * J / K / kg,
        conductivity=45.0 * W / K / m,
    )
)
case.boundary("*").wall()
""",
    "files": f"""\
import autan
from autan.units import Pa, m, mm

case = autan.Case("files", mesh="crm-hl.cgns")
case.physics = autan.Compressible(turbulence=autan.Laminar())
case.freestream(mach=0.2, pressure=101325.0 * Pa)
case.time = autan.Unsteady(steps=50, cfl=4.0)
case.initial_state = autan.StateFile(
    path="start.initial-state.h5",
    sha256="{SHA}",
)
case.boundary("*").farfield()
case.monitor(autan.forces("wing*"))
case.monitor(
    autan.forces(
        "flap*", name="flap", ref_area=2.0 * m**2, moment_center=(250.0 * mm, 0.0 * m, 0.0 * m)
    ),
    every=5,
)
""",
    "verification": f"""\
import autan
from autan.units import Pa

case = autan.Case("verification", mesh="crm-hl.cgns")
case.physics = autan.Compressible(turbulence=autan.Inviscid())
case.freestream(mach=0.2, pressure=101325.0 * Pa)
case.verification = autan.Manufactured(
    path="euler.mms.json", sha256="{SHA}"
)
""",
}


def body(source: str) -> str:
    first = ast.parse(source).body[0]
    assert isinstance(first, ast.Expr)
    assert first.end_lineno is not None
    return "".join(source.splitlines(keepends=True)[first.end_lineno + 1 :])


@pytest.mark.parametrize("example", sorted(path.name for path in EXAMPLES.glob("*.py")))
def test_examples_are_canonical(example: str) -> None:
    source = (EXAMPLES / example).read_text()
    assert autan.load(EXAMPLES / example).to_python() == body(source)


@pytest.mark.parametrize("name", sorted(CASES))
def test_canonical_code_survives_load_toml_and_json(
    name: str, write_case: Callable[..., Path]
) -> None:
    source = CASES[name]
    case = autan.load(write_case(source))
    assert case.to_python() == source
    for suffix, text in ((".toml", case.to_toml()), (".json", case.to_json())):
        reloaded = autan.load(write_case(text, f"case{suffix}"))
        assert reloaded.spec == case.spec
        assert reloaded.to_python() == source


def ruff_clean(source: str) -> bool:
    ruff = [sys.executable, "-m", "ruff"]
    imports = subprocess.run(
        [*ruff, "check", "--select", "I,F401", "--stdin-filename", "case.py"],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    formatted = subprocess.run(
        # Without magic trailing commas, ruff lays the code out from scratch.
        [
            *ruff,
            "format",
            "--line-length",
            "100",
            "--config",
            "format.skip-magic-trailing-comma = true",
            "-",
        ],
        input=source,
        capture_output=True,
        text=True,
        check=True,
    )
    return imports.returncode == 0 and formatted.stdout == source


@pytest.mark.parametrize("name", sorted(CASES))
def test_canonical_code_is_ruff_clean(name: str) -> None:
    assert ruff_clean(CASES[name])


def co2_viscosity_in_base_units(case: autan.Case) -> None:
    gas = autan.gases.CO2.model_dump()
    gas["viscosity"]["mu_ref"] = gas["viscosity"]["mu_ref"].to("kg / m / s")
    case.cell_zone("co2").fluid(autan.IdealGas(**gas))


# Cases built in Python whose values the canonical form must keep as the document writes them:
# units autan.units does not export, values equal to a default or a library material in another
# unit or sign, characters a Python string must escape, calls too long for one line.
EDGES: dict[str, Callable[[autan.Case], None]] = {
    "micrometre": lambda case: case.freestream(
        mach=0.2, pressure=Q(101325, "Pa"), ref_length=Q(1, "um")
    ),
    "angstrom": lambda case: case.freestream(
        mach=0.2, pressure=Q(101325, "Pa"), ref_length=Q(1, "angstrom")
    ),
    "percent": lambda case: case.freestream(
        mach=0.2, pressure=Q(101325, "Pa"), ref_length=Q(1, "percent * m")
    ),
    "integer": lambda case: case.freestream(
        mach=0.2, pressure=Q(101325, "Pa"), ref_length=Q(2**53 + 1, "m")
    ),
    "aoa-radian": lambda case: case.freestream(mach=0.2, pressure=Q(101325, "Pa"), aoa=Q(0, "rad")),
    "aoa-negative-zero": lambda case: case.freestream(
        mach=0.2, pressure=Q(101325, "Pa"), aoa=Q(-0.0, "deg")
    ),
    "celsius": lambda case: case.freestream(
        mach=0.2, pressure=Q(101325, "Pa"), temperature=Q(15, "degC")
    ),
    "library-units": co2_viscosity_in_base_units,
    "long-zone": lambda case: case.boundary("w" * 80).wall(),
    "empty-call-at-the-width": lambda case: case.boundary("w" * 77).wall(),
    "long-zone-arguments": lambda case: case.boundary("w" * 80).wall(temperature=Q(400, "K")),
    "long-zone-long-arguments": lambda case: case.cell_zone("w" * 80).fluid(
        autan.gases.AIR,
        initial=autan.State(
            pressure=Q(101325, "Pa"), temperature=Q(300, "K"), velocity=(Q(1, "m/s"),) * 3
        ),
    ),
    "broken-tuple": lambda case: case.monitor(
        autan.forces("*", moment_center=(Q(1.23456789012345e-100, "m"),) * 3)
    ),
    "positive-exponent": lambda case: case.freestream(
        mach=0.2, pressure=Q(101325, "Pa"), ref_length=Q(1e20, "m")
    ),
    "long-unit": lambda case: case.freestream(
        mach=0.2,
        pressure=Q(101325, "Pa"),
        ref_length=Q(
            1,
            "kPa**0.8125 * m**0.65625 * s**3.09375 * W**0.59375 / J**0.0625 / kg**1.25"
            " / Pa**0.09375",
        ),
    ),
}
NAMES = {
    "quote": 'a"b',
    "both-quotes": "a\"b'c",
    "more-double-quotes": 'a""b\'c',
    "carriage-return": "a\rb",
    "nul": "a\x00b",
    "wide": "n" + "\u754c" * 40,
    "combining": "e\u0301" * 55,
}


MESH_UNITS = {"mesh-in-inches": "in", "mesh-in-micrometres": "um"}


def edge_case(edge: str) -> autan.Case:
    unit = MESH_UNITS.get(edge, "m")
    case = autan.Case(NAMES.get(edge, "edge"), mesh="crm-hl.cgns", mesh_unit=unit)
    case.physics = autan.Compressible(turbulence=autan.SA())
    case.freestream(mach=0.2, pressure=Q(101325, "Pa"))
    if edge in EDGES:
        EDGES[edge](case)
    return case


@pytest.mark.parametrize("edge", sorted(EDGES) + sorted(NAMES) + sorted(MESH_UNITS))
def test_rendering_keeps_the_document(edge: str, write_case: Callable[..., Path]) -> None:
    case = edge_case(edge)
    source = case.to_python()
    reloaded = autan.load(write_case(source))
    assert reloaded.spec == case.spec
    assert reloaded.spec.hash() == case.spec.hash()
    assert ruff_clean(source)


def test_rendered_code_type_checks_against_the_builders(tmp_path: Path) -> None:
    for edge in sorted(EDGES) + sorted(NAMES) + sorted(MESH_UNITS):
        (tmp_path / f"{edge.replace('-', '_')}.py").write_text(edge_case(edge).to_python())
    config = Path(__file__).parents[2] / "pyproject.toml"
    checked = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(config), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout
