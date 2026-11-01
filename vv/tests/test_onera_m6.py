"""ONERA M6: geometry against its published definition, meshes, and the section comparators
(Schmitt and Charpin, AGARD AR-138, 1979; Destarac and Dumont, TMR, 2016)."""

import math
from pathlib import Path
from typing import Literal

import autan
import numpy as np
import pytest
from autan.units import Pa, deg, m
from autan_vv.cases import fetch, get_case, read_columns
from autan_vv.cases import onera_m6 as cases
from autan_vv.errors import VVError
from autan_vv.meshes import onera_m6
from autan_vv.meshes.facemesh import FaceMesh
from autan_vv.runs import mesh_info, write_run

gmsh = pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")


def test_planform_reproduces_the_published_figures() -> None:
    b, c0 = onera_m6.SEMISPAN, onera_m6.ROOT_CHORD
    c_tip = float(onera_m6.chord(b))
    # Sweeps 30 deg and 15.8 deg, taper 0.562 (Schmitt and Charpin).
    assert math.degrees(math.atan(float(onera_m6.leading_edge(1.0)))) == pytest.approx(
        30.0, abs=0.01
    )
    te_sweep = math.atan((float(onera_m6.leading_edge(b)) + c_tip - c0) / b)
    assert math.degrees(te_sweep) == pytest.approx(15.8, abs=0.05)
    assert c_tip / c0 == pytest.approx(0.562, abs=6e-4)
    # The reference area and mean aerodynamic chord are those of the trapezoid.
    assert b * (c0 + c_tip) / 2.0 == pytest.approx(onera_m6.REFERENCE_AREA, abs=5e-5)
    taper = c_tip / c0
    mac = 2.0 / 3.0 * c0 * (1 + taper + taper**2) / (1 + taper)
    assert mac == pytest.approx(onera_m6.MEAN_AERODYNAMIC_CHORD, abs=5e-5)
    # The sharp trailing edge extends the chord by about 0.55 % (TMR model).
    assert onera_m6.sharp_section()[-1, 0] - 1.0 == pytest.approx(0.0055, abs=4e-4)


# Rows of AGARD AR-138 table B1-1 typed from Destarac and Dumont (2016), table 1.
TABLE_ROWS = [
    (0.0098413, 0.0168984),
    (0.1649976, 0.0419089),
    (0.3761446, 0.0489296),
    (0.6268104, 0.0387613),
    (0.8868235, 0.0145051),
    (1.0000000, 0.0007052),
]


def test_section_holds_the_published_rows() -> None:
    for row in TABLE_ROWS:
        assert np.any(np.all(row == onera_m6.SECTION, axis=1)), row
    assert onera_m6.SECTION.shape == (72, 2)


@pytest.fixture(scope="module")
def coarse() -> FaceMesh:
    return onera_m6.generate_level(4)[0]


def test_mesh_bounds_the_wing_in_a_half_sphere(coarse: FaceMesh) -> None:
    mesh = coarse
    g = mesh.geometry
    assert g.cell_volume.min() > 0.0
    zones = {z.name: z for z in mesh.zones}
    assert list(zones) == list(onera_m6.ZONES)

    def nodes(name: str) -> np.ndarray:
        z = zones[name]
        ids = np.concatenate([mesh.face(f) for f in range(z.start, z.start + z.count)])
        points: np.ndarray = mesh.points[np.unique(ids)]
        return points

    np.testing.assert_array_equal(nodes("symmetry")[:, 1], 0.0)
    r = np.linalg.norm(nodes("farfield") - np.array(onera_m6.CENTRE), axis=1)
    assert np.all(np.abs(r - onera_m6.FARFIELD_RADIUS) < 1e-6 * onera_m6.FARFIELD_RADIUS)
    wing = nodes("wing")
    # Tip of the half body of revolution: b + the tip section's half thickness.
    s_tip = float(onera_m6.scale(onera_m6.SEMISPAN))
    half = s_tip * onera_m6.SECTION[:, 1].max() * onera_m6.ROOT_CHORD
    assert wing[:, 1].max() == pytest.approx(onera_m6.SEMISPAN + half, abs=2e-3 * half)
    # Wing nodes inside the span lie on the conical surface: z/c on the table, to the chordal
    # deviation of the spline between table points.
    inside = wing[wing[:, 1] < onera_m6.SEMISPAN - 1e-9]
    y = inside[:, 1]
    xc = (inside[:, 0] - onera_m6.leading_edge(y)) / onera_m6.chord(y)
    zc = inside[:, 2] / onera_m6.chord(y)
    sec = onera_m6.sharp_section()
    table = np.interp(np.clip(xc, 0.0, sec[-1, 0]), sec[:, 0], sec[:, 1])
    assert np.max(np.abs(np.abs(zc) - table)) < 2e-4
    # The wing closes the domain: the outward areas of all boundary faces sum to zero.
    total = sum(g.face_area[z.start : z.start + z.count].sum(axis=0) for z in mesh.zones)
    assert np.linalg.norm(total) < 1e-9 * onera_m6.FARFIELD_RADIUS**2


def test_family_is_reproducible_and_refines_by_two(tmp_path: Path) -> None:
    first, meshes = onera_m6.generate_family((3, 6), tmp_path / "a")
    again, _ = onera_m6.generate_family((3,), tmp_path / "b")
    assert again.levels[0] == first.levels[0]
    ratio = (first.levels[1].n_cells / first.levels[0].n_cells) ** (1.0 / 3.0)
    assert 1.7 < ratio < 2.2
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(tmp_path / "a" / first.levels[0].file))
        names = {gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()}
        assert names == {*onera_m6.ZONES, "fluid"}
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(coords).reshape(-1, 3)[np.argsort(node_tags)]
        np.testing.assert_array_equal(xyz, meshes[0].points)
    finally:
        gmsh.finalize()


def test_critical_cp_is_the_sonic_isentropic_pressure() -> None:
    gamma, mach = 1.4, 0.8395
    # p/p0 = (1 + (gamma-1)/2 M^2)^(-gamma/(gamma-1)) at M and at M = 1.
    p_inf = (1 + 0.2 * mach**2) ** -3.5
    p_star = 1.2**-3.5
    q_inf = 0.5 * gamma * p_inf * mach**2
    assert cases.critical_cp(mach) == pytest.approx((p_star - p_inf) / q_inf, rel=1e-14)


def test_shock_is_where_cp_rises_through_the_critical_value() -> None:
    x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    cp = np.array([0.5, -0.8, -0.2, -0.6, -0.7, 0.1, 0.2])
    # Rises through -0.4 between 0.1 and 0.2 and between 0.4 and 0.5.
    assert cases.shock_positions(x, cp, -0.4) == pytest.approx(
        [0.1 + 0.1 * 4 / 6, 0.4 + 0.1 * 3 / 8]
    )
    assert cases.shock_positions(x, cp, -0.9) == []


def test_cut_interpolates_linear_fields_exactly(coarse: FaceMesh) -> None:
    mesh = coarse
    zone = next(z for z in mesh.zones if z.name == "wing")
    faces = [mesh.face(f) for f in range(zone.start, zone.start + zone.count)]
    field = 2.0 * mesh.points[:, 0] - 3.0 * mesh.points[:, 2] + 0.5
    y = 0.44 * onera_m6.SEMISPAN
    xzv = cases.cut(mesh.points, faces, field, y)
    assert xzv.shape[0] > 20
    np.testing.assert_allclose(xzv[:, 2], 2.0 * xzv[:, 0] - 3.0 * xzv[:, 1] + 0.5, atol=1e-14)
    # A constant face field gives the same constant at every node.
    nodes, values = cases.node_values(mesh.points, faces, np.full(len(faces), 0.25))
    np.testing.assert_allclose(values, 0.25, rtol=1e-15)
    assert nodes.size == np.unique(np.concatenate(faces)).size


def _case_without_monitor(
    path: Path, alpha: float = 3.06, precision: Literal["fp64", "fp32+acc64"] = "fp64"
) -> autan.Case:
    """Run 2308 as `cases.case` writes it, without the forces monitor that write_run refuses."""
    case = autan.Case("onera-m6-2308", mesh=str(path))
    case.physics = autan.Compressible(turbulence=autan.Inviscid(), precision=precision)
    case.freestream(mach=0.8395, aoa=alpha * deg, pressure=101325.0 * Pa, ref_length=0.64607 * m)
    case.time = autan.Steady(max_iterations=10, residual_orders=10.0)
    case.boundary("wing").slip()
    case.boundary("symmetry").symmetry()
    case.boundary("farfield").farfield()
    return case


def test_study_cuts_the_wing_of_a_run(tmp_path: Path) -> None:
    path = tmp_path / "m6.msh"
    mesh, _ = onera_m6.generate_level(4, path)
    info = mesh_info(mesh, path, "gmsh")
    fields = {
        "Density": np.ones(mesh.n_cells),
        "VelocityX": np.zeros(mesh.n_cells),
        "VelocityY": np.zeros(mesh.n_cells),
        "VelocityZ": np.zeros(mesh.n_cells),
        "Pressure": np.ones(mesh.n_cells),
        "Temperature": np.ones(mesh.n_cells),
    }
    mach = cases.conditions()["mach"]
    run = write_run(tmp_path / "runs", _case_without_monitor(path), mesh, info, fields)
    experiment = tmp_path / "case_2308.dat"
    lines = ['VARIABLES = "Section", "Tap", "X/L", "Z/L", "CP"']
    for k in range(7):
        lines.append(f'ZONE T="Section {k + 1}", I=4, F=POINT')
        lines += [
            f"{k + 1} 1 0.1 0.05 -0.6",
            f"{k + 1} 2 0.5 0.04 0.0",
            f"{k + 1} 3 0.3 -0.04 0.1",
            f"{k + 1} 4 0.7 -0.03 0.1",
        ]
    experiment.write_text("\n".join(lines) + "\n")
    report = cases.study(run.path, experiment)
    # write_run marks the run converged; its case asks for 10 orders, so the fp64 gate passes.
    assert report["gate"] == {"id": "onera-m6-residual-drop", "pass": True}
    assert report["residual_drop"] == dict.fromkeys(report["residual_drop"], 0.0)
    assert [row["eta"] for row in report["sections"]] == cases.conditions()["eta"]
    # The experiment's rise through Cp* between taps 0.1 and 0.5 (linear interpolation).
    cp_star = cases.critical_cp(mach)
    expected = 0.1 + 0.4 * (cp_star + 0.6) / 0.6
    assert report["sections"][0]["shocks_experiment"] == pytest.approx([expected], rel=1e-12)
    # The wall pressure written is 0 Pa: Cp = -p_inf / q_inf = -2 / (gamma M^2) on every
    # section, which spans the chord from the leading to the sharp trailing edge.
    for station in report["sections_cp"].values():
        for surface in ("upper", "lower"):
            xc_cp = np.array(station[surface])
            np.testing.assert_allclose(xc_cp[:, 1], -2.0 / (1.4 * mach**2), rtol=1e-12)
            assert xc_cp[:, 0].min() < 0.1
            assert xc_cp[:, 0].max() > 0.9
    mixed = _case_without_monitor(path, precision="fp32+acc64")
    mixed_run = write_run(tmp_path / "runs", mixed, mesh, info, fields)
    assert cases.study(mixed_run.path, experiment)["gate"] is None
    for alpha in (0.0, -3.06, 6.06):
        other = write_run(tmp_path / "runs", _case_without_monitor(path, alpha), mesh, info, fields)
        with pytest.raises(VVError) as err:
            cases.study(other.path, experiment)
        assert err.value.code == "V0801"


@pytest.mark.network
def test_section_is_the_nparc_table(tmp_path: Path) -> None:
    paths = {p.name: p for p in fetch(get_case("onera-m6"), tmp_path, ("grid",))}
    np.testing.assert_array_equal(read_columns(paths["airfoil.txt"]).data, onera_m6.SECTION)


@pytest.mark.network
def test_tmr_and_nparc_transcriptions_give_the_same_shocks(tmp_path: Path) -> None:
    case = get_case("onera-m6")
    paths = {p.name: p for p in fetch(case, tmp_path, ("experiment",))}
    tmr = cases.experiment(paths["case_2308.dat"])
    cp_star = cases.critical_cp(cases.conditions()["mach"])
    for k, eta in enumerate(cases.conditions()["eta"], start=1):
        nparc = read_columns(paths[f"cp{k}u.ex"]).data
        from_nparc = cases.shock_positions(nparc[:, 0], -nparc[:, 1], cp_star)  # NPARC stores -Cp
        sec = tmr[eta].upper
        assert cases.shock_positions(sec[:, 0], sec[:, 1], cp_star)[-1] == pytest.approx(
            from_nparc[-1], abs=2e-3
        )
