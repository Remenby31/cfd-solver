"""The case document through its public seams: the builder, files, TOML/JSON, JSON Schema."""

import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import jsonschema
import pytest
from autan.cli import json_schema
from autan.units import K, Pa, s
from pydantic import ValidationError

from tests.conftest import CRM_HL, EXAMPLES


def codes(error: pytest.ExceptionInfo[autan.CaseError]) -> list[str]:
    return error.value.codes


@pytest.mark.parametrize("example", ["crm_hl.py", "heat_exchanger.py"])
def test_example_round_trips_through_toml_and_json(example: str, tmp_path: Path) -> None:
    case = autan.load(EXAMPLES / example)
    for suffix, text in ((".toml", case.to_toml()), (".json", case.to_json())):
        path = tmp_path / f"case{suffix}"
        path.write_text(text)
        reloaded = autan.load(path)
        assert reloaded.spec == case.spec
        assert reloaded.spec.hash() == case.spec.hash()


@pytest.mark.parametrize("example", ["crm_hl.py", "heat_exchanger.py"])
def test_published_json_schema_accepts_examples_and_rejects_unknown_keys(example: str) -> None:
    schema = json_schema("case")
    validator = jsonschema.Draft202012Validator(schema)
    document = autan.load(EXAMPLES / example).spec.document()
    validator.validate(document)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({**document, "solver_magic": 1})
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({**document, "freestream": {**document["freestream"], "aoa": 19.57}})


def test_case_hash_ignores_layout_and_follows_values(tmp_path: Path) -> None:
    document = autan.load(EXAMPLES / "crm_hl.py").spec.document()
    reordered = dict(reversed(list(document.items())))
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(reordered, indent=7))
    assert autan.load(path).spec.hash() == autan.load(EXAMPLES / "crm_hl.py").spec.hash()

    document["freestream"]["ref_length"] = "700.0 cm"
    path.write_text(json.dumps(document))
    assert autan.load(path).spec.hash() != autan.load(EXAMPLES / "crm_hl.py").spec.hash()


def test_contract_example_toml_is_the_crm_hl_example() -> None:
    contract = (Path(__file__).parents[2] / "docs" / "contracts" / "case-schema.md").read_text()
    block = contract.split("```toml\n", 1)[1].split("```", 1)[0]
    assert (
        autan.CaseSpec.from_document(tomllib.loads(block), autan.Location())
        == autan.load(EXAMPLES / "crm_hl.py").spec
    )


@pytest.mark.parametrize(
    ("call", "code"),
    [
        ("case.freestream(mach=50 * m / s, re=5e6, ref_length=7 * m)", "E0301"),
        ("case.freestream(mach=0.2, aoa=19.57, re=5e6, ref_length=7 * m)", "E0302"),
        ("case.freestream(mach=0.2, aoa=3 * m, re=5e6, ref_length=7 * m)", "E0301"),
        ("case.freestream(mach=0.2, re=5e6, ref_length=7 * deg)", "E0301"),
        ("case.freestream(mach=-0.2, re=5e6, ref_length=7 * m)", "E0304"),
        ("case.freestream(mach=0.2, temperature=-10 * K, re=5e6, ref_length=7 * m)", "E0304"),
        ("case.freestream(mach=0.2, re=float('nan'), ref_length=7 * m)", "E0304"),
        ("case.boundary('w').wall(temperature=300 * Pa)", "E0301"),
        ("case.boundary('w').outlet(static_pressure=101325)", "E0302"),
        # Its document would write the exponent as 1: a length.
        (
            "case.freestream(mach=0.2, re=5e6, ref_length=autan.units.Quantity(1, 'm ** 1.000001'))",
            "E0301",
        ),
        # Its document would write each exponent 1/3 as 0.333333: [length] ** 0.999999.
        (
            (
                "case.freestream(mach=0.2, re=5e6, "
                "ref_length=autan.units.Quantity(1, 'm ** (1/3) * mm ** (1/3) * cm ** (1/3)'))"
            ),
            "E0301",
        ),
    ],
)
def test_builder_rejects_invalid_values_at_the_calling_line(
    call: str, code: str, write_case: Callable[[str], Path]
) -> None:
    path = write_case(
        "import autan\nfrom autan.units import K, Pa, deg, m, s\n"
        "case = autan.Case('c', mesh='m.cgns')\n" + call + "\n"
    )
    with pytest.raises(autan.CaseError) as error:
        autan.load(path)
    [found] = error.value.diagnostics
    assert found.code == code
    assert found.location is not None
    assert (found.location.file, found.location.line) == (str(path), 4)


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("7.0 parsecs_per_fortnight", "E0303"),
        ("7.0", "E0302"),
        ("7.0 s", "E0301"),
        ("m", "E0303"),
        ("7 m + 1 m", "E0303"),
        ("1 (m", "E0303"),
    ],
)
def test_toml_quantities_are_checked(value: str, code: str, tmp_path: Path) -> None:
    document = autan.load(EXAMPLES / "crm_hl.py").spec.document()
    document["freestream"]["ref_length"] = value
    path = tmp_path / "case.json"
    path.write_text(json.dumps(document))
    with pytest.raises(autan.CaseError) as error:
        autan.load(path)
    [found] = error.value.diagnostics
    assert found.code == code
    assert found.location is not None
    assert found.location.path == "freestream.ref_length"


@pytest.mark.parametrize(
    ("edit", "codes_expected"),
    [
        (lambda d: d.update(schema="autan.case/2"), ["E0004"]),
        (lambda d: d.update(turbo=True), ["E0003"]),
        (lambda d: d["freestream"].pop("re"), ["E0306"]),
        (lambda d: d["freestream"].update(pressure="1 bar"), ["E0306"]),
        (lambda d: d["freestream"].pop("ref_length"), ["E0306"]),
        (lambda d: d["freestream"].update(velocity="50 m/s"), ["E0306"]),
        (lambda d: d["physics"].update(turbulence={"model": "wmles"}), ["E0305"]),
        (lambda d: d["physics"].update(kind="incompressible"), ["E0003", "E0003"]),
        (lambda d: d.update(time={"kind": "unsteady", "steps": 10}), ["E0003"]),
        (lambda d: d["physics"]["gas"].update(gamma=0.9), ["E0304"]),
    ],
)
def test_document_errors_have_codes(
    edit: Callable[[dict[str, Any]], None], codes_expected: list[str], tmp_path: Path
) -> None:
    document = autan.load(EXAMPLES / "crm_hl.py").spec.document()
    edit(document)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(document))
    with pytest.raises(autan.CaseError) as error:
        autan.load(path)
    assert sorted(codes(error)) == codes_expected


def test_incompressible_freestream_rules(write_case: Callable[[str], Path]) -> None:
    head = (
        "import autan\nfrom autan.units import K, Pa, kg, m, s\n"
        "water = autan.ConstantDensityFluid(name='water', density=998 * kg / m**3,"
        " viscosity=1e-3 * Pa * s)\n"
        "case = autan.Case('c', mesh='m.cgns')\n"
        "case.physics = autan.Incompressible(turbulence=autan.SST(), fluid=water)\n"
    )
    assert autan.load(write_case(head + "case.freestream(velocity=2 * m / s)\n")).spec
    with pytest.raises(autan.CaseError) as error:
        autan.load(write_case(head + "case.freestream(velocity=2 * m / s, mach=0.1)\ncase.spec\n"))
    assert codes(error) == ["E0306"]


@pytest.mark.parametrize(
    ("body", "code", "line"),
    [
        ("import autan\n", "E0002", None),
        (CRM_HL + "other = autan.Case('b', mesh='m.cgns')\n", "E0002", None),
        ("import autan\ncase = autan.Case('a', mesh='m')\n1 / 0\n", "E0001", 3),
        ("import autan\ncase = (\n", "E0001", 2),
    ],
)
def test_case_file_errors(
    body: str, code: str, line: int | None, write_case: Callable[[str], Path]
) -> None:
    with pytest.raises(autan.CaseError) as error:
        autan.load(write_case(body))
    [found] = error.value.diagnostics
    assert found.code == code
    assert found.location is not None
    assert found.location.line == line


def test_missing_physics_is_reported_at_the_case_constructor(
    write_case: Callable[[str], Path],
) -> None:
    path = write_case("import autan\n\ncase = autan.Case('a', mesh='m')\ncase.spec\n")
    with pytest.raises(autan.CaseError) as error:
        autan.load(path)
    assert codes(error) == ["E0003", "E0003"]
    assert {d.location.line for d in error.value.diagnostics if d.location} == {3}


def test_missing_case_file() -> None:
    with pytest.raises(autan.CaseError) as error:
        autan.load("no/such/case.py")
    assert codes(error) == ["E0001"]


def test_presets_are_physical() -> None:
    def sutherland(gas: autan.IdealGas, temperature: float) -> float:
        law = gas.viscosity
        assert isinstance(law, autan.Sutherland)
        t_ref, s_ = law.t_ref.to(K).magnitude, law.s.to(K).magnitude
        mu_ref = law.mu_ref.to(Pa * s).magnitude
        return float(mu_ref * (temperature / t_ref) ** 1.5 * (t_ref + s_) / (temperature + s_))

    # Air viscosity at 300 K: 1.846e-5 Pa s (Incropera et al., 7th ed., Table A.4).
    assert sutherland(autan.gases.AIR, 300.0) == pytest.approx(1.846e-5, rel=0.01)
    # CO2 viscosity at 300 K: 1.49e-5 Pa s (Incropera et al., 7th ed., Table A.4).
    assert sutherland(autan.gases.CO2, 300.0) == pytest.approx(1.49e-5, rel=0.02)


def test_quantities_from_another_registry_are_rejected(write_case: Callable[[str], Path]) -> None:
    body = (
        "import autan, pint\nother = pint.UnitRegistry()\nother.define('inch = 0.03 * meter')\n"
        "case = autan.Case('c', mesh='m.cgns')\n"
        "case.freestream(mach=0.2, re=5e6, ref_length=other.Quantity(1, 'inch'))\n"
    )
    with pytest.raises(autan.CaseError) as error:
        autan.load(write_case(body))
    assert codes(error) == ["E0303"]


def test_cases_do_not_share_mutable_values() -> None:
    first = autan.load(EXAMPLES / "crm_hl.py").spec
    second = autan.load(EXAMPLES / "crm_hl.py").spec
    before = second.hash()
    assert isinstance(first.physics, autan.Compressible)
    law = first.physics.gas.viscosity
    assert isinstance(law, autan.Sutherland)
    law.mu_ref.ito("mPa * s")
    assert second.hash() == before
    assert autan.gases.AIR.viscosity.model_dump(mode="json")["mu_ref"] == "1.716e-05 Pa * s"


def test_defaults_are_materialized_before_hashing(tmp_path: Path) -> None:
    document = autan.load(EXAMPLES / "crm_hl.py").spec.document()
    assert document["monitors"][0]["moment_center"] == ["0.0 m", "0.0 m", "0.0 m"]
    del document["monitors"][0]["moment_center"]
    path = tmp_path / "case.json"
    path.write_text(json.dumps(document))
    assert autan.load(path).spec.hash() == autan.load(EXAMPLES / "crm_hl.py").spec.hash()


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.pop("schema"),
        lambda d: d.update(freestream={}),
        lambda d: d["freestream"].update(mach=-0.2),
        lambda d: d["freestream"].update(pressure="1 bar"),
        lambda d: d["boundaries"][0]["condition"].pop("kind"),
        lambda d: d["physics"].update(turbulence={"model": "wmles"}),
        lambda d: d["time"].update(max_iterations=True),
        lambda d: d["physics"].update(
            turbulence={"model": "ddes", "base": {"model": "sa", "intensity": 0.1}}
        ),
    ],
)
def test_published_json_schema_and_reader_reject_contract_violations(
    edit: Callable[[dict[str, Any]], None], tmp_path: Path
) -> None:
    document = autan.load(EXAMPLES / "crm_hl.py").spec.document()
    edit(document)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(json_schema("case")).validate(document)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(document))
    with pytest.raises(autan.CaseError):
        autan.load(path)


def test_cross_field_rules_hold_for_plain_pydantic_validation() -> None:
    document = autan.load(EXAMPLES / "crm_hl.py").spec.document()
    del document["freestream"]["mach"]
    with pytest.raises(ValidationError) as error:
        autan.CaseSpec.model_validate(document)
    assert error.value.errors()[0]["type"] == "autan.consistency"


@pytest.mark.parametrize(
    ("edit", "code"),
    [
        (lambda d: d["time"].update(max_iterations=2**53), "E0304"),
        (lambda d: d["time"].update(max_iterations="12"), "E0003"),
    ],
)
def test_integers_are_strict_and_canonicalizable(
    edit: Callable[[dict[str, Any]], None], code: str, tmp_path: Path
) -> None:
    document = autan.load(EXAMPLES / "crm_hl.py").spec.document()
    edit(document)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(document))
    with pytest.raises(autan.CaseError) as error:
        autan.load(path)
    assert codes(error) == [code]


def test_error_paths_are_document_paths(tmp_path: Path) -> None:
    document = autan.load(EXAMPLES / "crm_hl.py").spec.document()
    document["boundaries"][0]["condition"]["temperature"] = "3 Pa"
    document["physics"]["turbulence"]["nu_tilde_ratio"] = -1
    path = tmp_path / "case.json"
    path.write_text(json.dumps(document))
    with pytest.raises(autan.CaseError) as error:
        autan.load(path)
    paths = sorted(d.location.path or "" for d in error.value.diagnostics if d.location)
    assert paths == ["boundaries[0].condition.temperature", "physics.turbulence.nu_tilde_ratio"]


@pytest.mark.parametrize(
    ("unit", "code"),
    [
        ("Pa", "E0301"),
        ("m**2", "E0301"),
        ("2 m", "E0303"),
        ("cubit_x", "E0303"),
        ("m*dB", "E0301"),  # a length, but logarithmic: no factor to metres
        ("km**200/m**199", "E0301"),  # a length whose factor overflows
        # A length whose text, with exponents to 6 digits, is [length] ** 0.999999.
        ("m**0.3333333333333333 * cm**0.3333333333333333 * mm**0.3333333333333334", "E0301"),
    ],
)
def test_mesh_unit_must_be_a_unit_of_length(
    unit: str, code: str, write_case: Callable[[str], Path]
) -> None:
    path = write_case(
        "import autan\nfrom autan.units import m\n"
        f"case = autan.Case('c', mesh='m.cgns', mesh_unit={unit!r})\n"
        "case.physics = autan.Compressible(turbulence=autan.Inviscid())\n"
        "case.freestream(mach=0.2, re=5e6, ref_length=7 * m)\n"
    )
    with pytest.raises(autan.CaseError) as error:
        autan.load(path)
    [found] = error.value.diagnostics
    assert found.code == code
    assert found.location is not None
    assert (found.location.file, found.location.line) == (str(path), 3)
