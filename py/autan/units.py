"""Physical units for case values.

Multiply a number by a unit to build a quantity: ``7.0 * m``, ``19.57 * deg``,
``101325 * Pa``. Every dimensional field of the case checks the dimension of
the quantity it receives; see ``docs/contracts/case-schema.md#quantities``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

import pint
from pydantic import PlainSerializer
from pydantic_core import PydanticCustomError, core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue

ureg = pint.UnitRegistry()
Quantity = ureg.Quantity

m = ureg.meter
mm = ureg.millimeter
s = ureg.second
kg = ureg.kilogram
mol = ureg.mole
K = ureg.kelvin
Pa = ureg.pascal
kPa = ureg.kilopascal
J = ureg.joule
W = ureg.watt
deg = ureg.degree
rad = ureg.radian

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_QUANTITY_PATTERN = rf"^\s*{_NUMBER}\s+\S.*$"
_QUANTITY = re.compile(rf"\s*({_NUMBER})\s+(\S.*?)\s*")
_BARE_NUMBER = re.compile(rf"\s*{_NUMBER}\s*")


def fail(code: str, template: str, **context: object) -> NoReturn:
    """Raise a pydantic error whose type carries a diagnostic code (``autan.E0301``)."""
    kind = f"autan.{code}"
    raise PydanticCustomError(kind, template, context)


@dataclass(frozen=True)
class Dimension:
    """Field marker: a quantity of ``dimension`` (a Pint dimensionality string).

    ``"[angle]"`` is special: Pint treats angles as dimensionless, so an angle
    is a quantity whose root unit is the radian.
    """

    dimension: str
    positive: bool = False

    def __get_pydantic_core_schema__(
        self, source: type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(self.validate)

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        bound = " > 0" if self.positive else ""
        return {
            "type": "string",
            "pattern": _QUANTITY_PATTERN,
            "description": f'Quantity "<number> <unit>" of dimension {self.dimension}{bound}.',
            "x-autan-dimension": self.dimension,
        }

    def _check_dimension(self, quantity: pint.Quantity) -> None:
        if not _has_dimension(quantity, self.dimension):
            fail(
                "E0301",
                "expected a quantity of dimension {expected}, got {got} ({actual})",
                expected=self.dimension,
                got=repr(format_quantity(quantity)),
                actual=_describe_dimension(quantity),
            )

    def validate(self, value: object) -> pint.Quantity:
        quantity = _to_quantity(value, self.dimension)
        self._check_dimension(quantity)
        if not math.isfinite(float(quantity.magnitude)):
            fail("E0304", "{got} is not finite", got=repr(format_quantity(quantity)))
        # A case holds each quantity as its document text reads back, so that the case, its
        # document and its hash agree; that text may round a unit's exponents.
        quantity = parse(format_quantity(quantity), self.dimension)
        self._check_dimension(quantity)
        if self.positive:
            absolute = quantity.to(ureg.kelvin) if self.dimension == "[temperature]" else quantity
            if float(absolute.magnitude) <= 0:
                fail("E0304", "{got} must be > 0", got=repr(format_quantity(quantity)))
        return quantity


class UnitOfLength:
    """Field marker: a unit of length, ``"m"``, ``"mm"``, ``"in"``, or a Pint unit; held as the
    unit its short text reads back."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls.validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                format_unit, when_used="json"
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {
            "type": "string",
            "minLength": 1,
            "description": "Unit of length, as Pint writes it.",
            "x-autan-dimension": "[length]",
        }

    @staticmethod
    def validate(value: object) -> pint.Unit:
        if isinstance(value, pint.Unit):
            if value._REGISTRY is not ureg:
                fail("E0303", "{got} comes from another unit registry", got=repr(str(value)))
            text = format_unit(value)
        elif isinstance(value, str):
            text = value
        else:
            fail("E0301", "expected a unit of length, got {got}", got=repr(value))
        try:
            unit = ureg.parse_units(text)
        except Exception as error:  # noqa: BLE001 -- Pint's parser raises many unrelated types
            fail(
                "E0303",
                "cannot read the unit {got}: {reason}",
                got=repr(text),
                reason=f"{type(error).__name__}: {error}",
            )
        canonical = ureg.parse_units(format_unit(unit))
        for candidate in (unit, canonical):
            if not ureg.Quantity(1.0, candidate).check("[length]"):
                fail("E0301", "expected a unit of length, got {got}", got=repr(text))
            try:
                one, two = (
                    ureg.Quantity(x, candidate).to(ureg.meter).magnitude for x in (1.0, 2.0)
                )
            except (pint.PintError, ArithmeticError):
                one = two = math.nan
            # A mesh scale: a finite, positive, proportional factor to metres (not dB, not
            # an exponent that overflows), the same after the text reads back.
            if not (math.isfinite(one) and one > 0 and math.isclose(two, 2 * one)):
                fail("E0301", "{got} has no finite factor to metres", got=repr(text))
        return canonical


def metres_per(unit: pint.Unit) -> float:
    """Metres in one ``unit`` of length, to 15 significant digits: Pint's chained factors
    can miss an exact decimal definition by an ulp (``ft``: 0.30479999999999996)."""
    return float(f"{ureg.Quantity(1.0, unit).to(ureg.meter).magnitude:.15g}")


def format_unit(unit: pint.Unit) -> str:
    """Canonical text of a unit: Pint's short form."""
    return f"{unit:~D}"


def _to_quantity(value: object, dimension: str) -> pint.Quantity:
    if isinstance(value, pint.Quantity):
        if value._REGISTRY is not ureg:
            fail(
                "E0303",
                "{got} comes from another unit registry; build it with autan.units",
                got=repr(str(value)),
            )
        return Quantity(value.magnitude, value.units)
    if isinstance(value, str):
        return parse(value, dimension)
    if isinstance(value, int | float) and not isinstance(value, bool):
        fail(
            "E0302",
            "{got} has no unit; expected a quantity of dimension {expected}",
            got=repr(value),
            expected=dimension,
        )
    fail(
        "E0301",
        "expected a quantity of dimension {expected}, got {got}",
        expected=dimension,
        got=repr(value),
    )


def _has_dimension(quantity: pint.Quantity, dimension: str) -> bool:
    if dimension == "[angle]":
        return bool(ureg.get_root_units(quantity.units)[1] == ureg.radian)
    return bool(quantity.check(dimension)) and ureg.get_root_units(quantity.units)[1] != ureg.radian


def _describe_dimension(quantity: pint.Quantity) -> str:
    if ureg.get_root_units(quantity.units)[1] == ureg.radian:
        return "[angle]"
    return str(quantity.dimensionality) or "dimensionless"


def parse(text: str, dimension: str) -> pint.Quantity:
    """Read ``"<number> <unit>"``; the unit part may only multiply, divide and raise units."""
    if _BARE_NUMBER.fullmatch(text):
        fail("E0302", "{got} has no unit; expected {expected}", got=repr(text), expected=dimension)
    match = _QUANTITY.fullmatch(text)
    if match is None:
        fail("E0303", "{got} is not written as '<number> <unit>'", got=repr(text))
    try:
        unit = ureg.parse_units(match.group(2))
    except Exception as error:  # noqa: BLE001 -- Pint's parser raises many unrelated types
        fail(
            "E0303",
            "cannot read the unit of {got}: {reason}",
            got=repr(text),
            reason=f"{type(error).__name__}: {error}",
        )
    return Quantity(float(match.group(1)), unit)


def quantity(dimension: str, *, positive: bool = False) -> Any:
    """Annotation metadata of a quantity field: dimension check, and canonical text in JSON."""
    return (Dimension(dimension, positive), PlainSerializer(format_quantity, when_used="json"))


def format_quantity(quantity: pint.Quantity) -> str:
    """Canonical text of a quantity: shortest round-trip number, a space, Pint's short unit."""
    return f"{float(quantity.magnitude)!r} {quantity.units:~D}"


def validate_number(value: object, *, positive: bool) -> float:
    """Validate a dimensionless number; a quantity with a dimension is rejected (E0301)."""
    if isinstance(value, pint.Quantity):
        if not value.dimensionless or ureg.get_root_units(value.units)[1] == ureg.radian:
            fail(
                "E0301",
                "expected a dimensionless number, got {got}",
                got=repr(format_quantity(value)),
            )
        value = float(value.to(ureg.dimensionless).magnitude)
    if isinstance(value, bool) or not isinstance(value, int | float):
        fail("E0301", "expected a dimensionless number, got {got}", got=repr(value))
    number = float(value)
    if not math.isfinite(number):
        fail("E0304", "{got} is not finite", got=repr(number))
    if positive and number <= 0:
        fail("E0304", "{got} must be > 0", got=repr(number))
    return number


def si(quantity: pint.Quantity, unit: str) -> float:
    """Magnitude of ``quantity`` in ``unit``."""
    return float(quantity.to(unit).magnitude)
