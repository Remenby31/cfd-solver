from __future__ import annotations

import io
import re
import tokenize
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pint
from pydantic import BaseModel

from autan import gases, materials, units
from autan.schema import CaseSpec, Compressible, Fluid, Freestream, Spec, Steady
from autan.units import format_quantity, format_unit

if TYPE_CHECKING:
    from collections.abc import Iterable

WIDTH = 100
INDENT = "    "
# The temperature CaseSpec gives a compressible freestream that has none.
_FILLED_TEMPERATURE = "288.15 K"
_EXPORTED_UNITS = sorted(
    (name for name, value in vars(units).items() if isinstance(value, pint.Unit)),
    key=len,
    reverse=True,
)
_UNIT = "(?:" + "|".join(map(re.escape, _EXPORTED_UNITS)) + r")(?: \*\* [0-9.]+)?"
# Pint's short unit text when every unit in it is one autan.units exports.
_EXPORTED_TEXT = re.compile(rf"(?:1 / )?{_UNIT}(?: [*/] {_UNIT})*")
_LIBRARY = [
    (f"autan.{module.__name__.rsplit('.', 1)[1]}.{name}", value)
    for module in (gases, materials)
    for name, value in sorted(vars(module).items())
    if name.isupper() and isinstance(value, BaseModel)
]


def to_python(spec: CaseSpec) -> str:
    body = "\n".join(_statements(spec, _Values())) + "\n"
    names = {
        token.string
        for token in tokenize.generate_tokens(io.StringIO(body).readline)
        if token.type == tokenize.NAME
    }
    imports = ["import autan"]
    if used := names & set(_EXPORTED_UNITS):
        imports.append(f"from autan.units import {', '.join(_sorted_names(used))}")
    return "\n".join([*imports, "", body])


def _sorted_names(names: Iterable[str]) -> list[str]:
    # ruff's isort order: constants (uppercase, longer than one character), classes, the rest.
    def kind(name: str) -> int:
        if len(name) > 1 and name.isupper():
            return 0
        return 1 if name[0].isupper() else 2

    return sorted(names, key=lambda name: (kind(name), name.lower(), name))


def _statements(spec: CaseSpec, value: _Values) -> list[str]:
    given = [value(spec.name), _Keyword("mesh", value(spec.mesh))]
    unit = CaseSpec.model_fields["mesh_unit"].get_default(call_default_factory=True)
    if not _same(spec.mesh_unit, unit):
        given.append(_Keyword("mesh_unit", value(spec.mesh_unit)))
    case = _Call("autan.Case", tuple(given))
    out = [_statement("case", case), _statement("case.physics", value(spec.physics))]
    out.append(_statement("", _Call("case.freestream", value.keywords(_freestream(spec)))))
    if not _same(spec.time, Steady()):
        out.append(_statement("case.time", value(spec.time)))
    if spec.initial_state is not None:
        out.append(_statement("case.initial_state", value(spec.initial_state)))
    if spec.verification is not None:
        out.append(_statement("case.verification", value(spec.verification.manufactured)))
    for zone in spec.cell_zones:
        medium = zone.medium
        selector = _Call("case.cell_zone", (value(zone.zones),))
        arguments: tuple[_Expr | _Keyword, ...] = (value(medium.material),)
        if isinstance(medium, Fluid) and medium.initial is not None:
            arguments += (_Keyword("initial", value(medium.initial)),)
        out.append(_statement("", _Call(f".{medium.kind}", arguments, selector)))
    for rule in spec.boundaries:
        condition = rule.condition
        selector = _Call("case.boundary", (_selection(value, rule.zones, rule.group),))
        method = f".{condition.kind.replace('-', '_')}"
        out.append(_statement("", _Call(method, value.keywords(condition), selector)))
    for monitor in spec.monitors:
        chosen = value.keywords(monitor, skip=frozenset({"surfaces", "group", "every"}))
        selection = _selection(value, monitor.surfaces, monitor.group)
        forces = _Call("autan.forces", (selection, *chosen))
        every = (_Keyword("every", value(monitor.every)),) if monitor.every != 1 else ()
        out.append(_statement("", _Call("case.monitor", (forces, *every))))
    return out


def _selection(value: _Values, zones: str | None, group: str | None) -> _Expr | _Keyword:
    # A glob over zone names is positional, a group glob a keyword.
    return value(zones) if zones is not None else _Keyword("group", value(group))


def _freestream(spec: CaseSpec) -> Freestream:
    fs = spec.freestream
    if (
        isinstance(spec.physics, Compressible)
        and fs.temperature is not None
        and format_quantity(fs.temperature) == _FILLED_TEMPERATURE
    ):
        return fs.model_copy(update={"temperature": None})
    return fs


def _document(value: Any) -> Any:
    if isinstance(value, Spec):
        return (type(value), value.model_dump(mode="json", by_alias=True))
    if isinstance(value, pint.Quantity):
        return format_quantity(value)
    if isinstance(value, pint.Unit):
        return format_unit(value)
    if isinstance(value, tuple | list):
        return [_document(item) for item in value]
    if isinstance(value, float):
        return repr(value)
    return value


def _same(a: Any, b: Any) -> bool:
    # Equal as the case document writes them: 0 rad is not 0 deg, -0.0 is not 0.0.
    return bool(_document(a) == _document(b))


class _Values:
    def __call__(self, value: Any) -> _Expr:
        if isinstance(value, pint.Quantity):
            return self.quantity(value)
        if isinstance(value, pint.Unit):
            return _string(format_unit(value))
        if isinstance(value, float):
            return _number(value)
        if isinstance(value, bool | int) or value is None:
            return repr(value)
        if isinstance(value, str):
            return _string(value)
        if isinstance(value, tuple | list):
            return _Call("", tuple(self(item) for item in value))
        if isinstance(value, Spec):
            return self.model(value)
        message = f"cannot render {type(value).__name__}"
        raise TypeError(message)

    def model(self, model: Spec) -> _Expr:
        for name, library in _LIBRARY:
            if _same(model, library):
                return name
        return _Call(f"autan.{type(model).__name__}", self.keywords(model))

    def keywords(
        self, model: BaseModel, skip: frozenset[str] = frozenset()
    ) -> tuple[_Keyword, ...]:
        found = []
        for key, field in type(model).model_fields.items():
            if key in skip or key in ("kind", "model"):
                continue
            mine = getattr(model, key)
            if not field.is_required() and _same(
                mine, field.get_default(call_default_factory=True)
            ):
                continue
            found.append(_Keyword(key, self(mine)))
        return tuple(found)

    def quantity(self, quantity: pint.Quantity) -> _Expr:
        text = format_quantity(quantity)
        unit = text.split(" ", 1)[1]
        if not _EXPORTED_TEXT.fullmatch(unit):
            return _string(text)
        magnitude = _number(float(quantity.magnitude))
        expression = unit.replace(" ** ", "**")
        if expression.startswith("1 / "):
            return _Quantity(f"{magnitude} / {expression.removeprefix('1 / ')}", _string(text))
        return _Quantity(f"{magnitude} * {expression}", _string(text))


def _number(value: float) -> str:
    # ruff writes a positive exponent without its sign.
    return repr(value).replace("e+", "e")


def _string(text: str) -> str:
    # ruff's preferred quotes: double, unless single quotes need fewer escapes.
    quote = "'" if text.count('"') > text.count("'") else '"'
    body = "".join(
        "\\\\"
        if char == "\\"
        else f"\\{quote}"
        if char == quote
        else char
        if char.isprintable()
        else repr(char)[1:-1]
        for char in text
    )
    return f"{quote}{body}{quote}"


@dataclass(frozen=True)
class _Keyword:
    name: str
    value: _Expr


@dataclass(frozen=True)
class _Call:
    # ``receiver.head(arguments)``; a tuple when ``head`` is empty.
    head: str
    arguments: tuple[_Expr | _Keyword, ...]
    receiver: _Call | None = None


@dataclass(frozen=True)
class _Quantity:
    # A quantity as an expression of autan.units, or as its text where the expression would
    # need a line of its own that it does not fit on.
    expression: str
    text: str


_Expr = str | _Call | _Quantity


def _width(text: str) -> int:
    # Display width, as ruff measures lines: East Asian wide characters take two columns.
    # Combining marks take none.
    return sum(
        0
        if unicodedata.category(char) in ("Mn", "Me")
        else 2
        if unicodedata.east_asian_width(char) in "WF"
        else 1
        for char in text
    )


def _flat(expr: _Expr | _Keyword) -> str:
    if isinstance(expr, str):
        return expr
    if isinstance(expr, _Quantity):
        return expr.expression
    if isinstance(expr, _Keyword):
        return f"{expr.name}={_flat(expr.value)}"
    return f"{_head(expr)}({_joined(expr)})"


def _head(call: _Call) -> str:
    return (_flat(call.receiver) if call.receiver else "") + call.head


def _joined(call: _Call) -> str:
    inner = ", ".join(_flat(argument) for argument in call.arguments)
    return inner + "," if not call.head and len(call.arguments) == 1 else inner


def _layout(expr: _Expr | _Keyword, indent: int, column: int, tail: int) -> str:
    # ``expr`` from ``column`` on a line indented by ``indent``, ``tail`` columns following it,
    # broken as ruff format breaks it: the last brackets first, unless what precedes them does
    # not fit, then the receiver's.
    flat = _flat(expr)
    if isinstance(expr, str) or column + _width(flat) + tail <= WIDTH:
        return flat
    if isinstance(expr, _Quantity):
        return expr.text
    if isinstance(expr, _Keyword):
        prefix = f"{expr.name}="
        return prefix + _layout(expr.value, indent, column + len(prefix), tail)
    if expr.receiver and (not expr.arguments or column + _width(_head(expr)) + 1 > WIDTH):
        opening = len(expr.head) + (1 if expr.arguments else 2)
        receiver = _layout(expr.receiver, indent, column, opening)
        last = _width(receiver.rsplit("\n", 1)[-1])
        if "\n" not in receiver:
            last += column
        return receiver + _layout(_Call(expr.head, expr.arguments), indent, last, tail)
    head = _head(expr)
    inner = indent + len(INDENT)
    joined = _joined(expr)
    if expr.head and inner + _width(joined) <= WIDTH:
        return f"{head}(\n{' ' * inner}{joined}\n{' ' * indent})"
    if expr.head and len(expr.arguments) == 1:
        # No comma to split at: ruff breaks inside the argument, without a trailing comma.
        argument = _layout(expr.arguments[0], inner, inner, 0)
        return f"{head}(\n{' ' * inner}{argument}\n{' ' * indent})"
    lines = "".join(
        f"{' ' * inner}{_layout(argument, inner, inner, 1)},\n" for argument in expr.arguments
    )
    return f"{head}(\n{lines}{' ' * indent})"


def _statement(target: str, expr: _Expr) -> str:
    prefix = f"{target} = " if target else ""
    return prefix + _layout(expr, 0, len(prefix), 0)
