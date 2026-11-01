import json
import math
from dataclasses import dataclass, field
from typing import Any

from autan.restart import FLOOR

from autan_ui.confine import Folder
from autan_ui.history import read_history

TREND_POINTS = 48


@dataclass
class _Trend:
    offset: int = 0
    first: dict[str, float] | None = None
    iteration: int | None = None
    orders: list[float] = field(default_factory=list)
    current: float | None = None
    woken: set[str] = field(default_factory=set)
    rule: str = ""


def residual_orders(
    first: dict[str, float],
    residuals: dict[str, float],
    woken: set[str],
    floors: dict[str, float] | None = None,
) -> float | None:
    """Orders of magnitude the slowest equation not yet converged has dropped since iteration 1,
    by the steady convergence rule of case-schema.md#time: `math.inf` once every equation is.

    An equation below the floor at iteration 1 (FLOOR times the largest first residual) is
    exempt while it stays at or below it; once above, it joins `woken`, which this call
    updates, and is never exempt again. An equation at or below its round-off floor (`floors`,
    run.json `convergence.floor`) has converged, and so has one at exactly zero. None when an
    equation cannot be compared: a residual that is not a positive finite number, or that grew
    from an exact zero; None for an empty line."""
    floor = FLOOR * max((v for v in first.values() if _positive(v)), default=0.0)
    drops = []
    comparable = True
    for name, value in residuals.items():
        start = first.get(name)
        if name not in woken and _number(start) and 0 <= start < floor:  # type: ignore[operator]
            if _number(value) and 0 <= value <= floor:
                continue
            woken.add(name)
        limit = (floors or {}).get(name)
        if _number(value) and limit is not None and 0 <= value <= limit:
            continue  # At its round-off floor.
        if value == 0 and (start == 0 or _positive(start)):
            continue  # Reached exactly zero.
        if _positive(start) and _positive(value):
            drops.append(math.log10(start) - math.log10(value))  # type: ignore[arg-type]
        else:
            comparable = False  # Keep going: the equations after it may still wake.
    if not comparable or not residuals:
        return None
    return min(drops, default=math.inf)


def _number(value: object) -> bool:
    return isinstance(value, float | int) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value: object) -> bool:
    return isinstance(value, float | int) and not isinstance(value, bool) and 0 < value < math.inf


def downsample(values: list[float], points: int = TREND_POINTS) -> list[float]:
    if len(values) <= points:
        return [round(v, 3) for v in values]
    step = (len(values) - 1) / (points - 1)
    return [round(values[round(k * step)], 3) for k in range(points)]


class Summaries:
    """Per-run iteration, residual drop and its trend, read incrementally from history.jsonl."""

    def __init__(self) -> None:
        self._trends: dict[str, _Trend] = {}

    def update(
        self,
        name: str,
        folder: Folder,
        *,
        live: bool,
        reference: dict[str, float] | None = None,
        woken: list[str] | None = None,
        floors: dict[str, float] | None = None,
        target: float | None = None,
    ) -> dict[str, Any] | None:
        """`reference` and `woken`: `start.residual_reference` and `start.woken` of a steady
        `continue` run (results.md), the residuals the drop is measured from and the identically
        zero equations no longer exempt; `floors`: run.json `convergence.floor`; `target`: the
        case's `residual_orders`, which a run with every equation converged reports."""
        # Floors arrive once the runtime has built its physics (results.md): a change in the
        # rule's inputs recomputes the whole history.
        rule = json.dumps([reference, woken, floors, target], sort_keys=True)
        trend = self._trends.setdefault(name, _Trend(rule=rule))
        stat = folder.stat("history.jsonl")
        if (stat.st_size if stat else 0) < trend.offset or trend.rule != rule:
            trend = self._trends[name] = _Trend(rule=rule)
        try:
            chunk = read_history(folder, trend.offset, live=live)
        except (OSError, ValueError):
            return None
        for record in chunk.records:
            iteration = record.get("iteration") if isinstance(record, dict) else None
            if type(iteration) is not int:
                continue
            residuals = record.get("residuals")
            if not isinstance(residuals, dict):
                residuals = {}
            if trend.first is None:
                trend.first = reference or residuals
                trend.woken = set(woken or ())
            trend.iteration = iteration
            orders = residual_orders(trend.first, residuals, trend.woken, floors)
            if orders == math.inf:
                orders = target
            trend.current = orders
            if orders is not None:
                trend.orders.append(orders)
        trend.offset = chunk.offset
        # A run with no recorded iteration yet still has a summary, its settings come with it;
        # its iteration is then run.json's (a continuation may not have written one yet).
        return {
            "iteration": trend.iteration,
            "orders": None if trend.current is None else round(trend.current, 3),
            "trend": downsample(trend.orders),
        }

    def forget(self, names: set[str]) -> None:
        for name in names:
            self._trends.pop(name, None)
