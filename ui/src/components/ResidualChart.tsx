import { useEffect, useRef, useState } from "react";
import uPlot from "uplot";
import type { HistoryColumns } from "../api/runs";
import { Tooltip } from "./ui/Tooltip";

const css = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const series = (k: number) => css(`--series-${(k % 8) + 1}`);

// An equation satisfied identically (the spanwise momentum of a one-cell-thick slab) leaves
// round-off residuals near 1e-33, or exactly zero. The axis spans whole decades, at most DECADES
// below its top; zero, which a log axis cannot place, is a gap.
const DECADES = 20;
const positive = (values: number[]) => values.map((v) => (v > 0 ? v : null));
const range = (
  _: uPlot,
  data: number | null,
  max: number | null,
  floors: number[],
): uPlot.Range.MinMax => {
  const min = Math.min(data ?? Number.POSITIVE_INFINITY, ...floors);
  // A floor can sit above every curve once divided by its R₀: the axis reaches it too.
  const highest = Math.max(max ?? 0, ...floors);
  if (!Number.isFinite(min) || !(highest > 0)) return [data, max];
  const top = Math.ceil(Math.log10(highest));
  const bottom = Math.max(Math.floor(Math.log10(Math.max(min, Number.MIN_VALUE))), top - DECADES);
  return [10 ** Math.min(bottom, top - 1), 10 ** top];
};

// Ticks, labels and grid lines on whole decades, every n-th past LABELS of them. uPlot's own
// log splits are not used: they overflow on decades far below 1.
const LABELS = 12;
const SUPERSCRIPT = "⁰¹²³⁴⁵⁶⁷⁸⁹";
/** A power of ten as 10ⁿ. */
const decade = (value: number) => {
  const k = Math.round(Math.log10(value));
  const digits = String(Math.abs(k))
    .split("")
    .map((d) => SUPERSCRIPT[Number(d)])
    .join("");
  return `10${k < 0 ? "⁻" : ""}${digits}`;
};
const short = (value: number) => (value === 0 ? "0" : value.toExponential(2));
const decades = (_: uPlot, __: number, min: number, max: number) => {
  if (!(min > 0 && max > min)) return [];
  const low = Math.ceil(Math.log10(min) - 1e-9);
  const high = Math.floor(Math.log10(max) + 1e-9);
  const step = Math.ceil((high - low + 1) / LABELS);
  const splits = [];
  for (let k = Math.ceil(low / step) * step; k <= high; k += step) splits.push(10 ** k);
  return splits;
};

/** Residuals relative to their first value, R/R₀, on a log axis with a label per decade (or
 * per few), the round-off floor of each equation dashed in its colour (run.json
 * `convergence.floor`), and the values at the cursor; redraws in place as the history grows.
 * A continued run divides by its `start.residual_reference`, the first residuals of the run it
 * continues. An equation exempt from the convergence rule (identically zero) is not drawn. */
export function ResidualChart({
  history,
  floors,
  reference,
  steady,
}: {
  history: HistoryColumns;
  floors?: Record<string, number> | undefined;
  reference?: Record<string, number> | null | undefined;
  /** The convergence rule, and so its exemptions, applies to steady runs only. */
  steady: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const floorsRef = useRef(floors);
  floorsRef.current = floors;
  const names = [...history.residuals.keys()];
  const key = names.join(",");
  // R₀ of each equation: the reference a continued run carries, or its first positive residual.
  // A zero reference (an equation woken after the run it continues began) gives no scale.
  const first = (name: string) => {
    const given = reference?.[name];
    return given !== undefined && given > 0
      ? given
      : (history.residuals.get(name)?.find((v) => v > 0) ?? 1);
  };
  const r0 = useRef<Record<string, number>>({});
  r0.current = Object.fromEntries(names.map((name) => [name, first(name)]));
  const hidden = (name: string) => steady && history.exempt.has(name);
  // What the uPlot hooks read, as of the latest render.
  const latest = useRef({ history, hidden });
  latest.current = { history, hidden };
  const value = (name: string) => readout?.values.find(([n]) => n === name)?.[1];
  const [readout, setReadout] = useState<{ iteration: number; values: [string, number][] } | null>(
    null,
  );

  // biome-ignore lint/correctness/useExhaustiveDependencies: the plot is rebuilt only when the equations change
  useEffect(() => {
    const el = host.current;
    if (!el || !names.length) return;
    const chart = new uPlot(
      {
        width: el.clientWidth,
        height: el.clientHeight,
        scales: {
          x: { time: false },
          y: {
            distr: 3,
            range: (u, min, max) =>
              range(
                u,
                min,
                max,
                Object.entries(floorsRef.current ?? {})
                  .filter(([name]) => !latest.current.hidden(name))
                  .map(([name, v]) => v / (r0.current[name] ?? 1))
                  .filter((v) => v > 0),
              ),
          },
        },
        hooks: {
          setCursor: [
            (u) => {
              const index = u.cursor.idx;
              const iteration = index == null ? undefined : u.data[0][index];
              if (index == null || iteration == null) return setReadout(null);
              // What is drawn, and a drawn equation's exact zero, which a log axis leaves as a gap.
              const values = names.flatMap((name, k): [string, number][] => {
                const v = u.data[k + 1]?.[index];
                if (v != null) return [[name, v]];
                const raw = latest.current.history.residuals.get(name)?.[index];
                return raw === 0 && !latest.current.hidden(name) ? [[name, 0]] : [];
              });
              setReadout({ iteration, values });
            },
          ],
          draw: [
            (u) => {
              const ctx = u.ctx;
              ctx.save();
              ctx.setLineDash([4 * devicePixelRatio, 4 * devicePixelRatio]);
              ctx.lineWidth = devicePixelRatio;
              names.forEach((name, k) => {
                const raw = floorsRef.current?.[name];
                if (raw === undefined || !(raw > 0) || latest.current.hidden(name)) return;
                const floor = raw / (r0.current[name] ?? 1);
                const y = Math.round(u.valToPos(floor, "y", true));
                ctx.strokeStyle = series(k);
                ctx.beginPath();
                ctx.moveTo(u.bbox.left, y);
                ctx.lineTo(u.bbox.left + u.bbox.width, y);
                ctx.stroke();
              });
              ctx.restore();
            },
          ],
        },
        axes: [
          {
            stroke: css("--fg-3"),
            grid: { stroke: css("--line") },
            label: "iteration",
            labelSize: 20,
          },
          {
            stroke: css("--fg-3"),
            grid: { stroke: css("--line") },
            size: 56,
            label: "R / R₀",
            labelSize: 20,
            splits: decades,
            values: (_, splits) => splits.map((v) => (v == null ? "" : decade(v))),
          },
        ],
        legend: { show: false },
        cursor: { drag: { x: true, y: false } },
        series: [
          { label: "iteration" },
          ...names.map((name, k) => ({
            label: name,
            stroke: series(k),
            points: { show: false },
            width: 1.5,
          })),
        ],
      },
      [[], ...names.map(() => [])],
      el,
    );
    plot.current = chart;
    const resize = new ResizeObserver(() =>
      chart.setSize({ width: el.clientWidth, height: el.clientHeight }),
    );
    resize.observe(el);
    return () => {
      resize.disconnect();
      chart.destroy();
      plot.current = null;
    };
  }, [key]);

  // A new floor rescales the axis: the data is set again, which recomputes the range.
  const floorsKey = JSON.stringify(floors ?? null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: `version` changes on every append
  useEffect(() => {
    plot.current?.setData([
      history.iteration,
      ...names.map((name) => {
        if (hidden(name)) return history.iteration.map(() => null);
        const r = r0.current[name] ?? 1;
        return positive(history.residuals.get(name) ?? []).map((v) => (v == null ? null : v / r));
      }),
    ]);
  }, [history.version, key, floorsKey, steady, [...history.exempt].join()]);

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-fg-3 text-xs">
        {readout && (
          <span className="text-fg-2" data-testid="residual-iteration">
            iteration {readout.iteration}
          </span>
        )}
        {names.map((name, k) =>
          steady && history.exempt.has(name) ? (
            <Tooltip
              key={name}
              content="Identically zero: exempt while ≤ 1e-12 × the largest initial residual"
            >
              <span className="flex items-center gap-1.5 opacity-60" data-exempt>
                <Swatch k={k} />
                {name}
                <span className="font-sans">exempt: identically zero, not drawn</span>
                {value(name) !== undefined && (
                  <span className="text-fg-2 tabular-nums" data-testid={`residual-${name}`}>
                    {short(value(name) as number)}
                  </span>
                )}
              </span>
            </Tooltip>
          ) : (
            <span key={name} className="flex items-center gap-1.5">
              <Swatch k={k} />
              {name}
              {value(name) !== undefined && (
                <span className="text-fg-2 tabular-nums" data-testid={`residual-${name}`}>
                  {short(value(name) as number)}
                </span>
              )}
            </span>
          ),
        )}
      </div>
      <div ref={host} className="min-h-0 flex-1" data-testid="residual-chart" />
    </div>
  );
}

function Swatch({ k }: { k: number }) {
  return (
    <i className="inline-block h-0.5 w-3" style={{ background: `var(--series-${(k % 8) + 1})` }} />
  );
}
