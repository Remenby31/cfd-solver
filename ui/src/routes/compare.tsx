import { useQueries, useQuery } from "@tanstack/react-query";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { GitCompareArrows, Plus, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Run } from "../api/contracts";
import { devicesLabel } from "../api/devices";
import { flatten, polars } from "../api/polars";
import { runQuery, runsQuery, shownOrders, soleForces, useRunsHistory } from "../api/runs";
import { coefficient } from "../api/units";
import { OverlayChart, type OverlaySeries } from "../components/OverlayChart";
import { DASHES, PolarChart } from "../components/PolarChart";
import { TopBar } from "../components/shell/TopBar";
import { Button } from "../components/ui/Button";
import { cn } from "../components/ui/cn";
import { EmptyState } from "../components/ui/EmptyState";
import { Menu, MenuItem } from "../components/ui/Menu";
import { StateGlyph } from "../components/ui/StateGlyph";

const MAX_RUNS = 8;

export const Route = createFileRoute("/compare")({
  validateSearch: (raw: Record<string, unknown>): { runs: string[] } => ({
    runs: Array.isArray(raw.runs)
      ? raw.runs.filter((r): r is string => typeof r === "string").slice(0, MAX_RUNS)
      : [],
  }),
  component: Compare,
});

const seriesColor = (k: number) => `var(--series-${(k % 8) + 1})`;
const resolvedColor = (k: number) =>
  getComputedStyle(document.documentElement)
    .getPropertyValue(`--series-${(k % 8) + 1}`)
    .trim();

function Compare() {
  const { runs: ids } = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });
  const registry = useQuery(runsQuery);
  // Only accepted runs are followed: one unavailable run must not stop the others' stream.
  const followed = ids.filter((id) => registry.data?.runs.some((r) => r.run_id === id));
  const live = useRunsHistory(followed);
  const details = useQueries({ queries: ids.map((id) => runQuery(id)) });
  const [all, setAll] = useState(false);
  const heading = useRef<HTMLHeadingElement>(null);
  // Arriving from Runs, focus starts on the screen's heading rather than on the body.
  useEffect(() => heading.current?.focus(), []);
  const runs = ids.map((id) => registry.data?.runs.find((r) => r.run_id === id));
  const setRuns = (next: string[]) => void navigate({ search: { runs: next }, replace: true });
  const version = ids.map((id) => live.get(id)?.history.version ?? 0).join(",");

  // Histories grow in place: `version` is what changes when they do.
  const charts = useMemo(() => {
    const drop: OverlaySeries[] = [];
    const cl: OverlaySeries[] = [];
    const cd: OverlaySeries[] = [];
    ids.forEach((id, k) => {
      const history = live.get(id)?.history;
      if (!history) return;
      const label = id.slice(-8);
      const color = resolvedColor(k);
      const target = registry.data?.summaries[id]?.target_orders ?? undefined;
      const y = history.orders.map((o) => shownOrders(o, target));
      drop.push({ label, color, x: history.iteration, y });
      const forces = soleForces(history);
      if (forces) {
        cl.push({ label, color, x: forces.iteration, y: forces.cl });
        cd.push({ label, color, x: forces.iteration, y: forces.cd });
      }
    });
    const targets = ids.map((id) => registry.data?.summaries[id]?.target_orders ?? "").join(",");
    return { drop, cl, cd, dropVersion: `${version}|${targets}` };
  }, [ids, live, version, registry.data]);

  const polar = useMemo(
    () =>
      polars(
        ids.map((id, k) => ({
          id,
          role: String.fromCharCode(65 + k),
          run: runs[k],
          case: details[k]?.data?.case,
          pending: details[k]?.isPending || !registry.data,
        })),
      ),
    [ids, runs, details, registry.data],
  );
  const plotted = polar.series.flatMap((s) => s.points.map((p) => ({ ...p, series: s.label })));
  const runColor = (p: { role: string }) => seriesColor(p.role.charCodeAt(0) - 65);

  const diff = useMemo(() => {
    const flat = details.map((d) =>
      d.data
        ? new Map([
            ...flatten(d.data.case),
            ...flatten({
              run: {
                devices: devicesLabel(d.data.run.devices),
                backend: d.data.run.solver.backend,
                solver: d.data.run.solver.commit,
              },
            }),
          ])
        : null,
    );
    const keys = [...new Set(flat.flatMap((f) => (f ? [...f.keys()] : [])))].sort();
    // A setting absent from a case differs from every value, "—" included.
    return keys.map((key) => {
      const raw = flat.map((f) => (f ? (f.has(key) ? `=${f.get(key)}` : "absent") : "loading"));
      const values = raw.map((r) => (r === "absent" ? "—" : r === "loading" ? "…" : r.slice(1)));
      return { key, values, raw, differs: new Set(raw).size > 1 };
    });
  }, [details]);

  const others = (registry.data?.runs ?? []).filter((r) => !ids.includes(r.run_id));

  return (
    <>
      <TopBar crumbs={[{ label: "Runs", to: "/" }, { label: "Compare" }]} />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto flex max-w-[1200px] flex-col gap-6 px-6 py-8">
          <h1
            ref={heading}
            tabIndex={-1}
            className="m-0 font-semibold text-xl outline-none focus-visible:shadow-none"
          >
            Compare
          </h1>
          <div className="flex flex-wrap items-center gap-2" data-testid="compared">
            {ids.map((id, k) => (
              <RunChip
                key={id}
                id={id}
                run={runs[k]}
                unavailable={Boolean(registry.data) && !runs[k]}
                role={String.fromCharCode(65 + k)}
                color={seriesColor(k)}
                onRemove={() => {
                  setRuns(ids.filter((other) => other !== id));
                  // Focus stays in the chip row: the next chip's remove button, else the heading.
                  requestAnimationFrame(() => {
                    const buttons = document.querySelectorAll<HTMLElement>("[data-chip-remove]");
                    (buttons[Math.min(k, buttons.length - 1)] ?? heading.current)?.focus();
                  });
                }}
              />
            ))}
            {ids.length < MAX_RUNS && others.length > 0 && (
              <Menu
                label="Add a run"
                trigger={<Button size="sm" variant="tertiary" icon={<Plus className="size-4" />} />}
              >
                {others.slice(0, 20).map((run) => (
                  <MenuItem
                    key={run.run_id}
                    icon={<StateGlyph status={run.status} />}
                    onClick={() => setRuns([...ids, run.run_id])}
                  >
                    {run.case.name}{" "}
                    <span className="font-mono text-fg-3">{run.run_id.slice(-8)}</span>
                  </MenuItem>
                ))}
              </Menu>
            )}
          </div>

          {ids.length < 2 ? (
            <EmptyState
              icon={<GitCompareArrows className="size-5" />}
              title="Pick two runs or more"
              description="Select runs on the Runs page with x, then press c."
            >
              <Link to="/" className="self-center">
                <Button>Go to Runs</Button>
              </Link>
            </EmptyState>
          ) : (
            <>
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
                <ChartCard title="Residual drop" unit="orders" testId="chart-drop">
                  <OverlayChart
                    series={charts.drop}
                    xLabel="iteration"
                    sync="compare"
                    version={charts.dropVersion}
                  />
                </ChartCard>
                <ChartCard title="CL" unit="iteration">
                  <OverlayChart
                    series={charts.cl}
                    xLabel="iteration"
                    sync="compare"
                    version={version}
                  />
                </ChartCard>
                <ChartCard title="CD" unit="iteration">
                  <OverlayChart
                    series={charts.cd}
                    xLabel="iteration"
                    sync="compare"
                    version={version}
                  />
                </ChartCard>
              </div>

              {plotted.length > 0 && (
                <div className="flex flex-col gap-3" data-testid="polar">
                  <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-sm">
                    {polar.fixed && (
                      <span className="text-fg-3" data-testid="polar-fixed">
                        {polar.fixed}
                      </span>
                    )}
                    {polar.series.length > 1 &&
                      polar.series.map((s, k) => (
                        <span
                          key={s.key}
                          className="inline-flex items-center gap-2"
                          data-testid="polar-series"
                        >
                          <svg width="20" height="6" aria-hidden className="flex-none">
                            <line
                              x1="0"
                              x2="20"
                              y1="3"
                              y2="3"
                              stroke="var(--fg-3)"
                              strokeWidth="1.5"
                              strokeDasharray={DASHES[k % DASHES.length]}
                            />
                          </svg>
                          {s.label}
                          <span className="font-mono text-fg-3">
                            {s.points.map((p) => p.role).join(" ")}
                          </span>
                        </span>
                      ))}
                  </div>
                  <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                    <ChartCard title="Lift polar" unit="CL against α (deg), converged runs">
                      <PolarChart
                        series={polar.series}
                        x={(p) => p.alpha}
                        y={(p) => p.cl}
                        xLabel="α (deg)"
                        yLabel="CL"
                        color={runColor}
                        testId="lift-polar"
                      />
                    </ChartCard>
                    <ChartCard title="Drag polar" unit="CD against CL">
                      <PolarChart
                        series={polar.series}
                        x={(p) => p.cl}
                        y={(p) => p.cd}
                        xLabel="CL"
                        yLabel="CD"
                        color={runColor}
                        testId="drag-polar"
                      />
                    </ChartCard>
                  </div>
                  <table className="w-full border-collapse text-sm" data-testid="polar-table">
                    <thead>
                      <tr className="text-left text-fg-3">
                        <th className="h-8 px-2 font-normal">Run</th>
                        <th className="px-2 text-right font-normal">α (deg)</th>
                        <th className="px-2 text-right font-normal">CL</th>
                        <th className="px-2 text-right font-normal">CD</th>
                        {polar.series.length > 1 && <th className="px-2 font-normal">Series</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {plotted.map((p) => (
                        <tr key={p.id} className="border-line border-t font-mono">
                          <td className="h-8 px-2">
                            <span className="inline-flex items-center gap-2">
                              <i
                                className="inline-block size-2 rounded-full"
                                style={{ background: runColor(p) }}
                              />
                              {p.role}
                            </span>
                          </td>
                          <td className="px-2 text-right">{p.alpha}</td>
                          <td className="px-2 text-right">{coefficient(p.cl)}</td>
                          <td className="px-2 text-right">{coefficient(p.cd)}</td>
                          {polar.series.length > 1 && (
                            <td className="px-2 font-sans text-fg-2">{p.series}</td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {polar.unplotted.length > 0 && (
                <p className="m-0 text-fg-2 text-sm" data-testid="polar-missing">
                  Not on the polars:{" "}
                  {polar.unplotted.map((u, k) => (
                    <span key={u.id}>
                      {k > 0 && " · "}
                      <span className="font-mono">{u.role}</span> {u.reason}
                    </span>
                  ))}
                </p>
              )}

              <section className="overflow-hidden rounded-lg border border-line bg-bg-0">
                <header className="flex items-center gap-3 border-line border-b px-4 py-3">
                  <span className="font-medium">Settings</span>
                  <span className="text-fg-3 text-sm">
                    {diff.filter((d) => d.differs).length} differ of {diff.length}
                  </span>
                  <Button
                    size="sm"
                    variant="tertiary"
                    className="ml-auto"
                    aria-pressed={all}
                    onClick={() => setAll(!all)}
                  >
                    {all ? "Only differences" : "Show all"}
                  </Button>
                </header>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse text-sm" data-testid="settings-diff">
                    <thead>
                      <tr className="text-left text-fg-3">
                        <th className="h-9 px-4 font-normal">Setting</th>
                        {ids.map((id, k) => (
                          <th key={id} className="px-4 font-normal">
                            <span className="inline-flex items-center gap-2">
                              <i
                                className="inline-block size-2 rounded-full"
                                style={{ background: seriesColor(k) }}
                              />
                              {String.fromCharCode(65 + k)}
                            </span>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {diff
                        .filter((row) => all || row.differs)
                        .map((row) => (
                          <tr key={row.key} className="border-line border-t">
                            <td className="h-9 px-4 font-mono text-fg-2">{row.key}</td>
                            {row.values.map((value, k) => (
                              <td
                                key={ids[k]}
                                className={cn(
                                  "px-4 font-mono",
                                  row.differs &&
                                    k > 0 &&
                                    row.raw[k] !== row.raw[0] &&
                                    "bg-accent-soft",
                                )}
                              >
                                {value}
                              </td>
                            ))}
                          </tr>
                        ))}
                    </tbody>
                  </table>
                  {details.some((d) => d.isError) && (
                    <p className="m-0 px-4 py-3 text-danger text-sm">
                      The case of {details.filter((d) => d.isError).length} run(s) could not be
                      read.
                    </p>
                  )}
                  {!all && details.every((d) => d.isSuccess) && !diff.some((d) => d.differs) && (
                    <p className="m-0 px-4 py-6 text-center text-fg-3">The cases are identical.</p>
                  )}
                </div>
              </section>
            </>
          )}
        </div>
      </div>
    </>
  );
}

function ChartCard({
  title,
  unit,
  testId,
  children,
}: {
  title: string;
  unit: string;
  testId?: string;
  children: React.ReactNode;
}) {
  return (
    <section
      className="flex h-72 flex-col gap-2 rounded-lg border border-line bg-bg-0 p-4"
      data-testid={testId}
    >
      <header className="flex items-baseline gap-2">
        <span className="font-medium">{title}</span>
        <span className="text-fg-3 text-sm">{unit}</span>
      </header>
      <div className="min-h-0 flex-1">{children}</div>
    </section>
  );
}

function RunChip({
  id,
  run,
  role,
  color,
  onRemove,
  unavailable,
}: {
  id: string;
  run: Run | undefined;
  role: string;
  color: string;
  onRemove: () => void;
  unavailable: boolean;
}) {
  return (
    <span className="inline-flex h-8 items-center gap-2 rounded-full border border-line bg-bg-0 pr-1 pl-3 text-sm">
      <i className="inline-block size-2.5 rounded-full" style={{ background: color }} />
      <span className="font-mono text-fg-3">{role}</span>
      <Link
        to="/runs/$runId"
        params={{ runId: id }}
        className="flex items-center gap-2 hover:underline"
      >
        {run && <StateGlyph status={run.status} />}
        <span className="font-medium">{run?.case.name ?? (unavailable ? "unavailable" : "…")}</span>
        <span className="font-mono text-fg-3">{id.slice(-8)}</span>
      </Link>
      <button
        type="button"
        aria-label={`Remove ${id} from the comparison`}
        data-chip-remove
        onClick={onRemove}
        className="flex size-6 items-center justify-center rounded-full text-fg-3 hover:bg-bg-2 hover:text-fg"
      >
        <X className="size-3.5" />
      </button>
    </span>
  );
}
