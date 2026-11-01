import { useQuery, useQueryClient } from "@tanstack/react-query";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Box, Copy, FilePlus2 } from "lucide-react";
import { useEffect, useSyncExternalStore } from "react";
import type { CaseDocument } from "../api/cases";
import type { Run } from "../api/contracts";
import { devicesLabel } from "../api/devices";
import {
  type HistoryColumns,
  planQuery,
  runQuery,
  runsQuery,
  shownOrders,
  soleForces,
  useLiveRun,
} from "../api/runs";
import { coefficient, coefficientsCaveat, degrees } from "../api/units";
import { ResidualChart } from "../components/ResidualChart";
import { formatBytes, formatCount, remainingMs } from "../components/run/format";
import { LogView } from "../components/run/LogView";
import { MiddleTruncate } from "../components/run/MiddleTruncate";
import { MonitorTable } from "../components/run/MonitorTable";
import { OutputList } from "../components/run/OutputList";
import { PlanTable } from "../components/run/PlanTable";
import { TopBar } from "../components/shell/TopBar";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Collapse } from "../components/ui/Collapse";
import { cn } from "../components/ui/cn";
import { Gauge } from "../components/ui/Gauge";
import { LoadingDots } from "../components/ui/LoadingDots";
import { MetaGrid } from "../components/ui/MetaGrid";
import { Note } from "../components/ui/Note";
import { RelativeTime } from "../components/ui/RelativeTime";
import { Skeleton } from "../components/ui/Skeleton";
import { Snippet } from "../components/ui/Snippet";
import { formatDuration, Status } from "../components/ui/Status";
import { TabList, TabPanel, TabsRoot } from "../components/ui/Tabs";
import { useToast } from "../components/ui/Toast";
import { Tooltip } from "../components/ui/Tooltip";

const TABS = ["run", "log", "case", "outputs"] as const;
type Tab = (typeof TABS)[number];

export const Route = createFileRoute("/runs/$runId")({
  validateSearch: (raw: Record<string, unknown>): { tab?: Tab | undefined } =>
    TABS.includes(raw.tab as Tab) && raw.tab !== "run" ? { tab: raw.tab as Tab } : {},
  component: RunView,
});

const second = {
  subscribe: (listener: () => void) => {
    const timer = setInterval(listener, 1000);
    return () => clearInterval(timer);
  },
  now: () => Math.floor(Date.now() / 1000) * 1000,
};

function RunView() {
  const { runId } = Route.useParams();
  const { tab = "run" } = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });
  const { run, history, log, connected, invalid } = useLiveRun(runId);
  const warnings = log.lines.filter((line) => line.level === "warning").length;
  const errors = log.lines.filter((line) => line.level === "error").length;
  const details = useQuery({
    ...runQuery(runId),
    // Outputs grow while the run writes them; the status change brings solution.cgns.
    refetchInterval: run?.status === "running" ? 5000 : false,
  });
  const client = useQueryClient();
  // Refetch once the run has ended, or whenever the details lag its status: the registry
  // stream updates the cached run.json but not the outputs (solution.cgns comes at the end).
  const status = run?.status;
  const stale = Boolean(run && details.data && details.data.run.status !== run.status);
  // biome-ignore lint/correctness/useExhaustiveDependencies: refetch on each status change
  useEffect(() => {
    if (status && status !== "running")
      void client.invalidateQueries({ queryKey: runQuery(runId).queryKey });
  }, [status]);
  useEffect(() => {
    if (stale) void client.invalidateQueries({ queryKey: runQuery(runId).queryKey });
  }, [stale, client, runId]);
  const plan = useQuery(planQuery(runId));
  const toast = useToast();
  // While running, solution.cgns is not a result yet (results.md); after, it is the one to show.
  const hasSolution =
    run?.status !== "running" &&
    Boolean(details.data?.outputs.some((o) => o.name === "solution.cgns"));
  const now = useSyncExternalStore(second.subscribe, second.now);
  const time = details.data?.case.time;
  const targetOrders = time?.kind === "steady" ? time.residual_orders : undefined;
  const maxIterations = time?.kind === "steady" ? time.max_iterations : time?.steps;
  const iteration = history.iteration.at(-1) ?? 0;
  const orders = shownOrders(history.orders.at(-1), targetOrders);
  const forces = soleForces(history);
  const unsteady = (details.data?.case.time as { kind?: string } | undefined)?.kind === "unsteady";
  const caveat = run ? coefficientsCaveat(run.status, unsteady) : null;
  // An ended run's coefficients are its final state (results.md#runjson monitors), which can
  // come after the last recorded sample; a running one's are its last sample.
  const ended = run && run.status !== "running" ? run.monitors : null;
  const finalForces = ended
    ? Object.values(ended).filter((m) => m.cl !== undefined && m.cd !== undefined)
    : [];
  const [cl, cd] =
    finalForces.length === 1
      ? [finalForces[0]?.cl, finalForces[0]?.cd]
      : [forces?.cl.at(-1), forces?.cd.at(-1)];

  const crumbs = [
    { label: "Runs", to: "/" },
    {
      label: run?.case.name ?? "…",
      to: `/?case=${encodeURIComponent(JSON.stringify([run?.case.name ?? ""]))}`,
    },
    { label: <span className="font-mono">{runId.slice(-8)}</span> },
  ];

  return (
    <>
      <TopBar
        crumbs={crumbs}
        actions={
          <>
            <Button
              size="sm"
              icon={<Copy className="size-4" />}
              onClick={() =>
                void navigator.clipboard.writeText(runId).then(() => toast("Run id copied"))
              }
            >
              Copy Run Id
            </Button>
            <Button
              size="sm"
              icon={<FilePlus2 className="size-4" />}
              onClick={() => void navigate({ to: "/cases/new", search: { run: runId } })}
            >
              Edit a copy
            </Button>
            {hasSolution ? (
              <Button
                size="sm"
                icon={<Box className="size-4" />}
                onClick={() => void navigate({ to: "/viewer", search: { run: runId } })}
              >
                Open in Analysis
              </Button>
            ) : (
              <Tooltip content="The run has no solution.cgns yet">
                <span>
                  <Button size="sm" icon={<Box className="size-4" />} disabled>
                    Open in Analysis
                  </Button>
                </span>
              </Tooltip>
            )}
          </>
        }
      />
      <TabsRoot
        value={tab}
        onValueChange={(value) =>
          void navigate({
            search: { tab: value === "run" ? undefined : (value as Tab) },
            replace: true,
          })
        }
        className="flex min-h-0 flex-1 flex-col"
      >
        <div className="border-line border-b bg-bg-1 px-6 py-2">
          <TabList
            tabs={[
              { value: "run", label: "Run" },
              { value: "log", label: "Log" },
              { value: "case", label: "Case" },
              {
                value: "outputs",
                label: (
                  <span className="flex items-center gap-2">
                    Outputs
                    {details.data && <Badge>{details.data.outputs.length}</Badge>}
                  </span>
                ),
              },
            ]}
          />
        </div>
        <div className="min-h-0 flex-1 overflow-auto">
          <div className="mx-auto flex max-w-[1200px] flex-col gap-6 px-6 py-8">
            <header className="flex flex-wrap items-center gap-x-4 gap-y-1">
              <h1 className="m-0 font-semibold text-xl">
                {run?.case.name ?? <Skeleton className="h-8 w-40" />}
              </h1>
              {run && (
                <span data-testid="status">
                  <Status status={run.status} />
                </span>
              )}
              {run && (
                <Conditions
                  run={run}
                  document={details.data?.case}
                  cells={details.data?.mesh_info.counts.cells}
                />
              )}
              {run?.status === "running" && (
                <span className="flex items-center gap-2 text-fg-3 text-sm">
                  {connected ? (
                    "Live"
                  ) : (
                    <>
                      Reconnecting <LoadingDots />
                    </>
                  )}
                </span>
              )}
            </header>

            {invalid && (
              <Note tone="error" title="history.jsonl breaks the results contract">
                {invalid}
              </Note>
            )}
            {run?.diagnostics.map((d) => (
              <Note
                key={`${d.code}-${String(d.data.iteration)}`}
                tone={d.severity === "warning" ? "warning" : "error"}
                title={
                  <span>
                    <span className="font-mono">{d.code}</span> {d.message}
                  </span>
                }
              >
                {d.fix && <p className="m-0">{d.fix}</p>}
                {d.doc && (
                  <a
                    href={d.doc}
                    className="text-accent hover:underline"
                    target="_blank"
                    rel="noreferrer"
                  >
                    {d.code} reference
                  </a>
                )}
              </Note>
            ))}

            <TabPanel value="run" className="flex flex-col gap-6 outline-none">
              <section className="grid grid-cols-1 gap-6 rounded-lg border border-line bg-bg-0 p-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
                <div className="h-96 min-w-0 rounded-md border border-line bg-bg-1 p-3">
                  <ResidualChart
                    history={history}
                    floors={run?.convergence?.floor}
                    reference={run?.start?.residual_reference}
                    steady={time?.kind === "steady"}
                  />
                </div>
                {run ? (
                  <Facts
                    run={run}
                    now={now}
                    iteration={iteration}
                    maxIterations={maxIterations}
                    orders={orders}
                    targetOrders={targetOrders}
                    history={history}
                    cells={details.data?.mesh_info.counts.cells}
                    cl={cl}
                    cd={cd}
                    caveat={caveat}
                  />
                ) : (
                  <Skeleton className="h-48 w-full" />
                )}
              </section>

              <section className="rounded-lg border border-line bg-bg-0">
                <Collapse
                  title="Plan"
                  meta={
                    plan.data ? (
                      plan.data.ok ? (
                        <Badge tone="green">{plan.data.boundaries.length} zones covered</Badge>
                      ) : (
                        <Badge tone="red">{plan.data.diagnostics.length} problems</Badge>
                      )
                    ) : (
                      <LoadingDots />
                    )
                  }
                >
                  {plan.data && <PlanTable plan={plan.data} />}
                </Collapse>
                <Collapse
                  title="Monitors"
                  defaultOpen={history.monitors.size > 0 || !history.iteration.length}
                  meta={
                    <span
                      className={cn(
                        "font-mono text-sm tabular-nums",
                        caveat ? "text-fg-3" : "text-fg-2",
                      )}
                    >
                      {cl !== undefined && cd !== undefined
                        ? `CL ${coefficient(cl)} · CD ${coefficient(cd)}${caveat ? ` · ${caveat}` : ""}`
                        : "—"}
                    </span>
                  }
                >
                  <MonitorTable history={history} final={ended} caveat={caveat} />
                </Collapse>
                <Collapse
                  title="Outputs"
                  meta={
                    details.data && (
                      <span className="text-fg-2 text-sm">
                        {details.data.outputs.length} files ·{" "}
                        {formatBytes(details.data.outputs.reduce((s, o) => s + o.bytes, 0))}
                      </span>
                    )
                  }
                >
                  {details.data && <OutputList outputs={details.data.outputs} />}
                </Collapse>
                <Collapse
                  title="Log"
                  meta={
                    <span className="flex items-center gap-2 text-fg-2 text-sm tabular-nums">
                      {log.lines.length.toLocaleString("en")} lines
                      {warnings > 0 && <Badge tone="amber">{warnings} warnings</Badge>}
                      {errors > 0 && <Badge tone="red">{errors} errors</Badge>}
                    </span>
                  }
                >
                  <LogView
                    history={history}
                    log={log}
                    running={run?.status === "running"}
                    height={240}
                  />
                </Collapse>
              </section>
            </TabPanel>

            <TabPanel value="log" className="outline-none">
              <LogView history={history} log={log} running={run?.status === "running"} />
            </TabPanel>

            <TabPanel value="case" className="flex flex-col gap-4 outline-none">
              <Snippet text={`autan run case.toml --mesh-info mesh-info.json`} />
              <pre className="m-0 overflow-auto rounded-lg border border-line bg-bg-0 p-4 font-mono text-sm leading-6">
                {details.data?.case_toml}
              </pre>
            </TabPanel>

            <TabPanel
              value="outputs"
              className="rounded-lg border border-line bg-bg-0 px-4 outline-none"
            >
              {details.data && <OutputList outputs={details.data.outputs} />}
            </TabPanel>
          </div>
        </div>
      </TabsRoot>
    </>
  );
}

function Facts({
  run,
  now,
  iteration,
  maxIterations,
  orders,
  targetOrders,
  history,
  cells,
  cl,
  cd,
  caveat,
}: {
  run: Run;
  now: number;
  iteration: number;
  maxIterations: number | undefined;
  orders: number | undefined;
  targetOrders: number | undefined;
  history: HistoryColumns;
  cells: number | undefined;
  cl: number | undefined;
  cd: number | undefined;
  caveat: string | null;
}) {
  const registry = useQuery(runsQuery);
  const known = new Set(registry.data?.runs.map((r) => r.run_id));
  const started = new Date(run.started).getTime();
  const end = run.finished ? new Date(run.finished).getTime() : now;
  const left =
    run.status === "running"
      ? remainingMs({
          iteration,
          elapsedMs: now - started,
          maxIterations,
          orders: history.orders,
          iterations: history.iteration,
          targetOrders,
        })
      : null;
  return (
    <MetaGrid
      items={[
        {
          label: "Iteration",
          value:
            run.status === "running" ? (
              <>
                {maxIterations !== undefined && (
                  <Gauge
                    value={iteration}
                    max={maxIterations}
                    size={20}
                    label="Iteration budget used"
                  />
                )}
                <span className="font-mono tabular-nums" data-testid="iteration">
                  {iteration.toLocaleString("en")}
                </span>
                {maxIterations !== undefined && (
                  <span className="font-mono text-fg-3 tabular-nums">
                    / {maxIterations.toLocaleString("en")}
                  </span>
                )}
              </>
            ) : (
              <span className="font-mono tabular-nums" data-testid="iteration">
                {iteration.toLocaleString("en")}
              </span>
            ),
        },
        {
          label: "Residual drop",
          value:
            orders !== undefined && Number.isFinite(orders) ? (
              <>
                {targetOrders !== undefined && (
                  <Gauge
                    value={orders}
                    max={targetOrders}
                    size={20}
                    label="Residual drop towards the target"
                  />
                )}
                <span className="font-mono tabular-nums" data-testid="orders">
                  {orders.toFixed(2)}
                </span>
                {targetOrders !== undefined && (
                  <span className="font-mono text-fg-3">/ {targetOrders} orders</span>
                )}
              </>
            ) : (
              "—"
            ),
        },
        {
          label: "Devices",
          value: <span className="font-mono">{devicesLabel(run.devices)}</span>,
        },
        ...(run.timing
          ? [
              {
                label: "Iteration cost",
                value: (
                  <span className="flex flex-col" data-testid="timing">
                    <span className="font-mono tabular-nums">
                      {run.timing.ns_per_cell_iteration.toPrecision(3)} ns per cell
                    </span>
                    <span className="text-fg-3 text-sm tabular-nums">
                      {run.timing.solve_seconds.toPrecision(3)} s solving of{" "}
                      {run.timing.wall_seconds.toPrecision(3)} s
                    </span>
                  </span>
                ),
              },
            ]
          : []),
        { label: "Started", value: <RelativeTime iso={run.started} /> },
        ...(run.start
          ? [
              {
                label: run.start.mode === "continue" ? "Continues" : "Warm start from",
                // A predecessor read from another runs directory has no page here.
                value: known.has(run.start.run_id) ? (
                  <Link
                    to="/runs/$runId"
                    params={{ runId: run.start.run_id }}
                    className="font-mono text-accent hover:underline"
                  >
                    {run.start.run_id.slice(-8)}
                  </Link>
                ) : (
                  <Tooltip content={`${run.start.run_id}, not in this runs directory`}>
                    <span className="font-mono text-fg-3">{run.start.run_id.slice(-8)}</span>
                  </Tooltip>
                ),
              },
            ]
          : []),
        {
          label: run.status === "running" ? "Elapsed · left" : "Duration",
          value: (
            <span className="tabular-nums">
              {formatDuration(end - started)}
              {left !== null && <span className="text-fg-3"> · ≤ {formatDuration(left)}</span>}
            </span>
          ),
        },
        {
          label: "Mesh",
          value: (
            <>
              <span className="font-mono" data-testid="mesh-name">
                {run.mesh.path.split("/").pop()}
              </span>
              <MiddleTruncate value={run.mesh.sha256} keep={8} />
              {cells !== undefined && <span className="text-fg-3">{formatCount(cells)} cells</span>}
            </>
          ),
        },
        {
          label: "Forces",
          value:
            cl !== undefined && cd !== undefined ? (
              <span
                className={cn("flex flex-col font-mono tabular-nums", caveat && "text-fg-3")}
                data-testid="forces"
              >
                <span>CL {coefficient(cl)}</span>
                <span>CD {coefficient(cd)}</span>
                {caveat && <span className="font-sans text-sm">{caveat}</span>}
              </span>
            ) : (
              "—"
            ),
        },
        {
          label: "Solver",
          value: (
            <span className="flex items-center gap-2">
              <MiddleTruncate value={run.solver.commit} keep={8} />
              {run.solver.backend === "fake" && <Badge tone="amber">fake</Badge>}
            </span>
          ),
        },
      ]}
    />
  );
}

/** What the run computed, in a line, as its stored case writes it: Mach or velocity, angles,
 * Reynolds number or pressure and temperature, model, precision, and the mesh's cells. */
function Conditions({
  run,
  document,
  cells,
}: {
  run: Run;
  document: CaseDocument | undefined;
  cells: number | undefined;
}) {
  const freestream = (document?.freestream ?? {}) as Record<string, string | number | undefined>;
  const physics = (document?.physics ?? {}) as { turbulence?: { model?: string } };
  const angle = (value: unknown) => (typeof value === "string" ? degrees(value) : null);
  const alpha = angle(freestream.aoa);
  const beta = angle(freestream.sideslip);
  const parts = [
    freestream.mach !== undefined && `M ${freestream.mach}`,
    freestream.velocity !== undefined && `V ${freestream.velocity}`,
    alpha !== null && `α ${alpha}°`,
    beta !== null && beta !== 0 && `β ${beta}°`,
    typeof freestream.re === "number" && `Re ${freestream.re.toExponential(2)}`,
    freestream.pressure !== undefined && `p ${freestream.pressure}`,
    freestream.temperature !== undefined && `T ${freestream.temperature}`,
    physics.turbulence?.model,
    run.precision,
    cells !== undefined && `${formatCount(cells)} cells`,
  ].filter(Boolean);
  return (
    <p className="m-0 basis-full font-mono text-fg-2 text-sm" data-testid="conditions">
      {parts.join(" · ")}
    </p>
  );
}
