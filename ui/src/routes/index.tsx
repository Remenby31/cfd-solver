import { useQuery } from "@tanstack/react-query";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import {
  ChevronRight,
  FolderSearch,
  GitCompareArrows,
  HardDrive,
  ListFilter,
  SearchX,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  RUN_STATUSES,
  type Run,
  type RunSettings,
  type RunStatus,
  type RunSummary,
} from "../api/contracts";
import { finalCoefficients } from "../api/polars";
import { runsQuery } from "../api/runs";
import { coefficient, degrees } from "../api/units";
import { Peek } from "../components/runs/Peek";
import { SelectionBar } from "../components/runs/SelectionBar";
import { SelectionContext, useSelection } from "../components/runs/selection";
import { TopBar } from "../components/shell/TopBar";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Checkbox } from "../components/ui/Checkbox";
import { Collapse } from "../components/ui/Collapse";
import { cn } from "../components/ui/cn";
import { EmptyState } from "../components/ui/EmptyState";
import { FilterChip } from "../components/ui/FilterChip";
import { Menu, MenuItem } from "../components/ui/Menu";
import { Note } from "../components/ui/Note";
import { RelativeTime } from "../components/ui/RelativeTime";
import { SearchInput } from "../components/ui/SearchInput";
import { Skeleton } from "../components/ui/Skeleton";
import { Snippet } from "../components/ui/Snippet";
import { Sparkline } from "../components/ui/Sparkline";
import { StateGlyph } from "../components/ui/StateGlyph";
import { formatDuration, Status, useTicker } from "../components/ui/Status";
import { useToast } from "../components/ui/Toast";
import { Tooltip } from "../components/ui/Tooltip";

interface Search {
  status?: RunStatus[] | undefined;
  case?: string[] | undefined;
  q?: string | undefined;
  flat?: boolean | undefined;
}

const strings = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((v): v is string => typeof v === "string") : [];

export const Route = createFileRoute("/")({
  validateSearch: (raw: Record<string, unknown>): Search => {
    const search: Search = {};
    const status = strings(raw.status).filter((s): s is RunStatus =>
      RUN_STATUSES.includes(s as RunStatus),
    );
    if (status.length) search.status = status;
    const cases = strings(raw.case);
    if (cases.length) search.case = cases;
    if (typeof raw.q === "string" && raw.q) search.q = raw.q;
    if (raw.flat === true) search.flat = true;
    return search;
  },
  component: Runs,
});

function matches(run: Run, search: Search): boolean {
  if (search.status?.length && !search.status.includes(run.status)) return false;
  if (search.case?.length && !search.case.includes(run.case.name)) return false;
  if (!search.q) return true;
  const q = search.q.toLowerCase();
  return [run.case.name, run.run_id, run.precision, run.solver.backend, run.status].some((v) =>
    v.toLowerCase().includes(q),
  );
}

const COLUMNS =
  "grid grid-cols-[minmax(140px,1.4fr)_minmax(140px,1fr)_40px_56px_64px_64px_minmax(100px,1fr)_64px_88px_40px] items-center gap-4";

function Runs() {
  const search = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });
  const { data, error, isPending } = useQuery(runsQuery);
  const filter = useRef<HTMLInputElement>(null);
  const [selected, setSelected] = useState<ReadonlySet<string>>(() => new Set());
  const [peek, setPeek] = useState<string | null>(null);
  const toggle = useCallback(
    (runId: string) =>
      setSelected((s) => {
        const next = new Set(s);
        if (next.has(runId)) next.delete(runId);
        else next.add(runId);
        return next;
      }),
    [],
  );
  const rows = useRef<HTMLDivElement>(null);
  const runs = useMemo(() => (data?.runs ?? []).filter((r) => matches(r, search)), [data, search]);
  const cases = useMemo(
    () => [...new Set((data?.runs ?? []).map((r) => r.case.name))].sort(),
    [data],
  );
  const groups = useMemo(() => {
    const byCase = new Map<string, Run[]>();
    for (const run of runs) byCase.set(run.case.name, [...(byCase.get(run.case.name) ?? []), run]);
    return [...byCase.entries()];
  }, [runs]);
  const setSearch = (patch: Partial<Search>) =>
    void navigate({ search: (s: Search) => ({ ...s, ...patch }), replace: true });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.closest("input, textarea, [contenteditable], [role=dialog]")) return;
      if (e.key === "/") {
        e.preventDefault();
        filter.current?.focus();
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const links = [...(rows.current?.querySelectorAll<HTMLAnchorElement>("a[data-row]") ?? [])];
      const row = (document.activeElement as HTMLElement | null)?.closest("li[data-run-row]");
      const focused = links.find((link) => link === document.activeElement || row?.contains(link));
      const runId = focused?.dataset.run;
      if (e.key === "j" || e.key === "k") {
        const at = focused ? links.indexOf(focused) : -1;
        const next = e.key === "j" ? Math.min(at + 1, links.length - 1) : Math.max(at - 1, 0);
        links[next]?.focus();
        // An open preview follows the focused row.
        setPeek((open) => (open ? (links[next]?.dataset.run ?? open) : open));
      } else if (e.key === "x" && runId) {
        toggle(runId);
      } else if (e.key === " " && runId && !target.closest("button, [role=checkbox]")) {
        e.preventDefault();
        setPeek((open) => (open === runId ? null : runId));
      } else if (e.key === "c") {
        compareSelected.current();
      } else if (e.key === "Escape") {
        // The innermost layer first: an open preview closes before the selection clears.
        if (peekOpen.current) setPeek(null);
        else setSelected(new Set());
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle]);

  const compare = (ids: string[]) =>
    void navigate({ to: "/compare", search: { runs: ids.slice(0, 8) } });
  // Compares in selection order, whatever the filters show; read through refs by the key handler.
  const compareSelected = useRef(() => {});
  compareSelected.current = () => {
    const known = new Set(data?.runs.map((r) => r.run_id));
    const ids = [...selected].filter((id) => known.has(id));
    if (ids.length >= 2) compare(ids);
  };
  const peekOpen = useRef(false);
  peekOpen.current = peek !== null;
  const peekRun = peek ? data?.runs.find((r) => r.run_id === peek) : undefined;

  const filtered = Boolean(search.status?.length || search.case?.length || search.q);
  const grouped = !search.flat;

  return (
    <SelectionContext.Provider value={{ selected, toggle, peek }}>
      <TopBar crumbs={[{ label: "Runs" }]} />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto flex max-w-[1200px] flex-col gap-5 px-6 py-8">
          <h1 className="m-0 font-semibold text-2xl tracking-tight">Runs</h1>
          <div className="flex flex-wrap items-center gap-2">
            <SearchInput
              ref={filter}
              value={search.q ?? ""}
              onChange={(q) => setSearch({ q: q || undefined })}
              placeholder="Search runs…"
              label="Search runs"
              shortcut="/"
            />
            <FilterChip
              label="State"
              options={RUN_STATUSES}
              selected={search.status ?? []}
              onChange={(status) => setSearch({ status: status.length ? status : undefined })}
              render={(s) => (
                <span className="inline-flex items-center gap-1.5">
                  <StateGlyph status={s} />
                  {s}
                </span>
              )}
            />
            <FilterChip
              label="Case"
              options={cases}
              selected={search.case ?? []}
              onChange={(c) => setSearch({ case: c.length ? c : undefined })}
              render={(c) => c}
            />
            {cases.length > 0 && (
              <Button
                size="md"
                variant="tertiary"
                icon={<ListFilter className="size-4" />}
                aria-pressed={grouped}
                onClick={() => setSearch({ flat: grouped ? true : undefined })}
                className="ml-auto aria-pressed:bg-bg-3"
              >
                Group by case
              </Button>
            )}
          </div>

          {error && (
            <Note tone="error" title="The server did not answer">
              {String(error)}
            </Note>
          )}

          {isPending ? (
            <TableFrame>
              {["a", "b", "c", "d"].map((k) => (
                <li
                  key={k}
                  className={cn(COLUMNS, "h-14 border-line border-t px-4 first:border-t-0")}
                >
                  <Skeleton className="h-4 w-32" />
                  <Skeleton className="h-4 w-24" />
                  <Skeleton className="h-4 w-28" />
                  <Skeleton className="h-4 w-12" />
                  <Skeleton className="h-4 w-24" />
                  <Skeleton className="h-4 w-14" />
                </li>
              ))}
            </TableFrame>
          ) : !data?.runs.length ? (
            <EmptyState
              icon={<HardDrive className="size-5" />}
              title="No runs yet"
              description="Runs started from the SDK, the CLI or an agent appear here as soon as their directory is written."
            >
              <Snippet text="autan run case.py --mesh-info mesh-info.json" />
            </EmptyState>
          ) : !runs.length ? (
            <EmptyState icon={<SearchX className="size-5" />} title="No runs match these filters">
              <Button
                className="self-center"
                onClick={() => setSearch({ q: undefined, status: undefined, case: undefined })}
              >
                Clear filters
              </Button>
            </EmptyState>
          ) : (
            <div ref={rows}>
              <TableFrame
                header={
                  <div
                    aria-hidden
                    className={cn(
                      COLUMNS,
                      "h-10 border-line border-b pr-4 pl-10 text-fg-3 text-sm",
                    )}
                  >
                    <span>{grouped && !filtered ? "Run" : "Case"}</span>
                    <span>State</span>
                    <span className="text-right">M</span>
                    <span className="text-right">α</span>
                    <span className="text-right">CL</span>
                    <span className="text-right">CD</span>
                    <span>Drop (orders)</span>
                    <span className="text-right">Iteration</span>
                    <span className="text-right">Started</span>
                    <span />
                  </div>
                }
              >
                {grouped && !filtered ? (
                  groups.map(([name, members]) => (
                    <Group key={name} name={name} runs={members} summaries={data.summaries} />
                  ))
                ) : (
                  <AnimatePresence initial={false}>
                    {runs.map((run) => (
                      <RunRow key={run.run_id} run={run} summary={data.summaries[run.run_id]} />
                    ))}
                  </AnimatePresence>
                )}
              </TableFrame>
            </div>
          )}

          {data?.rejected.length ? (
            <div className="rounded-lg border border-line bg-bg-0" data-testid="rejected">
              <Collapse
                title={
                  <span className="flex items-center gap-2">
                    <FolderSearch className="size-4 text-warning" />
                    {data.rejected.length} unreadable run{" "}
                    {data.rejected.length === 1 ? "directory" : "directories"}
                  </span>
                }
              >
                <ul className="m-0 flex list-none flex-col gap-2 p-0 font-mono text-sm">
                  {data.rejected.map((r) => (
                    <li key={r.dir}>
                      <span>{r.dir}</span>
                      <span className="block text-fg-3">{r.reason}</span>
                    </li>
                  ))}
                </ul>
              </Collapse>
            </div>
          ) : null}
        </div>
      </div>
      <AnimatePresence>
        {peekRun && (
          <Peek
            key="peek"
            run={peekRun}
            summary={data?.summaries[peekRun.run_id]}
            onClose={() => setPeek(null)}
          />
        )}
        {selected.size > 0 && (
          <SelectionBar
            key="selection"
            count={selected.size}
            onCompare={() => compareSelected.current()}
            onClear={() => setSelected(new Set())}
          />
        )}
      </AnimatePresence>
    </SelectionContext.Provider>
  );
}

function TableFrame({ header, children }: { header?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-line bg-bg-0">
      <div className="min-w-[980px]">
        {header}
        <ul aria-label="Runs" data-testid="runs" className="m-0 list-none p-0">
          {children}
        </ul>
      </div>
    </div>
  );
}

function Group({
  name,
  runs,
  summaries,
}: {
  name: string;
  runs: Run[];
  summaries: Record<string, RunSummary>;
}) {
  const [open, setOpen] = useState(true);
  const navigate = useNavigate();
  const latest = runs[0];
  const sweep = study(runs.map((run) => summaries[run.run_id]?.settings));
  return (
    <li className="border-line border-t first:border-t-0">
      <div className="flex h-10 items-center gap-2 bg-bg-1 pr-2 text-sm">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
          className="flex h-full min-w-0 flex-1 items-center gap-2 px-4 text-left transition-colors hover:bg-bg-2"
        >
          <ChevronRight
            className={cn(
              "size-4 flex-none text-fg-3 transition-transform duration-[var(--dur-base)]",
              open && "rotate-90",
            )}
          />
          <span className="font-medium">{name}</span>
          <span className="text-fg-3">
            {runs.length} {runs.length === 1 ? "run" : "runs"}
          </span>
          {sweep && (
            <span className="flex min-w-0 items-center gap-2" data-testid="study">
              <Badge tone="purple">study</Badge>
              <span className="truncate font-mono text-fg-2 text-xs">{sweep}</span>
            </span>
          )}
          {latest && (
            <span className="ml-auto flex flex-none items-center gap-2 text-fg-3">
              latest <StateGlyph status={latest.status} />{" "}
              <RelativeTime iso={latest.started} card={false} />
            </span>
          )}
        </button>
        {runs.length > 1 && (
          <Button
            size="sm"
            variant="tertiary"
            icon={<GitCompareArrows className="size-4" />}
            onClick={() =>
              void navigate({
                to: "/compare",
                search: { runs: runs.slice(0, 8).map((r) => r.run_id) },
              })
            }
          >
            Compare
          </Button>
        )}
      </div>
      <ul className="m-0 list-none p-0">
        <AnimatePresence initial={false}>
          {open &&
            runs.map((run) => (
              <RunRow key={run.run_id} run={run} summary={summaries[run.run_id]} grouped />
            ))}
        </AnimatePresence>
      </ul>
    </li>
  );
}

/** "α 7.05–21.47 deg" for the settings that vary across a group; null when none does. Angles
 * are compared in degrees, whatever unit each case wrote. */
export function study(settings: (RunSettings | undefined)[]): string | null {
  const parts: string[] = [];
  const labels = {
    aoa: "α",
    sideslip: "β",
    mach: "Mach",
    re: "Re",
    turbulence: "turbulence",
    precision: "precision",
  };
  for (const key of ["aoa", "sideslip", "mach", "re", "turbulence", "precision"] as const) {
    const raw = settings.map((s) => s?.[key]).filter((v) => v != null);
    const angle = key === "aoa" || key === "sideslip";
    const numbers = raw.map((v) => {
      const n = angle ? degrees(String(v)) : typeof v === "number" ? v : null;
      // Equal settings written in other units must not look like a sweep after conversion.
      return n === null ? null : Math.round(n * 1e9) / 1e9;
    });
    if (numbers.length && numbers.every((n) => n !== null)) {
      const distinct = [...new Set(numbers as number[])];
      if (distinct.length < 2) continue;
      const range = `${Math.min(...distinct)}–${Math.max(...distinct)}`;
      parts.push(`${labels[key]} ${range}${angle ? " deg" : ""}`);
    } else {
      const distinct = [...new Set(raw.map(String))];
      if (distinct.length > 1) parts.push(`${labels[key]} ${distinct.join(" · ")}`);
    }
  }
  return parts.length ? parts.join(", ") : null;
}

function duration(run: Run): string {
  const end = run.finished ? new Date(run.finished).getTime() : Date.now();
  return formatDuration(end - new Date(run.started).getTime());
}

function RunRow({
  run,
  summary,
  grouped = false,
}: {
  run: Run;
  summary: RunSummary | undefined;
  grouped?: boolean;
}) {
  const toast = useToast();
  const { selected, toggle, peek } = useSelection();
  const isSelected = selected.has(run.run_id);
  useTicker(run.status === "running");
  const iteration = summary?.iteration ?? run.iterations;
  const target = summary?.target_orders ?? undefined;
  const alpha = degrees(summary?.settings?.aoa);
  const shownAlpha = alpha === null ? "" : String(Number(alpha.toFixed(2)));
  // The final coefficients once the run has ended (results.md#runjson); greyed unless converged.
  const coefficients = run.status === "running" ? "still running" : finalCoefficients(run);
  return (
    <motion.li
      data-run-row
      layout="position"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="group relative border-line border-t"
    >
      <Link
        to="/runs/$runId"
        params={{ runId: run.run_id }}
        data-row
        data-run={run.run_id}
        aria-current={peek === run.run_id ? "true" : undefined}
        className={cn(
          COLUMNS,
          isSelected && "bg-accent-soft hover:bg-accent-soft",
          "h-12 pr-4 pl-10 outline-none transition-colors duration-[var(--dur-fast)] hover:bg-bg-1 focus-visible:bg-bg-1 focus-visible:shadow-[inset_2px_0_0_var(--accent)]",
        )}
      >
        <span className="flex min-w-0 items-baseline gap-2">
          {!grouped && <span className="truncate font-medium">{run.case.name}</span>}
          <span className={cn("truncate font-mono", grouped ? "text-fg" : "text-fg-3 text-sm")}>
            {run.run_id.slice(-8)}
          </span>
          {run.solver.backend === "fake" && (
            <Tooltip content="Synthetic results from the fake backend">
              <span className="rounded-xs bg-warning-soft px-1 text-warning text-xs">fake</span>
            </Tooltip>
          )}
        </span>
        <span className="flex min-w-0 items-center gap-2">
          {/* The status shrinks first: the diagnostic's code always shows whole. */}
          <span className="min-w-0 overflow-hidden whitespace-nowrap">
            <Status status={run.status} duration={duration(run)} />
          </span>
          {run.diagnostics[0] && (
            <span className="flex-none font-mono text-danger text-xs">
              {run.diagnostics[0].code}
            </span>
          )}
        </span>
        <span className="text-right font-mono text-sm tabular-nums">
          <span className="sr-only">Mach </span>
          <span data-testid="mach">
            {summary?.settings?.mach == null
              ? "—"
              : String(Number(summary.settings.mach.toPrecision(3)))}
          </span>
        </span>
        <span className="text-right font-mono text-sm tabular-nums">
          <span className="sr-only">angle of attack </span>
          <span data-testid="alpha">{alpha === null ? "—" : `${shownAlpha}°`}</span>
        </span>
        {typeof coefficients === "string" ? (
          <>
            <span className="text-right text-fg-3">
              —<span className="sr-only"> CL: {coefficients}</span>
            </span>
            <span className="text-right text-fg-3">
              —<span className="sr-only"> CD: {coefficients}</span>
            </span>
          </>
        ) : (
          <>
            <span
              className={cn(
                "text-right font-mono text-sm tabular-nums",
                run.status !== "converged" && "text-fg-3",
              )}
            >
              <span className="sr-only">CL </span>
              <span data-testid="cl">{coefficient(coefficients.cl)}</span>
              {run.status !== "converged" && <span className="sr-only">, not converged</span>}
            </span>
            <span
              className={cn(
                "text-right font-mono text-sm tabular-nums",
                run.status !== "converged" && "text-fg-3",
              )}
            >
              <span className="sr-only">CD </span>
              <span data-testid="cd">{coefficient(coefficients.cd)}</span>
              {run.status !== "converged" && <span className="sr-only">, not converged</span>}
            </span>
          </>
        )}
        <span className="flex items-center gap-2">
          <Sparkline
            values={summary?.trend ?? []}
            {...(target !== undefined ? { target } : {})}
            label="Residual drop along the run"
            color={
              run.status === "diverged" || run.status === "failed"
                ? "var(--danger)"
                : run.status === "converged"
                  ? "var(--success)"
                  : "var(--accent)"
            }
          />
          {summary?.orders != null && (
            <span className="font-mono text-fg-2 text-xs tabular-nums">
              {summary.orders.toFixed(1)}
              {target !== undefined && <span className="text-fg-3">/{target}</span>}
            </span>
          )}
        </span>
        <span className="text-right font-mono tabular-nums" data-testid="iterations">
          {iteration.toLocaleString("en")}
        </span>
        <span className="text-right text-fg-2 text-sm">
          <RelativeTime iso={run.started} card={false} />
        </span>
        <span />
      </Link>
      <div
        className={cn(
          "absolute top-1/2 left-4 -translate-y-1/2 transition-opacity",
          selected.size
            ? "opacity-100"
            : "opacity-0 group-focus-within:opacity-100 group-hover:opacity-100",
        )}
      >
        <Checkbox
          checked={isSelected}
          onChange={() => toggle(run.run_id)}
          label={`Select ${run.case.name} ${run.run_id.slice(-8)}`}
        />
      </div>
      <div className="absolute top-1/2 right-3 -translate-y-1/2 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100 has-[[data-popup-open]]:opacity-100">
        <Menu label={`Actions for ${run.run_id}`}>
          <MenuItem
            onClick={() =>
              void navigator.clipboard.writeText(run.run_id).then(() => toast("Run id copied"))
            }
          >
            Copy run id
          </MenuItem>
          <MenuItem
            onClick={() =>
              void navigator.clipboard
                .writeText(run.mesh.sha256)
                .then(() => toast("Mesh hash copied"))
            }
          >
            Copy mesh hash
          </MenuItem>
        </Menu>
      </div>
    </motion.li>
  );
}
