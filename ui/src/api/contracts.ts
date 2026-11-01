// Shapes of docs/contracts/results.md (autan.results/1) and case-schema.md diagnostics.

export type RunStatus = "running" | "converged" | "completed" | "stopped" | "diverged" | "failed";

export const RUN_STATUSES: readonly RunStatus[] = [
  "running",
  "converged",
  "completed",
  "stopped",
  "diverged",
  "failed",
];

export interface Diagnostic {
  code: string;
  severity: "error" | "warning" | "divergence";
  message: string;
  location: { file?: string; line?: number; path?: string } | null;
  fix: string | null;
  doc: string | null;
  data: Record<string, unknown>;
}

export interface Run {
  schema: "autan.results/1";
  run_id: string;
  identity: string;
  status: RunStatus;
  case: { name: string; sha256: string };
  mesh: { path: string; sha256: string };
  solver: { version: string; commit: string; backend: "cuda" | "hip" | "cpu" | "fake" };
  precision: "fp32+acc64" | "fp64";
  deterministic: boolean;
  devices: { kind: "cuda" | "hip" | "cpu"; count: number };
  started: string;
  finished: string | null;
  iterations: number;
  time: number | null;
  averaging: Record<string, number> | null;
  diagnostics: Diagnostic[];
  /** Where the run started from, null from the freestream (results.md#runjson). */
  start: RunStart | null;
  /** Round-off floor of every equation of a steady fp32+acc64 run (case-schema.md#time). */
  convergence: { floor: Record<string, number> } | null;
  /** Timing of this run alone, null while running, without iterations and for the fake backend
   * (results.md#runjson); bench/ measures performance. */
  timing: {
    wall_seconds: number;
    solve_seconds: number;
    iterations: number;
    ns_per_cell_iteration: number;
  } | null;
  /** Every monitor's record on the final state, null while running and for diverged and
   * failed runs (results.md#runjson). */
  monitors: Record<string, Forces & { iteration: number }> | null;
}

export interface RunStart {
  run_id: string;
  identity: string;
  mode: "continue" | "warm";
  iterations: number;
  time: number | null;
  solution_sha256: string;
  residual_reference: Record<string, number> | null;
  woken: string[] | null;
}

export interface Forces {
  fx: number;
  fy: number;
  fz: number;
  mx: number;
  my: number;
  mz: number;
  cl?: number;
  cd?: number;
  cs?: number;
  cmx?: number;
  cmy?: number;
  cmz?: number;
}

export interface HistoryRecord {
  iteration: number;
  time: number | null;
  residuals: Record<string, number>;
  monitors?: Record<string, Forces>;
}

export interface RunRegistry {
  runs: Run[];
  rejected: { dir: string; reason: string }[];
  /** Per run id: last recorded iteration and residual drop, from history.jsonl. */
  summaries: Record<string, RunSummary>;
  /** Runs that have ended and written solution.cgns, which Analysis shows. */
  solutions: string[];
}

export interface RunSummary {
  /** Null until the history records an iteration; run.json's count holds then. */
  iteration: number | null;
  /** Orders of magnitude the slowest residual has dropped since iteration 1. */
  orders: number | null;
  /** `orders` along the run, at most 48 points. */
  trend: number[];
  /** The case's `time.residual_orders` (steady) and iteration or step budget. */
  target_orders: number | null;
  max_iterations: number | null;
  /** The settings studies usually sweep, as the case writes them ("19.57 deg"). */
  settings?: RunSettings;
}

export interface RunSettings {
  aoa: string | null;
  sideslip: string | null;
  mach: number | null;
  re: number | null;
  turbulence: string | null;
  precision: string | null;
}

/** One line of log.jsonl (results.md#log), with the page documenting its code. */
export interface LogLine {
  time: string;
  iteration: number | null;
  level: "info" | "warning" | "error";
  source: string;
  message: string;
  code: string | null;
  data: Record<string, unknown>;
  doc: string | null;
}
