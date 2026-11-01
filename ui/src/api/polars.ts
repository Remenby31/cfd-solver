import type { CaseDocument } from "./cases";
import type { Run } from "./contracts";
import { degrees } from "./units";

/** Every leaf of a case document under its dotted path; arrays of tables by index. */
export function flatten(value: unknown, prefix = ""): Map<string, string> {
  const out = new Map<string, string>();
  if (value !== null && typeof value === "object") {
    for (const [key, child] of Object.entries(value)) {
      const path = prefix ? (Array.isArray(value) ? `${prefix}[${key}]` : `${prefix}.${key}`) : key;
      for (const [k, v] of flatten(child, path)) out.set(k, v);
    }
  } else {
    out.set(
      prefix,
      value === undefined ? "—" : typeof value === "string" ? value : JSON.stringify(value),
    );
  }
  return out;
}

/** One compared run, with its letter on the Compare screen and the case it ran (run.json and
 * case.toml of its directory). */
export interface PolarRun {
  id: string;
  role: string;
  run: Run | undefined;
  case: CaseDocument | undefined;
  /** Its case is still being read: neither drawn nor reported yet. */
  pending?: boolean;
}

export interface PolarPoint {
  id: string;
  role: string;
  alpha: number;
  cl: number;
  cd: number;
}

/** Runs that solve the same problem but for the angle of attack, in angle order; `label`
 * names the settings that set it apart from the other series. */
export interface PolarSeries {
  key: string;
  label: string;
  points: PolarPoint[];
}

export interface Polars {
  series: PolarSeries[];
  /** Settings every series shares that a reader looks for first, e.g. "M 0.8 · inviscid". */
  fixed: string;
  /** The compared runs not on the polars, each with the reason. */
  unplotted: { id: string; role: string; reason: string }[];
}

// Settings that do not change the problem a polar point solves: the case's name, the swept
// angle, the iteration budget and stopping rule, how the solver reproduces itself, and the
// monitors (the one giving the coefficients is compared on its own, below). The mesh file's
// name is replaced by the hash of what it holds.
const IGNORED = /^(schema|name|mesh|freestream\.aoa|time\..*|physics\.reproducibility|monitors.*)$/;

const NAMES: Record<string, (v: string) => string> = {
  "freestream.mach": (v) => `M ${v}`,
  "freestream.re": (v) => `Re ${v}`,
  "freestream.sideslip": (v) => `β ${v}`,
  "physics.turbulence.model": (v) => v,
  "physics.precision": (v) => v,
  mesh: (v) => `mesh ${v.slice(0, 7)}`,
};
const say = (key: string, value: string) => NAMES[key]?.(value) ?? `${key} ${value}`;

/** The coefficients a converged run ends with: its one forces monitor giving CL and CD, on the
 * final state run.json records (results.md#runjson), with that monitor's name; otherwise why
 * there are none. */
function ended(run: Run): { name: string; cl: number; cd: number } | string {
  if (run.status !== "converged") return `${run.status}, not converged`;
  return finalCoefficients(run);
}

/** The CL and CD of an ended run's one forces monitor giving them, on its final state
 * (results.md#runjson monitors), with that monitor's name; otherwise why there are none. */
export function finalCoefficients(run: Run): { name: string; cl: number; cd: number } | string {
  const coefficients = Object.entries(run.monitors ?? {}).filter(
    ([, m]) => m.cl !== undefined && m.cd !== undefined,
  );
  const [only] = coefficients;
  if (!only) return "no forces monitor with CL and CD";
  if (coefficients.length > 1) return "several forces monitors with CL and CD";
  return { name: only[0], cl: only[1].cl as number, cd: only[1].cd as number };
}

// What the run solves, as dotted settings: the case without what IGNORED drops, the mesh by
// hash, and the coefficients' monitor without its name and interval.
function problem(run: Run, document: CaseDocument, monitor: string): Map<string, string> {
  const settings = new Map([...flatten(document)].filter(([key]) => !IGNORED.test(key)));
  settings.set("mesh", run.mesh.sha256);
  const monitors = Array.isArray(document.monitors) ? document.monitors : [];
  const forces =
    monitors.find((m) => (m as { name?: string }).name === monitor) ??
    (monitor === "forces" ? monitors.find((m) => !(m as { name?: string }).name) : undefined);
  for (const [key, value] of flatten(forces ?? {}, "forces"))
    if (key !== "forces.name" && key !== "forces.every") settings.set(key, value);
  return settings;
}

/** Lift and drag polars of the compared runs: one series per problem solved but for the angle
 * of attack (same case settings, mesh and coefficient monitor), every converged run with
 * coefficients a point of its series, none merged, and every other run listed with the reason
 * it is not there. Settings written differently but equal (`15 deg`, `0.2618 rad`) are
 * different series: the case is compared as written. */
export function polars(compared: PolarRun[]): Polars {
  const unplotted: Polars["unplotted"] = [];
  const groups = new Map<string, { settings: Map<string, string>; points: PolarPoint[] }>();
  for (const { id, role, run, case: document, pending } of compared) {
    if (pending) continue;
    if (!run || !document) {
      unplotted.push({ id, role, reason: "not available" });
      continue;
    }
    const coefficients = ended(run);
    if (typeof coefficients === "string") {
      unplotted.push({ id, role, reason: coefficients });
      continue;
    }
    const freestream = (document.freestream ?? {}) as { aoa?: string };
    // The SDK's default angle of attack is 0 (case-schema.md#freestream).
    const alpha = freestream.aoa === undefined ? 0 : degrees(freestream.aoa);
    if (alpha === null) {
      unplotted.push({ id, role, reason: "angle of attack not in deg or rad" });
      continue;
    }
    const settings = problem(run, document, coefficients.name);
    const key = JSON.stringify([...settings].sort(([a], [b]) => a.localeCompare(b)));
    const group = groups.get(key) ?? { settings, points: [] };
    group.points.push({ id, role, alpha, cl: coefficients.cl, cd: coefficients.cd });
    groups.set(key, group);
  }
  const all = [...groups.entries()];
  const keys = [...new Set(all.flatMap(([, g]) => [...g.settings.keys()]))].sort();
  const differing = keys.filter((k) => new Set(all.map(([, g]) => g.settings.get(k))).size > 1);
  const series = all.map(([key, g]) => ({
    key,
    label: differing.map((k) => say(k, g.settings.get(k) ?? "—")).join(" · "),
    points: g.points.sort((a, b) => a.alpha - b.alpha || a.role.localeCompare(b.role)),
  }));
  series.sort((a, b) => a.label.localeCompare(b.label, "en", { numeric: true }));
  const first = all[0]?.[1].settings;
  const fixed = first
    ? Object.keys(NAMES)
        .filter((k) => k !== "mesh" && first.has(k) && !differing.includes(k))
        // No sideslip is the usual case, not worth saying.
        .filter((k) => k !== "freestream.sideslip" || degrees(first.get(k)) !== 0)
        .map((k) => say(k, first.get(k) as string))
        .join(" · ")
    : "";
  return { series, fixed, unplotted };
}
