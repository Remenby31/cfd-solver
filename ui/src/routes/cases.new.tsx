import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { Plus, Trash2 } from "lucide-react";
import { type ReactNode, useDeferredValue, useEffect, useRef, useState } from "react";
import {
  type BoundaryForm,
  type CaseDocument,
  type CaseForm,
  type CellZoneForm,
  CONDITIONS,
  type ConditionKind,
  caseDocument,
  differences,
  EMPTY_CASE,
  formFromDocument,
  materialsQuery,
  meshesQuery,
  renderQuery,
  UNITS,
} from "../api/cases";
import type { Diagnostic } from "../api/contracts";
import { globEscape, globMatches } from "../api/glob";
import { type Plan, runQuery } from "../api/runs";
import { MeshView } from "../components/case/MeshView";
import { TopBar } from "../components/shell/TopBar";
import { Button } from "../components/ui/Button";
import { cn } from "../components/ui/cn";
import { Field, NumberInput, QuantityInput, SelectInput, TextInput } from "../components/ui/Field";
import { Note } from "../components/ui/Note";
import { TabList, TabPanel, TabsRoot } from "../components/ui/Tabs";

export const Route = createFileRoute("/cases/new")({
  // `webgl`: draw the mesh with WebGL2, as on CPU-only machines without WebGPU. `run`: start
  // from a copy of that run's case.
  validateSearch: (raw: Record<string, unknown>): { webgl?: boolean; run?: string } => ({
    ...(raw.webgl === true || raw.webgl === "1" ? { webgl: true } : {}),
    ...(typeof raw.run === "string" && raw.run ? { run: raw.run } : {}),
  }),
  component: CaseEditorRoute,
});

// A new editor for each source: a copy's values never outlive the run they came from.
function CaseEditorRoute() {
  const { run } = Route.useSearch();
  return <CaseEditor key={run ?? "new"} />;
}

const TURBULENCE = [
  { value: "inviscid", label: "Inviscid" },
  { value: "laminar", label: "Laminar" },
  { value: "sa", label: "Spalart–Allmaras" },
  { value: "sst", label: "k-ω SST" },
] as const;
const PRECISION = [
  { value: "fp32+acc64", label: "fp32 + fp64 accumulation" },
  { value: "fp64", label: "fp64" },
] as const;
const KINDS = Object.keys(CONDITIONS).map((kind) => ({
  value: kind as ConditionKind,
  label: kind,
}));
const LABEL: Record<string, string> = {
  temperature: "Wall temperature",
  total_pressure: "Total pressure",
  total_temperature: "Total temperature",
  static_pressure: "Static pressure",
};

/** A case built field by field against a mesh, read back by the SDK as it is typed. */
function CaseEditor() {
  const { webgl = false, run: from } = Route.useSearch();
  const meshes = useQuery(meshesQuery);
  const materials = useQuery(materialsQuery);
  const library = materials.data ?? [];
  const gases = library.filter((m) => m.document.kind === "ideal-gas");
  const solids = library.filter((m) => m.document.kind === "solid");
  const [form, setForm] = useState<CaseForm>(EMPTY_CASE);
  // A copy of a run's case: read once the library materials are known, its mesh chosen among
  // the listed MeshInfos by the run's mesh hash.
  const source = useQuery({ ...runQuery(from ?? ""), enabled: from !== undefined });
  const [imported, setImported] = useState<CaseDocument | null>(null);
  useEffect(() => {
    if (!source.data || !materials.isSuccess || !meshes.isSuccess || imported) return;
    const copy = formFromDocument(source.data.case, materials.data);
    const sha = source.data.mesh_info.source.sha256;
    const listed = meshes.data.find((m) => m.source.sha256 === sha);
    setForm({ ...copy, mesh: listed?.name ?? "" });
    setImported(source.data.case);
  }, [source.data, materials, meshes, imported]);
  // What the copy cannot carry: the SDK's case.toml of the run's case against that of the form
  // it was read into; a line only in the first is a setting editing here would drop.
  const original = useQuery({ ...renderQuery(imported ?? {}, null), enabled: imported !== null });
  const reread = useQuery({
    ...renderQuery(
      imported ? caseDocument(formFromDocument(imported, library), library) : {},
      null,
    ),
    enabled: imported !== null,
  });
  // Settled only when both readings are in: a pending or failed comparison says so.
  const comparison: Comparison =
    imported === null
      ? { state: "none" }
      : original.isError || reread.isError
        ? { state: "failed", message: String(original.error ?? reread.error) }
        : !original.data || !reread.data
          ? { state: "checking" }
          : !original.data.ok
            ? { state: "failed", message: "the SDK refuses the run's own case" }
            : !reread.data.ok
              ? { state: "unrepresentable", diagnostics: reread.data.diagnostics }
              : {
                  state: "compared",
                  dropped: differences(original.data.document, reread.data.document),
                };
  const importing = from !== undefined && imported === null;
  const deferred = useDeferredValue(form);
  const [settled, setSettled] = useState(form);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(deferred), 250);
    return () => clearTimeout(timer);
  }, [deferred]);
  const rendered = useQuery(renderQuery(caseDocument(settled, library), settled.mesh || null));
  const mesh = meshes.data?.find((m) => m.name === form.mesh);
  const zones = mesh?.boundary_zones.map((z) => z.name) ?? [];
  const [picked, setPicked] = useState<string | null>(null);
  // After a change of mesh, the rules whose pattern matches none of its zones, offered for
  // removal; rules typed afterwards are the user's own business (the plan reports them).
  const [inherited, setInherited] = useState<{ mesh: string; rules: Set<object> } | null>(null);
  const cellNames = mesh?.cell_zones.map((z) => z.name) ?? [];
  const matchesNone = (pattern: string, names: string[]) =>
    !names.some((n) => globMatches(pattern, n));
  const isStale = (rule: { zones: string }, names: string[]) =>
    mesh !== undefined &&
    inherited?.mesh === form.mesh &&
    inherited.rules.has(rule) &&
    matchesNone(rule.zones, names);
  const stale = [
    ...form.boundaries.filter((r) => isStale(r, zones)),
    ...form.cellZones.filter((r) => isStale(r, cellNames)),
  ];
  const removeStale = () => {
    setForm((f) => ({
      ...f,
      boundaries: f.boundaries.filter((r) => !isStale(r, zones)),
      cellZones: f.cellZones.filter((r) => !isStale(r, cellNames)),
    }));
    setInherited(null);
  };

  // A zone picked in 3D: the rule whose pattern matches it gets the focus, or a new rule for
  // it, its condition suggested by the type the mesh file gives the zone. A periodic zone takes
  // no condition (case-schema.md#boundaries): nothing is added.
  const [focusRule, setFocusRule] = useState<number | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const pick = (zone: string) => {
    setPicked(zone);
    setNotice(null);
    const type = mesh?.boundary_zones.find((z) => z.name === zone)?.type ?? "";
    if (type === "periodic") {
      setNotice(
        `${zone} is periodic: it takes no condition, the solver pairs it with its partner.`,
      );
      return;
    }
    const index = form.boundaries.findIndex((r) => globMatches(r.zones, zone));
    if (index >= 0) {
      setFocusRule(index);
      return;
    }
    const typed: ConditionKind = type in CONDITIONS ? (type as ConditionKind) : "wall";
    // A no-slip wall needs viscous flow (E0414): an inviscid case walls it with slip.
    const kind = typed === "wall" && form.physics.turbulence === "inviscid" ? "slip" : typed;
    set("boundaries", [...form.boundaries, { zones: globEscape(zone), kind, values: {} }]);
    setFocusRule(form.boundaries.length);
  };
  // Focused once React has committed the rule's fieldset.
  useEffect(() => {
    if (focusRule === null) return;
    document
      .querySelector<HTMLInputElement>(
        `fieldset[aria-label="Boundary rule ${focusRule + 1}"] input`,
      )
      ?.focus();
    setFocusRule(null);
  }, [focusRule]);

  // The freestream is given by a Reynolds number or by a pressure (case-schema.md#freestream,
  // E0306); an inviscid flow has no Reynolds number to give, unless the case already has one.
  const [givenBy, setGivenBy] = useState<"re" | "pressure">("re");
  const inviscid = form.physics.turbulence === "inviscid";
  const given: "re" | "pressure" = form.freestream.re.trim()
    ? "re"
    : form.freestream.pressure.trim() || inviscid
      ? "pressure"
      : givenBy;
  const shown = rendered.data;
  const uncoveredZones =
    shown?.ok && shown.plan ? shown.plan.boundaries.filter((b) => b.condition === null).length : 0;
  const announcement =
    shown?.ok === false
      ? "The case is not complete yet."
      : uncoveredZones > 0
        ? `${uncoveredZones} ${uncoveredZones === 1 ? "zone" : "zones"} without a condition.`
        : shown?.plan?.ok
          ? "The plan is ready."
          : "";
  const set = <K extends keyof CaseForm>(key: K, value: CaseForm[K]) =>
    setForm((f) => ({ ...f, [key]: value }));
  const setIn = <K extends "physics" | "freestream" | "time">(
    key: K,
    field: keyof CaseForm[K],
    value: string,
  ) => setForm((f) => ({ ...f, [key]: { ...f[key], [field]: value } }));
  const cellZones = mesh?.cell_zones.map((z) => z.name) ?? [];
  const setCellZone = (k: number, zone: CellZoneForm) =>
    set(
      "cellZones",
      form.cellZones.map((z, i) => (i === k ? zone : z)),
    );
  const setRule = (k: number, rule: BoundaryForm) =>
    set(
      "boundaries",
      form.boundaries.map((r, i) => (i === k ? rule : r)),
    );

  return (
    <>
      <TopBar crumbs={[{ label: "Cases" }, { label: "New case" }]} />
      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_minmax(0,1fr)] overflow-hidden">
        <form
          className="flex flex-col gap-8 overflow-auto px-8 py-6"
          onSubmit={(e) => e.preventDefault()}
          aria-label="Case"
        >
          {source.isError && (
            <Note
              tone="error"
              title={`The run ${from} could not be read`}
              action={
                <Button size="sm" onClick={() => source.refetch()}>
                  Retry
                </Button>
              }
            >
              {source.error.message}
            </Note>
          )}
          {source.data && <ImportNote run={source.data.run.run_id} comparison={comparison} />}
          {materials.isError && (
            <Note
              tone="error"
              title="The material libraries could not be listed"
              action={
                <Button size="sm" onClick={() => materials.refetch()}>
                  Retry
                </Button>
              }
            >
              Gases and solids come from the SDK: {materials.error.message}
            </Note>
          )}
          {meshes.isError && (
            <Note tone="error" title="The meshes could not be listed">
              {meshes.error.message}
            </Note>
          )}
          <fieldset disabled={importing} className="m-0 flex flex-col gap-8 border-0 p-0">
            <Section title="Case">
              <Field label="Name">
                <TextInput value={form.name} onChange={(v) => set("name", v)} mono />
              </Field>
              <Field label="Mesh" hint={mesh && `${mesh.counts.cells.toLocaleString("en")} cells`}>
                <SelectInput
                  value={form.mesh}
                  onChange={(name) => {
                    const chosen = meshes.data?.find((m) => m.name === name);
                    setPicked(null);
                    setNotice(null);
                    setInherited({
                      mesh: name,
                      rules: new Set([...form.boundaries, ...form.cellZones]),
                    });
                    setForm((f) => ({
                      ...f,
                      mesh: name,
                      meshFile: chosen ? (chosen.source.path.split("/").pop() ?? "") : "",
                    }));
                  }}
                  options={(meshes.data ?? []).map((m) => ({ value: m.name, label: m.name }))}
                  placeholder="Choose a mesh…"
                />
              </Field>
            </Section>

            <Section title="Physics">
              <Field label="Turbulence">
                <SelectInput
                  value={form.physics.turbulence as (typeof TURBULENCE)[number]["value"] | ""}
                  onChange={(v) => setIn("physics", "turbulence", v)}
                  options={TURBULENCE}
                  placeholder="Choose…"
                />
              </Field>
              <Field label="Precision">
                <SelectInput
                  value={form.physics.precision as (typeof PRECISION)[number]["value"] | ""}
                  onChange={(v) => setIn("physics", "precision", v)}
                  options={PRECISION}
                  placeholder="Default (fp32 + fp64 accumulation)"
                />
              </Field>
              <Field label="Gas">
                <SelectInput
                  value={form.physics.gas}
                  onChange={(v) => setIn("physics", "gas", v)}
                  options={gases.map((m) => ({ value: m.name, label: m.document.name }))}
                  placeholder="Default (air)"
                />
              </Field>
            </Section>

            <Section
              title="Cell zones"
              action={
                <Button
                  size="sm"
                  icon={<Plus className="size-4" />}
                  onClick={() =>
                    set("cellZones", [
                      ...form.cellZones,
                      { zones: "", kind: "fluid", material: "" },
                    ])
                  }
                >
                  Add zone
                </Button>
              }
            >
              {form.cellZones.map((zone, k) => (
                <fieldset
                  // biome-ignore lint/suspicious/noArrayIndexKey: rules have no identity but their place
                  key={k}
                  aria-label={`Cell zone ${k + 1}`}
                  className="col-span-2 m-0 grid grid-cols-[minmax(0,1fr)_8rem_minmax(0,1fr)_auto] items-end gap-3 border-0 p-0"
                >
                  <Field label="Zones">
                    <TextInput
                      value={zone.zones}
                      onChange={(v) => setCellZone(k, { ...zone, zones: v })}
                      placeholder="plate"
                      mono
                      options={{ id: `cells-${k}`, values: cellZones }}
                    />
                  </Field>
                  <Field label="Medium">
                    <SelectInput
                      value={zone.kind}
                      onChange={(v) => v && setCellZone(k, { ...zone, kind: v, material: "" })}
                      options={[
                        { value: "fluid", label: "fluid" },
                        { value: "solid", label: "solid" },
                      ]}
                    />
                  </Field>
                  <Field label="Material">
                    <SelectInput
                      value={zone.material}
                      onChange={(v) => setCellZone(k, { ...zone, material: v })}
                      options={(zone.kind === "fluid" ? gases : solids).map((m) => ({
                        value: m.name,
                        label: m.document.name,
                      }))}
                      placeholder="Choose…"
                    />
                  </Field>
                  <Button
                    square
                    variant="tertiary"
                    aria-label={`Remove cell zone ${k + 1}`}
                    icon={<Trash2 className="size-4" />}
                    onClick={() =>
                      set(
                        "cellZones",
                        form.cellZones.filter((_, i) => i !== k),
                      )
                    }
                  />
                </fieldset>
              ))}
            </Section>

            <Section title="Freestream">
              <Field label="Mach">
                <NumberInput
                  value={form.freestream.mach}
                  onChange={(v) => setIn("freestream", "mach", v)}
                />
              </Field>
              <Field label="Angle of attack">
                <QuantityInput
                  value={form.freestream.aoa}
                  onChange={(v) => setIn("freestream", "aoa", v)}
                  units={UNITS.angle}
                />
              </Field>
              {(!inviscid || form.freestream.re.trim() !== "") && (
                <Field label="Given by" className="col-span-2">
                  <SelectInput
                    value={given}
                    onChange={(v) => {
                      if (!v || v === given) return;
                      setGivenBy(v);
                      // The case states exactly one of the two (E0306): the other is cleared.
                      setForm((f) => ({
                        ...f,
                        freestream:
                          v === "pressure"
                            ? { ...f.freestream, re: "" }
                            : { ...f.freestream, pressure: "" },
                      }));
                    }}
                    options={[
                      { value: "re", label: "Reynolds number" },
                      { value: "pressure", label: "Pressure" },
                    ]}
                  />
                </Field>
              )}
              {given === "re" ? (
                <Field label="Reynolds number" hint="per reference length">
                  <NumberInput
                    value={form.freestream.re}
                    onChange={(v) => setIn("freestream", "re", v)}
                  />
                </Field>
              ) : (
                <Field label="Pressure">
                  <QuantityInput
                    value={form.freestream.pressure}
                    onChange={(v) => setIn("freestream", "pressure", v)}
                    units={UNITS.pressure}
                  />
                </Field>
              )}
              <Field label="Reference length" hint={given === "re" ? undefined : "for moments"}>
                <QuantityInput
                  value={form.freestream.ref_length}
                  onChange={(v) => setIn("freestream", "ref_length", v)}
                  units={UNITS.length}
                />
              </Field>
              <Field label="Temperature" hint="default 288.15 K">
                <QuantityInput
                  value={form.freestream.temperature}
                  onChange={(v) => setIn("freestream", "temperature", v)}
                  units={UNITS.temperature}
                />
              </Field>
            </Section>

            <Section title="Time">
              <Field label="Maximum iterations" hint="default 10 000">
                <NumberInput
                  value={form.time.max_iterations}
                  onChange={(v) => setIn("time", "max_iterations", v)}
                />
              </Field>
              <Field label="Residual drop (orders)" hint="default 8">
                <NumberInput
                  value={form.time.residual_orders}
                  onChange={(v) => setIn("time", "residual_orders", v)}
                />
              </Field>
            </Section>

            <Section
              title="Boundaries"
              action={
                <Button
                  size="sm"
                  icon={<Plus className="size-4" />}
                  onClick={() =>
                    set("boundaries", [...form.boundaries, { zones: "", kind: "wall", values: {} }])
                  }
                >
                  Add rule
                </Button>
              }
            >
              {form.boundaries.map((rule, k) => (
                <fieldset
                  // biome-ignore lint/suspicious/noArrayIndexKey: rules have no identity but their place
                  key={k}
                  aria-label={`Boundary rule ${k + 1}`}
                  className="col-span-2 m-0 grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] items-end gap-3 border-0 p-0"
                >
                  <Field label="Zones">
                    <TextInput
                      value={rule.zones}
                      onChange={(v) => setRule(k, { ...rule, zones: v })}
                      placeholder="wing.*"
                      mono
                      options={{ id: `zones-${k}`, values: zones }}
                    />
                  </Field>
                  <Field label="Condition">
                    <SelectInput
                      value={rule.kind}
                      onChange={(v) => v && setRule(k, { ...rule, kind: v, values: {} })}
                      options={KINDS}
                    />
                  </Field>
                  <Button
                    square
                    variant="tertiary"
                    aria-label={`Remove boundary rule ${k + 1}`}
                    icon={<Trash2 className="size-4" />}
                    onClick={() =>
                      set(
                        "boundaries",
                        form.boundaries.filter((_, i) => i !== k),
                      )
                    }
                  />
                  {CONDITIONS[rule.kind].map(({ key, dimension, optional }) => (
                    <Field
                      key={key}
                      label={LABEL[key] ?? key}
                      hint={optional ? "optional" : undefined}
                    >
                      <QuantityInput
                        value={rule.values[key] ?? ""}
                        onChange={(v) =>
                          setRule(k, { ...rule, values: { ...rule.values, [key]: v } })
                        }
                        units={UNITS[dimension]}
                      />
                    </Field>
                  ))}
                </fieldset>
              ))}
            </Section>

            <Section
              title="Monitors"
              action={
                <Button
                  size="sm"
                  icon={<Plus className="size-4" />}
                  onClick={() =>
                    set("monitors", [
                      ...form.monitors,
                      { name: "", surfaces: "", ref_area: "", every: "" },
                    ])
                  }
                >
                  Add forces
                </Button>
              }
            >
              {form.monitors.map((monitor, k) => (
                <fieldset
                  // biome-ignore lint/suspicious/noArrayIndexKey: monitors have no identity but their place
                  key={k}
                  aria-label={`Forces monitor ${k + 1}`}
                  className="col-span-2 m-0 grid grid-cols-[8rem_minmax(0,1fr)_minmax(0,1fr)_6rem_auto] items-end gap-3 border-0 p-0"
                >
                  <Field label="Name">
                    <TextInput
                      value={monitor.name}
                      onChange={(v) =>
                        set(
                          "monitors",
                          form.monitors.map((m, i) => (i === k ? { ...m, name: v } : m)),
                        )
                      }
                      placeholder={k === 0 ? "forces" : `forces-${k + 1}`}
                      mono
                    />
                  </Field>
                  <Field label="Surfaces">
                    <TextInput
                      value={monitor.surfaces}
                      onChange={(v) =>
                        set(
                          "monitors",
                          form.monitors.map((m, i) => (i === k ? { ...m, surfaces: v } : m)),
                        )
                      }
                      placeholder="wing.*"
                      mono
                      options={{ id: `surfaces-${k}`, values: zones }}
                    />
                  </Field>
                  <Field label="Reference area">
                    <QuantityInput
                      value={monitor.ref_area}
                      onChange={(v) =>
                        set(
                          "monitors",
                          form.monitors.map((m, i) => (i === k ? { ...m, ref_area: v } : m)),
                        )
                      }
                      units={UNITS.area}
                    />
                  </Field>
                  <Field label="Every" hint="iterations">
                    <NumberInput
                      value={monitor.every}
                      onChange={(v) =>
                        set(
                          "monitors",
                          form.monitors.map((m, i) => (i === k ? { ...m, every: v } : m)),
                        )
                      }
                    />
                  </Field>
                  <Button
                    square
                    variant="tertiary"
                    aria-label={`Remove forces monitor ${k + 1}`}
                    icon={<Trash2 className="size-4" />}
                    onClick={() =>
                      set(
                        "monitors",
                        form.monitors.filter((_, i) => i !== k),
                      )
                    }
                  />
                </fieldset>
              ))}
            </Section>
          </fieldset>
        </form>

        <aside className="flex min-h-0 min-w-0 flex-col gap-4 overflow-y-auto border-line border-l bg-bg-1 px-6 py-6">
          {form.mesh && (
            <MeshView
              mesh={form.mesh}
              zones={zones}
              selected={picked}
              onPick={pick}
              webgl={webgl}
            />
          )}
          {stale.length > 0 && (
            <Note
              tone="warning"
              title={`${stale.length} ${stale.length === 1 ? "rule matches" : "rules match"} no zone of ${form.mesh}`}
              action={
                <Button size="sm" onClick={removeStale}>
                  Remove
                </Button>
              }
            >
              <span className="font-mono" data-testid="stale-rules">
                {stale.map((r) => r.zones).join(" · ")}
              </span>
            </Note>
          )}
          {notice && (
            <Note tone="info" title="Periodic zone">
              {notice}
            </Note>
          )}
          {rendered.isError && (
            <Note tone="error" title="The SDK could not be reached">
              The code below is from an earlier version of the case: {rendered.error.message}
            </Note>
          )}
          {/* Mounted for good, so that a screen reader hears what blocks the case as it changes. */}
          <p role="status" className="sr-only">
            {announcement}
          </p>
          <div className="flex justify-end">
            <CaseActions
              name={form.name}
              // Only the case shown, once it is the form's: an edit not yet rendered waits.
              rendered={
                rendered.data?.ok && !rendered.isPlaceholderData && settled === form
                  ? rendered.data
                  : null
              }
            />
          </div>
          {rendered.data?.ok === false ? (
            <Refusal
              diagnostics={rendered.data.diagnostics}
              document={caseDocument(form, library)}
            />
          ) : (
            <TabsRoot
              defaultValue="python"
              className={cn(
                "flex min-w-0 flex-none flex-col gap-3 transition-opacity",
                (rendered.isError || rendered.isPlaceholderData) && "opacity-60",
              )}
            >
              <div className="flex items-center gap-3">
                <TabList
                  tabs={[
                    { value: "python", label: <span className="font-mono">case.py</span> },
                    { value: "toml", label: <span className="font-mono">case.toml</span> },
                  ]}
                />
                <span className="text-fg-3 text-sm">written by the SDK</span>
              </div>
              <TabPanel value="python" className="min-w-0 outline-none">
                <Code testId="case-py" text={rendered.data?.ok ? rendered.data.python : ""} />
              </TabPanel>
              <TabPanel value="toml" className="min-w-0 outline-none">
                <Code testId="case-toml" text={rendered.data?.ok ? rendered.data.toml : ""} />
              </TabPanel>
            </TabsRoot>
          )}
          {rendered.data?.ok && rendered.data.plan && (
            <PlanSummary plan={rendered.data.plan} current={!rendered.isPlaceholderData} />
          )}
        </aside>
      </div>
    </>
  );
}

type Comparison =
  | { state: "none" }
  | { state: "checking" }
  | { state: "failed"; message: string }
  | { state: "unrepresentable"; diagnostics: Diagnostic[] }
  | { state: "compared"; dropped: ReturnType<typeof differences> };

const show = (value: unknown) => (value === undefined ? "absent" : JSON.stringify(value));

function ImportNote({ run, comparison }: { run: string; comparison: Comparison }) {
  const title = `A copy of ${run}`;
  if (comparison.state === "none" || comparison.state === "checking")
    return (
      <Note tone="info" title={title}>
        Comparing the copy with the run's case…
      </Note>
    );
  if (comparison.state === "failed")
    return (
      <Note tone="error" title={title}>
        The copy could not be compared with the run's case: {comparison.message}
      </Note>
    );
  if (comparison.state === "unrepresentable")
    return (
      <Note tone="warning" title={title}>
        The form cannot say this case yet; read into it, the case is refused:
        <Diagnostics items={comparison.diagnostics} testId="unrepresentable" />
      </Note>
    );
  if (comparison.dropped.length === 0)
    return (
      <Note tone="info" title={title}>
        Every setting of the run's case is in the form.
      </Note>
    );
  return (
    <Note tone="warning" title={title}>
      The form cannot show these settings of the run's case yet; this copy changes them:
      <ul className="m-0 mt-2 p-0 font-mono text-sm" data-testid="dropped">
        {comparison.dropped.map((d) => (
          <li key={d.path} className="list-none">
            {d.path}: {show(d.before)} → {show(d.after)}
          </li>
        ))}
      </ul>
    </Note>
  );
}

// Lines never wrap: the code's own line breaks are part of what it shows. A line longer than the
// panel scrolls, and a fade on the right edge says so until its end is in view (overlay
// scrollbars stay hidden until a reader scrolls).
function Code({ text, testId }: { text: string; testId: string }) {
  const ref = useRef<HTMLPreElement>(null);
  const [more, setMore] = useState(false);
  const measure = (el: HTMLElement) => setMore(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
  // biome-ignore lint/correctness/useExhaustiveDependencies: measured again when the text changes
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    measure(el);
    const resize = new ResizeObserver(() => measure(el));
    resize.observe(el);
    return () => resize.disconnect();
  }, [text]);
  return (
    <div className="relative min-w-0">
      <pre
        ref={ref}
        onScroll={(e) => measure(e.currentTarget)}
        className="m-0 max-w-full overflow-x-auto whitespace-pre rounded-lg border border-line bg-bg-0 p-4 font-mono text-sm leading-6"
        data-testid={testId}
      >
        {text}
      </pre>
      {more && (
        <span
          aria-hidden
          data-testid={`${testId}-more`}
          className="pointer-events-none absolute inset-y-px right-px w-12 rounded-r-lg bg-gradient-to-l from-bg-0 to-transparent"
        />
      )}
    </div>
  );
}

function Section({
  title,
  action,
  children,
}: {
  title: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="m-0 font-medium text-base">{title}</h2>
        {action}
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-4">{children}</div>
    </section>
  );
}

// `current`: the plan is of the case as it now reads; otherwise it is being checked again.
/** What to do with a case the SDK accepts and plans: copy its code, or save its case file. */
function CaseActions({
  name,
  rendered,
}: {
  name: string;
  rendered: { python: string; toml: string; plan: Plan | null } | null;
}) {
  const [copied, setCopied] = useState<"copied" | "refused" | null>(null);
  const dismiss = useRef<number | undefined>(undefined);
  useEffect(() => () => clearTimeout(dismiss.current), []);
  const ready = rendered !== null && rendered.plan?.ok === true;
  const copy = async () => {
    if (!rendered) return;
    try {
      await navigator.clipboard.writeText(rendered.python);
      setCopied("copied");
    } catch {
      setCopied("refused");
    }
    clearTimeout(dismiss.current);
    dismiss.current = window.setTimeout(() => setCopied(null), 4000);
  };
  const save = () => {
    if (!rendered) return;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([rendered.toml], { type: "application/toml" }));
    link.download = `${name.trim() || "case"}.toml`;
    link.click();
    URL.revokeObjectURL(link.href);
  };
  return (
    <span className="ml-auto flex items-center gap-2">
      <span role="status" className="text-danger text-sm">
        {copied === "refused" && "The clipboard refused: select the code instead."}
      </span>
      <Button size="sm" variant="primary" disabled={!ready} onClick={() => void copy()}>
        {copied === "copied" ? "Copied" : "Copy case.py"}
      </Button>
      <Button size="sm" disabled={!ready} onClick={save}>
        Save case.toml
      </Button>
    </span>
  );
}

function PlanSummary({ plan, current }: { plan: Plan; current: boolean }) {
  const uncovered = plan.boundaries.filter((b) => b.condition === null);
  const errors = plan.diagnostics.filter((d) => d.severity !== "warning").length;
  return (
    <section className="flex flex-col gap-2" data-testid="plan">
      <h2 className="m-0 flex items-baseline gap-2 font-medium text-base">
        Plan
        {current ? (
          <span className={plan.ok ? "text-success" : "text-danger"} data-testid="plan-status">
            {plan.ok ? "ready" : `refused: ${errors} ${errors === 1 ? "error" : "errors"}`}
          </span>
        ) : (
          <span className="text-fg-3">checking…</span>
        )}
        <span className="font-normal text-fg-3 text-sm">
          {plan.boundaries.length - uncovered.length} of {plan.boundaries.length} zones covered
        </span>
      </h2>
      {uncovered.length > 0 && (
        <p className="m-0 text-fg-2 text-sm" data-testid="uncovered">
          {uncovered.length} {uncovered.length === 1 ? "zone" : "zones"} without a condition:{" "}
          <span className="font-mono">{uncovered.map((b) => b.zone).join(" · ")}</span>
        </p>
      )}
      <Diagnostics
        // The editor plans without the mesh file by design: W0205 says so of every case. E0412,
        // one per zone without a condition, is the line above.
        items={plan.diagnostics.filter(
          (d) => !(d.code === "W0205" && d.location?.path === "mesh") && d.code !== "E0412",
        )}
        testId="plan-diagnostics"
      />
    </section>
  );
}

// What the editor calls a setting, from its path in the case document.
const LABELS: Record<string, string> = {
  name: "Name",
  mesh: "Mesh",
  "physics.turbulence": "Turbulence",
  "freestream.mach": "Mach",
  "freestream.ref_length": "Reference length",
  freestream: "Reynolds number or pressure",
};
const OWNERS = { boundaries: "boundary rule", cell_zones: "cell zone", monitors: "monitor" };
function label(path: string): string {
  const known = LABELS[path];
  if (known) return known;
  const owned = /^(boundaries|cell_zones|monitors)\[(\d+)\]/.exec(path);
  const leaf = (path.split(".").pop() ?? path).replaceAll("_", " ");
  return owned
    ? `${OWNERS[owned[1] as keyof typeof OWNERS]} ${Number(owned[2]) + 1} ${leaf}`
    : leaf.charAt(0).toUpperCase() + leaf.slice(1);
}

// The value at a diagnostic's path in the case document; unset when absent or empty.
function isSet(document: CaseDocument, path: string): boolean {
  let value: unknown = document;
  for (const part of path.split(/\.|\[(\d+)\]/).filter(Boolean)) {
    value =
      value !== null && typeof value === "object"
        ? (value as Record<string, unknown>)[part]
        : undefined;
  }
  if (value === undefined || value === "") return false;
  return typeof value !== "object" || value === null || Object.keys(value).length > 0;
}

/** Why the SDK refuses the case: the settings still to fill in, in a line, and the problems
 * with what was set, each with its code. Nothing alarming before anything was typed. */
function Refusal({ diagnostics, document }: { diagnostics: Diagnostic[]; document: CaseDocument }) {
  // A setting not given yet is to be set, not an error; "freestream" is the choice of
  // Reynolds number or pressure, pending while neither is given.
  const pending = (d: Diagnostic) => {
    const path = d.location?.path;
    if (path === undefined) return false;
    if (path === "freestream")
      return !isSet(document, "freestream.re") && !isSet(document, "freestream.pressure");
    return !isSet(document, path);
  };
  const missing = [
    ...new Set(diagnostics.filter(pending).map((d) => label(d.location?.path ?? ""))),
  ];
  return (
    <div className="flex flex-col gap-2">
      {missing.length > 0 && (
        <p className="m-0 text-fg-2 text-sm" data-testid="still-to-set">
          Still to set: {missing.join(" · ")}
        </p>
      )}
      <Diagnostics items={diagnostics.filter((d) => !pending(d))} testId="diagnostics" />
    </div>
  );
}

function Diagnostics({ items, testId }: { items: Diagnostic[]; testId: string }) {
  return (
    <div className="flex flex-col gap-2" data-testid={testId}>
      {items.map((d, k) => (
        <Note
          // biome-ignore lint/suspicious/noArrayIndexKey: diagnostics repeat; their place is their identity
          key={k}
          tone={d.severity === "warning" ? "warning" : "error"}
          title={d.location?.path ? `${d.code} · ${d.location.path}` : d.code}
        >
          {d.message}
        </Note>
      ))}
    </div>
  );
}
