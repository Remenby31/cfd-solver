import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { RUNS } from "../playwright.config";

// py/examples/crm_hl.py without its module docstring and the blank line after it, as the
// SDK's own gate compares it with Case.to_python (py/tests/test_render.py).
function withoutDocstring(source: string): string {
  const lines = source.split(/(?<=\n)/);
  const end = lines.findIndex(
    (line, k) => (k === 0 ? line.trim().length > 6 : true) && line.trimEnd().endsWith('"""'),
  );
  return lines.slice(end + 2).join("");
}
const CRM_HL = withoutDocstring(readFileSync("../py/examples/crm_hl.py", "utf8"));
const HEAT_EXCHANGER = withoutDocstring(readFileSync("../py/examples/heat_exchanger.py", "utf8"));

async function rule(page: Page, k: number, zones: string, kind: string) {
  await page.getByRole("button", { name: "Add rule" }).click();
  const fieldset = page.getByRole("group", { name: `Boundary rule ${k}` });
  await fieldset.getByLabel("Zones").fill(zones);
  await fieldset.getByLabel("Condition").selectOption(kind);
}

async function forces(page: Page, k: number, surfaces: string, area: string, every: string) {
  await page.getByRole("button", { name: "Add forces" }).click();
  const monitor = page.getByRole("group", { name: `Forces monitor ${k}` });
  await monitor.getByLabel("Surfaces").fill(surfaces);
  await monitor.getByLabel("Reference area", { exact: true }).fill(area);
  await monitor.getByLabel("Every").fill(every);
}

// The CRM-HL case of py/examples/crm_hl.py, field by field.
async function crmHl(page: Page) {
  await page.getByLabel("Name").fill("crm-hl");
  await page.getByLabel("Mesh", { exact: true }).selectOption("crm-hl");
  await page.getByLabel("Turbulence").selectOption("sa");
  await page.getByLabel("Mach").fill("0.2");
  await page.getByLabel("Angle of attack", { exact: true }).fill("19.57");
  await page.getByLabel("Reynolds number", { exact: true }).fill("5490000.0");
  await page.getByLabel("Reference length", { exact: true }).fill("7.0");
  await rule(page, 1, "wing.*", "wall");
  await rule(page, 2, "symmetry", "symmetry");
  await forces(page, 1, "wing.*", "191.84", "10");
}

test("the CRM-HL case built in the editor writes py/examples/crm_hl.py", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("a[data-row]").first()).toBeVisible();
  await page.keyboard.press("ControlOrMeta+k");
  await page.keyboard.type("new case");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/cases\/new$/);

  await crmHl(page);

  // Byte for byte: toHaveText would collapse whitespace.
  await expect.poll(() => page.getByTestId("case-py").textContent()).toBe(CRM_HL);
  await expect(page.getByTestId("plan")).toContainText("ready");
  await expect(page.getByTestId("plan-diagnostics")).toBeEmpty();
});

test("a value the SDK refuses shows its code at its field", async ({ page }) => {
  await page.goto("/cases/new");
  await crmHl(page);
  await expect(page.getByTestId("plan")).toContainText("ready");

  await page.getByLabel("Mach").fill("-1");
  await expect(page.getByTestId("diagnostics")).toContainText("E0304 · freestream.mach");

  await page.getByLabel("Mach").fill("0.2");
  await page.getByLabel("Reynolds number", { exact: true }).fill("abc");
  await expect(page.getByTestId("diagnostics")).toContainText("E0301 · freestream.re");
});

test("a unit chosen first is kept, and monitors get names of their own", async ({ page }) => {
  await page.goto("/cases/new");
  await crmHl(page);
  await page.getByLabel("Reference length unit").selectOption("mm");
  await page.getByLabel("Reference length", { exact: true }).fill("");
  await page.getByLabel("Reference length", { exact: true }).fill("7000.0");
  await forces(page, 2, "wing.*", "191.84", "10");

  await expect(page.getByTestId("case-py")).toContainText("ref_length=7000.0 * mm");
  await expect(page.getByTestId("case-py")).toContainText('name="forces-2"');
  await expect(page.getByTestId("plan")).toContainText("ready");
});

test("errors leave with the rule that raised them", async ({ page }) => {
  await page.goto("/cases/new");
  await crmHl(page);
  await rule(page, 3, "wing.*", "slip");
  const errors = page.getByTestId("plan-diagnostics").getByText(/^E0411/);
  await expect(errors).toHaveCount(3); // wing.slat, wing.main and wing.flap, matched twice

  const zones = page.getByRole("group", { name: "Boundary rule 3" }).getByLabel("Zones");
  await zones.fill("missing");
  await expect(page.getByTestId("plan")).not.toContainText("checking");
  await zones.fill("symmetry");
  await expect(page.getByTestId("plan")).toContainText("refused");
  await expect(errors).toHaveCount(1); // symmetry alone

  await page.getByRole("button", { name: "Remove boundary rule 3" }).click();
  await expect(page.getByTestId("plan")).toContainText("ready");
  await expect(errors).toHaveCount(0);
});

test("a line longer than the code panel says so and scrolls to its end", async ({ page }) => {
  await page.goto("/cases/new");
  await crmHl(page);
  const code = page.getByTestId("case-py");
  await expect.poll(() => code.textContent()).toBe(CRM_HL);
  await expect(page.getByTestId("case-py-more")).toBeVisible();

  const end = await code.evaluate((el) => {
    el.scrollLeft = el.scrollWidth;
    el.dispatchEvent(new Event("scroll"));
    const aside = el.closest("aside")?.getBoundingClientRect();
    return {
      inside: aside !== undefined && el.getBoundingClientRect().right <= aside.right,
      atEnd: Math.abs(el.scrollLeft + el.clientWidth - el.scrollWidth) <= 1,
    };
  });
  expect(end).toEqual({ inside: true, atEnd: true });
  await expect(page.getByTestId("case-py-more")).toHaveCount(0);
});

const WEBGL = process.env.AUTAN_UI_E2E_BACKEND === "webgl2" ? "?webgl=1" : "";

// The default view looks at the box centre from direction (-0.7, -1, 0.75) (SurfaceViewer
// frameAll): the ray through the middle of the view enters the unit cube through ymin.
async function clickCentre(page: Page) {
  const view = page.getByTestId("mesh-view");
  await expect(view).toHaveAttribute("data-ready", "true");
  const box = await view.boundingBox();
  if (!box) throw new Error("the mesh view has no box");
  await view.click({ position: { x: box.width / 2, y: box.height / 2 } });
}

test("a zone clicked in 3D gets a boundary rule, its condition from the mesh", async ({ page }) => {
  await page.goto(`/cases/new${WEBGL}`);
  await page.getByLabel("Mesh", { exact: true }).selectOption("polyMesh");
  await clickCentre(page);

  // polyMesh.meshinfo.json types ymin as a wall.
  const rule = page.getByRole("group", { name: "Boundary rule 1" });
  await expect(rule.getByLabel("Zones")).toHaveValue("ymin");
  await expect(rule.getByLabel("Condition")).toHaveValue("wall");
  await expect(rule.getByLabel("Zones")).toBeFocused();

  await clickCentre(page);
  await expect(page.getByRole("group", { name: /^Boundary rule/ })).toHaveCount(1);
});

test("a zone a pattern already covers focuses that rule", async ({ page }) => {
  await page.goto(`/cases/new${WEBGL}`);
  await page.getByLabel("Mesh", { exact: true }).selectOption("polyMesh");
  await rule(page, 1, "*", "slip");
  await page.getByLabel("Name").focus();

  await clickCentre(page);

  await expect(page.getByRole("group", { name: /^Boundary rule/ })).toHaveCount(1);
  await expect(
    page.getByRole("group", { name: "Boundary rule 1" }).getByLabel("Zones"),
  ).toBeFocused();
});

test("picking keeps working on another mesh", async ({ page }) => {
  await page.goto(`/cases/new${WEBGL}`);
  await page.getByLabel("Mesh", { exact: true }).selectOption("polyMesh");
  await expect(page.getByTestId("mesh-view")).toHaveAttribute("data-ready", "true");
  await page.getByLabel("Mesh", { exact: true }).selectOption("hybrid-cgns");

  await clickCentre(page);

  // hybrid.cgns gives its zones no type: the suggestion is a wall.
  const rule = page.getByRole("group", { name: "Boundary rule 1" });
  await expect(rule.getByLabel("Zones")).toHaveValue("ymin");
  await expect(rule.getByLabel("Condition")).toHaveValue("wall");
});

test("an inviscid case walls a picked wall with slip", async ({ page }) => {
  await page.goto(`/cases/new${WEBGL}`);
  await page.getByLabel("Turbulence").selectOption("inviscid");
  await page.getByLabel("Mesh", { exact: true }).selectOption("polyMesh");
  await clickCentre(page);

  const rule = page.getByRole("group", { name: "Boundary rule 1" });
  await expect(rule.getByLabel("Zones")).toHaveValue("ymin");
  await expect(rule.getByLabel("Condition")).toHaveValue("slip");
});

async function cellZone(page: Page, k: number, zones: string, kind: string, material: string) {
  await page.getByRole("button", { name: "Add zone" }).click();
  const fieldset = page.getByRole("group", { name: `Cell zone ${k}` });
  await fieldset.getByLabel("Zones").fill(zones);
  await fieldset.getByLabel("Medium").selectOption(kind);
  await fieldset.getByLabel("Material").selectOption(material);
}

async function quantity(page: Page, group: string, label: string, value: string, unit: string) {
  const fieldset = page.getByRole("group", { name: group });
  await fieldset.getByLabel(`${label} unit`).selectOption(unit);
  await fieldset.getByLabel(label, { exact: true }).fill(value);
}

test("the heat exchanger built in the editor writes py/examples/heat_exchanger.py", async ({
  page,
}) => {
  await page.goto("/cases/new");
  await page.getByLabel("Name").fill("heat-exchanger");
  await page.getByLabel("Mesh", { exact: true }).selectOption("heat-exchanger");
  await page.getByLabel("Turbulence").selectOption("sst");
  await page.getByLabel("Mach").fill("0.1");
  await page.getByLabel("Temperature", { exact: true }).fill("300.0");
  await page.getByLabel("Given by").selectOption("pressure");
  await page.getByLabel("Pressure", { exact: true }).fill("101325.0");
  await cellZone(page, 1, "co2", "fluid", "CO2");
  await cellZone(page, 2, "plate", "solid", "COPPER");
  for (const [k, zone, temperature] of [
    [1, "air-inlet", "400.0"],
    [2, "co2-inlet", "280.0"],
  ] as const) {
    await rule(page, k, zone, "inlet");
    await quantity(page, `Boundary rule ${k}`, "Total pressure", "103.0", "kPa");
    await quantity(page, `Boundary rule ${k}`, "Total temperature", temperature, "K");
  }
  await rule(page, 3, "*-outlet", "outlet");
  await quantity(page, "Boundary rule 3", "Static pressure", "101325.0", "Pa");

  await expect.poll(() => page.getByTestId("case-py").textContent()).toBe(HEAT_EXCHANGER);
  await expect(page.getByTestId("plan")).toContainText("ready");
});

test("gas and medium choices change the case and can be undone", async ({ page }) => {
  await page.goto("/cases/new");
  await crmHl(page);
  await page.getByLabel("Gas").selectOption("CO2");
  await expect(page.getByTestId("case-py")).toContainText("gas=autan.gases.CO2");
  await page.getByLabel("Gas").selectOption("");
  await expect.poll(() => page.getByTestId("case-py").textContent()).toBe(CRM_HL);

  await cellZone(page, 1, "fluid", "fluid", "CO2");
  const zone = page.getByRole("group", { name: "Cell zone 1" });
  await zone.getByLabel("Medium").selectOption("solid");
  await expect(zone.getByLabel("Material")).toHaveValue("");
  await zone.getByLabel("Material").selectOption("COPPER");
  await expect(page.getByTestId("case-py")).toContainText(
    'case.cell_zone("fluid").solid(autan.materials.COPPER)',
  );
});

test("material libraries that fail to load say so and can be retried", async ({ page }) => {
  let fail = true;
  await page.route("**/api/materials", (route) =>
    fail ? route.fulfill({ status: 503, body: "" }) : route.continue(),
  );
  await page.goto("/cases/new");
  await expect(page.getByText("The material libraries could not be listed")).toBeVisible({
    timeout: 15_000,
  });

  fail = false;
  await page.getByRole("button", { name: "Retry" }).click();
  await expect(page.getByText("The material libraries could not be listed")).toHaveCount(0);
  await expect(page.getByLabel("Gas").locator("option", { hasText: "co2" })).toHaveCount(1);
});

test("a copy of every sample run's case writes that case's SDK code", async ({ page }) => {
  const listed: { run_id: string; status: string }[] = (
    await (await page.request.get("/api/runs")).json()
  ).runs;
  expect(listed.length).toBeGreaterThan(0);
  for (const { run_id } of listed) {
    // The oracle: the SDK's code of the case the run stored, rendered by the server.
    const stored = (await (await page.request.get(`/api/runs/${run_id}`)).json()).case;
    const sdk = await (
      await page.request.post("/api/cases/render", { data: { case: stored } })
    ).json();
    expect(sdk.ok, run_id).toBe(true);

    await page.goto(`/runs/${run_id}`);
    await page.getByRole("button", { name: "Edit a copy" }).click();
    await expect(page).toHaveURL(new RegExp(`/cases/new\\?run=${run_id}`));
    await expect.poll(() => page.getByTestId("case-py").textContent()).toBe(sdk.python);
    await expect(page.getByText("Every setting of the run's case is in the form.")).toBeVisible();
  }
});

test("a copy of a case the form cannot say lists what it changes", async ({ page }) => {
  // A real run of a case with an unsteady time, which the editor does not offer yet, found by
  // its own case name and removed afterwards.
  const dir = mkdtempSync(join(tmpdir(), "autan-unsteady-"));
  const name = `unsteady-${Date.now()}`;
  writeFileSync(
    join(dir, "case.py"),
    readFileSync("../py/examples/crm_hl.py", "utf8").replace('"crm-hl"', `"${name}"`) +
      "case.time = autan.Unsteady(steps=3, dt=1e-3 * autan.units.s)\n",
  );
  execFileSync("uv", [
    ...["run", "--locked", "--project", "server", "autan", "run", join(dir, "case.py")],
    ...["--mesh-info", "../py/tests/fixtures/crm-hl.meshinfo.json", "--runs", RUNS],
  ]);
  const listed: { run_id: string; case: { name: string } }[] = (
    await (await page.request.get("/api/runs")).json()
  ).runs;
  const run = listed.find((r) => r.case.name === name);
  if (!run) throw new Error("the unsteady run is not listed");
  try {
    await page.goto(`/cases/new?run=${run.run_id}`);

    await expect(page.getByTestId("dropped")).toContainText('time.kind: "unsteady" → "steady"');
  } finally {
    rmSync(join(RUNS, run.run_id), { recursive: true, force: true });
    rmSync(dir, { recursive: true, force: true });
  }
});

test("the form waits for the copy, and a run it cannot read says so", async ({ page }) => {
  let release = () => {};
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/materials", async (route) => {
    await held;
    await route.continue();
  });
  const listed: { run_id: string }[] = (await (await page.request.get("/api/runs")).json()).runs;
  await page.goto(`/cases/new?run=${listed[0]?.run_id}`);
  // The case's own name, not a monitor's.
  const caseName = page
    .getByRole("group")
    .filter({ hasNotText: "Forces monitor" })
    .getByLabel("Name")
    .first();
  await expect(caseName).toBeDisabled();
  release();
  await expect(caseName).toBeEnabled();
  await expect(caseName).not.toHaveValue("");

  await page.goto("/cases/new?run=run-20000101T000000Z-00000000");
  await expect(
    page.getByText("The run run-20000101T000000Z-00000000 could not be read"),
  ).toBeVisible({
    timeout: 15_000,
  });
});

test("a planned case is copied as case.py or saved as case.toml", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/cases/new");
  const copy = page.getByRole("button", { name: "Copy case.py" });
  const save = page.getByRole("button", { name: "Save case.toml" });
  await expect(copy).toBeDisabled();

  await crmHl(page);
  await expect(page.getByTestId("plan-status")).toHaveText("ready");
  // An edit not yet rendered is not what the buttons would hand over.
  await page.getByLabel("Mach").fill("0.25");
  await expect(copy).toBeDisabled();
  await page.getByLabel("Mach").fill("0.2");
  await copy.click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(CRM_HL);
  const download = page.waitForEvent("download");
  await save.click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("crm-hl.toml");
  const saved = readFileSync(await file.path(), "utf8");
  await page.getByRole("tab", { name: "case.toml" }).click();
  expect(saved).toBe(await page.getByTestId("case-toml").textContent());
});

test("a clipboard that refuses the copy says so", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: () => Promise.reject(new DOMException("denied", "NotAllowedError")) },
    });
  });
  await page.goto("/cases/new");
  await crmHl(page);
  await page.getByRole("button", { name: "Copy case.py" }).click();
  await expect(page.getByRole("status").filter({ hasText: "clipboard" })).toBeVisible();
});

test("the plan sits below the code, and its status counts its errors", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/cases/new");
  await page.getByLabel("Name").fill("box");
  await page.getByLabel("Mesh", { exact: true }).selectOption("hybrid-cgns");
  await page.getByLabel("Turbulence").selectOption("inviscid");
  await page.getByLabel("Mach").fill("0.8");
  await page.getByLabel("Pressure", { exact: true }).fill("101325");

  const plan = page.getByTestId("plan");
  await expect(page.getByTestId("plan-status")).toContainText("refused");
  // One error per note, and one per zone on the line of zones without a condition (E0412).
  const notes = await page.getByTestId("plan-diagnostics").getByRole("alert").count();
  const uncovered = Number((await page.getByTestId("uncovered").textContent())?.split(" ")[0]);
  await expect(page.getByTestId("plan-status")).toHaveText(`refused: ${notes + uncovered} errors`);
  const code = await page.getByTestId("case-py").boundingBox();
  const status = await plan.boundingBox();
  if (!code || !status) throw new Error("the code or the plan is not laid out");
  expect(status.y).toBeGreaterThanOrEqual(code.y + code.height);
});

test("rules a new mesh does not have are offered for removal", async ({ page }) => {
  await page.goto("/cases/new");
  await page.getByLabel("Mesh", { exact: true }).selectOption("polyMesh");
  await rule(page, 1, "zmax", "symmetry");
  await rule(page, 2, "wing.*", "wall");
  await rule(page, 3, "xmin", "farfield");

  await page.getByLabel("Mesh", { exact: true }).selectOption("crm-hl");

  // crm-hl.meshinfo.json has no zone zmax or xmin; a rule added since is not the mesh's doing.
  await expect(page.getByTestId("stale-rules")).toHaveText("zmax · xmin");
  await page.getByRole("button", { name: "Add rule" }).click();
  await page.getByRole("button", { name: "Remove", exact: true }).click();
  const rules = page.getByRole("group", { name: /^Boundary rule/ });
  await expect(rules).toHaveCount(2);
  await expect(rules.first().getByLabel("Zones")).toHaveValue("wing.*");
  await expect(rules.last().getByLabel("Zones")).toHaveValue("");
  await expect(page.getByTestId("stale-rules")).toHaveCount(0);
});

test("a mesh without a 3D view lists its zones to pick from", async ({ page }) => {
  await page.goto(`/cases/new${WEBGL}`);
  await page.getByLabel("Mesh", { exact: true }).selectOption("polyMesh");
  await expect(page.getByTestId("mesh-view")).toHaveAttribute("data-ready", "true");
  await page.getByLabel("Mesh", { exact: true }).selectOption("crm-hl");
  const list = page.getByTestId("zone-list");
  await expect(list.getByRole("button")).toHaveText(
    JSON.parse(
      readFileSync("../py/tests/fixtures/crm-hl.meshinfo.json", "utf8"),
    ).boundary_zones.map((z: { name: string }) => z.name),
  );

  await expect(page.getByRole("group", { name: /^Boundary rule/ })).toHaveCount(0);
  const symmetry = list.getByRole("button", { name: "symmetry" });
  await symmetry.focus();
  await page.keyboard.press("Shift+Tab");
  await page.keyboard.press("Tab");
  await expect(symmetry).toBeFocused();
  const ring = await symmetry.evaluate((el) => getComputedStyle(el).boxShadow);
  await page.getByLabel("Name").focus();
  expect(ring).not.toBe(await symmetry.evaluate((el) => getComputedStyle(el).boxShadow));
  await symmetry.click();
  const rule = page.getByRole("group", { name: "Boundary rule 1" });
  await expect(rule.getByLabel("Zones")).toHaveValue("symmetry");
  await expect(rule.getByLabel("Condition")).toHaveValue("symmetry");
});

test("a new case asks for what it lacks before showing errors", async ({ page }) => {
  await page.goto("/cases/new");
  // Nothing typed yet: what the SDK requires (case-schema.md), not its errors.
  await expect(page.getByTestId("still-to-set")).toHaveText(
    "Still to set: Name · Mesh · Turbulence",
  );
  await expect(page.getByTestId("diagnostics").getByRole("alert")).toHaveCount(0);

  await page.getByLabel("Name").fill("box");
  await page.getByLabel("Mesh", { exact: true }).selectOption("hybrid-cgns");
  await page.getByLabel("Turbulence").selectOption("sa");
  await page.getByLabel("Mach").fill("0.3");
  await expect(page.getByTestId("still-to-set")).toHaveText(
    "Still to set: Reynolds number or pressure",
  );
  // A value set and refused is an error at once.
  await page.getByLabel("Mach").fill("-0.3");
  await expect(page.getByTestId("diagnostics")).toContainText("freestream.mach");
  await page.getByLabel("Mach").fill("0.3");

  // A Reynolds number needs its reference length (E0306): still to set, not an error.
  await page.getByLabel("Reynolds number", { exact: true }).fill("1000000");
  await expect(page.getByTestId("still-to-set")).toHaveText("Still to set: Reference length");
  await expect(page.getByTestId("diagnostics").getByRole("alert")).toHaveCount(0);
  await page.getByLabel("Reference length", { exact: true }).fill("1");
  // Inviscid with a Reynolds number already given: the choice stays, to give a pressure.
  await page.getByLabel("Turbulence").selectOption("inviscid");
  await expect(page.getByLabel("Given by")).toBeVisible();
  await page.getByLabel("Turbulence").selectOption("sa");
  // One of Reynolds number and pressure (E0306): choosing one clears the other; the reference
  // length stays, the moments need it.
  await page.getByLabel("Given by").selectOption("pressure");
  await expect(page.getByLabel("Reference length", { exact: true })).toHaveValue("1");
  await expect(page.getByLabel("Reynolds number", { exact: true })).toHaveCount(0);
  await page.getByLabel("Pressure", { exact: true }).fill("101325");
  await expect(page.getByTestId("case-py")).toContainText("pressure=101325");
  expect(await page.getByTestId("case-py").textContent()).not.toMatch(/\bre=/);
  // An inviscid flow has no Reynolds number to give.
  await page.getByLabel("Turbulence").selectOption("inviscid");
  await expect(page.getByLabel("Given by")).toHaveCount(0);

  // Zones without a condition are one line, not one error each.
  await expect(page.getByTestId("uncovered")).toContainText("6 zones without a condition");
  await expect(page.getByRole("status").filter({ hasText: "without a condition" })).toHaveText(
    "6 zones without a condition.",
  );
  await expect(page.getByTestId("plan-diagnostics")).not.toContainText("E0412");
});
