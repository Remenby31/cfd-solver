import { expect, test } from "vitest";
import { flatten } from "../api/polars";
import { degrees } from "../api/units";
import { study } from "./index";

test("angles are read in degrees from the case's units", () => {
  expect(degrees("19.57 deg")).toBe(19.57);
  expect(degrees(`${Math.PI / 2} rad`)).toBeCloseTo(90);
  expect(degrees("100 mrad")).toBeCloseTo(5.729578);
  expect(degrees("15 arcmin")).toBe(0.25);
  expect(degrees("0.5 turn")).toBe(180);
  expect(degrees("10.0 grad")).toBe(9);
  expect(degrees("100000.0 μrad")).toBeCloseTo(5.729578);
  expect(degrees("19.57 m")).toBeNull();
  expect(degrees("3 constructor")).toBeNull();
  expect(degrees("19.57")).toBeNull();
  expect(degrees(null)).toBeNull();
});

test("a case document flattens to dotted settings with arrays by index", () => {
  const flat = flatten({
    freestream: { aoa: "7 deg", mach: 0.2 },
    boundaries: [{ zones: "wing.*" }],
  });
  expect(Object.fromEntries(flat)).toEqual({
    "freestream.aoa": "7 deg",
    "freestream.mach": "0.2",
    "boundaries[0].zones": "wing.*",
  });
});

test("a study names the settings that vary across its runs", () => {
  const base = {
    sideslip: "0 deg",
    mach: 0.2,
    re: 5.49e6,
    turbulence: "sa",
    precision: "fp32+acc64",
  };
  expect(study([{ ...base, aoa: "7.05 deg" }, { ...base, aoa: "21.47 deg" }, undefined])).toBe(
    "α 7.05–21.47 deg",
  );
  expect(
    study([
      { ...base, aoa: "1 deg", turbulence: "sst" },
      { ...base, aoa: "1 deg" },
    ]),
  ).toBe("turbulence sst · sa");
  expect(
    study([
      { ...base, aoa: "1 deg" },
      { ...base, aoa: "1 deg" },
    ]),
  ).toBeNull();
});
