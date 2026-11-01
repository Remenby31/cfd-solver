// Degrees per unit of the angle units a case may write (Pint's spellings).
const ANGLE: Record<string, number> = {
  deg: 1,
  degree: 1,
  degrees: 1,
  rad: 180 / Math.PI,
  radian: 180 / Math.PI,
  radians: 180 / Math.PI,
  mrad: 0.18 / Math.PI,
  milliradian: 0.18 / Math.PI,
  arcmin: 1 / 60,
  arcminute: 1 / 60,
  arcsec: 1 / 3600,
  arcsecond: 1 / 3600,
  turn: 360,
  grad: 0.9,
  gradian: 0.9,
  gon: 0.9,
  urad: 1.8e-4 / Math.PI,
  μrad: 1.8e-4 / Math.PI,
  microradian: 1.8e-4 / Math.PI,
};

/** An angle written as the case writes it ("19.57 deg", "0.34 rad", "100 mrad"), in degrees;
 * null if the value has no angle unit. */
export function degrees(value: string | null | undefined): number | null {
  const match = /^\s*(-?[\d.eE+-]+)\s*([a-zμ]+)\s*$/u.exec(value ?? "");
  const factor =
    match?.[2] !== undefined && Object.hasOwn(ANGLE, match[2]) ? ANGLE[match[2]] : undefined;
  if (!match || factor === undefined) return null;
  const number = Number(match[1]);
  return Number.isFinite(number) ? number * factor : null;
}

/** A force coefficient to four decimals, one drag count; never "-0.0000". */
export function coefficient(value: number): string {
  const text = value.toFixed(4);
  return /^-0\.0+$/.test(text) ? text.slice(1) : text;
}

/** Why a run's last coefficients are not those of a converged steady solution, or null when
 * they are (results.md#runjson status). */
export function coefficientsCaveat(status: string, unsteady: boolean): string | null {
  if (status === "converged") return null;
  if (status === "running") return "still running";
  if (status === "completed" && unsteady) return "last step of an unsteady run";
  return "not converged";
}
