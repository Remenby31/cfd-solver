import { useEffect, useRef, useState } from "react";
import { fetchSurface, type Surface } from "../../api/surface";
import { SurfaceViewer } from "../../viewer/SurfaceViewer";
import { useResolvedTheme } from "../shell/theme";

// The colour of each condition kind on the faces and in the legend. Walls are a neutral grey,
// and no kind is blue, the colour of the selection.
const CONDITION_COLOR: Record<string, string> = {
  wall: "--series-8",
  slip: "--series-6",
  symmetry: "--series-3",
  farfield: "--series-4",
  inlet: "--series-2",
  outlet: "--series-5",
};

/** The design token a condition kind is drawn in. */
export function conditionColor(kind: string): string {
  return Object.hasOwn(CONDITION_COLOR, kind) ? (CONDITION_COLOR[kind] as string) : "--series-7";
}

/** The boundary of a mesh in 3D, one zone per boundary zone; a click picks the zone under the
 * cursor. `selected` is a zone name, highlighted when the mesh has it. Without a surface file,
 * the mesh's boundary `zones` are listed to pick from instead. */
export function MeshView({
  mesh,
  zones,
  conditions,
  selected,
  onPick,
  webgl,
}: {
  mesh: string;
  zones: string[];
  /** The condition kind of each zone that has one; its faces take that kind's colour. */
  conditions: Map<string, string>;
  selected: string | null;
  onPick: (zone: string) => void;
  webgl: boolean;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const viewer = useRef<SurfaceViewer | null>(null);
  const [surface, setSurface] = useState<Surface | null>(null);
  const [error, setError] = useState<string | null>(null);

  // One renderer for the canvas's life: disposing it loses the WebGL context, which a new
  // renderer on the same canvas cannot get back. Disposal waits a tick, so that a remount at
  // once (React's development StrictMode replays effects) takes the same renderer back.
  const [ready, setReady] = useState(false);
  const life = useRef<{ creating: Promise<SurfaceViewer>; dispose?: number } | null>(null);
  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    if (life.current?.dispose !== undefined) clearTimeout(life.current.dispose);
    else life.current = { creating: SurfaceViewer.create(el, { forceWebGL: webgl }) };
    const current = life.current;
    if (!current) return;
    delete current.dispose;
    let live = true;
    current.creating
      .then((v) => {
        if (!live) return;
        viewer.current = v;
        setReady(true);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
      setReady(false);
      current.dispose = window.setTimeout(() => {
        current.creating.then((v) => v.dispose()).catch(() => {});
        if (life.current === current) life.current = null;
        viewer.current = null;
      }, 0);
    };
  }, [webgl]);

  useEffect(() => {
    if (!ready) return;
    const abort = new AbortController();
    viewer.current?.clear();
    setSurface(null);
    setError(null);
    fetchSurface(`/api/meshes/${encodeURIComponent(mesh)}/surface`, abort.signal)
      .then((s) => {
        if (abort.signal.aborted) return;
        viewer.current?.setSurface(s, "");
        setSurface(s);
      })
      .catch((e: unknown) => {
        if (!abort.signal.aborted) setError(e instanceof Error ? e.message : String(e));
      });
    return () => abort.abort();
  }, [mesh, ready]);

  const conditionKey = JSON.stringify([...conditions]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: conditionKey stands for conditions
  useEffect(() => {
    viewer.current?.setZoneColors(
      surface?.zones.map((z) => {
        const kind = conditions.get(z.name);
        return kind === undefined ? null : conditionColor(kind);
      }) ?? [],
    );
  }, [surface, conditionKey]);

  useEffect(() => {
    const index = surface?.zones.findIndex((z) => z.name === selected) ?? -1;
    viewer.current?.setSelectedZone(index >= 0 ? index : null);
  }, [surface, selected]);

  const theme = useResolvedTheme();
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs when the displayed theme changes
  useEffect(() => {
    const frame = requestAnimationFrame(() => viewer.current?.applyTheme());
    return () => cancelAnimationFrame(frame);
  }, [theme]);

  return (
    <div className="relative h-72 flex-none overflow-hidden rounded-lg border border-line bg-bg-0">
      <canvas
        ref={canvas}
        className="block size-full"
        data-testid="mesh-view"
        data-ready={surface ? "true" : undefined}
        aria-label={`Boundary of ${mesh}; click a zone to set its condition`}
        onClick={async (e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const index = await viewer.current?.pick(e.clientX - rect.left, e.clientY - rect.top);
          const zone = index == null ? undefined : surface?.zones[index]?.name;
          if (zone) onPick(zone);
        }}
      />
      {error && (
        <div className="absolute inset-0 flex flex-col gap-3 overflow-auto bg-bg-0 p-4">
          <p className="m-0 text-fg-2 text-sm">
            {error.includes("404")
              ? "No 3D view of this mesh (autan-mesh-info --surface writes one). Its boundary zones:"
              : `${error}. Its boundary zones:`}
          </p>
          <ul className="m-0 flex list-none flex-wrap gap-2 p-0" data-testid="zone-list">
            {zones.map((zone) => (
              <li key={zone}>
                <button
                  type="button"
                  aria-pressed={selected === zone}
                  onClick={() => onPick(zone)}
                  className="rounded-sm px-2 py-1 font-mono text-sm shadow-[0_0_0_1px_var(--line)] hover:bg-bg-2 focus-visible:shadow-[var(--focus-ring)] aria-pressed:bg-accent-soft aria-pressed:text-accent"
                >
                  {zone}
                  {conditions.has(zone) && (
                    <span className="ml-2 font-sans text-fg-3">{conditions.get(zone)}</span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      {selected && surface?.zones.some((z) => z.name === selected) && (
        <span className="absolute top-2 left-2 rounded-sm bg-bg-1/90 px-2 py-1 font-mono text-sm">
          {selected}
        </span>
      )}
    </div>
  );
}
