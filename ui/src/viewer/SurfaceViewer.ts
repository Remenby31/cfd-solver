import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { attribute, clamp, float, mix, texture, uniform, vec2, vec4 } from "three/tsl";
import * as THREE from "three/webgpu";
import type { Surface } from "../api/surface";
import { COOL_WARM, colormapFor, displayRange, sampleColormap } from "./colormap";

export type Backend = "webgpu" | "webgl2";

const token = (name: string) =>
  new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue(name).trim());

interface ZoneMaterial {
  material: THREE.MeshStandardNodeMaterial;
  highlight: { value: number };
  tint: { value: THREE.Color };
  tinted: { value: number };
}

/** Renders one surface with a node field through a colormap; zones are pickable on the GPU. */
export class SurfaceViewer {
  readonly backend: Backend;
  private readonly renderer: THREE.WebGPURenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000);
  private readonly controls: OrbitControls;
  private readonly resize: ResizeObserver;
  private readonly lut: THREE.DataTexture;
  private readonly range = { min: uniform(0), span: uniform(1) };
  private readonly accent = uniform(new THREE.Color());
  private mesh: THREE.Mesh | null = null;
  private surface: Surface | null = null;
  private geometry: THREE.BufferGeometry | null = null;
  private zones: ZoneMaterial[] = [];
  private zoneColors: (string | null)[] = [];
  private pickMaterials: THREE.MeshBasicNodeMaterial[] = [];
  private pickTarget = new THREE.RenderTarget(1, 1);
  private frame = 0;

  static async create(
    canvas: HTMLCanvasElement,
    options: { forceWebGL?: boolean } = {},
  ): Promise<SurfaceViewer> {
    const adapter =
      !options.forceWebGL && "gpu" in navigator ? await navigator.gpu.requestAdapter() : null;
    // Surfaces of 10+ M triangles exceed the 256 MiB default buffer limit.
    const requiredLimits = adapter
      ? {
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        }
      : undefined;
    const renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: true,
      forceWebGL: options.forceWebGL ?? false,
      ...(requiredLimits ? { requiredLimits } : {}),
    });
    await renderer.init();
    return new SurfaceViewer(canvas, renderer);
  }

  private constructor(canvas: HTMLCanvasElement, renderer: THREE.WebGPURenderer) {
    this.renderer = renderer;
    const backend = renderer.backend as { isWebGPUBackend?: boolean };
    this.backend = backend.isWebGPUBackend ? "webgpu" : "webgl2";
    if (this.backend === "webgl2")
      console.warn("viewer: WebGPU unavailable, rendering with WebGL2");
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.applyTheme();
    this.scene.add(new THREE.HemisphereLight("#dfe8f2", "#2a323c", 1.6));
    const headlight = new THREE.DirectionalLight("#ffffff", 1.6);
    headlight.position.set(0.3, 0.6, 1);
    this.camera.add(headlight);
    this.scene.add(this.camera);
    this.camera.up.set(0, 0, 1);

    this.lut = new THREE.DataTexture(sampleColormap(COOL_WARM), 256, 1);
    this.lut.colorSpace = THREE.SRGBColorSpace;
    this.lut.magFilter = THREE.LinearFilter;
    this.lut.needsUpdate = true;

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.addEventListener("change", () => this.requestRender());
    this.resize = new ResizeObserver(() => this.fit());
    this.resize.observe(canvas);
    this.fit();
  }

  setSurface(surface: Surface, fieldName: string): void {
    this.clearMesh();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(surface.positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(surface.indices, 1));
    this.surface = surface;
    this.geometry = geometry;
    if (surface.fields.length === 0) {
      // A bare mesh: every node at the middle of the colormap, a neutral tone the selection
      // highlight stands out from.
      geometry.setAttribute(
        "value",
        new THREE.BufferAttribute(new Float32Array(surface.positions.length / 3), 1),
      );
      this.range.min.value = -1;
      this.range.span.value = 2;
    }
    this.setField(fieldName);
    geometry.computeBoundingSphere();

    const t = clamp(
      attribute<"float">("value", "float").sub(this.range.min).div(this.range.span),
      0,
      1,
    );
    const fieldColor = texture(this.lut, vec2(t, float(0.5))).rgb;
    this.zoneColors = [];
    this.zones = surface.zones.map((zone, index) => {
      geometry.addGroup(zone.first_triangle * 3, zone.triangles * 3, index);
      const highlight = uniform(0);
      const tint = uniform(new THREE.Color());
      const tinted = uniform(0);
      const material = new THREE.MeshStandardNodeMaterial({
        flatShading: true,
        roughness: 0.65,
        side: THREE.DoubleSide,
      });
      material.colorNode = mix(mix(fieldColor, tint, tinted), this.accent, highlight);
      return { material, highlight, tint, tinted };
    });
    this.pickMaterials = surface.zones.map((_, index) => {
      const id = index + 1;
      const material = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide });
      material.colorNode = vec4((id % 256) / 255, Math.floor(id / 256) / 255, 0, 1);
      return material;
    });
    this.mesh = new THREE.Mesh(
      geometry,
      this.zones.map((z) => z.material),
    );
    this.scene.add(this.mesh);
    this.frameAll();
  }

  /** Draw nothing, until the next setSurface. */
  clear(): void {
    this.clearMesh();
    this.surface = null;
    this.requestRender();
  }

  /** Re-read the background and selection colours from the design tokens. */
  applyTheme(): void {
    this.scene.background = token("--bg-0");
    this.accent.value.copy(token("--accent"));
    this.setZoneColors(this.zoneColors);
    this.requestRender();
  }

  /** Colour the surface by another of its fields: a new attribute, range and colormap, no
   * geometry rebuild. */
  setField(fieldName: string): void {
    const field = this.surface?.fields.find((f) => f.name === fieldName);
    if (!field || !this.geometry) return;
    this.geometry.setAttribute("value", new THREE.BufferAttribute(field.values, 1));
    const range = displayRange(field.min, field.max);
    this.range.min.value = range.min;
    this.range.span.value = range.max - range.min || 1;
    (this.lut.image.data as Uint8Array).set(sampleColormap(colormapFor(field.min, field.max)));
    this.lut.needsUpdate = true;
    this.requestRender();
  }

  /** Paint zones in a flat colour, by index: a CSS colour or design token (`--series-1`), or
   * null to show the field. */
  setZoneColors(colors: (string | null)[]): void {
    this.zoneColors = colors;
    this.zones.forEach((zone, k) => {
      const color = colors[k] ?? null;
      zone.tinted.value = color === null ? 0 : 1;
      if (color !== null)
        zone.tint.value.copy(color.startsWith("--") ? token(color) : new THREE.Color(color));
    });
    this.requestRender();
  }

  setSelectedZone(index: number | null): void {
    this.zones.forEach((zone, k) => {
      zone.highlight.value = k === index ? 0.55 : 0;
    });
    this.requestRender();
  }

  /** Zone index under the canvas pixel (CSS pixels), or null for the background. */
  async pick(x: number, y: number): Promise<number | null> {
    if (!this.mesh) return null;
    const canvas = this.renderer.domElement;
    const ratio = this.renderer.getPixelRatio();
    const width = canvas.clientWidth * ratio;
    const height = canvas.clientHeight * ratio;
    const materials = this.mesh.material;
    const background = this.scene.background;
    this.mesh.material = this.pickMaterials;
    this.scene.background = new THREE.Color(0, 0, 0);
    this.camera.setViewOffset(width, height, Math.round(x * ratio), Math.round(y * ratio), 1, 1);
    this.renderer.setRenderTarget(this.pickTarget);
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(null);
    this.camera.clearViewOffset();
    this.mesh.material = materials;
    this.scene.background = background;
    const pixel = (await this.renderer.readRenderTargetPixelsAsync(
      this.pickTarget,
      0,
      0,
      1,
      1,
    )) as Uint8Array;
    this.requestRender();
    const id = (pixel[0] ?? 0) + 256 * (pixel[1] ?? 0);
    return id === 0 ? null : id - 1;
  }

  /** Milliseconds per frame, each frame rendered and waited for on the GPU, orbiting the camera. */
  async benchmark(frames: number): Promise<number[]> {
    const times: number[] = [];
    for (let i = 0; i < frames; i++) {
      const start = performance.now();
      this.orbit(0.01);
      this.renderer.render(this.scene, this.camera);
      await this.finish();
      times.push(performance.now() - start);
    }
    return times;
  }

  private async finish(): Promise<void> {
    const backend = this.renderer.backend as {
      device?: GPUDevice;
      gl?: WebGL2RenderingContext;
    };
    if (backend.device) {
      await backend.device.queue.onSubmittedWorkDone();
    } else if (backend.gl) {
      const gl = backend.gl;
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    }
  }

  orbit(radians: number): void {
    const offset = this.camera.position.clone().sub(this.controls.target);
    offset.applyAxisAngle(this.camera.up, radians);
    this.camera.position.copy(this.controls.target).add(offset);
    this.camera.lookAt(this.controls.target);
  }

  requestRender(): void {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      const moving = this.controls.update();
      this.renderer.render(this.scene, this.camera);
      if (moving) this.requestRender();
    });
  }

  dispose(): void {
    cancelAnimationFrame(this.frame);
    this.resize.disconnect();
    this.controls.dispose();
    this.clearMesh();
    this.lut.dispose();
    this.pickTarget.dispose();
    this.renderer.dispose();
  }

  private clearMesh(): void {
    if (!this.mesh) return;
    this.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    for (const zone of this.zones) zone.material.dispose();
    for (const material of this.pickMaterials) material.dispose();
    this.mesh = null;
    this.zones = [];
    this.pickMaterials = [];
  }

  private frameAll(): void {
    const sphere = this.mesh?.geometry.boundingSphere;
    if (!sphere) return;
    const distance = sphere.radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2));
    this.controls.target.copy(sphere.center);
    this.camera.position
      .copy(sphere.center)
      .add(new THREE.Vector3(-0.7, -1, 0.75).normalize().multiplyScalar(distance));
    this.camera.near = distance / 100;
    this.camera.far = distance * 10;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.requestRender();
  }

  private fit(): void {
    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }
}
