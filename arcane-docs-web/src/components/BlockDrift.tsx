"use client";

import { useEffect, useRef, type CSSProperties } from "react";
import * as THREE from "three";

const DEFAULTS = {
  near: "#0E0318",
  far: "#944FDE",
  edge: "#000000",
  grid: 17,
  blockSize: 12,
  gap: 20,
  layers: 15,
  density: 10,
  cluster: 10,
  edgeWidth: 1,
  fade: 1,
  shade: 20,
  clearCentre: 3,
  speed: 12,
  direction: "front" as const,
};

const MAX_GRID = 41;

type Config = {
  near: string;
  far: string;
  edge: string;
  grid: number;
  blockSize: number;
  gap: number;
  layers: number;
  density: number;
  cluster: number;
  edgeWidth: number;
  fade: number;
  shade: number;
  clearCentre: number;
  speed: number;
  direction: "front" | "back";
};

export type BlockDriftProps = Partial<Config> & {
  className?: string;
  style?: CSSProperties;
};

function clamp(value: number, minimum: number, maximum: number, fallback: number) {
  const safeValue = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.max(minimum, Math.min(maximum, safeValue));
}

function settingsFor(config: Config) {
  const requestedGrid = Math.round(clamp(config.grid, 3, 31, DEFAULTS.grid));
  const block = 0.12 + clamp(config.blockSize, 1, 20, DEFAULTS.blockSize) * 0.05;

  return {
    grid: requestedGrid % 2 === 0 ? requestedGrid + 1 : requestedGrid,
    layers: Math.round(clamp(config.layers, 5, 15, DEFAULTS.layers)),
    block,
    spacing: block,
    gap: 0.4 + clamp(config.gap, 1, 20, DEFAULTS.gap) * 0.11,
    density: clamp(config.density, 1, 20, DEFAULTS.density) / 20,
    cluster: 1.4 - clamp(config.cluster, 1, 20, DEFAULTS.cluster) * 0.055,
    edgeWidth: clamp(config.edgeWidth, 0, 20, DEFAULTS.edgeWidth) * 0.011,
    fade: clamp(config.fade, 1, 20, DEFAULTS.fade) * 0.011,
    shade: clamp(config.shade, 0, 20, DEFAULTS.shade) * 0.03,
    speed: clamp(config.speed, 0, 20, DEFAULTS.speed) * 0.09,
    heading: config.direction === "back" ? -1 : 1,
    clearCentre: Math.round(clamp(config.clearCentre, 0, 5, DEFAULTS.clearCentre)),
  };
}

function buildBlocks(grid: number, layers: number) {
  const cells = grid * grid;
  const count = cells * layers;
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const cell = new Float32Array(count);
  const layer = new Float32Array(count);

  let index = 0;
  for (let layerIndex = 0; layerIndex < layers; layerIndex += 1) {
    for (let cellIndex = 0; cellIndex < cells; cellIndex += 1) {
      cell[index] = cellIndex;
      layer[index] = layerIndex;
      index += 1;
    }
  }

  geometry.setAttribute("aCell", new THREE.InstancedBufferAttribute(cell, 1));
  geometry.setAttribute("aLayer", new THREE.InstancedBufferAttribute(layer, 1));
  return geometry;
}

const BLOCK_VERTEX = `
  attribute float aCell;
  attribute float aLayer;

  uniform float uTime;
  uniform float uGrid;
  uniform float uBlock;
  uniform float uSpacing;
  uniform float uGap;
  uniform float uDensity;
  uniform float uCluster;
  uniform float uClearCentre;
  uniform float uHeading;
  uniform float uLayers;

  varying float vDepth;
  varying vec3 vNormal;
  varying float vAlive;
  varying vec2 vFace;

  float hash(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 34.56);
    return fract(p.x * p.y * 95.43);
  }

  float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
      mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
      u.y
    );
  }

  void main() {
    float cx = mod(aCell, uGrid);
    float cy = floor(aCell / uGrid);
    float mid = (uGrid - 1.0) * 0.5;
    vec2 cell = vec2(cx, cy) - mid;
    vec2 xy = cell * uSpacing;

    float beat = floor(uTime);
    float progress = fract(uTime);
    float slide = uHeading > 0.0 ? progress : 1.0 - progress;
    float depth = aLayer + 1.0 - slide;
    float seed = beat * uHeading + aLayer;
    float noise = vnoise(vec2(cx, cy) * uCluster + seed * 19.3);
    float alive = step(noise, uDensity);

    float ring = max(abs(cell.x), abs(cell.y));
    alive *= step(uClearCentre + 0.5, ring);

    float depthNormal = depth / max(1.0, uLayers);
    float grow = smoothstep(1.0, 0.84, depthNormal);
    vec2 drift = vec2(
      sin(uTime * 0.55 + depth * 0.18),
      cos(uTime * 0.42 + depth * 0.14)
    ) * 0.06;
    vec3 positionOut = position * uBlock * alive * grow;
    positionOut.xy += xy + drift;
    positionOut.z += uGap - depth * uGap;

    vec4 modelView = modelViewMatrix * vec4(positionOut, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vDepth = clamp(depthNormal, 0.0, 1.0);
    vAlive = alive;
    vFace = uv;
    gl_Position = projectionMatrix * modelView;
  }
`;

const BLOCK_FRAGMENT = `
  uniform vec3 uNear;
  uniform vec3 uFar;
  uniform vec3 uEdge;
  uniform float uFade;
  uniform float uShade;
  uniform float uEdgeWidth;

  varying float vDepth;
  varying vec3 vNormal;
  varying float vAlive;
  varying vec2 vFace;

  void main() {
    if (vAlive < 0.5) discard;

    vec3 colour = mix(uNear, uFar, pow(vDepth, 1.0 - uFade * 8.0 + 0.9));
    vec3 normalised = normalize(vNormal);
    float face = 0.5 + 0.5 * dot(normalised, normalize(vec3(-0.4, 0.55, 0.75)));
    colour *= 1.0 - uShade + uShade * face * 2.0;

    if (uEdgeWidth > 0.0001) {
      float edge = min(min(vFace.x, 1.0 - vFace.x), min(vFace.y, 1.0 - vFace.y));
      float antialias = max(fwidth(edge), 0.0001);
      float border = 1.0 - smoothstep(uEdgeWidth - antialias, uEdgeWidth + antialias, edge);
      colour = mix(colour, uEdge, border * (1.0 - vDepth * 0.65));
    }

    gl_FragColor = vec4(max(colour, 0.0), 1.0);
  }
`;

class BlockScene {
  private readonly container: HTMLElement;
  private config: Config;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(60, 1, 0.02, 400);
  private geometry: THREE.BufferGeometry;
  private readonly material: THREE.ShaderMaterial;
  private mesh: THREE.InstancedMesh;
  private gridUsed = 0;
  private layersUsed = 0;
  private time = 0;
  private width = 0;
  private height = 0;
  private frameId = 0;
  private lastFrameTime = 0;
  private motionScale = 1;
  private disposed = false;

  constructor(container: HTMLElement, config: Config) {
    this.container = container;
    this.config = config;
    const settings = settingsFor(config);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor(0x000000, 0);

    const canvas = this.renderer.domElement;
    canvas.style.position = "absolute";
    canvas.style.inset = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.pointerEvents = "none";
    container.appendChild(canvas);

    this.material = new THREE.ShaderMaterial({
      vertexShader: BLOCK_VERTEX,
      fragmentShader: BLOCK_FRAGMENT,
      uniforms: {
        uTime: { value: 0 },
        uGrid: { value: settings.grid },
        uBlock: { value: settings.block },
        uSpacing: { value: settings.spacing },
        uGap: { value: settings.gap },
        uDensity: { value: settings.density },
        uCluster: { value: settings.cluster },
        uClearCentre: { value: settings.clearCentre },
        uHeading: { value: settings.heading },
        uLayers: { value: settings.layers },
        uNear: { value: new THREE.Color(config.near) },
        uFar: { value: new THREE.Color(config.far) },
        uEdge: { value: new THREE.Color(config.edge) },
        uFade: { value: settings.fade },
        uShade: { value: settings.shade },
        uEdgeWidth: { value: settings.edgeWidth },
      },
    });

    this.gridUsed = settings.grid;
    this.layersUsed = settings.layers;
    this.geometry = buildBlocks(settings.grid, settings.layers);
    this.mesh = this.makeMesh(settings.grid * settings.grid * settings.layers);
    this.scene.add(this.mesh);
  }

  private makeMesh(count: number) {
    const mesh = new THREE.InstancedMesh(this.geometry, this.material, count);
    const identity = new THREE.Matrix4();
    for (let index = 0; index < count; index += 1) mesh.setMatrixAt(index, identity);
    mesh.instanceMatrix.needsUpdate = true;
    mesh.frustumCulled = false;
    return mesh;
  }

  start() {
    this.motionScale = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0.25 : 1;
    this.lastFrameTime = performance.now();
    const loop = () => {
      this.frameId = requestAnimationFrame(loop);
      this.step();
    };
    loop();
  }

  setSize(width: number, height: number) {
    if (this.disposed || width <= 0 || height <= 0) return;
    this.width = width;
    this.height = height;
    this.renderer.setSize(width, height, false);
    this.updateCamera();
    this.renderer.render(this.scene, this.camera);
  }

  updateConfig(config: Config) {
    if (this.disposed) return;
    this.config = config;
    const settings = settingsFor(config);
    const uniforms = this.material.uniforms;

    uniforms.uGrid.value = settings.grid;
    uniforms.uBlock.value = settings.block;
    uniforms.uSpacing.value = settings.spacing;
    uniforms.uGap.value = settings.gap;
    uniforms.uDensity.value = settings.density;
    uniforms.uCluster.value = settings.cluster;
    uniforms.uClearCentre.value = settings.clearCentre;
    uniforms.uHeading.value = settings.heading;
    uniforms.uLayers.value = settings.layers;
    uniforms.uFade.value = settings.fade;
    uniforms.uShade.value = settings.shade;
    uniforms.uEdgeWidth.value = settings.edgeWidth;
    uniforms.uNear.value.set(config.near);
    uniforms.uFar.value.set(config.far);
    uniforms.uEdge.value.set(config.edge);
    this.updateCamera();
  }

  private resolveGrid(settings: ReturnType<typeof settingsFor>, aspect: number, fieldOfView: number) {
    const tangent = Math.tan((fieldOfView * Math.PI) / 360);
    const reach = tangent * settings.gap * 1.25 * Math.max(1, aspect);
    const cells = 2 * Math.ceil(reach / settings.block) + 1;
    return Math.min(MAX_GRID, Math.max(settings.grid, cells));
  }

  private updateCamera() {
    const width = Math.max(1, this.width);
    const height = Math.max(1, this.height);
    const aspect = width / height;
    const settings = settingsFor(this.config);
    const fieldOfView = aspect < 1 ? 78 : 62;
    const grid = this.resolveGrid(settings, aspect, fieldOfView);

    if (grid !== this.gridUsed || settings.layers !== this.layersUsed) {
      this.gridUsed = grid;
      this.layersUsed = settings.layers;
      this.scene.remove(this.mesh);
      this.mesh.dispose();
      this.geometry.dispose();
      this.geometry = buildBlocks(grid, settings.layers);
      this.mesh = this.makeMesh(grid * grid * settings.layers);
      this.scene.add(this.mesh);
    }

    this.material.uniforms.uGrid.value = grid;
    this.camera.aspect = aspect;
    this.camera.fov = fieldOfView;
    this.camera.position.set(0, 0, 0);
    this.camera.lookAt(0, 0, -1);
    this.camera.near = 0.02;
    this.camera.far = settings.gap * (settings.layers + 2);
    this.camera.updateProjectionMatrix();
  }

  private step() {
    if (this.disposed) return;
    const now = performance.now();
    let delta = (now - this.lastFrameTime) / 1000;
    this.lastFrameTime = now;
    if (!Number.isFinite(delta) || delta < 0) delta = 0;
    if (delta > 0.05) delta = 0.05;

    this.time += delta * settingsFor(this.config).speed * this.motionScale;
    this.material.uniforms.uTime.value = this.time;
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.frameId);
    this.mesh.dispose();
    this.geometry.dispose();
    this.material.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    const canvas = this.renderer.domElement;
    if (canvas.parentNode === this.container) this.container.removeChild(canvas);
  }
}

export default function BlockDrift({
  near = DEFAULTS.near,
  far = DEFAULTS.far,
  edge = DEFAULTS.edge,
  grid = DEFAULTS.grid,
  blockSize = DEFAULTS.blockSize,
  gap = DEFAULTS.gap,
  layers = DEFAULTS.layers,
  density = DEFAULTS.density,
  cluster = DEFAULTS.cluster,
  edgeWidth = DEFAULTS.edgeWidth,
  fade = DEFAULTS.fade,
  shade = DEFAULTS.shade,
  clearCentre = DEFAULTS.clearCentre,
  speed = DEFAULTS.speed,
  direction = DEFAULTS.direction,
  className,
  style,
}: BlockDriftProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<BlockScene | null>(null);
  const configRef = useRef<Config>({
    near,
    far,
    edge,
    grid,
    blockSize,
    gap,
    layers,
    density,
    cluster,
    edgeWidth,
    fade,
    shade,
    clearCentre,
    speed,
    direction,
  });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let scene: BlockScene;
    try {
      scene = new BlockScene(container, configRef.current);
    } catch {
      return;
    }

    sceneRef.current = scene;
    scene.setSize(container.clientWidth, container.clientHeight);
    scene.start();

    const resizeObserver = new ResizeObserver(() => {
      scene.setSize(container.clientWidth, container.clientHeight);
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      scene.dispose();
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    const config = {
      near,
      far,
      edge,
      grid,
      blockSize,
      gap,
      layers,
      density,
      cluster,
      edgeWidth,
      fade,
      shade,
      clearCentre,
      speed,
      direction,
    };
    configRef.current = config;
    sceneRef.current?.updateConfig(config);
  }, [near, far, edge, grid, blockSize, gap, layers, density, cluster, edgeWidth, fade, shade, clearCentre, speed, direction]);

  return (
    <div
      ref={containerRef}
      className={className}
      aria-hidden="true"
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        minWidth: 120,
        minHeight: 120,
        overflow: "hidden",
        pointerEvents: "none",
        ...style,
      }}
    />
  );
}
