"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Camera, Geometry, Mesh, Program, Renderer, Transform } from "ogl";
import {
  ALL_REGION_MASK,
  BRAIN_REGIONS,
  buildBackgroundBrain,
  buildCircuitArbors,
  buildVncNeuropilMesh,
  buildVncNeuronMesh,
  type CircuitNeuron,
} from "@/lib/drosophila-brain";
import {
  MAX_CIRCUIT_NEURONS,
  buildFlywireCircuit,
  loadFlywireGeometry,
  type FlywireGeometry,
} from "@/lib/flywire-geometry";
import {
  bindVncArborsToCircuit,
  loadMancVncGeometry,
  type MancVncGeometry,
} from "@/lib/manc-vnc-geometry";
import { FruitflyCircuitSimulator } from "@/lib/fruitfly-simulator";
import { inkMass, predictMnistRsaa } from "@/lib/mnist-rsaa";
import circuitFallback from "@/data/flywire_escape_circuit.json";

type Neuron = CircuitNeuron & {
  super_class: string;
  nt: string;
  x: number;
  y: number;
  z: number;
};

type Circuit = {
  name: string;
  neurons: Neuron[];
  edges: { pre: string; post: string; synapses: number; sign: number }[];
};

type Prediction = {
  digit: number;
  confidence: number;
  probabilities: number[];
  source?: "api" | "rsaa";
};

const CIRCUIT_URL = "/api/flywire/circuit";
const FOCUS_Y = -1.2;
const DEFAULT_ZOOM = 11.2;
const PAD = 112;
const MODEL_SIZE = 28;

const LINE_VERTEX = `
attribute vec3 position;
attribute vec3 color;
attribute float region;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform float uMask;
varying vec3 vColor;
varying float vVisible;
void main() {
  float bit = mod(floor(uMask / pow(2.0, region) + 0.001), 2.0);
  vVisible = bit;
  vColor = color;
  if (bit < 0.5) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const BG_FRAGMENT = `
precision highp float;
varying vec3 vColor;
varying float vVisible;
uniform float uAlpha;
uniform float uTime;
uniform float uActivity;
void main() {
  if (vVisible < 0.5) discard;
  // Background neurites only flash when stimulated by real circuit firing
  float spike = pow(max(0.0, sin(uTime * 9.0 + vColor.r * 16.0 + vColor.g * 10.0)), 6.0) * uActivity;
  vec3 lit = vColor * (0.30 + 1.8 * uActivity + 2.2 * spike) + vec3(0.02, 0.018, 0.035);
  gl_FragColor = vec4(lit, uAlpha * (0.35 + 0.65 * uActivity));
}
`;

const CIRCUIT_FRAGMENT = `
precision highp float;
varying vec3 vColor;
varying float vVisible;
uniform float uAlpha;
void main() {
  if (vVisible < 0.5) discard;
  gl_FragColor = vec4(vColor, uAlpha);
}
`;

/** Shared region-mask test: bit `region` of uMask decides visibility. */
const MASK_TEST = `
float maskBit(float mask, float region) {
  return mod(floor(mask / pow(2.0, region) + 0.001), 2.0);
}
`;

/**
 * Neuropil shell (navis-flybrains FLYWIRE mesh, real triangles + normals).
 * Drawn as a fresnel rim so it reads as glass and never hides the neurons.
 */
const NEUROPIL_VERTEX = `
attribute vec3 position;
attribute vec3 normal;
attribute float region;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;
uniform float uMask;
varying float vRim;
varying float vVisible;
varying float vRegion;
${MASK_TEST}
void main() {
  float bit = maskBit(uMask, region);
  vVisible = bit;
  vRegion = region;
  if (bit < 0.5) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vec3 n = normalize(normalMatrix * normal);
  vRim = 1.0 - abs(dot(n, normalize(-mv.xyz)));
  gl_Position = projectionMatrix * mv;
}
`;

const NEUROPIL_FRAGMENT = `
precision highp float;
varying float vRim;
varying float vVisible;
varying float vRegion;
uniform float uAlpha;
uniform float uActivity;
void main() {
  // ~3% of shell triangles straddle a region boundary. Their masked-out vertex
  // is pushed off-clip, which would leave a stretched wedge, so anything not
  // fully inside a visible region is dropped rather than part-drawn.
  if (vVisible < 0.999) discard;
  vec3 tint = vec3(0.30, 0.20, 0.44);
  if (vRegion < 1.5) tint = vec3(0.16, 0.40, 0.54);
  else if (vRegion < 2.5) tint = vec3(0.34, 0.20, 0.48);
  else if (vRegion < 3.5) tint = vec3(0.26, 0.46, 0.24);
  else if (vRegion < 4.5) tint = vec3(0.52, 0.40, 0.16);
  else if (vRegion < 5.5) tint = vec3(0.36, 0.22, 0.46);
  float rim = pow(clamp(vRim, 0.0, 1.0), 3.0);
  gl_FragColor = vec4(tint * rim * (0.85 + 2.4 * uActivity), uAlpha);
}
`;

/**
 * Real v783 neuron arbors, as tube meshes swept along the published skeletons.
 *
 * The action-potential wavefront is computed here rather than on the CPU: with
 * ~660k vertices, re-writing a colour buffer every frame was the whole budget.
 * `uState[i]` carries (calcium, seconds since spike, homeostatic gain) for the
 * neuron in slot i, so the per-frame upload is a few dozen floats.
 */
const ARBOR_VERTEX = `
attribute vec3 position;
attribute vec3 normal;
attribute vec3 color;
attribute float region;
attribute float aNeuron;
attribute float aDist;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;
uniform float uMask;
uniform vec3 uState[${MAX_CIRCUIT_NEURONS}];
varying vec3 vColor;
varying float vVisible;
varying vec3 vNormal;
varying vec3 vView;
${MASK_TEST}
void main() {
  float bit = maskBit(uMask, region);
  vVisible = bit;
  if (bit < 0.5) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  vec3 st = uState[int(aNeuron + 0.5)];
  float ca = st.x;
  float dtSpike = st.y;
  float gain = st.z;

  // Travelling wavefront: a gaussian in soma-distance that sweeps outward at
  // the conduction velocity used by the CPU path (3.6 display units / s).
  // The window runs past 1s so a descending spike can finish the VNC (~3.6
  // units from the neck); head arbors are shorter and go dark sooner.
  float pulse = 0.0;
  if (dtSpike >= 0.0 && dtSpike < 1.15) {
    float diff = aDist - dtSpike * 3.6;
    pulse = exp(-(diff * diff) / 0.024) * 3.4;
  }

  float glow = 0.11 + (ca * 0.82 + pulse) * gain;
  vec3 lit = color * glow;
  if (pulse > 0.3) {
    lit += min(1.0, pulse * 0.35 * gain) * vec3(0.7, 0.85, 1.0);
  }
  vColor = lit;

  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  // Interpolating the radial normal per fragment shades even a 5-sided tube as
  // a round filament; only the silhouette stays polygonal.
  vNormal = normalMatrix * normal;
  vView = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}
`;

const ARBOR_FRAGMENT = `
precision highp float;
varying vec3 vColor;
varying float vVisible;
varying vec3 vNormal;
varying vec3 vView;
void main() {
  if (vVisible < 0.5) discard;
  float ndv = abs(dot(normalize(vNormal), normalize(vView)));
  // Headlight cylinder shading: brightest along the tube's centre line,
  // falling off toward the edges. The previous shader did the opposite, which
  // read as a hollow glowing pipe rather than a solid neurite.
  float body = 0.42 + 0.58 * pow(ndv, 0.7);
  float sheen = pow(ndv, 20.0) * 0.30;
  gl_FragColor = vec4(vColor * body + vec3(sheen) * vColor, 1.0);
}
`;

function padImage(canvas: HTMLCanvasElement) {
  const off = document.createElement("canvas");
  off.width = MODEL_SIZE;
  off.height = MODEL_SIZE;
  const ctx = off.getContext("2d");
  if (!ctx) return { left: 0.5, right: 0.5, pixels: new Array(MODEL_SIZE * MODEL_SIZE).fill(255) };
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  ctx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const pixels = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const gray: number[] = [];
  let left = 0;
  let right = 0;
  let nL = 0;
  let nR = 0;
  for (let y = 0; y < MODEL_SIZE; y++) {
    for (let x = 0; x < MODEL_SIZE; x++) {
      const v = pixels[(y * MODEL_SIZE + x) * 4];
      gray.push(v);
      const ink = 1 - v / 255;
      if (x < MODEL_SIZE / 2) {
        left += ink;
        nL++;
      } else {
        right += ink;
        nR++;
      }
    }
  }
  return { left: left / nL, right: right / nR, pixels: gray };
}

type OrbitQuat = { x: number; y: number; z: number; w: number };

function quatMul(a: OrbitQuat, b: OrbitQuat): OrbitQuat {
  return {
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  };
}

function quatAxisAngle(ax: number, ay: number, az: number, angle: number): OrbitQuat {
  const h = angle * 0.5;
  const s = Math.sin(h);
  return { x: ax * s, y: ay * s, z: az * s, w: Math.cos(h) };
}

function quatNormalize(q: OrbitQuat): OrbitQuat {
  const n = Math.hypot(q.x, q.y, q.z, q.w) || 1;
  return { x: q.x / n, y: q.y / n, z: q.z / n, w: q.w / n };
}

function quatFromYawPitch(yaw: number, pitch: number): OrbitQuat {
  return quatMul(quatAxisAngle(0, 1, 0, yaw), quatAxisAngle(1, 0, 0, pitch));
}

const ORBIT_SENSITIVITY = 0.008;
const IDLE_SPIN = 0.00045;

export function FlyWireConnectome() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const padRef = useRef<HTMLCanvasElement>(null);
  const circuitRef = useRef<Circuit | null>(null);
  const simulatorRef = useRef<FruitflyCircuitSimulator | null>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const orbitRef = useRef<OrbitQuat>(quatFromYawPitch(0.15, -0.22));
  const zoomRef = useRef(DEFAULT_ZOOM);
  const draggingRef = useRef(false);
  const drawingRef = useRef(false);
  const maskRef = useRef(ALL_REGION_MASK);
  const liveFrameRef = useRef(0);
  const predictionRef = useRef<Prediction | null>(null);

  const [circuit, setCircuit] = useState<Circuit | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [regionMask, setRegionMask] = useState(ALL_REGION_MASK);
  /** "real" once the v783 blob is decoded; "fallback" if it could not load. */
  const [geometrySource, setGeometrySource] = useState<"loading" | "real" | "fallback">("loading");
  const [geometryProgress, setGeometryProgress] = useState(0);
  const [stats, setStats] = useState({
    firingRateHz: 0,
    activeCount: 0,
    meanActivity: 0,
    meanGain: 1.0,
    divergence: 0,
  });

  const applyCircuitData = useCallback((data: Circuit) => {
    setCircuit(data);
    circuitRef.current = data;
    simulatorRef.current = new FruitflyCircuitSimulator(data.neurons, data.edges);
  }, []);

  const loadCircuit = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(CIRCUIT_URL);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.error === "string" ? data.error : res.statusText);
      applyCircuitData(data as Circuit);
    } catch {
      const data = circuitFallback as Circuit;
      applyCircuitData(data);
    } finally {
      setLoading(false);
    }
  }, [applyCircuitData]);

  useEffect(() => {
    loadCircuit();
  }, [loadCircuit]);

  useEffect(() => {
    maskRef.current = regionMask;
  }, [regionMask]);

  const sense = useCallback(() => {
    const pad = padRef.current;
    if (!pad) return;
    const means = padImage(pad);
    const mass = inkMass(means.pixels);
    if (mass < 6) {
      setPrediction(null);
      predictionRef.current = null;
      if (simulatorRef.current) {
        simulatorRef.current.reset();
      }
      setStats({ firingRateHz: 0, activeCount: 0, meanActivity: 0, meanGain: 1.0, divergence: 0 });
      return;
    }
    const next = predictMnistRsaa(means.pixels);
    predictionRef.current = next;
    setPrediction(next);
  }, []);

  const startSensing = useCallback(() => {
    const tick = () => {
      liveFrameRef.current = 0;
      sense();
      if (drawingRef.current) {
        liveFrameRef.current = requestAnimationFrame(tick);
      }
    };
    if (!liveFrameRef.current) liveFrameRef.current = requestAnimationFrame(tick);
  }, [sense]);

  const clearPad = useCallback(() => {
    const pad = padRef.current;
    const ctx = pad?.getContext("2d");
    if (!ctx || !pad) return;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, PAD, PAD);
    setPrediction(null);
    predictionRef.current = null;
    if (simulatorRef.current) {
      simulatorRef.current.reset();
    }
    setStats({ firingRateHz: 0, activeCount: 0, meanActivity: 0, meanGain: 1.0, divergence: 0 });
  }, []);

  useLayoutEffect(() => {
    const pad = padRef.current;
    if (!pad) return;
    const ctx = pad.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, PAD, PAD);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let disposed = false;

    const renderer = new Renderer({
      canvas,
      dpr: Math.min(2, window.devicePixelRatio || 1),
      antialias: true,
      alpha: false,
      depth: true,
    });
    const gl = renderer.gl;
    gl.clearColor(0.025, 0.027, 0.04, 1);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    const camera = new Camera(gl, { fov: 28, near: 0.1, far: 60 });
    camera.position.set(0, FOCUS_Y, DEFAULT_ZOOM);
    camera.lookAt([0, FOCUS_Y, 0]);

    const scene = new Transform();
    scene.scale.set(0.86, 0.86, 0.86);

    // --- Background: procedural until the real geometry arrives -------------
    // Both paths write uActivity, so the render loop does not care which is up.
    const backgroundPrograms: Program[] = [];
    let backgroundMesh: Mesh | null = null;

    const buildProceduralBackground = () => {
      const bg = buildBackgroundBrain(window.innerWidth < 700 ? 6500 : 11000);
      const bgGeom = new Geometry(gl, {
        position: { size: 3, data: bg.positions },
        color: { size: 3, data: bg.colors },
        region: { size: 1, data: bg.regions },
      });
      const bgProgram = new Program(gl, {
        vertex: LINE_VERTEX,
        fragment: BG_FRAGMENT,
        uniforms: {
          uAlpha: { value: 0.55 },
          uTime: { value: 0 },
          uMask: { value: maskRef.current },
          uActivity: { value: 0 },
        },
        transparent: true,
        depthTest: false,
        depthWrite: false,
        cullFace: false,
      });
      const mesh = new Mesh(gl, { geometry: bgGeom, program: bgProgram, mode: gl.LINES });
      mesh.frustumCulled = false;
      mesh.setParent(scene);
      backgroundMesh = mesh;
      backgroundPrograms.push(bgProgram);
    };

    buildProceduralBackground();

    // --- Circuit arbors ----------------------------------------------------
    let circuitMesh: Mesh | null = null;
    let circuitGeom: Geometry | null = null;
    let circuitProgram: Program | null = null;
    /** Non-null only on the procedural fallback, which colours on the CPU. */
    let cpuColoring: {
      vertexNeuron: string[];
      vertexDistances: Float32Array;
      baseColors: Float32Array;
      spikeTarget: Float32Array;
    } | null = null;
    /**
     * Non-null only on the real-geometry path, which colours on the GPU.
     * `state` must be a plain array: ogl skips Float32Array uniform arrays.
     */
    let gpuColoring: { neuronIds: string[]; state: number[] } | null = null;

    const clearCircuit = () => {
      if (circuitMesh) circuitMesh.setParent(null);
      circuitMesh = null;
      circuitGeom = null;
      circuitProgram = null;
      cpuColoring = null;
      gpuColoring = null;
    };

    /** Procedural arbors: seeded random walks, coloured per vertex on the CPU. */
    const buildFallbackCircuit = (data: Circuit) => {
      clearCircuit();
      const built = buildCircuitArbors(data.neurons);
      cpuColoring = {
        vertexNeuron: built.vertexNeuron,
        vertexDistances: built.vertexDistances,
        baseColors: built.baseColors,
        spikeTarget: new Float32Array(built.cloud.colors.length),
      };
      circuitGeom = new Geometry(gl, {
        position: { size: 3, data: built.cloud.positions },
        color: { size: 3, data: built.cloud.colors },
        region: { size: 1, data: built.cloud.regions },
      });
      circuitProgram = new Program(gl, {
        vertex: LINE_VERTEX,
        fragment: CIRCUIT_FRAGMENT,
        uniforms: { uAlpha: { value: 0.92 }, uMask: { value: maskRef.current } },
        transparent: true,
        depthTest: false,
        depthWrite: false,
        cullFace: false,
      });
      circuitMesh = new Mesh(gl, { geometry: circuitGeom, program: circuitProgram, mode: gl.LINES });
      circuitMesh.frustumCulled = false;
      circuitMesh.setParent(scene);
    };

    /** Real v783 skeletons: opaque tube meshes, coloured in the vertex shader. */
    const buildRealCircuit = (geometry: FlywireGeometry, data: Circuit) => {
      const merged = buildFlywireCircuit(geometry, data.neurons);
      if (merged.count === 0) return false;
      clearCircuit();

      gpuColoring = {
        neuronIds: merged.neuronIds,
        state: new Array(MAX_CIRCUIT_NEURONS * 3).fill(0),
      };
      // Slots with no neuron must look permanently silent, not freshly spiked.
      for (let i = merged.neuronIds.length; i < MAX_CIRCUIT_NEURONS; i++) {
        gpuColoring.state[i * 3 + 1] = 1e3;
        gpuColoring.state[i * 3 + 2] = 1;
      }

      circuitGeom = new Geometry(gl, {
        position: { size: 3, data: merged.positions },
        normal: { size: 3, data: merged.normals },
        color: { size: 3, data: merged.colors },
        region: { size: 1, data: merged.regions },
        aNeuron: { size: 1, data: merged.neuronIndex },
        aDist: { size: 1, data: merged.distances },
      });
      circuitProgram = new Program(gl, {
        vertex: ARBOR_VERTEX,
        fragment: ARBOR_FRAGMENT,
        uniforms: {
          uMask: { value: maskRef.current },
          uState: { value: gpuColoring.state },
        },
        // Solid geometry: let the depth buffer resolve overlap instead of
        // additively stacking every tube in front of every other one.
        transparent: false,
        depthTest: true,
        depthWrite: true,
        cullFace: false,
      });
      circuitMesh = new Mesh(gl, { geometry: circuitGeom, program: circuitProgram });
      circuitMesh.frustumCulled = false;
      circuitMesh.setParent(scene);
      return true;
    };

    /** Swap the procedural fiber cloud for the FlyWire neuropil + Male CNS VNC. */
    const installRealBackground = (geometry: FlywireGeometry, vnc: MancVncGeometry | null) => {
      if (backgroundMesh) backgroundMesh.setParent(null);
      backgroundPrograms.length = 0;

      const shellGeom = new Geometry(gl, {
        position: { size: 3, data: geometry.meshPositions },
        normal: { size: 3, data: geometry.meshNormals },
        region: { size: 1, data: geometry.meshRegions },
      });
      const shellProgram = new Program(gl, {
        vertex: NEUROPIL_VERTEX,
        fragment: NEUROPIL_FRAGMENT,
        uniforms: {
          uAlpha: { value: 0.85 },
          uMask: { value: maskRef.current },
          uActivity: { value: 0 },
        },
        transparent: true,
        // Reads depth so neurons occlude the far wall, but writes none so the
        // shell never hides an arbor behind it.
        depthTest: true,
        depthWrite: false,
        cullFace: false,
      });
      shellProgram.setBlendFunc(gl.SRC_ALPHA, gl.ONE);
      const shell = new Mesh(gl, { geometry: shellGeom, program: shellProgram });
      shell.frustumCulled = false;
      shell.renderOrder = 1;
      shell.setParent(scene);
      backgroundPrograms.push(shellProgram);

      // FAFB stops at the neck. Prefer the Male CNS VNC neuropil mesh; fall back
      // to the procedural cord only if that blob failed to load.
      const vncShell = vnc
        ? { positions: vnc.meshPositions, normals: vnc.meshNormals, regions: vnc.meshRegions }
        : buildVncNeuropilMesh();
      const vncArbor = vnc
        ? {
            positions: vnc.arborPositions,
            normals: vnc.arborNormals,
            colors: vnc.arborColors,
            regions: vnc.arborRegions,
          }
        : buildVncNeuronMesh();

      const vncShellGeom = new Geometry(gl, {
        position: { size: 3, data: vncShell.positions },
        normal: { size: 3, data: vncShell.normals },
        region: { size: 1, data: vncShell.regions },
      });
      const vncShellProgram = new Program(gl, {
        vertex: NEUROPIL_VERTEX,
        fragment: NEUROPIL_FRAGMENT,
        uniforms: {
          uAlpha: { value: 0.85 },
          uMask: { value: maskRef.current },
          uActivity: { value: 0 },
        },
        transparent: true,
        depthTest: true,
        depthWrite: false,
        cullFace: false,
      });
      vncShellProgram.setBlendFunc(gl.SRC_ALPHA, gl.ONE);
      const vncShellMesh = new Mesh(gl, { geometry: vncShellGeom, program: vncShellProgram });
      vncShellMesh.frustumCulled = false;
      vncShellMesh.renderOrder = 1;
      vncShellMesh.setParent(scene);
      backgroundPrograms.push(vncShellProgram);

      const vncBound = bindVncArborsToCircuit(
        vncArbor.positions,
        vncArbor.colors,
        vncArbor.positions.length / 3,
        (gpuColoring?.neuronIds ?? []).map((id, index) => {
          const n = circuitRef.current?.neurons.find((cell) => cell.id === id);
          return {
            index,
            cell_type: n?.cell_type ?? "",
            layer: n?.layer ?? "",
            side: n?.side ?? "",
          };
        })
      );

      const vncNeuronGeom = new Geometry(gl, {
        position: { size: 3, data: vncArbor.positions },
        normal: { size: 3, data: vncArbor.normals },
        color: { size: 3, data: vncBound.colors },
        region: { size: 1, data: vncArbor.regions },
        aNeuron: { size: 1, data: vncBound.neuronIndex },
        aDist: { size: 1, data: vncBound.distances },
      });
      const vncState = gpuColoring?.state ?? new Array(MAX_CIRCUIT_NEURONS * 3).fill(0);
      const vncNeuronProgram = new Program(gl, {
        vertex: ARBOR_VERTEX,
        fragment: ARBOR_FRAGMENT,
        uniforms: {
          uMask: { value: maskRef.current },
          uState: { value: vncState },
        },
        transparent: false,
        depthTest: true,
        depthWrite: true,
        cullFace: false,
      });
      const vncNeuronMesh = new Mesh(gl, { geometry: vncNeuronGeom, program: vncNeuronProgram });
      vncNeuronMesh.frustumCulled = false;
      vncNeuronMesh.renderOrder = 0;
      vncNeuronMesh.setParent(scene);
      backgroundPrograms.push(vncNeuronProgram);
    };

    if (circuitRef.current) buildFallbackCircuit(circuitRef.current);

    // Real geometry is a ~23 MB blob, so the procedural scene renders first and
    // is replaced in place once the download and decode finish. The Male CNS
    // VNC mesh loads in parallel and is swapped in with the brain shell.
    let realGeometry: FlywireGeometry | null = null;
    const vncPromise = loadMancVncGeometry().catch(() => null);
    loadFlywireGeometry((loaded, total) => {
      if (!disposed) setGeometryProgress(total > 0 ? loaded / total : 0);
    })
      .then(async (geometry) => {
        if (disposed) return;
        const data = circuitRef.current;
        if (!data || !buildRealCircuit(geometry, data)) {
          setGeometrySource("fallback");
          return;
        }
        realGeometry = geometry;
        const vnc = await vncPromise;
        if (disposed) return;
        installRealBackground(geometry, vnc);
        setGeometrySource("real");
      })
      .catch(() => {
        if (!disposed) setGeometrySource("fallback");
      });

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const next = zoomRef.current * (e.deltaY > 0 ? 1.08 : 0.92);
      zoomRef.current = Math.max(4.2, Math.min(24, next));
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });

    let frame = 0;
    const t0 = performance.now();
    let lastTime = performance.now();
    let lastHudTime = performance.now();

    const loop = () => {
      frame = requestAnimationFrame(loop);
      const now = performance.now();
      const dt = Math.min(0.05, Math.max(0.001, (now - lastTime) / 1000));
      lastTime = now;
      const time = (now - t0) / 1000;

      const rect = canvas.getBoundingClientRect();
      const w = Math.max(320, Math.floor(rect.width));
      const h = Math.max(280, Math.floor(rect.height));
      if (gl.canvas.width !== Math.floor(w * renderer.dpr) || gl.canvas.height !== Math.floor(h * renderer.dpr)) {
        renderer.setSize(w, h);
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        camera.perspective({ aspect: w / h });
      }
      if (!draggingRef.current) {
        orbitRef.current = quatMul(quatAxisAngle(0, 1, 0, IDLE_SPIN), orbitRef.current);
      }
      const q = orbitRef.current;
      scene.quaternion.set(q.x, q.y, q.z, q.w);
      camera.position.set(0, FOCUS_Y, zoomRef.current);
      camera.lookAt([0, FOCUS_Y, 0]);

      const data = circuitRef.current;
      if (data && !circuitMesh) {
        if (!realGeometry || !buildRealCircuit(realGeometry, data)) buildFallbackCircuit(data);
      }

      let activity = 0;
      const sim = simulatorRef.current;
      if (sim) {
        const pad = padRef.current;
        if (pad) {
          const padData = padImage(pad);
          sim.updateSensoryInput(padData.pixels, drawingRef.current);
        }
        if (predictionRef.current) {
          sim.updateResonanceFeedback(predictionRef.current.probabilities);
        } else {
          sim.updateResonanceFeedback(null);
        }
        sim.step(dt, time);

        if (gpuColoring) {
          // ~40 floats per frame; the wavefront itself is evaluated per vertex
          // in ARBOR_VERTEX.
          sim.writeNeuronState(gpuColoring.state, gpuColoring.neuronIds, time);
        } else if (cpuColoring && circuitGeom) {
          sim.applyColors(
            cpuColoring.spikeTarget,
            cpuColoring.vertexNeuron,
            cpuColoring.vertexDistances,
            cpuColoring.baseColors,
            time
          );
          const attr = circuitGeom.attributes.color as { data: Float32Array; needsUpdate: boolean };
          attr.data.set(cpuColoring.spikeTarget);
          attr.needsUpdate = true;
        }

        const s = sim.getStats();
        activity = s.meanActivity;

        if (now - lastHudTime > 120) {
          lastHudTime = now;
          setStats({
            firingRateHz: s.firingRateHz,
            activeCount: s.activeCount,
            meanActivity: s.meanActivity,
            meanGain: s.meanGain,
            divergence: s.divergence,
          });
        }
      }

      for (const program of backgroundPrograms) {
        if (program.uniforms.uActivity) program.uniforms.uActivity.value = activity;
        if (program.uniforms.uTime) program.uniforms.uTime.value = time;
        if (program.uniforms.uMask) program.uniforms.uMask.value = maskRef.current;
      }
      if (circuitProgram) circuitProgram.uniforms.uMask.value = maskRef.current;

      renderer.render({ scene, camera });
    };
    frame = requestAnimationFrame(loop);

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      canvas.removeEventListener("wheel", onWheel);
    };
  }, [circuit]);

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = { x: e.clientX, y: e.clientY };
    draggingRef.current = true;
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    const dx = (e.clientX - dragRef.current.x) * ORBIT_SENSITIVITY;
    const dy = (e.clientY - dragRef.current.y) * ORBIT_SENSITIVITY;
    dragRef.current = { x: e.clientX, y: e.clientY };
    orbitRef.current = quatNormalize(
      quatMul(quatAxisAngle(0, 1, 0, dx), quatMul(quatAxisAngle(1, 0, 0, dy), orbitRef.current)),
    );
  };
  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    dragRef.current = null;
    draggingRef.current = false;
  };

  const padPointer = (e: React.PointerEvent<HTMLCanvasElement>, type: "down" | "move" | "up") => {
    const pad = padRef.current;
    if (!pad) return;
    const ctx = pad.getContext("2d");
    if (!ctx) return;
    const rect = pad.getBoundingClientRect();
    const x = ((e.clientX - rect.left) * pad.width) / rect.width;
    const y = ((e.clientY - rect.top) * pad.height) / rect.height;
    if (type === "down") {
      drawingRef.current = true;
      pad.setPointerCapture(e.pointerId);
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 10;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(x, y);
      startSensing();
    } else if (type === "move" && drawingRef.current) {
      ctx.lineTo(x, y);
      ctx.stroke();
    } else if (type === "up") {
      drawingRef.current = false;
      ctx.stroke();
      sense();
    }
  };

  const toggleRegion = (bit: number) => {
    setRegionMask((mask) => mask ^ (1 << bit));
  };

  return (
    <div className="not-prose">
      <div className="relative overflow-hidden border border-zinc-800 bg-[#07070c]">
        <canvas
          ref={canvasRef}
          className="block h-[480px] w-full touch-none cursor-grab active:cursor-grabbing sm:h-[620px]"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          aria-label="ARCANE fruitfly brain"
        />

        <div className="pointer-events-none absolute inset-x-0 top-0 bg-gradient-to-b from-black/80 via-black/35 to-transparent px-2.5 pb-10 pt-2.5">
          <div className="flex items-center justify-between gap-3 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-500">
            <p className="text-zinc-300">
              ARCANE fruitfly brain
              <span className="mx-1.5 text-zinc-700">·</span>
              {geometrySource === "real" ? (
                <span className="text-[#C785F2]" title="Traced skeletons from FlyWire FAFB v783">
                  v783 skeletons
                </span>
              ) : (
                <span
                  className="text-amber-300/80"
                  title="The v783 geometry blob could not be loaded; showing procedural stand-in arbors"
                >
                  {geometrySource === "loading" ? "loading geometry" : "schematic arbors"}
                </span>
              )}
            </p>
            <p className="whitespace-nowrap">
              {stats.firingRateHz > 0 || stats.activeCount > 0 ? (
                <>
                  <span className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse align-middle" />
                  firing
                  <span className="mx-1 text-zinc-700">·</span>
                  {stats.firingRateHz} spikes/s
                  <span className="mx-1 text-zinc-700">·</span>
                  {stats.activeCount} cells
                  <span className="mx-1 text-zinc-700">·</span>
                  h-gain {stats.meanGain}x
                </>
              ) : (
                <>
                  <span className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-zinc-600 align-middle" />
                  quiescent
                  <span className="mx-1 text-zinc-700">·</span>
                  0 spikes/s
                  <span className="mx-1 text-zinc-700">·</span>
                  draw to stimulate
                </>
              )}
              <span className="mx-1.5 text-zinc-700">·</span>
              <button
                type="button"
                onClick={() => setRegionMask(ALL_REGION_MASK)}
                className="pointer-events-auto text-zinc-400 hover:text-zinc-100"
              >
                All
              </button>
              <span className="mx-1 text-zinc-700">/</span>
              <button
                type="button"
                onClick={() => setRegionMask(0)}
                className="pointer-events-auto text-zinc-400 hover:text-zinc-100"
              >
                None
              </button>
            </p>
          </div>
          <div className="pointer-events-auto mt-2 flex flex-wrap gap-1">
            {BRAIN_REGIONS.map((region) => {
              const on = (regionMask & (1 << region.bit)) !== 0;
              return (
                <button
                  key={region.id}
                  type="button"
                  aria-pressed={on}
                  onClick={() => toggleRegion(region.bit)}
                  className={`border px-2 py-1 text-[10px] font-medium leading-none tracking-tight ${
                    on
                      ? "border-[#C785F2]/55 bg-[#C785F2]/15 text-[#ead2fb]"
                      : "border-white/10 bg-black/50 text-zinc-500 hover:border-zinc-500 hover:text-zinc-200"
                  }`}
                >
                  {region.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 via-black/45 to-transparent p-2.5">
          <div className="flex items-end gap-3">
            <div className="pointer-events-auto relative shrink-0">
              <canvas
                ref={padRef}
                width={PAD}
                height={PAD}
                className="h-[96px] w-[96px] cursor-crosshair bg-white sm:h-[112px] sm:w-[112px]"
                onPointerDown={(e) => padPointer(e, "down")}
                onPointerMove={(e) => padPointer(e, "move")}
                onPointerUp={(e) => padPointer(e, "up")}
                onPointerCancel={(e) => padPointer(e, "up")}
                aria-label="Draw a digit for ARCANE MNIST"
              />
              <button
                type="button"
                onClick={clearPad}
                className="absolute right-1 top-1 bg-black/75 px-1 py-px font-mono text-[8px] uppercase tracking-[0.14em] text-zinc-200 hover:bg-black"
              >
                Clear
              </button>
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-end gap-3">
                <p className="min-w-[0.7em] font-mono text-5xl leading-none tabular-nums text-[#C785F2] sm:text-6xl">
                  {prediction ? prediction.digit : <span className="text-zinc-700">—</span>}
                </p>
                <div className="pb-0.5 font-mono text-[9px] uppercase leading-tight tracking-[0.16em] text-zinc-500">
                  <p>class</p>
                  <p className="text-zinc-200">
                    {prediction ? `${Math.round(prediction.confidence * 100)}%` : "idle"}
                  </p>
                  {prediction &&
                    (() => {
                      const runner = prediction.probabilities
                        .map((p, d) => ({ d, p }))
                        .sort((a, b) => b.p - a.p)[1];
                      if (!runner || runner.p < 0.04) return null;
                      return (
                        <p className="normal-case tracking-normal text-zinc-400">
                          {runner.d} {Math.round(runner.p * 100)}%
                        </p>
                      );
                    })()}
                </div>
              </div>
              <div className="mt-2 flex h-7 items-end gap-px">
                {Array.from({ length: 10 }, (_, digit) => {
                  const p = prediction?.probabilities[digit] ?? 0;
                  return (
                    <div key={digit} className="flex h-full min-w-0 flex-1 flex-col justify-end" title={`${digit}: ${Math.round(p * 100)}%`}>
                      <div
                        className={`w-full ${digit === prediction?.digit ? "bg-[#C785F2]" : "bg-zinc-600"}`}
                        style={{ height: `${prediction ? Math.max(2, p * 100) : 3}%` }}
                      />
                    </div>
                  );
                })}
              </div>
              <div className="mt-1 flex justify-between font-mono text-[8px] uppercase tracking-[0.14em] text-zinc-600">
                <span>0</span>
                <span>
                  {stats.activeCount > 0
                    ? `${stats.activeCount} cells firing · orbit · zoom`
                    : "draw on whiteboard to fire neurons"}
                </span>
                <span>9</span>
              </div>
            </div>
          </div>
        </div>

        {(loading || geometrySource === "loading") && (
          <p className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-500">
            {loading
              ? "waking circuit…"
              : `loading v783 skeletons ${Math.round(geometryProgress * 100)}%`}
          </p>
        )}
        {error && (
          <p className="absolute bottom-28 left-2.5 right-2.5 font-mono text-[10px] text-amber-200/90">{error}</p>
        )}
      </div>
    </div>
  );
}
