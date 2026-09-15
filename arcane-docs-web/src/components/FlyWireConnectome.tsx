"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Camera, Geometry, Mesh, Program, Renderer, Transform } from "ogl";
import {
  ALL_REGION_MASK,
  BRAIN_REGIONS,
  buildBackgroundBrain,
  buildCircuitArbors,
  type CircuitNeuron,
} from "@/lib/drosophila-brain";
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

export function FlyWireConnectome() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const padRef = useRef<HTMLCanvasElement>(null);
  const circuitRef = useRef<Circuit | null>(null);
  const simulatorRef = useRef<FruitflyCircuitSimulator | null>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const yawRef = useRef(0.15);
  const pitchRef = useRef(-0.12);
  const zoomRef = useRef(8.2);
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

    const renderer = new Renderer({
      canvas,
      dpr: Math.min(2, window.devicePixelRatio || 1),
      antialias: true,
      alpha: false,
    });
    const gl = renderer.gl;
    gl.clearColor(0.025, 0.027, 0.04, 1);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.disable(gl.DEPTH_TEST);

    const camera = new Camera(gl, { fov: 28, near: 0.1, far: 40 });
    camera.position.set(0, 0.06, 8.2);
    camera.lookAt([0, 0.06, 0]);

    const scene = new Transform();
    scene.scale.set(0.86, 0.86, 0.86);

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
      cullFace: false,
    });
    const bgMesh = new Mesh(gl, { geometry: bgGeom, program: bgProgram, mode: gl.LINES });
    bgMesh.frustumCulled = false;
    bgMesh.setParent(scene);

    let circuitMesh: Mesh | null = null;
    let circuitGeom: Geometry | null = null;
    let circuitProgram: Program | null = null;
    let vertexNeuron: string[] = [];
    let vertexDistances: Float32Array = new Float32Array(0);
    let baseColors: Float32Array = new Float32Array(0);
    let spikeTarget: Float32Array | null = null;

    const rebuildCircuit = (data: Circuit) => {
      if (circuitMesh) circuitMesh.setParent(null);
      const built = buildCircuitArbors(data.neurons);
      vertexNeuron = built.vertexNeuron;
      vertexDistances = built.vertexDistances;
      baseColors = built.baseColors;
      spikeTarget = new Float32Array(built.cloud.colors.length);
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
        cullFace: false,
      });
      circuitMesh = new Mesh(gl, { geometry: circuitGeom, program: circuitProgram, mode: gl.LINES });
      circuitMesh.frustumCulled = false;
      circuitMesh.setParent(scene);
    };

    if (circuitRef.current) rebuildCircuit(circuitRef.current);

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const next = zoomRef.current * (e.deltaY > 0 ? 1.08 : 0.92);
      zoomRef.current = Math.max(3.4, Math.min(14.5, next));
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
      if (!draggingRef.current) yawRef.current += 0.0004;
      scene.rotation.y = yawRef.current;
      scene.rotation.x = pitchRef.current;
      camera.position.set(0, 0.06, zoomRef.current);
      camera.lookAt([0, 0.06, 0]);

      const data = circuitRef.current;
      if (data && !circuitMesh) rebuildCircuit(data);

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
        if (circuitGeom && spikeTarget && vertexDistances.length > 0) {
          sim.applyColors(spikeTarget, vertexNeuron, vertexDistances, baseColors, time);
          const attr = circuitGeom.attributes.color as { data: Float32Array; needsUpdate: boolean };
          attr.data.set(spikeTarget);
          attr.needsUpdate = true;
        }
        const s = sim.getStats();
        bgProgram.uniforms.uActivity.value = s.meanActivity;

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
      } else {
        bgProgram.uniforms.uActivity.value = 0.0;
      }

      bgProgram.uniforms.uTime.value = time;
      bgProgram.uniforms.uMask.value = maskRef.current;
      if (circuitProgram) circuitProgram.uniforms.uMask.value = maskRef.current;

      renderer.render({ scene, camera });
    };
    frame = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(frame);
      canvas.removeEventListener("wheel", onWheel);
    };
  }, [circuit]);

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    dragRef.current = { x: e.clientX, y: e.clientY };
    draggingRef.current = true;
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    yawRef.current += (e.clientX - dragRef.current.x) * 0.007;
    pitchRef.current += (e.clientY - dragRef.current.y) * 0.007;
    pitchRef.current = Math.max(-0.85, Math.min(0.65, pitchRef.current));
    dragRef.current = { x: e.clientX, y: e.clientY };
  };
  const onPointerUp = () => {
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
          className="block h-[420px] w-full touch-none cursor-grab sm:h-[560px]"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={onPointerUp}
          aria-label="ARCANE fruitfly brain"
        />

        <div className="pointer-events-none absolute inset-x-0 top-0 bg-gradient-to-b from-black/80 via-black/35 to-transparent px-2.5 pb-10 pt-2.5">
          <div className="flex items-center justify-between gap-3 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-500">
            <p className="text-zinc-300">ARCANE fruitfly brain</p>
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

        {loading && (
          <p className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-500">
            waking circuit…
          </p>
        )}
        {error && (
          <p className="absolute bottom-28 left-2.5 right-2.5 font-mono text-[10px] text-amber-200/90">{error}</p>
        )}
      </div>
    </div>
  );
}
