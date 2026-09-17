/**
 * Procedural stand-in geometry for the adult Drosophila CNS.
 *
 * Nothing in this file is measured data. The shapes are ellipsoid fields and
 * seeded random walks tuned to look like the right anatomy. Two things still
 * use it:
 *
 * 1. The ventral nerve cord fallback. The live viewer loads the Male CNS
 *    JRCFIB2022M_vnc mesh; these builders only run if that blob fails.
 * 2. The fallback arbors, used only when the real geometry blob fails to load.
 *
 * The real neuropil shell and the real v783 neuron skeletons live in
 * `flywire-geometry.ts`.
 */

export type Vec3 = { x: number; y: number; z: number };

export type CircuitNeuron = {
  id: string;
  label: string;
  cell_type: string;
  layer: string;
  side: string;
};

export type LineCloud = {
  positions: Float32Array;
  colors: Float32Array;
  regions: Float32Array;
  count: number;
};

export const BRAIN_REGIONS = [
  { bit: 0, id: "opticLeft", label: "Left optic lobe" },
  { bit: 1, id: "opticRight", label: "Right optic lobe" },
  { bit: 2, id: "central", label: "Central brain" },
  { bit: 3, id: "mushroom", label: "Mushroom body" },
  { bit: 4, id: "antennal", label: "Antennal lobe" },
  { bit: 5, id: "vnc", label: "Ventral nerve cord" },
  { bit: 6, id: "giant", label: "Giant Fiber" },
  { bit: 7, id: "descending", label: "Descending" },
] as const;

export const VNC_REGION = 5;

export const ALL_REGION_MASK = BRAIN_REGIONS.reduce((mask, region) => mask | (1 << region.bit), 0);

export type MeshCloud = {
  positions: Float32Array;
  normals: Float32Array;
  regions: Float32Array;
  count: number;
};

export type TubeCloud = {
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array;
  regions: Float32Array;
  distances: Float32Array;
  count: number;
};

type FlyLeg = {
  neuromere: "T1" | "T2" | "T3";
  side: 1 | -1;
  root: Vec3;
  coxa: Vec3;
  femur: Vec3;
  tibia: Vec3;
  tarsus: Vec3;
  neuropilR: number;
  femurR: number;
  tibiaR: number;
};

function makeLeg(
  neuromere: FlyLeg["neuromere"],
  side: 1 | -1,
  y: number,
  reach: number,
  back: number,
  drop: number,
  nR: number
): FlyLeg {
  const root = { x: side * 0.17, y, z: 0.016 };
  const coxa = { x: side * 0.4, y: y - back * 0.08, z: -0.05 };
  const femur = { x: side * (0.4 + reach * 0.48), y: y - back * 0.4, z: -0.12 - drop * 0.35 };
  const tibia = { x: side * (0.4 + reach * 0.8), y: y - back * 0.88, z: -0.08 - drop * 0.18 };
  const tarsus = { x: side * (0.4 + reach), y: y - back * 1.18, z: -0.14 - drop * 0.45 };
  return {
    neuromere,
    side,
    root,
    coxa,
    femur,
    tibia,
    tarsus,
    neuropilR: nR,
    femurR: nR * 0.34,
    tibiaR: nR * 0.24,
  };
}

/** Six thoracic legs: prothoracic (T1), mesothoracic (T2), metathoracic (T3). */
const FLY_LEGS: FlyLeg[] = [
  makeLeg("T1", -1, -1.1, 1.02, -0.05, 0.16, 0.125),
  makeLeg("T1", 1, -1.1, 1.02, -0.05, 0.16, 0.125),
  makeLeg("T2", -1, -1.48, 1.18, 0.26, 0.22, 0.145),
  makeLeg("T2", 1, -1.48, 1.18, 0.26, 0.22, 0.145),
  makeLeg("T3", -1, -1.86, 1.06, 0.4, 0.18, 0.125),
  makeLeg("T3", 1, -1.86, 1.06, 0.4, 0.18, 0.125),
];

function smoothMin(a: number, b: number, k: number) {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

function capsuleDist(p: Vec3, a: Vec3, b: Vec3) {
  const pax = p.x - a.x;
  const pay = p.y - a.y;
  const paz = p.z - a.z;
  const bax = b.x - a.x;
  const bay = b.y - a.y;
  const baz = b.z - a.z;
  const baba = bax * bax + bay * bay + baz * baz || 1;
  const h = Math.max(0, Math.min(1, (pax * bax + pay * bay + paz * baz) / baba));
  return Math.hypot(pax - bax * h, pay - bay * h, paz - baz * h);
}

function capsuleField(p: Vec3, a: Vec3, b: Vec3, r: number) {
  const n = noise(p) + noise({ x: p.x * 2.4, y: p.y * 2.1, z: p.z * 2.7 }) * 0.65;
  return capsuleDist(p, a, b) / (r * (1 + n * 0.85));
}

export type SomaMarker = {
  id: string;
  label: string;
  cell_type: string;
  layer: string;
  position: Vec3;
};

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function ellipsoid(p: Vec3, c: Vec3, r: Vec3) {
  const x = (p.x - c.x) / r.x;
  const y = (p.y - c.y) / r.y;
  const z = (p.z - c.z) / r.z;
  return x * x + y * y + z * z;
}

function noise(p: Vec3) {
  return (
    Math.sin(p.x * 6.7 + p.y * 3.1 + p.z * 2.4) +
    Math.sin(p.y * 5.9 + p.z * 4.2) +
    Math.sin(p.z * 7.3 + p.x * 2.8)
  ) * 0.045;
}

export function brainField(p: Vec3) {
  const ol = ellipsoid(p, { x: -1.38, y: 0.02, z: 0.02 }, { x: 0.6, y: 0.54, z: 0.52 });
  const or_ = ellipsoid(p, { x: 1.38, y: 0.02, z: 0.02 }, { x: 0.6, y: 0.54, z: 0.52 });
  const central = ellipsoid(p, { x: 0, y: 0.04, z: 0 }, { x: 0.9, y: 0.48, z: 0.38 });
  const hornL = ellipsoid(p, { x: -0.2, y: 0.58, z: -0.02 }, { x: 0.17, y: 0.26, z: 0.15 });
  const hornR = ellipsoid(p, { x: 0.2, y: 0.58, z: -0.02 }, { x: 0.17, y: 0.26, z: 0.15 });
  const alL = ellipsoid(p, { x: -0.34, y: -0.26, z: 0.1 }, { x: 0.22, y: 0.17, z: 0.16 });
  const alR = ellipsoid(p, { x: 0.34, y: -0.26, z: 0.1 }, { x: 0.22, y: 0.17, z: 0.16 });
  const mbL = ellipsoid(p, { x: -0.42, y: 0.08, z: 0.06 }, { x: 0.22, y: 0.16, z: 0.16 });
  const mbR = ellipsoid(p, { x: 0.42, y: 0.08, z: 0.06 }, { x: 0.22, y: 0.16, z: 0.16 });
  const hole = ellipsoid(p, { x: 0, y: -0.02, z: 0.02 }, { x: 0.13, y: 0.2, z: 0.15 });
  const d = Math.min(ol, or_, central, hornL, hornR, alL, alR, mbL, mbR) + noise(p);
  return { d, hole, ol, or_, central, hornL, hornR, alL, alR, mbL, mbR };
}

export function insideBrain(p: Vec3) {
  const f = brainField(p);
  return f.d < 1 && f.hole > 1;
}

/**
 * Adult VNC occupancy: a fused cervical–thoracic–abdominal cord plus six
 * thoracic leg neuropils. Capsules are blended with a smooth minimum so the
 * cord reads as one irregular ganglion, not a stack of ellipsoids.
 *
 * `includeLegs` adds the peripheral nerve trunks. The glass shell uses the
 * cord + neuropil lobes only — wrapping the whole leg in a capsule made the
 * limbs look like smooth sausages, unlike the head's filament arbors.
 */
function vncOccupancy(p: Vec3, includeLegs: boolean) {
  const neck = capsuleField(p, { x: 0, y: -0.46, z: 0.02 }, { x: 0.01, y: -0.98, z: 0.02 }, 0.088);
  const thorax = capsuleField(p, { x: -0.01, y: -0.88, z: 0.025 }, { x: 0.02, y: -2.12, z: -0.01 }, 0.155);
  const ridgeL = capsuleField(p, { x: -0.14, y: -1.04, z: 0.01 }, { x: -0.16, y: -1.94, z: -0.01 }, 0.1);
  const ridgeR = capsuleField(p, { x: 0.14, y: -1.04, z: 0.01 }, { x: 0.16, y: -1.94, z: -0.01 }, 0.1);
  const abd = capsuleField(p, { x: 0.02, y: -2.05, z: -0.02 }, { x: 0.04, y: -2.64, z: -0.05 }, 0.068);
  let d = smoothMin(neck, thorax, 0.18);
  d = smoothMin(d, ridgeL, 0.14);
  d = smoothMin(d, ridgeR, 0.14);
  d = smoothMin(d, abd, 0.16);
  let trunk = 99;
  if (includeLegs) {
    for (const L of FLY_LEGS) {
      const root = capsuleField(p, L.root, L.coxa, 0.055);
      const femur = capsuleField(p, L.coxa, L.femur, L.femurR * 0.7);
      const tibia = capsuleField(p, L.femur, L.tibia, L.tibiaR * 0.68);
      const tarsus = capsuleField(p, L.tibia, L.tarsus, L.tibiaR * 0.48);
      trunk = Math.min(trunk, root, femur, tibia, tarsus);
    }
    d = smoothMin(d, trunk, 0.08);
  }
  d += noise(p) * 0.24 + noise({ x: p.x * 2.1, y: p.y * 1.7, z: p.z * 2.4 }) * 0.13;
  d += noise({ x: p.x * 3.4, y: p.y * 2.8, z: p.z * 3.1 }) * 0.07;
  return { d, neck, thorax, abd, leg: trunk };
}

export function vncField(p: Vec3) {
  return vncOccupancy(p, true);
}

function vncShellField(p: Vec3) {
  return vncOccupancy(p, false);
}

export function insideVnc(p: Vec3) {
  return vncField(p).d < 1;
}

export function regionIdAt(p: Vec3): number {
  const f = brainField(p);
  if (f.ol < 1.05 && f.ol <= f.or_) return 0;
  if (f.or_ < 1.05) return 1;
  if (f.hornL < 1.05 || f.hornR < 1.05 || f.mbL < 1.05 || f.mbR < 1.05) return 3;
  if (f.alL < 1.05 || f.alR < 1.05) return 4;
  return 2;
}

export function circuitRegion(n: CircuitNeuron): number {
  if (n.cell_type === "DNp01") return 6;
  if (n.layer === "sensory") return n.side === "left" ? 0 : 1;
  return 7;
}

function vncColor(p: Vec3): [number, number, number] {
  const f = vncField(p);
  if (f.leg < 1.02 && f.leg <= f.thorax && f.leg <= f.neck) return [0.25, 0.78, 0.82];
  if (f.neck < 1.05 && f.neck <= f.thorax) return [0.82, 0.4, 0.95];
  if (f.abd < 1.05 && f.abd <= f.thorax) return [0.78, 0.5, 0.28];
  return [0.58, 0.34, 0.78];
}

function regionColor(p: Vec3): [number, number, number] {
  const f = brainField(p);
  if (f.ol < 1.05 || f.or_ < 1.05) {
    const optic = f.ol < f.or_ ? f.ol : f.or_;
    if (optic > 0.72) return [0.22, 0.74, 0.84];
    if (optic > 0.42) return [0.62, 0.28, 0.78];
    return [0.86, 0.22, 0.42];
  }
  if (f.hornL < 1 || f.hornR < 1) return [0.52, 0.28, 0.86];
  if (f.mbL < 1 || f.mbR < 1) return [0.45, 0.78, 0.38];
  if (f.alL < 1.05 || f.alR < 1.05) return [0.9, 0.72, 0.28];
  if (Math.abs(p.x) < 0.18 && p.y < 0) return [0.55, 0.82, 0.4];
  return [0.68, 0.28, 0.8];
}

function randDir(rng: () => number): Vec3 {
  const z = rng() * 2 - 1;
  const t = rng() * Math.PI * 2;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  return { x: r * Math.cos(t), y: r * Math.sin(t), z };
}

function add(a: Vec3, b: Vec3, s = 1): Vec3 {
  return { x: a.x + b.x * s, y: a.y + b.y * s, z: a.z + b.z * s };
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function normalize(v: Vec3): Vec3 {
  const l = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / l, y: v.y / l, z: v.z / l };
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x };
}

function catmullRom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: number): Vec3 {
  const t2 = t * t;
  const t3 = t2 * t;
  return {
    x: 0.5 * (2 * p1.x + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
    y: 0.5 * (2 * p1.y + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3),
    z: 0.5 * (2 * p1.z + (-p0.z + p2.z) * t + (2 * p0.z - 5 * p1.z + 4 * p2.z - p3.z) * t2 + (-p0.z + 3 * p1.z - 3 * p2.z + p3.z) * t3),
  };
}

/** Dense FlyWire-like neurite: spline through waypoints, then meander in the local tract. */
function meanderTract(waypoints: Vec3[], rng: () => number, samplesPerSeg = 16, wobble = 0.032): Vec3[] {
  if (waypoints.length < 2) return waypoints.slice();
  const padded = [waypoints[0], ...waypoints, waypoints[waypoints.length - 1]];
  const phase = rng() * Math.PI * 2;
  const freq = 1.6 + rng() * 2.4;
  const amp = wobble * (0.75 + rng() * 0.5);
  const out: Vec3[] = [];
  for (let i = 0; i < padded.length - 3; i++) {
    const p0 = padded[i];
    const p1 = padded[i + 1];
    const p2 = padded[i + 2];
    const p3 = padded[i + 3];
    const last = i === padded.length - 4;
    const steps = last ? samplesPerSeg : samplesPerSeg - 1;
    for (let s = 0; s <= steps; s++) {
      const t = s / samplesPerSeg;
      const p = catmullRom(p0, p1, p2, p3, t);
      const ahead = catmullRom(p0, p1, p2, p3, Math.min(1, t + 0.04));
      const tangent = normalize(sub(ahead, p));
      const helper = Math.abs(tangent.y) < 0.92 ? { x: 0, y: 1, z: 0 } : { x: 1, y: 0, z: 0 };
      const n1 = normalize(cross(tangent, helper));
      const n2 = normalize(cross(tangent, n1));
      const along = i + t;
      const w1 = Math.sin(along * freq + phase) * amp;
      const w2 = Math.sin(along * freq * 1.41 + phase * 1.9) * amp * 0.58;
      const w3 = Math.sin(along * 7.4 + phase * 0.6) * amp * 0.22;
      out.push({
        x: p.x + n1.x * (w1 + w3) + n2.x * w2,
        y: p.y + n1.y * (w1 + w3) + n2.y * w2,
        z: p.z + n1.z * (w1 + w3) + n2.z * w2,
      });
    }
  }
  return out;
}

function sampleInside(rng: () => number): Vec3 {
  for (let i = 0; i < 80; i++) {
    const p = { x: (rng() - 0.5) * 4.1, y: (rng() - 0.5) * 2.1, z: (rng() - 0.5) * 1.5 };
    if (insideBrain(p)) return p;
  }
  return { x: 0, y: 0, z: 0 };
}

function walkIn(
  start: Vec3,
  steps: number,
  step: number,
  rng: () => number,
  inside: (p: Vec3) => boolean,
  tangentAround?: Vec3
) {
  const pts: Vec3[] = [start];
  let dir = randDir(rng);
  let p = start;
  for (let i = 0; i < steps; i++) {
    if (tangentAround) {
      const radial = sub(p, tangentAround);
      dir = normalize(add(cross(radial, randDir(rng)), randDir(rng), 0.35));
    } else {
      dir = normalize(add(dir, randDir(rng), 0.55));
    }
    let next = add(p, dir, step);
    let ok = inside(next);
    if (!ok) {
      for (let k = 0; k < 6; k++) {
        dir = randDir(rng);
        next = add(p, dir, step);
        if (inside(next)) {
          ok = true;
          break;
        }
      }
    }
    if (!ok) break;
    p = next;
    pts.push(p);
  }
  return pts;
}

function walk(
  start: Vec3,
  steps: number,
  step: number,
  rng: () => number,
  tangentAround?: Vec3
) {
  return walkIn(start, steps, step, rng, insideBrain, tangentAround);
}

function sampleInsideVnc(rng: () => number): Vec3 {
  for (let i = 0; i < 80; i++) {
    const p = { x: (rng() - 0.5) * 3.2, y: -0.48 - rng() * 2.25, z: (rng() - 0.5) * 0.9 };
    if (insideVnc(p)) return p;
  }
  return { x: 0, y: -1.5, z: 0 };
}

function pushPolyline(
  pos: number[],
  col: number[],
  regions: number[],
  pts: Vec3[],
  rgb: [number, number, number],
  region: number,
  jitter = 0
) {
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    pos.push(a.x, a.y, a.z, b.x, b.y, b.z);
    const fade = 0.72 + jitter;
    col.push(rgb[0] * fade, rgb[1] * fade, rgb[2] * fade, rgb[0] * fade, rgb[1] * fade, rgb[2] * fade);
    regions.push(region, region);
  }
}

let backgroundCache: LineCloud | null = null;
let vncCache: LineCloud | null = null;
let vncMeshCache: MeshCloud | null = null;
let vncTubeCache: TubeCloud | null = null;

type VncFiber = { pts: Vec3[]; rgb: [number, number, number]; radius: number };

function muteRgb(rgb: [number, number, number], sat = 0.32, bright = 0.78): [number, number, number] {
  const grey = (rgb[0] + rgb[1] + rgb[2]) / 3;
  return [
    (rgb[0] * (1 - sat) + grey * sat) * bright,
    (rgb[1] * (1 - sat) + grey * sat) * bright,
    (rgb[2] * (1 - sat) + grey * sat) * bright * 1.08,
  ];
}

function jitter3(rng: () => number, s = 0.04): Vec3 {
  return { x: (rng() - 0.5) * s, y: (rng() - 0.5) * s, z: (rng() - 0.5) * s };
}

function sampleInsideCord(rng: () => number): Vec3 {
  for (let i = 0; i < 80; i++) {
    const p = { x: (rng() - 0.5) * 0.7, y: -0.7 - rng() * 1.85, z: (rng() - 0.5) * 0.32 };
    if (insideVnc(p) && Math.abs(p.x) < 0.42) return p;
  }
  return { x: 0, y: -1.4, z: 0 };
}

/**
 * Schematic VNC + leg neurons as irregular arbors, matching the head's
 * traced-looking filaments rather than filling neuromere ellipsoids.
 */
function buildVncFibers(): VncFiber[] {
  const rng = mulberry32(20260917);
  const fibers: VncFiber[] = [];
  const longRgb = muteRgb([0.72, 0.36, 0.9]);
  const localRgb = muteRgb([0.56, 0.3, 0.8]);
  const motorRgb = muteRgb([0.38, 0.86, 0.52], 0.18, 0.92);
  const senseRgb = muteRgb([0.22, 0.82, 0.92], 0.18, 0.92);
  const neckRgb = muteRgb([0.78, 0.4, 0.92]);

  for (let i = 0; i < 42; i++) {
    const side = i % 2 === 0 ? -1 : 1;
    const xOff = side * (0.035 + rng() * 0.07);
    const zOff = (rng() - 0.5) * 0.055;
    const y0 = -0.5 - rng() * 0.1;
    const y1 = -2.42 - rng() * 0.16;
    const waypoints: Vec3[] = [];
    const steps = 8 + Math.floor(rng() * 4);
    for (let k = 0; k <= steps; k++) {
      const t = k / steps;
      waypoints.push({
        x: xOff + Math.sin(t * 4.4 + i) * 0.03 + (rng() - 0.5) * 0.018,
        y: y0 + (y1 - y0) * t,
        z: zOff + Math.sin(t * 3.2 + i * 0.6) * 0.028,
      });
    }
    fibers.push({ pts: meanderTract(waypoints, rng, 8, 0.026), rgb: longRgb, radius: 0.0056 });
  }

  const stations = [-1.1, -1.3, -1.48, -1.68, -1.86, -2.12, -2.32];
  for (let i = 0; i < 24; i++) {
    const y = stations[i % stations.length] + (rng() - 0.5) * 0.05;
    const z = (rng() - 0.5) * 0.07;
    fibers.push({
      pts: meanderTract(
        [
          { x: -0.18 - rng() * 0.07, y: y + (rng() - 0.5) * 0.03, z },
          { x: 0, y: y + (rng() - 0.5) * 0.025, z: z + (rng() - 0.5) * 0.03 },
          { x: 0.18 + rng() * 0.07, y: y + (rng() - 0.5) * 0.03, z },
        ],
        rng,
        12,
        0.022
      ),
      rgb: localRgb,
      radius: 0.0044,
    });
  }

  for (let i = 0; i < 24; i++) {
    const soma = sampleInsideCord(rng);
    const branches = 3 + Math.floor(rng() * 2);
    for (let d = 0; d < branches; d++) {
      const pts = walkIn(add(soma, randDir(rng), 0.014), 8 + Math.floor(rng() * 6), 0.026, rng, insideVnc);
      if (pts.length > 2) fibers.push({ pts, rgb: localRgb, radius: 0.004 });
    }
  }

  for (let i = 0; i < 28; i++) {
    const side = i % 2 === 0 ? -1 : 1;
    const start = { x: side * 0.045, y: -0.47, z: (rng() - 0.5) * 0.05 };
    const pts = walkIn(start, 18 + Math.floor(rng() * 8), 0.034, rng, insideVnc);
    if (pts.length > 2) fibers.push({ pts, rgb: neckRgb, radius: 0.0052 });
  }

  for (const leg of FLY_LEGS) {
    for (let m = 0; m < 7; m++) {
      const soma = add(leg.root, randDir(rng), 0.035);
      for (let d = 0; d < 4; d++) {
        const pts = walkIn(add(soma, randDir(rng), 0.012), 8, 0.02, rng, insideVnc);
        if (pts.length > 2) fibers.push({ pts, rgb: motorRgb, radius: 0.0038 });
      }
      const axon = meanderTract(
        [
          soma,
          add(leg.coxa, jitter3(rng, 0.04)),
          add(leg.femur, jitter3(rng, 0.055)),
          add(leg.tibia, jitter3(rng, 0.05)),
          add(leg.tarsus, jitter3(rng, 0.04)),
        ],
        rng,
        11,
        0.034
      );
      fibers.push({ pts: axon, rgb: motorRgb, radius: 0.0064 });
      const femurAt = axon[Math.floor(axon.length * 0.38)];
      const tibiaAt = axon[Math.floor(axon.length * 0.68)];
      for (const joint of [femurAt, tibiaAt]) {
        if (!joint) continue;
        fibers.push({
          pts: meanderTract(
            [joint, add(joint, randDir(rng), 0.08 + rng() * 0.06), add(joint, randDir(rng), 0.14 + rng() * 0.08)],
            rng,
            7,
            0.022
          ),
          rgb: motorRgb,
          radius: 0.0038,
        });
      }
      const end = axon[axon.length - 1];
      for (let t = 0; t < 3; t++) {
        const tuft = meanderTract(
          [end, add(end, randDir(rng), 0.07 + rng() * 0.05), add(end, randDir(rng), 0.14 + rng() * 0.07)],
          rng,
          7,
          0.02
        );
        fibers.push({ pts: tuft, rgb: motorRgb, radius: 0.0036 });
      }
    }
    for (let s = 0; s < 6; s++) {
      const start = add(leg.tarsus, randDir(rng), 0.03);
      const axon = meanderTract(
        [
          start,
          add(leg.tibia, jitter3(rng, 0.04)),
          add(leg.femur, jitter3(rng, 0.05)),
          add(leg.coxa, jitter3(rng, 0.04)),
          add(leg.root, jitter3(rng, 0.035)),
        ],
        rng,
        11,
        0.032
      );
      fibers.push({ pts: axon, rgb: senseRgb, radius: 0.0054 });
      const term = axon[axon.length - 1];
      for (let t = 0; t < 3; t++) {
        const tuft = walkIn(add(term, randDir(rng), 0.01), 8, 0.018, rng, insideVnc);
        if (tuft.length > 2) fibers.push({ pts: tuft, rgb: senseRgb, radius: 0.0036 });
      }
    }
  }

  return fibers;
}

let vncFiberCache: VncFiber[] | null = null;
function getVncFibers(): VncFiber[] {
  if (!vncFiberCache) vncFiberCache = buildVncFibers();
  return vncFiberCache;
}

const TUBE_SIDES = 5;

function sweepTube(
  pts: Vec3[],
  radius: number,
  rgb: [number, number, number],
  pos: number[],
  nrm: number[],
  col: number[],
  regs: number[],
  dist: number[]
) {
  const path: Vec3[] = [pts[0]];
  for (let i = 1; i < pts.length; i++) {
    const prev = path[path.length - 1];
    if (Math.hypot(pts[i].x - prev.x, pts[i].y - prev.y, pts[i].z - prev.z) > 1e-5) path.push(pts[i]);
  }
  if (path.length < 2) return;

  const tangents: Vec3[] = [];
  for (let i = 0; i < path.length; i++) {
    const a = path[Math.max(0, i - 1)];
    const b = path[Math.min(path.length - 1, i + 1)];
    tangents.push(normalize(sub(b, a)));
  }
  const us: Vec3[] = [];
  const vs: Vec3[] = [];
  let u =
    Math.abs(tangents[0].z) < 0.9
      ? normalize(cross(tangents[0], { x: 0, y: 0, z: 1 }))
      : normalize(cross(tangents[0], { x: 1, y: 0, z: 0 }));
  for (let i = 0; i < path.length; i++) {
    const t = tangents[i];
    const d = u.x * t.x + u.y * t.y + u.z * t.z;
    u = { x: u.x - t.x * d, y: u.y - t.y * d, z: u.z - t.z * d };
    const n = Math.hypot(u.x, u.y, u.z);
    if (n < 1e-6) {
      u =
        Math.abs(t.z) < 0.9
          ? normalize(cross(t, { x: 0, y: 0, z: 1 }))
          : normalize(cross(t, { x: 1, y: 0, z: 0 }));
    } else {
      u = { x: u.x / n, y: u.y / n, z: u.z / n };
    }
    us.push(u);
    vs.push(cross(t, u));
  }

  const rings: Vec3[][] = [];
  const radials: Vec3[][] = [];
  const dists = [0];
  for (let i = 1; i < path.length; i++) {
    dists.push(
      dists[i - 1] + Math.hypot(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y, path[i].z - path[i - 1].z)
    );
  }
  for (let i = 0; i < path.length; i++) {
    const taper = radius * (0.84 + 0.16 * (1 - i / (path.length - 1)));
    const ring: Vec3[] = [];
    const rad: Vec3[] = [];
    for (let s = 0; s < TUBE_SIDES; s++) {
      const a = (s / TUBE_SIDES) * Math.PI * 2;
      const c = Math.cos(a);
      const si = Math.sin(a);
      const nx = us[i].x * c + vs[i].x * si;
      const ny = us[i].y * c + vs[i].y * si;
      const nz = us[i].z * c + vs[i].z * si;
      rad.push({ x: nx, y: ny, z: nz });
      ring.push({ x: path[i].x + nx * taper, y: path[i].y + ny * taper, z: path[i].z + nz * taper });
    }
    rings.push(ring);
    radials.push(rad);
  }

  for (let i = 0; i < path.length - 1; i++) {
    for (let s = 0; s < TUBE_SIDES; s++) {
      const j = (s + 1) % TUBE_SIDES;
      const verts = [rings[i][s], rings[i + 1][s], rings[i + 1][j], rings[i][s], rings[i + 1][j], rings[i][j]];
      const norms = [radials[i][s], radials[i + 1][s], radials[i + 1][j], radials[i][s], radials[i + 1][j], radials[i][j]];
      const ds = [dists[i], dists[i + 1], dists[i + 1], dists[i], dists[i + 1], dists[i]];
      for (let k = 0; k < 6; k++) {
        pos.push(verts[k].x, verts[k].y, verts[k].z);
        nrm.push(norms[k].x, norms[k].y, norms[k].z);
        col.push(rgb[0], rgb[1], rgb[2]);
        regs.push(VNC_REGION);
        dist.push(ds[k]);
      }
    }
  }
}

const TET_CORNERS: [number, number, number][] = [
  [0, 0, 0],
  [1, 0, 0],
  [1, 1, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [1, 1, 1],
  [0, 1, 1],
];
const TETS: [number, number, number, number][] = [
  [0, 1, 2, 6],
  [0, 2, 3, 6],
  [0, 3, 7, 6],
  [0, 7, 4, 6],
  [0, 4, 5, 6],
  [0, 5, 1, 6],
];

function interpEdge(a: Vec3, da: number, b: Vec3, db: number, iso: number): Vec3 {
  const t = Math.max(0, Math.min(1, (iso - da) / (db - da || 1e-9)));
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t };
}

function interpGrad(a: Vec3, da: number, b: Vec3, db: number, iso: number): Vec3 {
  const t = Math.max(0, Math.min(1, (iso - da) / (db - da || 1e-9)));
  return normalize({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t });
}

/**
 * Glass neuropil shell for the schematic VNC, including the six leg trunks.
 * Marching tetrahedra over the occupancy field, so the surface is irregular
 * like the FLYWIRE brain mesh rather than a stack of ellipsoids.
 */
export function buildVncNeuropilMesh(): MeshCloud {
  if (vncMeshCache) return vncMeshCache;
  const min = { x: -1.72, y: -2.82, z: -0.62 };
  const max = { x: 1.72, y: -0.38, z: 0.3 };
  const nx = 46;
  const ny = 66;
  const nz = 26;
  const dx = (max.x - min.x) / nx;
  const dy = (max.y - min.y) / ny;
  const dz = (max.z - min.z) / nz;
  const sx = nx + 1;
  const sy = ny + 1;
  const n = sx * sy * (nz + 1);
  const values = new Float32Array(n);
  const at = (ix: number, iy: number, iz: number) => values[ix + iy * sx + iz * sx * sy];
  for (let iz = 0; iz <= nz; iz++) {
    for (let iy = 0; iy <= ny; iy++) {
      for (let ix = 0; ix <= nx; ix++) {
        values[ix + iy * sx + iz * sx * sy] = vncShellField({
          x: min.x + ix * dx,
          y: min.y + iy * dy,
          z: min.z + iz * dz,
        }).d;
      }
    }
  }
  const gradAt = (ix: number, iy: number, iz: number): Vec3 => {
    const ip = Math.min(nx, ix + 1);
    const im = Math.max(0, ix - 1);
    const jp = Math.min(ny, iy + 1);
    const jm = Math.max(0, iy - 1);
    const kp = Math.min(nz, iz + 1);
    const km = Math.max(0, iz - 1);
    return normalize({
      x: at(ip, iy, iz) - at(im, iy, iz),
      y: at(ix, jp, iz) - at(ix, jm, iz),
      z: at(ix, iy, kp) - at(ix, iy, km),
    });
  };

  const pos: number[] = [];
  const nrm: number[] = [];
  const regions: number[] = [];
  const iso = 1;
  const corners: Vec3[] = new Array(8);
  const grads: Vec3[] = new Array(8);
  const vals = new Float32Array(8);

  const pushTri = (p0: Vec3, n0: Vec3, p1: Vec3, n1: Vec3, p2: Vec3, n2: Vec3) => {
    pos.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
    nrm.push(n0.x, n0.y, n0.z, n1.x, n1.y, n1.z, n2.x, n2.y, n2.z);
    regions.push(VNC_REGION, VNC_REGION, VNC_REGION);
  };

  for (let iz = 0; iz < nz; iz++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        let inside = 0;
        for (let c = 0; c < 8; c++) {
          const [cx, cy, cz] = TET_CORNERS[c];
          const vx = ix + cx;
          const vy = iy + cy;
          const vz = iz + cz;
          vals[c] = at(vx, vy, vz);
          if (vals[c] < iso) inside++;
          corners[c] = { x: min.x + vx * dx, y: min.y + vy * dy, z: min.z + vz * dz };
          grads[c] = gradAt(vx, vy, vz);
        }
        if (inside === 0 || inside === 8) continue;
        for (const tet of TETS) {
          const tv = [vals[tet[0]], vals[tet[1]], vals[tet[2]], vals[tet[3]]];
          let mask = 0;
          for (let i = 0; i < 4; i++) if (tv[i] < iso) mask |= 1 << i;
          if (mask === 0 || mask === 15) continue;
          const pc = [corners[tet[0]], corners[tet[1]], corners[tet[2]], corners[tet[3]]];
          const pg = [grads[tet[0]], grads[tet[1]], grads[tet[2]], grads[tet[3]]];
          const edge = (i: number, j: number) => ({
            p: interpEdge(pc[i], tv[i], pc[j], tv[j], iso),
            n: interpGrad(pg[i], tv[i], pg[j], tv[j], iso),
          });
          const insideIdx = [0, 1, 2, 3].filter((i) => (mask & (1 << i)) !== 0);
          if (insideIdx.length === 1 || insideIdx.length === 3) {
            const i = insideIdx.length === 1 ? insideIdx[0] : [0, 1, 2, 3].find((j) => (mask & (1 << j)) === 0)!;
            const o = [0, 1, 2, 3].filter((j) => j !== i);
            const a = edge(i, o[0]);
            const b = edge(i, o[1]);
            const c = edge(i, o[2]);
            if (insideIdx.length === 1) pushTri(a.p, a.n, b.p, b.n, c.p, c.n);
            else pushTri(a.p, a.n, c.p, c.n, b.p, b.n);
          } else {
            const i0 = insideIdx[0];
            const i1 = insideIdx[1];
            const o = [0, 1, 2, 3].filter((j) => j !== i0 && j !== i1);
            const a = edge(i0, o[0]);
            const b = edge(i0, o[1]);
            const c = edge(i1, o[1]);
            const d = edge(i1, o[0]);
            pushTri(a.p, a.n, b.p, b.n, c.p, c.n);
            pushTri(a.p, a.n, c.p, c.n, d.p, d.n);
          }
        }
      }
    }
  }

  vncMeshCache = {
    positions: new Float32Array(pos),
    normals: new Float32Array(nrm),
    regions: new Float32Array(regions),
    count: pos.length / 3,
  };
  return vncMeshCache;
}

/** Tube meshes swept along the schematic VNC / leg arbors. */
export function buildVncNeuronMesh(): TubeCloud {
  if (vncTubeCache) return vncTubeCache;
  const pos: number[] = [];
  const nrm: number[] = [];
  const col: number[] = [];
  const regs: number[] = [];
  const dist: number[] = [];
  for (const fiber of getVncFibers()) {
    sweepTube(fiber.pts, fiber.radius, fiber.rgb, pos, nrm, col, regs, dist);
  }
  vncTubeCache = {
    positions: new Float32Array(pos),
    normals: new Float32Array(nrm),
    colors: new Float32Array(col),
    regions: new Float32Array(regs),
    distances: new Float32Array(dist),
    count: pos.length / 3,
  };
  return vncTubeCache;
}

/**
 * Line-fiber fallback for the VNC (used when the procedural brain cloud is
 * drawn instead of the real neuropil). Same arbors as the tube mesh.
 */
export function buildVncScaffold(_fiberCount = 1100): LineCloud {
  if (vncCache) return vncCache;
  const pos: number[] = [];
  const col: number[] = [];
  const regions: number[] = [];
  for (const fiber of getVncFibers()) {
    pushPolyline(pos, col, regions, fiber.pts, fiber.rgb, VNC_REGION, 0.06);
  }
  vncCache = {
    positions: new Float32Array(pos),
    colors: new Float32Array(col),
    regions: new Float32Array(regions),
    count: pos.length / 3,
  };
  return vncCache;
}

function sampleOpticShell(rng: () => number, side: number): Vec3 {
  const c = { x: side * 1.38, y: 0.02, z: 0.02 };
  for (let i = 0; i < 40; i++) {
    const dir = randDir(rng);
    const r = 0.78 + rng() * 0.18;
    const p = { x: c.x + dir.x * 0.58 * r, y: c.y + dir.y * 0.52 * r, z: c.z + dir.z * 0.5 * r };
    if (insideBrain(p)) return p;
  }
  return { x: side * 1.38, y: 0.02, z: 0.02 };
}

export function buildBackgroundBrain(fiberCount = 11000): LineCloud {
  if (backgroundCache?.regions) return backgroundCache;
  const rng = mulberry32(20240915);
  const pos: number[] = [];
  const col: number[] = [];
  const regions: number[] = [];

  for (let i = 0; i < 2400; i++) {
    const side = i % 2 === 0 ? -1 : 1;
    const p = sampleOpticShell(rng, side);
    const center = { x: side * 1.38, y: 0.02, z: 0.02 };
    const pts = walk(p, 10 + Math.floor(rng() * 8), 0.032, rng, center);
    pushPolyline(pos, col, regions, pts, regionColor(p), regionIdAt(p), rng() * 0.12);
  }

  for (let i = 0; i < fiberCount; i++) {
    const p = sampleInside(rng);
    const f = brainField(p);
    const optic = f.ol < 1.02 || f.or_ < 1.02;
    const center = optic ? (f.ol < f.or_ ? { x: -1.38, y: 0.02, z: 0.02 } : { x: 1.38, y: 0.02, z: 0.02 }) : undefined;
    const shell = optic && rng() < 0.45;
    const pts = walk(p, 7 + Math.floor(rng() * 10), shell ? 0.034 : 0.028, rng, shell ? center : undefined);
    const rgb = regionColor(p);
    pushPolyline(pos, col, regions, pts, rgb, regionIdAt(p), rng() * 0.18);
  }

  for (let i = 0; i < 280; i++) {
    const left = { x: -1.05, y: (rng() - 0.5) * 0.35, z: (rng() - 0.5) * 0.2 };
    const pts = walk(left, 28, 0.042, rng);
    pushPolyline(pos, col, regions, pts, regionColor(left), regionIdAt(left), 0.1);
  }

  for (const fiber of getVncFibers()) {
    pushPolyline(pos, col, regions, fiber.pts, fiber.rgb, VNC_REGION, 0.08);
  }

  backgroundCache = {
    positions: new Float32Array(pos),
    colors: new Float32Array(col),
    regions: new Float32Array(regions),
    count: pos.length / 3,
  };
  return backgroundCache;
}

function originFor(n: CircuitNeuron, index: number, ofType: number): Vec3 {
  const side = n.side === "left" ? -1 : 1;
  const fan = (index - (ofType - 1) / 2) * 0.07;
  switch (n.cell_type) {
    case "LC4":
      return { x: side * (1.32 + fan * 0.15), y: 0.06 + fan, z: 0.1 + fan * 0.3 };
    case "LPLC2":
      return { x: side * (1.18 + fan * 0.12), y: -0.04 + fan * 0.4, z: -0.08 };
    case "DNp01":
      return { x: side * 0.1, y: 0.06, z: 0.02 };
    case "DNp09":
      return { x: side * 0.22, y: -0.12, z: 0.08 };
    case "DNa01":
      return { x: side * 0.38, y: -0.08, z: -0.06 };
    case "DNa02":
      return { x: side * 0.48, y: 0.02, z: -0.1 };
    case "MDN":
      return { x: side * 0.12, y: -0.22, z: 0.12 };
    default:
      return { x: side * 0.3, y: 0, z: 0 };
  }
}

function circuitRgb(n: CircuitNeuron): [number, number, number] {
  if (n.cell_type === "DNp01") return [0.95, 0.82, 1];
  if (n.layer === "sensory") return [0.2, 0.8, 0.9];
  return [0.82, 0.38, 0.95];
}

function tagArbor(
  vertexNeuron: string[],
  vertexDistances: number[],
  pos: number[],
  before: number,
  after: number,
  id: string,
  origin: Vec3
) {
  for (let v = before; v < after; v++) {
    vertexNeuron.push(id);
    const dx = pos[v * 3] - origin.x;
    const dy = pos[v * 3 + 1] - origin.y;
    const dz = pos[v * 3 + 2] - origin.z;
    vertexDistances.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
  }
}

function sensoryProjectionAxon(n: CircuitNeuron, origin: Vec3, rng: () => number): Vec3[] {
  const side = n.side === "left" ? -1 : 1;
  const lplc = n.cell_type === "LPLC2";
  const waypoints: Vec3[] = [
    origin,
    {
      x: side * (1.14 + rng() * 0.05),
      y: origin.y * 0.35 - 0.08 - rng() * 0.05,
      z: 0.16 + origin.z * 0.25 + rng() * 0.05,
    },
    {
      x: side * (0.86 + rng() * 0.06),
      y: -0.12 - rng() * 0.05 - (lplc ? 0.05 : 0),
      z: 0.18 + rng() * 0.04,
    },
    {
      x: side * (0.58 + rng() * 0.05),
      y: -0.04 + rng() * 0.06,
      z: 0.1 + rng() * 0.03,
    },
    {
      x: side * (0.3 + rng() * 0.04),
      y: 0.04 + rng() * 0.04,
      z: 0.05,
    },
    {
      x: side * (0.08 + rng() * 0.03),
      y: 0.07 + (lplc ? -0.03 : 0.02),
      z: 0.015,
    },
  ];
  return meanderTract(waypoints, rng, 18, 0.036);
}

function descendingVncAxon(n: CircuitNeuron, origin: Vec3, rng: () => number): Vec3[] {
  const side = n.side === "left" ? -1 : 1;
  const gf = n.cell_type === "DNp01";
  const mdn = n.cell_type === "MDN";
  const sway = (0.06 + rng() * 0.05) * (rng() < 0.5 ? -1 : 1);
  const waypoints: Vec3[] = gf
    ? [
        origin,
        { x: origin.x * 0.7 + side * 0.08, y: -0.12, z: 0.1 },
        { x: origin.x * 0.25 - side * 0.06, y: -0.38, z: 0.02 },
        { x: side * 0.05, y: -0.62, z: 0.07 },
        { x: -side * 0.04, y: -0.88, z: 0.01 },
        { x: side * 0.03, y: -1.18, z: 0.08 },
        { x: -side * 0.02, y: -1.48, z: 0.04 },
        { x: side * 0.05, y: -1.78, z: 0.09 },
      ]
    : mdn
      ? [
          origin,
          { x: origin.x * 0.6 + side * 0.1, y: -0.16, z: 0.12 },
          { x: side * 0.14, y: -0.42, z: 0.02 },
          { x: -side * 0.04, y: -0.7, z: 0.07 },
          { x: side * 0.1, y: -1.02, z: 0.0 },
          { x: side * 0.22, y: -1.36, z: 0.05 },
          { x: side * 0.08, y: -1.68, z: -0.02 },
          { x: side * 0.12, y: -1.95, z: 0.02 },
        ]
      : [
          origin,
          { x: origin.x * 0.62 + side * 0.08, y: -0.12, z: origin.z * 0.5 + 0.08 },
          { x: origin.x * 0.28 + sway, y: -0.36, z: -0.02 },
          { x: side * 0.1 - sway, y: -0.64, z: 0.08 },
          { x: -side * 0.05 + sway * 0.5, y: -0.92, z: 0.0 },
          { x: side * (0.16 + rng() * 0.1), y: -1.22, z: 0.06 },
          { x: side * (0.28 + rng() * 0.1), y: -1.5 - rng() * 0.12, z: -0.02 },
          { x: side * (0.14 + rng() * 0.08), y: -1.78 - rng() * 0.1, z: 0.03 },
        ];
  return meanderTract(waypoints, rng, 22, gf ? 0.028 : 0.042);
}

export function buildCircuitArbors(neurons: CircuitNeuron[]): {
  cloud: LineCloud;
  somas: SomaMarker[];
  vertexNeuron: string[];
  vertexDistances: Float32Array;
  baseColors: Float32Array;
} {
  const rng = mulberry32(77);
  const pos: number[] = [];
  const col: number[] = [];
  const regions: number[] = [];
  const vertexNeuron: string[] = [];
  const vertexDistances: number[] = [];
  const somas: SomaMarker[] = [];
  const byType: Record<string, CircuitNeuron[]> = {};
  neurons.forEach((n) => {
    (byType[n.cell_type] ??= []).push(n);
  });

  neurons.forEach((n) => {
    const group = byType[n.cell_type];
    const origin = originFor(n, group.indexOf(n), group.length);
    somas.push({ id: n.id, label: n.label, cell_type: n.cell_type, layer: n.layer, position: origin });
    const rgb = circuitRgb(n);
    const region = circuitRegion(n);
    const gf = n.cell_type === "DNp01";
    const sensory = n.layer === "sensory";
    const dendrites = gf ? 14 : sensory ? 10 : 7;
    for (let d = 0; d < dendrites; d++) {
      const start = add(origin, randDir(rng), sensory ? 0.04 : 0.02);
      const center = sensory
        ? { x: Math.sign(origin.x) * 1.38, y: 0.02, z: 0.02 }
        : undefined;
      const pts = walk(start, gf ? 16 : sensory ? 12 : 8, 0.03, rng, center);
      const before = pos.length / 3;
      pushPolyline(pos, col, regions, pts, rgb, region, 0.08);
      const after = pos.length / 3;
      for (let v = before; v < after; v++) {
        vertexNeuron.push(n.id);
        const dx = pos[v * 3] - origin.x;
        const dy = pos[v * 3 + 1] - origin.y;
        const dz = pos[v * 3 + 2] - origin.z;
        vertexDistances.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
      }
    }
    if (sensory) {
      const axon = sensoryProjectionAxon(n, origin, rng);
      const before = pos.length / 3;
      pushPolyline(pos, col, regions, axon, rgb, region, 0.05);
      tagArbor(vertexNeuron, vertexDistances, pos, before, pos.length / 3, n.id, origin);
      const terminal = axon[axon.length - 1];
      for (let t = 0; t < 5; t++) {
        const tuft = walk(add(terminal, randDir(rng), 0.012), 8, 0.018, rng);
        const b = pos.length / 3;
        pushPolyline(pos, col, regions, tuft, rgb, region, 0.06);
        tagArbor(vertexNeuron, vertexDistances, pos, b, pos.length / 3, n.id, origin);
      }
    }
    if (!sensory) {
      const axon = descendingVncAxon(n, origin, rng);
      const before = pos.length / 3;
      pushPolyline(pos, col, regions, axon, rgb, VNC_REGION, 0.04);
      tagArbor(vertexNeuron, vertexDistances, pos, before, pos.length / 3, n.id, origin);
      const branchAt = axon[Math.floor(axon.length * (0.45 + rng() * 0.2))];
      const collateral = meanderTract(
        [
          branchAt,
          { x: branchAt.x + (rng() - 0.5) * 0.18, y: branchAt.y - 0.22, z: branchAt.z + (rng() - 0.5) * 0.1 },
          { x: branchAt.x + (rng() - 0.5) * 0.28, y: branchAt.y - 0.45, z: branchAt.z * 0.4 },
        ],
        rng,
        12,
        0.03,
      );
      const cb = pos.length / 3;
      pushPolyline(pos, col, regions, collateral, rgb, VNC_REGION, 0.05);
      tagArbor(vertexNeuron, vertexDistances, pos, cb, pos.length / 3, n.id, origin);
      const terminal = axon[axon.length - 1];
      const tufts = n.cell_type === "DNp01" ? 6 : 5;
      for (let t = 0; t < tufts; t++) {
        const tuft = walkIn(add(terminal, randDir(rng), 0.015), 8, 0.02, rng, insideVnc);
        const b = pos.length / 3;
        pushPolyline(pos, col, regions, tuft, rgb, VNC_REGION, 0.05);
        tagArbor(vertexNeuron, vertexDistances, pos, b, pos.length / 3, n.id, origin);
      }
    }
  });

  return {
    cloud: {
      positions: new Float32Array(pos),
      colors: new Float32Array(col),
      regions: new Float32Array(regions),
      count: pos.length / 3,
    },
    somas,
    vertexNeuron,
    vertexDistances: new Float32Array(vertexDistances),
    baseColors: new Float32Array(col),
  };
}

export function seedFiring(neurons: CircuitNeuron[]) {
  const out: Record<string, number> = {};
  for (const n of neurons) {
    out[n.id] = 0.0; // Rest baseline: 0 firing until stimulated by drawing
  }
  return out;
}

export function activityFromMnist(
  neurons: CircuitNeuron[],
  probs: number[],
  leftMean: number,
  rightMean: number
) {
  const digit = probs.reduce((best, p, i) => (p > probs[best] ? i : best), 0);
  const out: Record<string, number> = {};
  for (const n of neurons) {
    const visual = n.side === "left" ? leftMean : rightMean;
    if (n.layer === "sensory") out[n.id] = visual > 0.05 ? visual : 0;
    else if (n.cell_type === "DNp01") out[n.id] = probs[digit] > 0.3 ? probs[digit] : 0;
    else out[n.id] = probs[digit] > 0.4 ? probs[digit] * 0.7 : 0;
  }
  return out;
}

export function applySpikeColors(
  target: Float32Array,
  vertexNeuron: string[],
  activity: Record<string, number>,
  neurons: CircuitNeuron[],
  _time?: number
) {
  const byId = Object.fromEntries(neurons.map((n) => [n.id, n]));
  for (let i = 0; i < vertexNeuron.length; i++) {
    const id = vertexNeuron[i];
    const n = byId[id];
    const act = Math.max(0, Math.min(1, Math.abs(activity[id] ?? 0)));
    // Real biophysical glow proportional to genuine activity; resting baseline 0.11 when act == 0
    const glow = 0.11 + act * 1.6;
    const rgb = n ? circuitRgb(n) : ([0.85, 0.85, 1] as [number, number, number]);
    const o = i * 3;
    target[o] = Math.min(1, rgb[0] * glow);
    target[o + 1] = Math.min(1, rgb[1] * glow);
    target[o + 2] = Math.min(1, rgb[2] * glow);
  }
}
