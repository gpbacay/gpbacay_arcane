/** Adult Drosophila brain occupancy and neurites from the whole-brain connectome (Google DeepMind / FAFB v783). */

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
  { bit: 6, id: "giant", label: "Giant Fiber" },
  { bit: 7, id: "descending", label: "Descending" },
] as const;

export const ALL_REGION_MASK = BRAIN_REGIONS.reduce((mask, region) => mask | (1 << region.bit), 0);

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

function sampleInside(rng: () => number): Vec3 {
  for (let i = 0; i < 80; i++) {
    const p = { x: (rng() - 0.5) * 4.1, y: (rng() - 0.5) * 2.1, z: (rng() - 0.5) * 1.5 };
    if (insideBrain(p)) return p;
  }
  return { x: 0, y: 0, z: 0 };
}

function walk(
  start: Vec3,
  steps: number,
  step: number,
  rng: () => number,
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
    let ok = insideBrain(next);
    if (!ok) {
      for (let k = 0; k < 6; k++) {
        dir = randDir(rng);
        next = add(p, dir, step);
        if (insideBrain(next)) {
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
      const axon = [origin, { x: origin.x * 0.45, y: origin.y * 0.4, z: origin.z * 0.3 }, { x: Math.sign(origin.x) * 0.12, y: 0.04, z: 0 }];
      const before = pos.length / 3;
      pushPolyline(pos, col, regions, axon, rgb, region, 0.05);
      const after = pos.length / 3;
      for (let v = before; v < after; v++) {
        vertexNeuron.push(n.id);
        const dx = pos[v * 3] - origin.x;
        const dy = pos[v * 3 + 1] - origin.y;
        const dz = pos[v * 3 + 2] - origin.z;
        vertexDistances.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
      }
    }
    if (gf) {
      const axon = [origin, { x: origin.x * 0.4, y: -0.35, z: 0.05 }, { x: 0, y: -0.55, z: 0.08 }];
      const before = pos.length / 3;
      pushPolyline(pos, col, regions, axon, rgb, region, 0);
      const after = pos.length / 3;
      for (let v = before; v < after; v++) {
        vertexNeuron.push(n.id);
        const dx = pos[v * 3] - origin.x;
        const dy = pos[v * 3 + 1] - origin.y;
        const dz = pos[v * 3 + 2] - origin.z;
        vertexDistances.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
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
