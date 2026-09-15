function oval(cx: number, cy: number, rx: number, ry: number, n = 20): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const t = (i / n) * Math.PI * 2;
    pts.push([cx + Math.cos(t) * rx, cy + Math.sin(t) * ry]);
  }
  return pts;
}

export const DIGIT_PATHS: [number, number][][][] = [
  [oval(0.5, 0.5, 0.28, 0.34)],
  [[[0.42, 0.24], [0.52, 0.2], [0.52, 0.8]]],
  [[[0.3, 0.28], [0.62, 0.2], [0.68, 0.38], [0.38, 0.52], [0.3, 0.8], [0.7, 0.8]]],
  [
    [[0.3, 0.22], [0.64, 0.18], [0.72, 0.32], [0.48, 0.46]],
    [[0.48, 0.46], [0.74, 0.58], [0.64, 0.8], [0.28, 0.82]],
  ],
  [[[0.62, 0.8], [0.62, 0.2], [0.3, 0.58], [0.74, 0.58]]],
  [[[0.68, 0.22], [0.32, 0.22], [0.32, 0.48], [0.64, 0.48], [0.68, 0.72], [0.3, 0.8]]],
  [oval(0.52, 0.62, 0.22, 0.2), [[0.38, 0.22], [0.3, 0.62]]],
  [[[0.28, 0.22], [0.72, 0.22], [0.4, 0.8]]],
  [oval(0.5, 0.3, 0.2, 0.18), oval(0.5, 0.7, 0.2, 0.18)],
  [oval(0.5, 0.34, 0.22, 0.2), [[0.7, 0.4], [0.58, 0.84]]],
];

/** Extra handwritten styles people actually draw on the pad. */
const DIGIT_VARIANTS: [number, number][][][][] = [
  [
    [[[0.38, 0.18], [0.62, 0.18], [0.74, 0.5], [0.62, 0.82], [0.38, 0.82], [0.26, 0.5], [0.38, 0.18]]],
    [[[0.3, 0.32], [0.5, 0.16], [0.7, 0.32], [0.7, 0.68], [0.5, 0.84], [0.3, 0.68], [0.3, 0.32]]],
  ],
  [
    [[[0.5, 0.16], [0.5, 0.84]]],
    [[[0.34, 0.28], [0.5, 0.16], [0.5, 0.84], [0.32, 0.84], [0.68, 0.84]]],
  ],
  [
    [[[0.28, 0.32], [0.5, 0.16], [0.72, 0.3], [0.28, 0.78], [0.74, 0.78]]],
    [[[0.26, 0.22], [0.7, 0.22], [0.7, 0.48], [0.3, 0.78], [0.74, 0.82]]],
  ],
  [
    [[[0.28, 0.2], [0.7, 0.22], [0.42, 0.48], [0.7, 0.58], [0.28, 0.82]]],
    [[[0.32, 0.18], [0.66, 0.28], [0.36, 0.5], [0.66, 0.72], [0.3, 0.82]]],
  ],
  [
    [[[0.28, 0.62], [0.7, 0.62], [0.58, 0.18], [0.58, 0.84]]],
    [[[0.3, 0.22], [0.3, 0.55], [0.72, 0.55], [0.72, 0.22], [0.72, 0.84]]],
    [[[0.68, 0.16], [0.32, 0.62], [0.74, 0.62], [0.68, 0.16], [0.68, 0.84]]],
  ],
  [
    [[[0.7, 0.18], [0.3, 0.18], [0.3, 0.48], [0.66, 0.48], [0.7, 0.78], [0.3, 0.82]]],
    [[[0.68, 0.2], [0.32, 0.22], [0.34, 0.5], [0.62, 0.58], [0.34, 0.82]]],
  ],
  [
    [[[0.62, 0.2], [0.32, 0.52], [0.32, 0.78], [0.62, 0.78], [0.66, 0.52], [0.34, 0.52]]],
    [[[0.5, 0.16], [0.28, 0.5], [0.5, 0.84], [0.72, 0.5], [0.5, 0.16]]],
  ],
  [
    [[[0.26, 0.18], [0.74, 0.18], [0.36, 0.84]]],
    [[[0.28, 0.2], [0.72, 0.2], [0.5, 0.46], [0.42, 0.84]]],
  ],
  [
    [oval(0.5, 0.3, 0.18, 0.16), oval(0.5, 0.7, 0.18, 0.16)],
    [oval(0.5, 0.32, 0.22, 0.2), oval(0.5, 0.68, 0.22, 0.18)],
  ],
  [
    [[[0.34, 0.78], [0.66, 0.5], [0.66, 0.22], [0.34, 0.22], [0.34, 0.5], [0.66, 0.5]]],
    [[[0.3, 0.34], [0.5, 0.16], [0.7, 0.34], [0.5, 0.52], [0.3, 0.34], [0.58, 0.84]]],
  ],
];

export type MnistPrediction = {
  digit: number;
  confidence: number;
  probabilities: number[];
  source: "api" | "rsaa";
};

export type AccuracyReport = {
  accuracy: number;
  n: number;
  correct: number;
  perDigit: number[];
};

const SIZE = 28;
const INNER = 20;

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

function stamp(grid: Float32Array, x: number, y: number, radius = 1) {
  const xi = Math.round(x);
  const yi = Math.round(y);
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dy * dy > radius * radius + 0.2) continue;
      const px = xi + dx;
      const py = yi + dy;
      if (px >= 0 && py >= 0 && px < SIZE && py < SIZE) {
        grid[py * SIZE + px] = 1;
      }
    }
  }
}

function drawLine(grid: Float32Array, x0: number, y0: number, x1: number, y1: number, radius = 1) {
  const n = Math.max(2, Math.ceil(Math.hypot(x1 - x0, y1 - y0) * SIZE));
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    stamp(grid, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, radius);
  }
}

function mapPoint(
  x: number,
  y: number,
  tx: number,
  ty: number,
  scale: number,
  rot: number
): [number, number] {
  const dx = x - 0.5;
  const dy = y - 0.5;
  const c = Math.cos(rot);
  const s = Math.sin(rot);
  return [0.5 + (dx * c - dy * s) * scale + tx, 0.5 + (dx * s + dy * c) * scale + ty];
}

function rasterizePaths(
  paths: [number, number][][],
  tx = 0,
  ty = 0,
  scale = 1,
  rot = 0,
  radius = 1
): Float32Array {
  const grid = new Float32Array(SIZE * SIZE);
  for (const path of paths) {
    for (let i = 1; i < path.length; i++) {
      const a = mapPoint(path[i - 1][0], path[i - 1][1], tx, ty, scale, rot);
      const b = mapPoint(path[i][0], path[i][1], tx, ty, scale, rot);
      drawLine(grid, a[0] * (SIZE - 1), a[1] * (SIZE - 1), b[0] * (SIZE - 1), b[1] * (SIZE - 1), radius);
    }
  }
  return grid;
}

export function rasterizeDigit(digit: number): Float32Array {
  return rasterizePaths(DIGIT_PATHS[digit] ?? DIGIT_PATHS[3]);
}

function allPathsFor(digit: number) {
  return [DIGIT_PATHS[digit] ?? DIGIT_PATHS[3], ...(DIGIT_VARIANTS[digit] ?? [])];
}

export function strokeDigit(ctx: CanvasRenderingContext2D, digit: number, size: number) {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = size * 0.11;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const paths = DIGIT_PATHS[digit] ?? DIGIT_PATHS[3];
  for (const path of paths) {
    ctx.beginPath();
    path.forEach(([x, y], i) => {
      const px = x * size;
      const py = y * size;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
}

function inkFromGray(pixels: number[] | Float32Array) {
  const out = new Float32Array(SIZE * SIZE);
  const n = Math.min(out.length, pixels.length);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = pixels[i];
    out[i] = v > 1 ? v / 255 : v;
    sum += out[i];
  }
  const mean = sum / n;
  if (mean > 0.5) {
    for (let i = 0; i < n; i++) out[i] = 1 - out[i];
  }
  return out;
}

export function inkMass(pixels: number[] | Float32Array) {
  const ink = inkFromGray(pixels);
  let sum = 0;
  for (let i = 0; i < ink.length; i++) sum += ink[i];
  return sum;
}

function centerAndFit(grid: Float32Array) {
  let minX = SIZE;
  let minY = SIZE;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      if (grid[y * SIZE + x] < 0.12) continue;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }
  if (maxX < 0) return new Float32Array(SIZE * SIZE);
  const w = maxX - minX + 1;
  const h = maxY - minY + 1;
  const s = INNER / Math.max(w, h);
  const out = new Float32Array(SIZE * SIZE);
  const ox = (SIZE - w * s) / 2;
  const oy = (SIZE - h * s) / 2;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const sx = minX + (x - ox) / s;
      const sy = minY + (y - oy) / s;
      if (sx < 0 || sy < 0 || sx >= SIZE - 1 || sy >= SIZE - 1) continue;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const fx = sx - x0;
      const fy = sy - y0;
      const a = grid[y0 * SIZE + x0];
      const b = grid[y0 * SIZE + x0 + 1];
      const c = grid[(y0 + 1) * SIZE + x0];
      const d = grid[(y0 + 1) * SIZE + x0 + 1];
      out[y * SIZE + x] = a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
    }
  }
  return out;
}

function downsample(grid: Float32Array, from: number, to: number) {
  const out = new Float32Array(to * to);
  const scale = from / to;
  for (let y = 0; y < to; y++) {
    for (let x = 0; x < to; x++) {
      let acc = 0;
      let c = 0;
      const x0 = Math.floor(x * scale);
      const y0 = Math.floor(y * scale);
      const x1 = Math.min(from, Math.ceil((x + 1) * scale));
      const y1 = Math.min(from, Math.ceil((y + 1) * scale));
      for (let yy = y0; yy < y1; yy++) {
        for (let xx = x0; xx < x1; xx++) {
          acc += grid[yy * from + xx];
          c++;
        }
      }
      out[y * to + x] = c ? acc / c : 0;
    }
  }
  return out;
}

function shift(grid: Float32Array, dx: number, dy: number) {
  const out = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const sx = x - dx;
      const sy = y - dy;
      if (sx < 0 || sy < 0 || sx >= SIZE || sy >= SIZE) continue;
      out[y * SIZE + x] = grid[sy * SIZE + sx];
    }
  }
  return out;
}

function countHoles(grid: Float32Array) {
  const wall = 0.28;
  const seen = new Uint8Array(SIZE * SIZE);
  const stack: number[] = [];
  const push = (x: number, y: number) => {
    if (x < 0 || y < 0 || x >= SIZE || y >= SIZE) return;
    const i = y * SIZE + x;
    if (seen[i] || grid[i] >= wall) return;
    seen[i] = 1;
    stack.push(i);
  };
  for (let x = 0; x < SIZE; x++) {
    push(x, 0);
    push(x, SIZE - 1);
  }
  for (let y = 0; y < SIZE; y++) {
    push(0, y);
    push(SIZE - 1, y);
  }
  while (stack.length) {
    const i = stack.pop()!;
    const x = i % SIZE;
    const y = (i / SIZE) | 0;
    push(x + 1, y);
    push(x - 1, y);
    push(x, y + 1);
    push(x, y - 1);
  }
  let holes = 0;
  for (let i = 0; i < seen.length; i++) {
    if (seen[i] || grid[i] >= wall) continue;
    holes++;
    seen[i] = 1;
    stack.push(i);
    while (stack.length) {
      const j = stack.pop()!;
      const x = j % SIZE;
      const y = (j / SIZE) | 0;
      push(x + 1, y);
      push(x - 1, y);
      push(x, y + 1);
      push(x, y - 1);
    }
  }
  return holes;
}

function geometry(grid: Float32Array) {
  let mass = 0;
  let cx = 0;
  let cy = 0;
  let top = 0;
  let bot = 0;
  let left = 0;
  let right = 0;
  let mid = 0;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const v = grid[y * SIZE + x];
      if (v < 0.12) continue;
      mass += v;
      cx += x * v;
      cy += y * v;
      if (y < SIZE * 0.33) top += v;
      else if (y > SIZE * 0.66) bot += v;
      else mid += v;
      if (x < SIZE * 0.5) left += v;
      else right += v;
    }
  }
  if (mass < 1e-6) return { holes: 0, cy: 0.5, top: 0, bot: 0, left: 0, right: 0, mid: 0 };
  return {
    holes: countHoles(grid),
    cy: cy / mass / (SIZE - 1),
    top: top / mass,
    bot: bot / mass,
    left: left / mass,
    right: right / mass,
    mid: mid / mass,
  };
}

function softmax(xs: number[], temperature = 1) {
  const scaled = xs.map((v) => v / temperature);
  const m = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - m));
  const z = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / z);
}

type Template = { digit: number; small: Float32Array };

let templates: Template[] | null = null;

function l2neg(a: Float32Array, b: Float32Array) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return -s;
}

function getTemplates() {
  if (templates) return templates;
  const built: Template[] = [];
  const affines: [number, number, number, number][] = [
    [0, 0, 1, 0],
    [0.05, 0, 0.9, 0],
    [-0.05, 0.04, 1.08, 0],
    [0, -0.05, 0.86, 0.1],
  ];
  for (let d = 0; d < 10; d++) {
    for (const paths of allPathsFor(d)) {
      for (const [tx, ty, scale, rot] of affines) {
        const fitted = centerAndFit(rasterizePaths(paths, tx, ty, scale, rot, 1));
        built.push({ digit: d, small: downsample(fitted, SIZE, 14) });
      }
    }
  }
  templates = built;
  return built;
}

function geometricBonus(digit: number, g: ReturnType<typeof geometry>) {
  let bonus = 0;
  if (digit === 8) bonus += g.holes >= 2 ? 4.2 : -3.5;
  if (digit === 0) bonus += g.holes === 1 ? 2.4 : -2.0;
  if (digit === 6) bonus += g.holes === 1 && g.cy > 0.52 ? 3.0 : g.cy > 0.52 ? 0.8 : -1.4;
  if (digit === 9) bonus += g.holes === 1 && g.cy < 0.48 ? 3.0 : g.cy < 0.48 ? 0.8 : -1.4;
  if (digit === 4) bonus += g.right > 0.48 && g.mid > 0.25 ? 1.6 : 0;
  if (digit === 1) bonus += g.holes === 0 && Math.abs(g.left - g.right) < 0.2 ? 1.2 : 0;
  if (digit === 7) bonus += g.top > 0.3 && g.bot < 0.28 ? 2.0 : -1.0;
  if (digit === 3) bonus += g.holes === 0 && g.right > 0.52 && g.mid > 0.22 ? 2.4 : 0.3;
  if (digit === 2) bonus += g.bot > 0.3 && g.holes === 0 ? 1.6 : 0;
  if (digit === 5) bonus += g.top > 0.22 && g.left > 0.4 && g.holes === 0 ? 1.5 : 0;
  return bonus;
}

function bestScores(fitted: Float32Array) {
  const probe = downsample(fitted, SIZE, 14);
  const bank = getTemplates();
  const scores = Array.from({ length: 10 }, () => -1e9);
  const geo = geometry(fitted);
  for (const dy of [-1, 0, 1]) {
    for (const dx of [-1, 0, 1]) {
      const shifted = dx || dy ? downsample(shift(fitted, dx, dy), SIZE, 14) : probe;
      for (const t of bank) {
        const s = l2neg(shifted, t.small);
        if (s > scores[t.digit]) scores[t.digit] = s;
      }
    }
  }
  for (let d = 0; d < 10; d++) scores[d] += geometricBonus(d, geo);
  return scores;
}

/**
 * Softmax over class evidence. Temperature is fixed on the L2 scale so nearby
 * digits (1 vs 4, 3 vs 8) share probability instead of collapsing to 100%.
 */
function classDistribution(scores: number[]) {
  const max = Math.max(...scores);
  const relative = scores.map((s) => s - max);
  const temperature = 3.2;
  const raw = softmax(relative, temperature);
  const floor = 0.05;
  const mixed = raw.map((p) => p * (1 - floor) + floor / 10);
  const z = mixed.reduce((a, b) => a + b, 0);
  return mixed.map((p) => p / z);
}

/** ARCANE RSAA: digit templates resonate with the drawing; softmax is the readout. */
export function predictMnistRsaa(pixels: number[] | Float32Array): MnistPrediction {
  const fitted = centerAndFit(inkFromGray(pixels));
  let state = bestScores(fitted);
  for (let cycle = 0; cycle < 2; cycle++) {
    const out = softmax(state, 3.2);
    const mean = state.reduce((a, b) => a + b, 0) / 10;
    for (let d = 0; d < 10; d++) {
      state[d] += 0.08 * (state[d] - mean) * out[d];
    }
  }
  const probabilities = classDistribution(state);
  const digit = probabilities.reduce((best, p, i) => (p > probabilities[best] ? i : best), 0);
  return {
    digit,
    confidence: probabilities[digit],
    probabilities,
    source: "rsaa",
  };
}

function jitteredDigit(digit: number, rng: () => number) {
  const paths = allPathsFor(digit)[Math.floor(rng() * allPathsFor(digit).length)];
  const tx = (rng() - 0.5) * 0.4;
  const ty = (rng() - 0.5) * 0.4;
  const scale = 0.58 + rng() * 0.7;
  const rot = (rng() - 0.5) * 0.7;
  const radius = rng() < 0.4 ? 1 : 2;
  return rasterizePaths(paths, tx, ty, scale, rot, radius);
}

export function evaluateMnistRsaaAccuracy(samplesPerDigit = 24, seed = 20240915): AccuracyReport {
  templates = null;
  const rng = mulberry32(seed);
  const perDigit = Array.from({ length: 10 }, () => 0);
  let correct = 0;
  let n = 0;
  for (let d = 0; d < 10; d++) {
    for (let i = 0; i < samplesPerDigit; i++) {
      const pixels = jitteredDigit(d, rng);
      const pred = predictMnistRsaa(pixels);
      n++;
      if (pred.digit === d) {
        correct++;
        perDigit[d]++;
      }
    }
    perDigit[d] /= samplesPerDigit;
  }
  return { accuracy: correct / n, n, correct, perDigit };
}
