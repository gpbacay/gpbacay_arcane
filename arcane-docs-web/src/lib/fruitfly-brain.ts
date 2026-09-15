/**
 * Browser-side RSAA-style controller for the fruit-fly embodiment demo.
 *
 * This is not TuragaLab/flybody and not the Python TensorFlow ARCANE package.
 * It is a small, inspectable Resonant State Alignment loop: sensory state is
 * refined against higher-level posture/intent before a motor command is issued.
 */

export type LocomotionMode = "walk" | "fly";
export type ControllerKind = "feedforward" | "resonant";

export type MotorCommand = {
  thrust: number;
  lift: number;
  pitchTorque: number;
  rollTorque: number;
  strideRate: number;
  wingAmp: number;
};

export type LayerSnapshot = {
  name: string;
  state: number[];
  divergence: number;
};

export type BrainSnapshot = {
  layers: LayerSnapshot[];
  cycles: number;
  meanDivergence: number;
  motor: MotorCommand;
};

export type FlyBody = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  pitch: number;
  pitchVel: number;
  roll: number;
  rollVel: number;
  gaitPhase: number;
  wingPhase: number;
  gust: number;
  gustDir: number;
};

const LAYER_SIZE = 6;
const LAYER_NAMES = ["Sensory", "Local motor", "Posture", "Intent"] as const;

function tanh(x: number) {
  return Math.tanh(x);
}

function zeros(n: number) {
  return Array.from({ length: n }, () => 0);
}

function rms(a: number[], b: number[]) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum / a.length);
}

function encodeSensory(body: FlyBody, mode: LocomotionMode, targetY: number) {
  const s = zeros(LAYER_SIZE);
  s[0] = tanh(body.pitch * 2.2);
  s[1] = tanh(body.pitchVel * 0.45);
  s[2] = tanh((targetY - body.y) / 36);
  s[3] = tanh(body.vy / 10);
  s[4] = tanh(body.gust * 1.4 + body.roll * 1.6);
  s[5] = mode === "fly" ? 1 : -1;
  return s;
}

function projectDown(higher: number[]) {
  const projected = zeros(LAYER_SIZE);
  for (let i = 0; i < LAYER_SIZE; i++) {
    const neighbor = higher[(i + 1) % LAYER_SIZE];
    const opposite = higher[(i + 3) % LAYER_SIZE];
    projected[i] = tanh(0.72 * higher[i] + 0.18 * neighbor - 0.08 * opposite);
  }
  return projected;
}

function liftUp(lower: number[], higher: number[]) {
  const next = zeros(LAYER_SIZE);
  for (let i = 0; i < LAYER_SIZE; i++) {
    next[i] = tanh(0.55 * higher[i] + 0.45 * lower[i]);
  }
  return next;
}

function decodeMotor(local: number[], intent: number[], mode: LocomotionMode): MotorCommand {
  const flyBias = mode === "fly" ? 1 : 0;
  const pitchTorque = -1.55 * local[0] - 0.55 * local[1] - 0.25 * intent[0];
  const lift = 3.1 * local[2] - 0.85 * local[3] + 1.4 * flyBias;
  const thrust = 0.55 + 0.35 * (1 - Math.abs(local[0])) + 0.4 * flyBias;
  const rollTorque = -1.2 * local[4] - 0.3 * intent[4];
  const strideRate = mode === "walk" ? 9.5 + 1.8 * thrust : 4.2;
  const wingAmp = mode === "fly" ? 0.72 + 0.22 * Math.max(0, lift) : 0.12;
  return { thrust, lift, pitchTorque, rollTorque, strideRate, wingAmp };
}

export function createBody(): FlyBody {
  return {
    x: 0,
    y: 18,
    vx: 0,
    vy: 0,
    pitch: 0,
    pitchVel: 0,
    roll: 0,
    rollVel: 0,
    gaitPhase: 0,
    wingPhase: 0,
    gust: 0,
    gustDir: 1,
  };
}

export function applyGust(body: FlyBody, strength = 1) {
  const dir = Math.random() > 0.5 ? 1 : -1;
  body.gust = Math.min(1.8, body.gust + 0.95 * strength);
  body.gustDir = dir;
  body.pitchVel += 5.4 * dir * strength;
  body.rollVel += 3.8 * dir * strength;
  body.vy -= 6.5 * strength;
  body.vx += 18 * dir * strength;
}

export function stepBrain(
  body: FlyBody,
  mode: LocomotionMode,
  kind: ControllerKind,
  targetY: number
): BrainSnapshot {
  const sensory = encodeSensory(body, mode, targetY);
  const layers = [sensory, sensory.map((v) => tanh(0.9 * v)), zeros(LAYER_SIZE), zeros(LAYER_SIZE)];

  for (let i = 0; i < LAYER_SIZE; i++) {
    layers[2][i] = tanh(0.65 * layers[1][i] + (i === 2 || i === 5 ? 0.35 * sensory[5] : 0));
    layers[3][i] = tanh((i === 5 ? sensory[5] : 0.4 * layers[2][i]) + (i === 2 ? 0.5 * sensory[2] : 0));
  }

  const cycles = kind === "resonant" ? 8 : 1;
  let lastDivs = [0, 0, 0, 0];

  for (let n = 0; n < cycles; n++) {
    for (let i = 3; i >= 1; i--) {
      const projected = projectDown(layers[i]);
      lastDivs[i - 1] = rms(layers[i - 1], projected);
      const gamma = kind === "resonant" ? 0.42 : 0.12;
      for (let u = 0; u < LAYER_SIZE; u++) {
        layers[i - 1][u] += gamma * (projected[u] - layers[i - 1][u]);
      }
    }
    for (let i = 1; i < 4; i++) {
      layers[i] = liftUp(layers[i - 1], layers[i]);
    }
  }

  const motor = decodeMotor(layers[1], layers[3], mode);
  if (kind === "feedforward") {
    motor.pitchTorque *= 1.35;
    motor.lift *= 0.82;
    motor.rollTorque *= 1.4;
  }

  const snapshots: LayerSnapshot[] = layers.map((state, i) => ({
    name: LAYER_NAMES[i],
    state: state.slice(),
    divergence: lastDivs[i] ?? 0,
  }));

  const meanDivergence =
    snapshots.reduce((acc, layer) => acc + layer.divergence, 0) / snapshots.length;

  return { layers: snapshots, cycles, meanDivergence, motor };
}

export function stepBody(
  body: FlyBody,
  motor: MotorCommand,
  mode: LocomotionMode,
  targetY: number,
  dt: number
) {
  const gravity = mode === "fly" ? 42 : 70;
  const drag = 1.8;

  body.pitchVel += motor.pitchTorque * 9 * dt;
  body.pitchVel += body.gust * body.gustDir * 4.5 * dt;
  body.pitchVel *= Math.exp(-3.4 * dt);
  body.pitch += body.pitchVel * dt;
  body.pitch = Math.max(-0.85, Math.min(0.85, body.pitch));

  body.rollVel += motor.rollTorque * 8 * dt;
  body.rollVel *= Math.exp(-3.1 * dt);
  body.roll += body.rollVel * dt;
  body.roll = Math.max(-0.7, Math.min(0.7, body.roll));

  if (mode === "fly") {
    const lift = motor.lift * 38 + Math.cos(body.wingPhase) * motor.wingAmp * 6;
    body.vy += (lift - gravity) * dt;
    body.vx += (motor.thrust * 22 - body.vx * drag) * dt;
    body.vx += Math.sin(body.pitch) * 10 * dt;
  } else {
    body.vy += (motor.lift * 8 - gravity) * dt;
    body.vx += (motor.thrust * 16 - body.vx * 2.4) * dt;
  }

  body.y += body.vy * dt;
  body.x += body.vx * dt;

  if (mode === "walk") {
    const ground = 18 + Math.sin(body.gaitPhase * 2) * 1.4;
    if (body.y < ground) {
      body.y = ground;
      body.vy *= -0.12;
    }
    body.y += (ground - body.y) * 6 * dt;
  } else {
    body.y += (targetY - body.y) * 0.35 * dt;
    body.y = Math.max(24, Math.min(210, body.y));
  }

  body.gaitPhase += motor.strideRate * dt;
  body.wingPhase += (mode === "fly" ? 52 : 10) * dt;
  body.gust *= Math.exp(-1.25 * dt);
}
