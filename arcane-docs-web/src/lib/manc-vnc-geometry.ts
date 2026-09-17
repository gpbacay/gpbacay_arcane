/**
 * Loader for the Male CNS ventral nerve cord exported by
 * `scripts/export-manc-vnc-geometry.py`.
 *
 * The neuropil shell is the navis-flybrains JRCFIB2022M_vnc mesh (Janelia /
 * Google Research complete male Drosophila CNS). Interior tubes are traced
 * Male CNS v1.0 skeletons (motor, sensory, intrinsic, ascending, descending),
 * clipped to the VNC volume and swept the same way as the FlyWire head neurons.
 */

import meta from "@/data/manc-vnc-geometry-meta.json";

export type MancVncGeometry = {
  meshPositions: Float32Array;
  meshNormals: Float32Array;
  meshRegions: Float32Array;
  meshCount: number;
  arborPositions: Float32Array;
  arborNormals: Float32Array;
  arborColors: Float32Array;
  arborRegions: Float32Array;
  arborCount: number;
  byteLength: number;
};

let pending: Promise<MancVncGeometry> | null = null;

function u8ToFloat(bytes: Uint8Array, count: number): Float32Array {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = bytes[i];
  return out;
}

export function mancVncGeometryMeta() {
  return meta;
}

export function loadMancVncGeometry(): Promise<MancVncGeometry> {
  if (!pending) {
    pending = (async () => {
      const res = await fetch(`${meta.bin}?v=${meta.byteLength}`);
      if (!res.ok) throw new Error(`Male CNS VNC geometry missing (${res.status})`);

      const buf = await res.arrayBuffer();
      if (buf.byteLength < meta.byteLength) {
        throw new Error(`Male CNS VNC geometry truncated (${buf.byteLength}/${meta.byteLength})`);
      }

      const bytes = new Uint8Array(buf);
      const view = new DataView(buf);
      const f32 = (offset: number, count: number) => {
        const out = new Float32Array(count);
        for (let i = 0; i < count; i++) out[i] = view.getFloat32(offset + i * 4, true);
        return out;
      };

      const meshCount = meta.mesh.count;
      const arborCount = meta.arbors.count;

      return {
        meshPositions: f32(meta.mesh.posOffset, meshCount * 3),
        meshNormals: f32(meta.mesh.nrmOffset, meshCount * 3),
        meshRegions: u8ToFloat(
          bytes.subarray(meta.mesh.regOffset, meta.mesh.regOffset + meshCount),
          meshCount
        ),
        meshCount,
        arborPositions: f32(meta.arbors.posOffset, arborCount * 3),
        arborNormals: f32(meta.arbors.nrmOffset, arborCount * 3),
        arborColors: f32(meta.arbors.colOffset, arborCount * 3),
        arborRegions: u8ToFloat(
          bytes.subarray(meta.arbors.regOffset, meta.arbors.regOffset + arborCount),
          arborCount
        ),
        arborCount,
        byteLength: meta.byteLength,
      };
    })().catch((err) => {
      pending = null;
      throw err;
    });
  }
  return pending;
}

const NECK_Y = -0.4;
const HEAD_GF: [number, number, number] = [0.95, 0.82, 1];
const HEAD_SENSORY: [number, number, number] = [0.2, 0.8, 0.9];
const HEAD_DEFAULT: [number, number, number] = [0.82, 0.38, 0.95];

export type VncCircuitSlot = {
  index: number;
  cell_type: string;
  layer: string;
  side: string;
};

export type BoundVncArbors = {
  colors: Float32Array;
  neuronIndex: Float32Array;
  distances: Float32Array;
};

/**
 * Maps VNC tube vertices onto the live FlyWire circuit so the cord uses the
 * same resting palette and the same spike-wavefront shader as the head.
 *
 * Descending / Giant Fiber slots drive the wave from the neck; motor and
 * intrinsic cells follow with a small extra path delay so they light after the
 * descending axons, the way the escape circuit actually recruits the cord.
 */
export function bindVncArborsToCircuit(
  positions: Float32Array,
  sourceColors: Float32Array,
  count: number,
  slots: VncCircuitSlot[]
): BoundVncArbors {
  const descending = slots.filter((s) => s.layer === "descending" || s.cell_type === "DNp01");
  const sensory = slots.filter((s) => s.layer === "sensory");
  const gf = descending.filter((s) => s.cell_type === "DNp01");
  const leftDesc = descending.filter((s) => s.side === "left");
  const rightDesc = descending.filter((s) => s.side === "right");
  const leftSens = sensory.filter((s) => s.side === "left");
  const rightSens = sensory.filter((s) => s.side === "right");
  const fallback = (slots.length > 0 ? slots : [{ index: 0, cell_type: "", layer: "", side: "" }]);

  const sideGroup = (x: number, preferred: VncCircuitSlot[], other: VncCircuitSlot[]) => {
    if (x < 0 && preferred.length) return preferred;
    if (x >= 0 && other.length) return other;
    return preferred.length ? preferred : other.length ? other : fallback;
  };

  const colors = new Float32Array(count * 3);
  const neuronIndex = new Float32Array(count);
  const distances = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const o = i * 3;
    const x = positions[o];
    const y = positions[o + 1];
    const r = sourceColors[o];
    const g = sourceColors[o + 1];
    const b = sourceColors[o + 2];

    const isDescending = r > 0.85 && g > 0.7 && b < 0.7;
    const isMotor = g > 0.7 && r < 0.45;
    const isSensory = b > 0.85 && r < 0.5 && g < 0.7;
    const isAscending = r > 0.85 && b > 0.6 && g < 0.6;

    let rgb: [number, number, number] = HEAD_DEFAULT;
    let delay = 0.28;
    let group = fallback;
    if (isDescending) {
      rgb = HEAD_GF;
      delay = 0;
      group = gf.length > 0 ? gf : sideGroup(x, leftDesc, rightDesc);
    } else if (isSensory) {
      rgb = HEAD_SENSORY;
      delay = 0.08;
      group = sideGroup(x, leftSens, rightSens);
    } else if (isMotor) {
      rgb = HEAD_SENSORY;
      delay = 0.18;
      group = sideGroup(x, leftDesc, rightDesc);
    } else if (isAscending) {
      rgb = HEAD_DEFAULT;
      delay = 0.16;
      group = sideGroup(x, leftDesc, rightDesc);
    } else {
      group = sideGroup(x, leftDesc, rightDesc);
      if (group === fallback && sensory.length) group = sideGroup(x, leftSens, rightSens);
    }

    colors[o] = rgb[0];
    colors[o + 1] = rgb[1];
    colors[o + 2] = rgb[2];
    neuronIndex[i] = group[i % group.length].index;
    distances[i] = Math.max(0, NECK_Y - y) + delay;
  }

  return { colors, neuronIndex, distances };
}
