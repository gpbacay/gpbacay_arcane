/**
 * Loader for the real FlyWire FAFB v783 geometry exported by
 * `scripts/export-flywire-geometry.py`.
 *
 * Everything here is measured data, not procedural: the neuropil shell is the
 * navis-flybrains FLYWIRE mesh and each neuron is a tube mesh swept along its
 * published v783 skeleton. Both are triangle lists with per-vertex normals,
 * packed into one binary blob described by `flywire-geometry-meta.json`.
 *
 * FAFB is a brain-only volume. There is no ventral nerve cord in this data --
 * descending axons simply end where the sample was cut at the neck connective.
 */

import meta from "@/data/flywire-geometry-meta.json";

export type FlywireNeuronGeom = {
  id: string;
  label: string;
  cell_type: string;
  layer: string;
  side: string;
  soma: { x: number; y: number; z: number };
  /** Triangle-list vertices, 3 floats each. */
  positions: Float32Array;
  /** Unit normals, 3 floats each, parallel to `positions`. */
  normals: Float32Array;
  /** Path distance from the soma along the arbor, one float per vertex. */
  distances: Float32Array;
  /** Brain-region id per vertex, matching BRAIN_REGIONS bits. */
  regions: Float32Array;
  /** Vertex count (positions.length / 3). */
  count: number;
};

export type FlywireGeometry = {
  /** Neuropil shell, triangle list. */
  meshPositions: Float32Array;
  meshNormals: Float32Array;
  meshRegions: Float32Array;
  meshCount: number;
  neurons: Record<string, FlywireNeuronGeom>;
  /** Neuron ids in the order they appear in the blob. */
  order: string[];
  byteLength: number;
};

/**
 * Merged, GPU-ready buffers for every neuron we have geometry for.
 *
 * Colour animation is done on the GPU from a tiny per-neuron uniform array, so
 * `neuronIds` is only 36 entries long and none of these buffers is ever
 * re-uploaded after creation.
 */
export type MergedCircuitGeometry = {
  positions: Float32Array;
  normals: Float32Array;
  /** Resting colour per vertex; the shader scales it by live activity. */
  colors: Float32Array;
  regions: Float32Array;
  /** Index into `neuronIds` for each vertex, as a float attribute. */
  neuronIndex: Float32Array;
  /** Soma distance per vertex, driving the action-potential wavefront. */
  distances: Float32Array;
  count: number;
  /** Neuron ids in uniform-slot order; slot i is `uState[i]` in the shader. */
  neuronIds: string[];
  somas: { id: string; label: string; cell_type: string; layer: string; position: { x: number; y: number; z: number } }[];
};

/** Uniform array size compiled into the circuit shader. */
export const MAX_CIRCUIT_NEURONS = 64;

let pending: Promise<FlywireGeometry> | null = null;

function u8ToFloat(bytes: Uint8Array, count: number): Float32Array {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = bytes[i];
  return out;
}

export function flywireGeometryMeta() {
  return meta;
}

/**
 * Fetches and decodes the geometry blob. The result is cached for the page, and
 * a failure clears the cache so a later call can retry.
 *
 * @param onProgress receives bytes-downloaded / total while streaming.
 */
export function loadFlywireGeometry(
  onProgress?: (loaded: number, total: number) => void
): Promise<FlywireGeometry> {
  if (!pending) {
    pending = (async () => {
      const res = await fetch(`${meta.bin}?v=${meta.byteLength}`);
      if (!res.ok) throw new Error(`FlyWire geometry missing (${res.status})`);

      const total = Number(res.headers.get("content-length")) || meta.byteLength;
      let buf: ArrayBuffer;
      if (onProgress && res.body) {
        const reader = res.body.getReader();
        const parts: Uint8Array[] = [];
        let loaded = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          parts.push(value);
          loaded += value.byteLength;
          onProgress(loaded, total);
        }
        const merged = new Uint8Array(loaded);
        let at = 0;
        for (const part of parts) {
          merged.set(part, at);
          at += part.byteLength;
        }
        buf = merged.buffer;
      } else {
        buf = await res.arrayBuffer();
      }

      if (buf.byteLength < meta.byteLength) {
        throw new Error(`FlyWire geometry truncated (${buf.byteLength}/${meta.byteLength})`);
      }

      const bytes = new Uint8Array(buf);
      const view = new DataView(buf);
      const f32 = (offset: number, count: number) => {
        const out = new Float32Array(count);
        for (let i = 0; i < count; i++) out[i] = view.getFloat32(offset + i * 4, true);
        return out;
      };

      const meshCount = meta.mesh.count;
      const meshPositions = f32(meta.mesh.posOffset, meshCount * 3);
      const meshNormals = f32(meta.mesh.nrmOffset, meshCount * 3);
      const meshRegions = u8ToFloat(
        bytes.subarray(meta.mesh.regOffset, meta.mesh.regOffset + meshCount),
        meshCount
      );

      const neurons: Record<string, FlywireNeuronGeom> = {};
      const order: string[] = [];
      for (const n of meta.neurons) {
        order.push(n.id);
        neurons[n.id] = {
          id: n.id,
          label: n.label,
          cell_type: n.cell_type,
          layer: n.layer,
          side: n.side,
          soma: { x: n.soma[0], y: n.soma[1], z: n.soma[2] },
          positions: f32(n.posOffset, n.count * 3),
          normals: f32(n.nrmOffset, n.count * 3),
          distances: f32(n.distOffset, n.count),
          regions: u8ToFloat(bytes.subarray(n.regOffset, n.regOffset + n.count), n.count),
          count: n.count,
        };
      }

      return {
        meshPositions,
        meshNormals,
        meshRegions,
        meshCount,
        neurons,
        order,
        byteLength: meta.byteLength,
      };
    })().catch((err) => {
      pending = null;
      throw err;
    });
  }
  return pending;
}

/** Resting colour for a neuron, keyed the same way as the procedural fallback. */
function restingRgb(cell_type: string, layer: string): [number, number, number] {
  if (cell_type === "DNp01") return [0.95, 0.82, 1];
  if (layer === "sensory") return [0.2, 0.8, 0.9];
  return [0.82, 0.38, 0.95];
}

/**
 * Concatenates the per-neuron tube meshes into single interleaved buffers.
 *
 * Only neurons present in both the connectome graph and the geometry blob are
 * included, so a live API circuit with extra cells degrades to "render what we
 * actually traced" rather than inventing arbors for the rest.
 */
export function buildFlywireCircuit(
  geometry: FlywireGeometry,
  circuitNeurons: { id: string; label: string; cell_type: string; layer: string; side: string }[]
): MergedCircuitGeometry {
  const usable = circuitNeurons
    .filter((n) => geometry.neurons[n.id])
    .slice(0, MAX_CIRCUIT_NEURONS);

  let total = 0;
  for (const n of usable) total += geometry.neurons[n.id].count;

  const positions = new Float32Array(total * 3);
  const normals = new Float32Array(total * 3);
  const colors = new Float32Array(total * 3);
  const regions = new Float32Array(total);
  const neuronIndex = new Float32Array(total);
  const distances = new Float32Array(total);
  const neuronIds: string[] = [];
  const somas: MergedCircuitGeometry["somas"] = [];

  let at = 0;
  usable.forEach((n, slot) => {
    const g = geometry.neurons[n.id];
    const rgb = restingRgb(n.cell_type, n.layer);
    positions.set(g.positions, at * 3);
    normals.set(g.normals, at * 3);
    regions.set(g.regions, at);
    distances.set(g.distances, at);
    for (let i = 0; i < g.count; i++) {
      const o = (at + i) * 3;
      colors[o] = rgb[0];
      colors[o + 1] = rgb[1];
      colors[o + 2] = rgb[2];
      neuronIndex[at + i] = slot;
    }
    at += g.count;
    neuronIds.push(n.id);
    somas.push({
      id: n.id,
      label: n.label || g.label,
      cell_type: n.cell_type,
      layer: n.layer,
      position: g.soma,
    });
  });

  return {
    positions,
    normals,
    colors,
    regions,
    neuronIndex,
    distances,
    count: total,
    neuronIds,
    somas,
  };
}
