import meta from "@/data/flywire-geometry-meta.json";

export type FlywireNeuronGeom = {
  id: string;
  label: string;
  cell_type: string;
  layer: string;
  side: string;
  soma: { x: number; y: number; z: number };
  positions: Float32Array;
  distances: Float32Array;
  regions: Float32Array;
};

export type FlywireGeometry = {
  meshPositions: Float32Array;
  meshRegions: Float32Array;
  neurons: Record<string, FlywireNeuronGeom>;
};

let pending: Promise<FlywireGeometry> | null = null;

function u8ToFloat(bytes: Uint8Array, count: number): Float32Array {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = bytes[i];
  return out;
}

export function loadFlywireGeometry(): Promise<FlywireGeometry> {
  if (!pending) {
      pending = (async () => {
      const res = await fetch(`${meta.bin}?v=${meta.byteLength}`);
      if (!res.ok) throw new Error(`FlyWire geometry missing (${res.status})`);
      const buf = await res.arrayBuffer();
      const bytes = new Uint8Array(buf);
      const view = new DataView(buf);
      const f32 = (offset: number, count: number) => {
        const out = new Float32Array(count);
        for (let i = 0; i < count; i++) out[i] = view.getFloat32(offset + i * 4, true);
        return out;
      };
      const meshPositions = f32(meta.mesh.posOffset, meta.mesh.count * 3);
      const meshRegions = u8ToFloat(bytes.subarray(meta.mesh.regOffset, meta.mesh.regOffset + meta.mesh.count), meta.mesh.count);
      const neurons: Record<string, FlywireNeuronGeom> = {};
      for (const n of meta.neurons) {
        neurons[n.id] = {
          id: n.id,
          label: n.label,
          cell_type: n.cell_type,
          layer: n.layer,
          side: n.side,
          soma: { x: n.soma[0], y: n.soma[1], z: n.soma[2] },
          positions: f32(n.posOffset, n.count * 3),
          distances: f32(n.distOffset, n.count),
          regions: u8ToFloat(bytes.subarray(n.regOffset, n.regOffset + n.count), n.count),
        };
      }
      return { meshPositions, meshRegions, neurons };
    })().catch((err) => {
      pending = null;
      throw err;
    });
  }
  return pending;
}
