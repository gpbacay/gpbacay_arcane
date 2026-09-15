"use client";

import Link from "next/link";
import { FlyWireConnectome } from "@/components/FlyWireConnectome";

export default function FruitFlyPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <p className="mb-3 text-[11px] font-semibold uppercase tracking-[0.22em] text-[#C785F2]">
          Embodiment
        </p>
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          ARCANE Fruitfly Brain
        </h1>
        <p className="text-xl text-zinc-400">
          Adult <em>Drosophila</em> brain and ventral nerve cord (FAFB v783 + MANC/FANC) running ARCANE resonance on a
          handwritten digit.
        </p>
      </div>

      <div className="space-y-6 text-zinc-300">
        <h2 id="live-connectome" className="text-2xl font-bold tracking-tight text-zinc-100 mt-2 mb-4 border-b border-zinc-800 pb-2">
          Live connectome
        </h2>
        <FlyWireConnectome />

        <h2 id="what-it-does" className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          What it does
        </h2>
        <p className="leading-relaxed">
          The pad is the retina. Strokes drive LC4 and LPLC2 cells in the optic lobes; spikes propagate through the
          reconstructed FAFB brain and down descending axons into the ventral nerve cord — the body CNS that drives
          legs, wings, and abdomen.{" "}
          <Link href="/docs/research" className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium">
            RSAA
          </Link>{" "}
          reads a softmax over digits 0–9 from that state. Nearby shapes share probability. Cells stay quiet until there is ink.
        </p>
        <p className="leading-relaxed text-sm text-zinc-500">
          Connectome: Google DeepMind / FlyWire FAFB v783 (
          <a href="https://doi.org/10.1038/s41586-024-07558-y" target="_blank" rel="noreferrer" className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium">
            Dorkenwald et al., 2024
          </a>
          ;{" "}
          <a href="https://doi.org/10.1038/s41586-024-07686-5" target="_blank" rel="noreferrer" className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium">
            Schlegel et al., 2024
          </a>
          ). Body: ventral nerve cord (
          <a href="https://doi.org/10.1038/s41586-024-07389-x" target="_blank" rel="noreferrer" className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium">
            Azevedo et al., 2024
          </a>
          ; MANC). Dynamics:{" "}
          <Link href="/docs/activations" className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium">
            resonant spiking
          </Link>
          ,{" "}
          <Link href="/docs/layers" className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium">
            homeostatic gain
          </Link>
          , Hebbian synapses.
        </p>

        <h2 id="how-to-use" className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          How to use
        </h2>
        <ol className="list-decimal list-inside space-y-2 leading-relaxed">
          <li>Draw a digit. The large numeral is the top class; the ten bars are P(0)…P(9).</li>
          <li>Drag to orbit in any direction. Scroll to zoom. Region chips isolate optic lobes, neuropil, descending cells, or the ventral nerve cord.</li>
        </ol>
      </div>
    </div>
  );
}
