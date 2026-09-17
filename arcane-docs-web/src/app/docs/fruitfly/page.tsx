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
          Traced neurons from the adult <em>Drosophila</em> whole-brain connectome (FAFB v783), driven by ARCANE
          resonance while you draw a digit.
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
          The pad is the retina. Strokes drive LC4 and LPLC2 cells in the optic lobes; spikes propagate along the
          traced arbors through synapses whose weights are the published EM synapse counts, and out along the
          descending axons toward the neck connective.
        </p>
        <div className="border-l-2 border-[#C785F2]/40 bg-zinc-900/40 py-3 pl-4 text-sm leading-relaxed text-zinc-400">
          <p className="mb-2">
            <strong className="text-zinc-200">What is real, and what is not.</strong>
          </p>
          <ul className="m-0 list-disc space-y-1.5 pl-4">
            <li>
              <strong className="text-zinc-300">Real:</strong> the 36 neurons are FlyWire v783 cells with their
              published root IDs, and what you see are their traced skeletons swept into tubes — not stand-in shapes.
              The shell is the FLYWIRE neuropil mesh. Synapse counts, signs and neurotransmitter calls come from the
              public release.
            </li>
            <li>
              <strong className="text-zinc-300">Simulated:</strong> the spiking is ARCANE&apos;s leaky
              integrate-and-fire model running over that wiring. It is not a recording of a fly.
            </li>
            <li>
              <strong className="text-zinc-300">Ventral nerve cord:</strong> the glass shell is the Male CNS
              JRCFIB2022M VNC neuropil mesh. The neurites inside it are traced Male CNS v1.0 skeletons (motor,
              sensory, intrinsic, ascending and descending cells from the Janelia / Google map), clipped to the VNC
              volume — FAFB itself is brain-only, so this cord is a different EM specimen aligned at the neck.
            </li>
            <li>
              <strong className="text-zinc-300">Not the connectome:</strong> the digit readout. The classifier is a
              separate template matcher; this circuit is an escape-reflex pathway and does not recognise digits. The
              bars show what the classifier decided, and the cells respond to it.
            </li>
          </ul>
        </div>
        <p className="leading-relaxed">
          Cells stay quiet until there is ink. Firing rate, active-cell count and homeostatic gain in the header are
          read straight out of the simulator.
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
          ). Skeletons from the FlyWire v783 skeleton service; brain neuropil from navis-flybrains FLYWIRE; ventral
          nerve cord mesh and traced VNC neurons from Male CNS v1.0 (
          <a
            href="https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/"
            target="_blank"
            rel="noreferrer"
            className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium"
          >
            Berg et al., Cell 2026
          </a>
          ). Dynamics:{" "}
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
