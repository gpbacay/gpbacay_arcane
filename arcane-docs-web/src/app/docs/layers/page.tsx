"use client";

import { useState, useEffect } from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export default function BiologicalLayersPage() {
  const [openItem, setOpenItem] = useState<string | undefined>(undefined);

  useEffect(() => {
    const hash = typeof window !== "undefined" ? window.location.hash.slice(1) : "";
    if (hash === "predictive-resonant") {
      setOpenItem("predictive-resonant");
    }
  }, []);

  const layers = [
    {
      id: "resonant-gser",
      name: "ResonantGSER",
      description: "Hierarchical resonant layer with bi-directional feedback, spiking dynamics, and reservoir computing.",
      details: "ResonantGSER is an LSTM-based RNN cell that EMA-harmonizes its hidden state toward a top-down prototype (closed-form N-step map, skipped until alignment is set). Spikes use a straight-through estimator. Alignment is a batch-mean vector of shape (units,), not a per-example code. Spectral radius and elastic sizing belong to GSER, not this layer.",
      link: "/docs/resonant-gser",
      code: `from gpbacay_arcane import ResonantGSER

layer = ResonantGSER(
    units=128,
    resonance_factor=0.2,
    spike_threshold=0.35,
    resonance_cycles=5
)`
    },
    {
      id: "predictive-resonant",
      name: "PredictiveResonantLayer",
      description: "Implements Local Predictive Resonance inspired by Predictive Coding, modeling autonomous neural clusters that minimize prediction error.",
      details: "Unlike ResonantGSER's prototype alignment, this layer stores alignment in the recurrent state (h, c, align) per example. N harmonization steps are a closed-form EMA toward that vector. Optional persist_alignment carries a batch-mean memory across calls.",
      link: "/docs/predictive-resonant-layer",
      code: `from gpbacay_arcane import PredictiveResonantLayer

layer = PredictiveResonantLayer(
    units=128,
    resonance_cycles=3,
    resonance_step_size=0.2,
    spike_threshold=0.4,
    return_sequences=True,
    persist_alignment=False  # True for stateful resonance across calls
)`
    },
    {
      id: "bioplastic-dense",
      name: "BioplasticDenseLayer",
      description: "Dual-weight dense map: gradient kernel plus optional inference-time BCM plasticity.",
      details: "Effective weights are kernel + plastic_kernel. With enable_inference_plasticity=True and training=False, BCM uses the running activity trace θ: ΔW ∝ xᵀ[y ⊙ (y − θ)], then scales the plastic kernel toward target_avg. Gradient descent still updates kernel only.",
      code: `from gpbacay_arcane import BioplasticDenseLayer

layer = BioplasticDenseLayer(
    units=64,
    target_avg=0.12,
    homeostatic_rate=5e-5,
    bcm_tau=800.0,
    learning_rate=1e-3,
    enable_inference_plasticity=True
)`
    },
    {
      id: "gser",
      name: "GSER",
      description: "Gated spiking elastic reservoir; recurrent weights scaled to spectral_radius.",
      details: "GSER is an RNN cell with LIF reset, LSTM-style gates, and a semantic gate that pads to max_dynamic_reservoir_dim so elastic sizing does not crash. Grow/prune via DynamicSelfModelingReservoirCallback (requires reservoir_layer in the constructor; prune_rate is an absolute weight cutoff).",
      code: `from gpbacay_arcane import GSER

cell = GSER(
    input_dim=32,
    initial_reservoir_size=64,
    max_dynamic_reservoir_dim=128,
    spectral_radius=0.9,
    leak_rate=0.1,
    spike_threshold=0.5
)`
    },
    {
      id: "dense-gser",
      name: "DenseGSER",
      description: "Dense map with leak-controlled spike gating (not a reservoir).",
      details: "Applies GELU, then a sigmoid spike gate whose slope is 1/leak_rate and whose offset is spike_threshold. Optional conceptual (sigmoid) gate. spectral_radius is stored for API compatibility and is not applied to this rectangular map.",
      code: `from gpbacay_arcane import DenseGSER

layer = DenseGSER(
    units=256,
    leak_rate=0.1,
    spike_threshold=0.5,
    activation='gelu'
)`
    },
    {
      id: "latent-temporal-coherence",
      name: "LatentTemporalCoherence",
      description: "Distills temporal dynamics into coherence vectors for stable sequence processing.",
      details: "This layer focuses on the temporal stability of semantic representations. It ensures that the latent space evolves smoothly over time, reducing noise and capturing the long-term contextual essence of sequential information.",
      code: `from gpbacay_arcane import LatentTemporalCoherence

layer = LatentTemporalCoherence(d_coherence=64)`
    },
    {
      id: "relational-concept",
      name: "RelationalConceptModeling",
      description: "Multi-head attention mechanism for high-level semantic concept extraction.",
      details: "Uses specialized attention heads to identify and model relationships between different semantic concepts in the latent space, fostering a more structured and interpretable semantic hierarchy.",
      code: `from gpbacay_arcane import RelationalConceptModeling

layer = RelationalConceptModeling(d_model=64, num_heads=8)`
    },
    {
      id: "neuromimetic-activations",
      name: "Neuromimetic Activations",
      description: "Stateful, adaptive activation functions including Resonant Spiking and Homeostatic GELU.",
      details: "Keras wrapper for resonant_spike, homeostatic_gelu, or adaptive_softplus. Membrane / activity traces are feature-wise running averages, not per-sample RNN state. resonance_factor is a call() argument, not an __init__ kwarg. Spikes use a straight-through estimator.",
      link: "/docs/activations",
      code: `from gpbacay_arcane.activations import NeuromimeticActivation

activation = NeuromimeticActivation(
    activation_type='resonant_spike',
    threshold=0.5,
    leak_rate=0.1,
    name='rsa'
)
spikes = activation(inputs, resonance_factor=0.3)`
    }
  ];

  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Biological Layers
        </h1>
        <p className="text-xl text-zinc-400">
          Neural network layers inspired by biological principles.
        </p>
      </div>

      <div className="space-y-8 text-zinc-300 leading-7">
        <p>
          ARCANE provides a suite of custom Keras layers that mimic the dynamics of biological neurons. 
          Each layer is designed to bridge the gap between traditional connectionist AI and neuroscientific realism.
        </p>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Available Layers
        </h2>

        <Accordion type="single" collapsible className="w-full" value={openItem ?? ""} onValueChange={setOpenItem}>
          {layers.map((layer) => (
            <AccordionItem key={layer.id} value={layer.id} className="border-zinc-800">
              <AccordionTrigger className="hover:no-underline py-4 text-left group">
                <div className="flex flex-col gap-1">
                  <h3 
                    id={layer.id}
                    className="text-lg font-bold text-zinc-100 group-hover:text-purple-400 group-data-[state=open]:text-purple-400 transition-colors m-0"
                  >
                    {layer.name}
                  </h3>
                  <span className="text-sm font-normal text-zinc-400">
                    {layer.description}
                  </span>
                </div>
              </AccordionTrigger>
              <AccordionContent className="text-zinc-300 pb-6">
                <div className="space-y-4">
                  <p className="leading-relaxed m-0">
                    {layer.details}
                  </p>
                  
                  {layer.code && (
                    <div className="space-y-2">
                      <p className="text-xs font-bold uppercase tracking-wider text-zinc-500 mt-4">Quick Implementation</p>
                      <pre className="overflow-x-auto rounded-none border border-zinc-800 bg-zinc-950 p-4 text-sm text-zinc-300 shadow-inner">
                        <code>{layer.code}</code>
                      </pre>
                    </div>
                  )}

                  {layer.link && (
                    <div className="pt-2">
                      <a 
                        href={layer.link} 
                        className="inline-flex items-center text-sm font-medium text-[#C785F2] hover:text-[#C785F2]/80 transition-colors"
                      >
                        View detailed documentation <span className="ml-1">&rarr;</span>
                      </a>
                    </div>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>
    </div>
  );
}
