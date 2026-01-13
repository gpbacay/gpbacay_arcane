import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export default function BiologicalLayersPage() {
  const layers = [
    {
      id: "resonant-gser",
      name: "ResonantGSER",
      description: "Hierarchical resonant layer with bi-directional feedback, spiking dynamics, and reservoir computing.",
      details: "The ResonantGSER is the core of ARCANE's deliberative reasoning. It implements the Resonant State Alignment Algorithm (RSAA) to synchronize internal states between hierarchical levels, allowing the model to 'think' before committing to an output. It features spectral radius control, leak rates, and spike thresholds.",
      link: "/docs/resonant-gser",
      code: `from gpbacay_arcane.layers import ResonantGSER

layer = ResonantGSER(
    units=128,
    resonance_factor=0.2,
    spike_threshold=0.35,
    resonance_cycles=5
)`
    },
    {
      id: "bioplastic-dense",
      name: "BioplasticDenseLayer",
      description: "Implements Hebbian learning ('neurons that fire together, wire together') and homeostatic plasticity.",
      details: "This layer mimics biological synaptic adaptation by strengthening connections between co-active neurons. It incorporates homeostatic regulation to maintain stable activity levels, preventing runaway excitation or neural silence through synaptic scaling.",
      code: `from gpbacay_arcane.layers import BioplasticDenseLayer

layer = BioplasticDenseLayer(
    units=64,
    target_avg=0.12,
    homeostatic_rate=5e-5,
    learning_rate=1e-3
)`
    },
    {
      id: "gser",
      name: "GSER",
      description: "Gated Spiking Elastic Reservoir with dynamic structural adaptation (neurogenesis and pruning).",
      details: "The Gated Spiking Elastic Reservoir (GSER) supports dynamic reservoir sizing. During training, the layer can grow new neurons (neurogenesis) or prune weak connections and inactive neurons (apoptosis) based on performance metrics, allowing the architecture to evolve with the data.",
      code: `from gpbacay_arcane.layers import GSER

layer = GSER(
    units=256,
    spectral_radius=0.95,
    leak_rate=0.1,
    use_neurogenesis=True
)`
    },
    {
      id: "latent-temporal-coherence",
      name: "LatentTemporalCoherence",
      description: "Distills temporal dynamics into coherence vectors for stable sequence processing.",
      details: "This layer focuses on the temporal stability of semantic representations. It ensures that the latent space evolves smoothly over time, reducing noise and capturing the long-term contextual essence of sequential information.",
      code: `from gpbacay_arcane.layers import LatentTemporalCoherence

layer = LatentTemporalCoherence(
    coherence_factor=0.1,
    temporal_window=10
)`
    },
    {
      id: "relational-concept",
      name: "RelationalConceptModeling",
      description: "Multi-head attention mechanism for high-level semantic concept extraction.",
      details: "Uses specialized attention heads to identify and model relationships between different semantic concepts in the latent space, fostering a more structured and interpretable semantic hierarchy.",
      code: `from gpbacay_arcane.layers import RelationalConceptModeling

layer = RelationalConceptModeling(
    num_heads=8,
    key_dim=64
)`
    },
    {
      id: "neuromimetic-activations",
      name: "Neuromimetic Activations",
      description: "Stateful, adaptive activation functions including Resonant Spiking and Homeostatic GELU.",
      details: "Moving beyond static mappings, these activations maintain a memory of their potential state and self-regulate their gain. They facilitate granular, per-neuron Inference-Time Learning.",
      link: "/docs/activations",
      code: `from gpbacay_arcane.activations import NeuromimeticActivation

# Use Resonant Spike activation
activation = NeuromimeticActivation(
    activation_type='resonant_spike',
    resonance_factor=0.3
)`
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

        <Accordion type="single" collapsible className="w-full">
          {layers.map((layer) => (
            <AccordionItem key={layer.id} value={layer.id} className="border-zinc-800">
              <AccordionTrigger className="hover:no-underline py-4 text-left group">
                <div className="flex flex-col gap-1">
                  <h3 
                    id={layer.id}
                    className="text-lg font-bold text-zinc-100 group-hover:text-purple-400 transition-colors m-0"
                  >
                    {layer.name}
                  </h3>
                  <span className="text-sm font-normal text-zinc-400 line-clamp-1">
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
                        className="inline-flex items-center text-sm font-medium text-purple-400 hover:text-purple-300 transition-colors"
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
