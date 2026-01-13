import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Markdown } from "@/components/markdown";

export default function ActivationsPage() {
  const activations = [
    {
      id: "resonant-spike",
      name: "Resonant Spiking Activation (RSA)",
      description: "Mimics a biological neuron's membrane potential with top-down modulation and leaky integration.",
      details: (
        <ul className="list-disc pl-5 space-y-2 m-0">
          <li><strong>LIF Modeling</strong>: Implements the Leaky Integrate-and-Fire mechanism for biological realism.</li>
          <li><strong>Temporal Integration</strong>: Accumulates stimulus over time, maintaining a membrane potential state.</li>
          <li><strong>Resonant Modulation</strong>: Amplifies signals that align with top-down hierarchical expectations.</li>
          <li><strong>Sparse Communication</strong>: Only fires discrete spikes when internal potential exceeds a threshold.</li>
        </ul>
      ),
      code: `from gpbacay_arcane.activations import resonant_spike

# Current potential (state) is updated at each step
spikes, new_state = resonant_spike(
    inputs, 
    current_state, 
    threshold=0.5, 
    leak_rate=0.1, 
    resonance_factor=0.2
)`
    },
    {
      id: "homeostatic-gelu",
      name: "Homeostatic GELU (h-GELU)",
      description: "Self-regulates sensitivity based on historical activity to prevent saturation.",
      details: (
        <ul className="list-disc pl-5 space-y-2 m-0">
          <li><strong>Activity Tracking</strong>: Maintains a moving average of historical firing rates.</li>
          <li><strong>Dynamic Gain Control</strong>: Automatically adjusts sensitivity to keep activity in an optimal range.</li>
          <li><strong>Saturation Prevention</strong>: Prevents runaway excitation and the 'dead neuron' problem.</li>
          <li><strong>Entropy Maximization</strong>: Ensures the semantic space remains highly informative.</li>
        </ul>
      ),
      code: `from gpbacay_arcane.activations import homeostatic_gelu

# Gain is adjusted based on activity history
activated = homeostatic_gelu(
    inputs, 
    moving_average_activity, 
    target_activity=0.12,
    adaptation_rate=0.01
)`
    },
    {
      id: "adaptive-softplus",
      name: "Adaptive Softplus",
      description: "A smooth activation with a tunable saturation threshold for biological firing rate mimicry.",
      details: (
        <ul className="list-disc pl-5 space-y-2 m-0">
          <li><strong>Differentiable Spiking</strong>: Provides a smooth approximation of discrete spiking behavior.</li>
          <li><strong>Tunable Sharpness</strong>: Control the 'hardness' of the activation threshold.</li>
          <li><strong>Stable Gradients</strong>: Maintains strong gradient flow for deep network optimization.</li>
          <li><strong>Biological Mimicry</strong>: Closely follows the non-linear response curves of real neurons.</li>
        </ul>
      ),
      code: `from gpbacay_arcane.activations import adaptive_softplus

activated = adaptive_softplus(
    inputs, 
    threshold=1.0, 
    sharpness=1.0
)`
    },
    {
      id: "neuromimetic-wrapper",
      name: "NeuromimeticActivation Layer",
      description: "Keras-compatible layer wrapper for stateful neuromimetic activations.",
      details: (
        <ul className="list-disc pl-5 space-y-2 m-0">
          <li><strong>State Management</strong>: Automatically handles the persistence of membrane potentials.</li>
          <li><strong>Seamless Integration</strong>: Can be used as a drop-in replacement in Keras/TensorFlow models.</li>
          <li><strong>Configurable Dynamics</strong>: Support for both spiking and homeostatic activation types.</li>
          <li><strong>Hierarchical Support</strong>: Designed to work with ARCANE's resonance cycles.</li>
        </ul>
      ),
      code: `from gpbacay_arcane.activations import NeuromimeticActivation

# Add a stateful spiking activation to your model
model.add(NeuromimeticActivation(
    activation_type='resonant_spike',
    threshold=0.5,
    leak_rate=0.1
))`
    }
  ];

  const resonantMath = `
#### 1. Resonant Spiking Dynamics
Grounding in the Leaky Integrate-and-Fire (LIF) model with hierarchical resonance.

The internal potential $V$ at time $t$ integrates incoming stimulus $I(t)$ and decayed previous state:
$$V_{integrated}(t) = I(t) + V(t-1) \\cdot (1 - \\lambda)$$

**Variables:**
*   $I(t)$: Incoming stimulus (bottom-up input).
*   $\\lambda$: \`leak_rate\` (passive decay of membrane potential).
*   $\\rho$: Hierarchical resonance factor (top-down modulation).
*   $\\theta$: Firing threshold.

**State Evolution:**
$$V_{res}(t) = V_{integrated}(t) \\cdot (1 + \\rho)$$
$$S(t) = \\mathbb{I}(V_{res}(t) > \\theta)$$
$$V(t) = V_{res}(t) - S(t) \\cdot \\theta$$
  `;

  const homeostaticMath = `
#### 2. Homeostatic Activity Scaling
Synaptic scaling mechanism to maintain firing rates in optimal ranges.

The gain $G(t)$ is regulated by the deviation from the target activity $A_{target}$:
$$G(t) = 1 + \\eta \\cdot (A_{target} - \\bar{A})$$

**Parameters:**
*   $\\eta$: \`adaptation_rate\` (speed of homeostatic correction).
*   $A_{target}$: Target firing rate (e.g., 0.12).
*   $\\bar{A}$: Moving average of historical activity.
*   $\\alpha$: Smoothing factor for activity tracking.

**Activity Update:**
$$\\bar{A}(t) = (1 - \\alpha) \\cdot \\bar{A}(t-1) + \\alpha \\cdot A(t)$$
$$y = \\text{GELU}(x \\cdot G(t))$$
  `;

  const softplusMath = `
#### 3. Adaptive Softplus (Biological Approximation)
A smooth, differentiable activation for biological firing rate mimicry.

**Core Formula:**
$$f(x, \\theta, s) = \\frac{\\ln(1 + \\exp(s \\cdot (x - \\theta)))}{s}$$

**Benefits:**
*   **Sparsity**: Naturally suppresses low-intensity noise.
*   **Optimization**: Provides non-zero gradients even in the 'off' state.
*   **Flexibility**: Sharpness $s$ can be tuned for different semantic granularities.
  `;

  const stateEvolutionMath = `
#### 4. Inference-Time State Evolution
Neuromimetic activations shift from static mappings to dynamic state transitions.

ARCANE activations are stateful and contextual:
$$(y, V_t) = f(x, V_{t-1}, \\text{context})$$

**Key Differences:**
*   **Memory**: Standard activations are stateless ($y = f(x)$).
*   **Adaptation**: ARCANE activations refine their response during the 'Thinking Phase'.
*   **Resonance**: Responses are modulated by global semantic alignment.
  `;

  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Neuromimetic Activations
        </h1>
        <p className="text-xl text-zinc-400">
          Stateful and adaptive activation functions inspired by biological neural dynamics.
        </p>
      </div>

      <div className="space-y-8 text-zinc-300 leading-7">
        <p>
          In the ARCANE framework, activation functions are not merely static mathematical non-linearities. 
          They incorporate <strong>Inference-Time State Adaptation</strong>, <strong>Spiking Dynamics</strong>, 
          and <strong>Homeostatic Regulation</strong> to mirror the adaptive nature of biological neurons.
        </p>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Available Activations
        </h2>

        <Accordion type="single" collapsible className="w-full">
          {activations.map((activation) => (
            <AccordionItem key={activation.id} value={activation.id} className="border-zinc-800">
              <AccordionTrigger className="hover:no-underline py-4 text-left group">
                <div className="flex flex-col gap-1">
                  <h3 
                    id={activation.id}
                    className="text-lg font-bold text-zinc-100 group-hover:text-purple-400 transition-colors m-0"
                  >
                    {activation.name}
                  </h3>
                  <span className="text-sm font-normal text-zinc-400 line-clamp-1">
                    {activation.description}
                  </span>
                </div>
              </AccordionTrigger>
              <AccordionContent className="text-zinc-300 pb-6">
                <div className="space-y-4">
                  <div className="leading-relaxed m-0 text-zinc-300">
                    {activation.details}
                  </div>
                  
                  {activation.code && (
                    <div className="space-y-2">
                      <p className="text-xs font-bold uppercase tracking-wider text-zinc-500 mt-4">Quick Implementation</p>
                      <pre className="overflow-x-auto rounded-none border border-zinc-800 bg-zinc-950 p-4 text-sm text-zinc-300 shadow-inner">
                        <code>{activation.code}</code>
                      </pre>
                    </div>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Mathematical Foundation
        </h2>
        
        <div className="space-y-12">
          <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-none px-8 py-4">
            <Markdown content={resonantMath} />
          </div>

          <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-none px-8 py-4">
            <Markdown content={homeostaticMath} />
          </div>

          <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-none px-8 py-4">
            <Markdown content={softplusMath} />
          </div>

          <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-none px-8 py-4">
            <Markdown content={stateEvolutionMath} />
          </div>
        </div>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          The Deliberative Advantage
        </h2>
        <div className="grid gap-6 md:grid-cols-3 pb-10">
          <div className="bg-zinc-900/50 border border-zinc-800 p-6 rounded-none">
            <h4 className="text-zinc-100 font-bold mb-3 flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-none bg-purple-500"></span>
              Temporal Coherence
            </h4>
            <p className="text-sm text-zinc-400 leading-relaxed">
              Maintains internal state over time, allowing the model to naturally filter stimulus noise and integrate evidence across temporal windows.
            </p>
          </div>
          <div className="bg-zinc-900/50 border border-zinc-800 p-6 rounded-none">
            <h4 className="text-zinc-100 font-bold mb-3 flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-none bg-purple-500"></span>
              Active Thresholding
            </h4>
            <p className="text-sm text-zinc-400 leading-relaxed">
              Suppresses ambiguous or low-confidence signals until hierarchical resonance provides enough gain to exceed the deliberative threshold.
            </p>
          </div>
          <div className="bg-zinc-900/50 border border-zinc-800 p-6 rounded-none">
            <h4 className="text-zinc-100 font-bold mb-3 flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-none bg-purple-500"></span>
              Structural Stability
            </h4>
            <p className="text-sm text-zinc-400 leading-relaxed">
              Homeostatic scaling ensures neurons operate in their most informative 'sweet spot', maximizing entropy and preventing activation collapse.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

