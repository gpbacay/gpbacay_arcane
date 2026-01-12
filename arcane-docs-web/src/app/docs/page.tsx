import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import Link from "next/link";

export default function DocsPage() {
  return (
    <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="space-y-4">
        <h1 className="text-4xl font-extrabold tracking-tight text-white mb-2">Introduction</h1>
        <p className="text-xl text-zinc-400 font-medium leading-relaxed">
          Welcome to the official documentation for A.R.C.A.N.E.
        </p>
      </div>

      {/* What is Arcane */}
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-8 backdrop-blur-sm relative overflow-hidden group">
        <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:rotate-12 transition-transform duration-500">
          <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-[#C785F2]">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
        </div>
        <h2 id="what-is-arcane" className="text-2xl font-bold text-white mb-4">What is A.R.C.A.N.E.?</h2>
        <div className="space-y-4 text-zinc-300 leading-relaxed">
          <p>
            <strong>Augmented Reconstruction of Consciousness through Artificial Neural Evolution</strong> is a comprehensive Python library designed to bridge the gap between computational neuroscience and artificial intelligence.
          </p>
          <p>
            A.R.C.A.N.E. enables developers and researchers to build, train, and deploy <strong>neuromimetic AI models</strong> that incorporate biological neural principles such as bi-directional state alignment, spiking dynamics, and homeostatic plasticity.
          </p>
        </div>
      </section>

      {/* Developer and Origin */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-4">
          <h2 className="text-2xl font-bold text-white">Who developed it?</h2>
          <p className="text-zinc-400 leading-relaxed">
            A.R.C.A.N.E. was developed by <Link href="https://www.gpbacay.xyz/" target="_blank" rel="noopener noreferrer" className="text-[#C785F2] hover:underline font-bold">Gianne P. Bacay</Link>, a developer and researcher dedicated to redefining human-computer interactions through disruptive AI innovations.
          </p>
        </div>
        <div className="space-y-4">
          <h2 className="text-2xl font-bold text-white">Why was it developed?</h2>
          <p className="text-zinc-400 leading-relaxed">
            It was created to address the limitations of traditional "black-box" neural networks. By mimicking the brain's internal feedback loops and adaptive mechanisms, Arcane aims to create AI that is more resilient, efficient, and ultimately more aligned with the principles of natural intelligence.
          </p>
        </div>
      </section>

      {/* Mission, Vision, Goals */}
      <section className="space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 rounded-xl border border-zinc-800 bg-zinc-950/50">
            <h3 className="text-[#C785F2] font-bold mb-3 uppercase tracking-wider text-xs">Mission</h3>
            <p className="text-sm text-zinc-400 leading-relaxed">
              To provide an open-source framework for building AI that emulates the brain's internal dynamics, fostering a new era of biologically-inspired computation.
            </p>
          </div>
          <div className="p-6 rounded-xl border border-zinc-800 bg-zinc-950/50">
            <h3 className="text-[#C785F2] font-bold mb-3 uppercase tracking-wider text-xs">Vision</h3>
            <p className="text-sm text-zinc-400 leading-relaxed">
              A future where artificial intelligence is as adaptive and efficient as biological systems, bridging the gap between machine and mind.
            </p>
          </div>
          <div className="p-6 rounded-xl border border-zinc-800 bg-zinc-950/50">
            <h3 className="text-[#C785F2] font-bold mb-3 uppercase tracking-wider text-xs">Goals</h3>
            <ul className="text-sm text-zinc-400 space-y-2 list-disc list-inside">
              <li>Implement biologically-plausible neural layers.</li>
              <li>Enable real-time inference state alignment.</li>
              <li>Advance research in neuromorphic computing.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Philosophy */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-white">The Philosophy of Arcane Systems</h2>
        <div className="space-y-6 text-zinc-400 leading-relaxed max-w-4xl">
          <p>
            A.R.C.A.N.E. is philosophically grounded in the intersection of cognitive psychology and neuro-evolutionary theory, drawing significant inspiration from <Link href="https://en.wikipedia.org/wiki/Information_integration_theory" target="_blank" rel="noopener noreferrer" className="text-[#C785F2] hover:underline font-bold">Norman H. Anderson’s Information Integration Theory (IIT)</Link>. It posits that intelligence emerges not from isolated data points, but from the systematic integration of multiple stimuli through a weighted, hierarchical framework. This is further synthesized with <Link href="https://arxiv.org/abs/2409.14545" target="_blank" rel="noopener noreferrer" className="text-[#C785F2] hover:underline font-bold">Michael Timothy Bennett’s layered theory of consciousness</Link>, which provides the structural blueprint for Arcane’s multi-level neural architecture. By emulating these biological hierarchies, the system moves beyond simple pattern recognition toward a structured reconstruction of internal cognitive states.
          </p>
          <p>
            At its core, the project champions the principle of <strong>Neuromimetic Semantic Engineering</strong>, where intelligence is viewed as an active process of <Link href="https://en.wikipedia.org/wiki/Self-model" target="_blank" rel="noopener noreferrer" className="text-[#C785F2] hover:underline font-bold">self-modeling</Link> rather than a static computational mapping. Grounded in research on self-modeling systems, A.R.C.A.N.E. treats its own internal neural dynamics as a plastic medium that can be refined and optimized in real-time. This philosophy shifts the focus from "training a black box" to "engineering a semantic space," where every neural activation contributes to a coherent, evolving model of the world that is intrinsically linked to the system’s own structural plasticity and homeostatic regulation.
          </p>
          <p>
            Central to this paradigm is the commitment to closing the <Link href="https://en.wikipedia.org/wiki/AI_alignment" target="_blank" rel="noopener noreferrer" className="text-[#C785F2] hover:underline font-bold">"Alignment Gap"</Link> through <Link href="/docs/neural-resonance" className="text-[#C785F2] hover:underline font-bold">Bidirectional Neural Resonance</Link> and <strong>Inference-time Learning</strong>. Arcane rejects the traditional AI reliance on unidirectional, memoryless feed-forward passes. Instead, it implements a deliberative "Thinking Phase" where information resonates between higher and lower layers, allowing for real-time state adaptation before an output is ever committed. This bidirectional flow ensures that global semantic expectations and local sensory inputs achieve a state of mutual alignment, effectively bridging the gap between neuroscience and artificial intelligence through a dynamic, resonant feedback loop.
          </p>
        </div>
      </section>

      {/* FAQs */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-white">Frequently Asked Questions</h2>
        <Accordion type="single" collapsible className="w-full space-y-4">
          <AccordionItem value="item-1" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-xl">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">What is Neural Resonance?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              Neural Resonance is A.R.C.A.N.E.'s core innovation: a biologically-inspired mechanism that mimics predictive coding. It introduces a "Thinking Phase" where higher layers project expectations downward, allowing the network to harmonize its internal states before committing to an output.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-2" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-xl">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">How does it differ from traditional deep learning?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              Unlike feed-forward networks optimized solely by backpropagation, A.R.C.A.N.E. uses bi-directional feedback and spiking dynamics. This enables "Inference-Time Learning," allowing models to refine their understanding during the generation process itself.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-3" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-xl">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Is A.R.C.A.N.E. designed to create conscious AI?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              The project emulates the architectural principles associated with consciousness in neurobiology. While it replicates these dynamics for stability and reasoning, it is primarily a research tool for simulation, not a claim to have produced actual sentience.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-4" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-xl">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Is it dangerous to make a Sentient AI?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              Sentience brings complexities in alignment and ethics. However, A.R.C.A.N.E. posits that the danger often stems from lack of transparency. By using biologically-plausible constraints (like homeostatic plasticity), we create systems that are more predictable and stable than unconstrained "black-box" models, making the path toward high-level AI safer and more manageable.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-5" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-xl">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Why open-source A.R.C.A.N.E.?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              The goal of bridging neuroscience and AI is a monumental task that requires collective intelligence. We open-sourced A.R.C.A.N.E. to invite researchers, neuroscientists, and developers to contribute their unique insights, ensuring the library remains at the cutting edge of both fields while remaining accessible to everyone.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-6" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-xl">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Is A.R.C.A.N.E. an SDK?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              Yes, A.R.C.A.N.E. acts as a specialized Software Development Kit (SDK) for neuromimetic AI. It provides a set of tools, libraries, and pre-built models (like the Hierarchical Resonance Foundation Model) that allow you to integrate biological neural dynamics into your own applications with minimal overhead.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-7" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-xl">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Is this a paradigm shift in AI development?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              We see it as a shift from "Static Learning" to "Dynamic Resonance." Traditional models rely on fixed weights after training; A.R.C.A.N.E. introduces a paradigm where the model's internal state is constantly vibrating and aligning with input in real-time, much like the active processing seen in a living brain.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-8" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-xl">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Can I use A.R.C.A.N.E. with TensorFlow or PyTorch?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              A.R.C.A.N.E. is built on top of TensorFlow/Keras, making it highly compatible with existing Python AI ecosystems. Its layers and activations can be easily integrated into standard Keras models, allowing you to mix traditional deep learning with neuromimetic components.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-9" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-xl">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Why use TensorFlow and Keras? Why not PyTorch?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              A.R.C.A.N.E. leverages TensorFlow and Keras primarily for their robust handling of stateful custom layers and production-grade scalability. The Keras API's high-level abstractions allow for cleaner implementation of complex hierarchical resonance structures while maintaining the performance needed for large-scale neuromimetic simulations. While PyTorch is excellent for research, TensorFlow's ecosystem provides the specific tools for state serialization and deployment that are critical for Arcane's long-term vision.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-10" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-xl">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">What is "Neuromimetic Semantic Engineering"?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              In A.R.C.A.N.E., Semantic Engineering is reimagined as a dynamic process of "continuous semantic refinement." Unlike traditional AI that treats tokens as static vectors, Neuromimetic Semantic Engineering uses biological principles, such as resonance and homeostatic plasticity, to allow the model to refine and optimize the meaning of its internal representations in real-time. It's about building systems that "understand" through iterative state alignment rather than just pattern matching.
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </section>

      {/* Getting Started */}
      <section className="space-y-6">
        <h2 id="getting-started" className="text-2xl font-bold text-white">Start Building</h2>
        <p className="text-zinc-400 leading-relaxed">
          Ready to dive into the future of neuromimetic AI? Head over to our guides to get started.
        </p>
        <div className="flex gap-4">
          <Link href="/docs/installation" className="inline-flex items-center justify-center rounded-lg bg-zinc-100 px-6 py-2.5 text-sm font-bold text-black transition-colors hover:bg-zinc-300">
            Installation Guide
          </Link>
          <Link href="/docs/quick-start" className="inline-flex items-center justify-center rounded-lg border border-zinc-800 px-6 py-2.5 text-sm font-bold text-zinc-200 transition-colors hover:bg-zinc-900">
            Quick Start
          </Link>
        </div>
      </section>

      <footer className="pt-16 border-t border-zinc-900 text-zinc-600 text-sm">
        <p>A.R.C.A.N.E. Documentation • Built for the future of AI</p>
      </footer>
    </div>
  );
}
