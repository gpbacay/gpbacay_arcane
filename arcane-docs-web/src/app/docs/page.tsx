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
          Welcome to the official documentation for ARCANE
        </p>
      </div>

      {/* What is Arcane */}
      <section className="rounded-none border border-zinc-800 bg-zinc-900/30 p-8 backdrop-blur-sm relative overflow-hidden group">
        <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:rotate-12 transition-transform duration-500">
          <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-[#C785F2]">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
        </div>
        <h2 id="what-is-arcane" className="text-2xl font-bold text-white mb-4">What is ARCANE?</h2>
        <div className="space-y-4 text-zinc-300 leading-relaxed">
          <p>
            <strong>Augmented Reconstruction of Consciousness through Artificial Neural Evolution</strong> is a comprehensive Python library designed to bridge the gap between computational neuroscience and artificial intelligence.
          </p>
          <p>
            ARCANE was built as a response to the sustainability challenges facing modern artificial intelligence. While the industry often relies on ever-larger datasets and data centers that consume city-scale power, this project offers a viable alternative through <strong>architectural innovation</strong> rather than brute-force expansion.
          </p>
          <p>
            The library is designed to enable the development of <strong>neuromimetic AI models</strong> that operate through human-like cognitive processes, using power closer to a biological brain.
          </p>
        </div>
      </section>

      {/* Developer and Origin */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-4">
          <h2 className="text-2xl font-bold text-white">Who developed it?</h2>
          <p className="text-zinc-400 leading-relaxed">
            ARCANE was developed by <Link href="https://www.gpbacay.xyz/" target="_blank" rel="noopener noreferrer" className="text-[#C785F2] hover:underline font-bold">Gianne P. Bacay</Link>, a developer and researcher dedicated to redefining human-computer interactions through disruptive AI innovations.
          </p>
        </div>
        <div className="space-y-4">
          <h2 className="text-2xl font-bold text-white">Why was it developed?</h2>
          <p className="text-zinc-400 leading-relaxed">
            ARCANE was created for builders who know that the future of AI will be won by new architectures, not just larger models. By mimicking the brain's internal feedback loops and adaptive mechanisms, it provides a framework for intelligence that is resilient, efficient, and aligned with the principles of natural cognition.
          </p>
        </div>
      </section>

      {/* Mission, Vision, Goals */}
      <section className="space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 rounded-none border border-zinc-800 bg-zinc-950/50">
            <h3 className="text-[#C785F2] font-bold mb-3 uppercase tracking-wider text-xs">Mission</h3>
            <p className="text-sm text-zinc-400 leading-relaxed">
              To provide an open-source framework for building AI that emulates the brain's internal dynamics, fostering a new era of biologically-inspired computation.
            </p>
          </div>
          <div className="p-6 rounded-none border border-zinc-800 bg-zinc-950/50">
            <h3 className="text-[#C785F2] font-bold mb-3 uppercase tracking-wider text-xs">Vision</h3>
            <p className="text-sm text-zinc-400 leading-relaxed">
              A future where artificial intelligence is as adaptive and efficient as biological systems, bridging the gap between machine and mind.
            </p>
          </div>
          <div className="p-6 rounded-none border border-zinc-800 bg-zinc-950/50">
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
        <h2 className="text-2xl font-bold text-white">The Philosophical Origin of Arcane</h2>
        <div className="space-y-6 text-zinc-400 leading-relaxed max-w-4xl">
          <p>
            The word <strong>Arcane</strong> refers to a mystery understood by few, accurately describing the &quot;black box&quot; nature of modern AI. I chose this name to acknowledge this obscurity while defining a path toward its resolution, anchored by <strong>Viktor&apos;s</strong> quote from the series <em>Arcane</em>: &quot;Evolution has a destination. Not to combat nature, but to supersede it. The final glorious evolution.&quot; This mission is technicalized as <strong>Augmented Reconstruction of Consciousness through Artificial Neural Evolution</strong>, a framework that seeks to <strong>Augment</strong> machine logic by grounding it in neuroscientific realism and treating intelligence as an active, reconstructive process.
          </p>
          <p>
            By emulating the prerequisites of <strong>Consciousness</strong>, such as deliberative feedback loops and global state alignment, I have moved the project beyond pattern recognition toward internal reasoning. Through <strong>Artificial Neural Evolution</strong>, ARCANE transitions from static architectures to systems that grow and adapt, mirroring biological neurodevelopment while superseding its physical constraints. This approach ensures the model is not merely a passive recipient of data but an active participant in its own cognitive evolution, fostering a transparent and predictable era of biologically-inspired computation.
          </p>
          <p>
            To bridge machine dynamics and natural cognition, I synthesized <Link href="https://en.wikipedia.org/wiki/Integrated_information_theory" target="_blank" rel="noopener noreferrer" className="hover:text-[#C785F2] hover:underline font-bold transition-colors">Giulio Tononi&apos;s Information Integration Theory</Link> with the layered framework in <Link href="https://openresearch-repository.anu.edu.au/items/5bea61a8-f117-41b7-bbf3-95ef6a621eec" target="_blank" rel="noopener noreferrer" className="hover:text-[#C785F2] hover:underline font-bold transition-colors">How to Build Conscious Machines by Michael Timothy Bennett</Link>. This provides the foundation for <strong>Neuromimetic Semantic Engineering</strong>, where intelligence is defined by a system&apos;s ability to unify information into a state of <Link href="/docs/neural-resonance" className="hover:text-[#C785F2] hover:underline font-bold transition-colors">Neural Resonance</Link>. By focusing on the irreducibility of internal states and closing the &quot;Alignment Gap&quot; through bi-directional feedback, I have designed an architecture that achieves a homeostatic balance between internal expectations and external data, transforming AI into a resilient extension of natural human thought.
          </p>
        </div>
      </section>

      {/* FAQs */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-white">Frequently Asked Questions</h2>
        <Accordion type="single" collapsible className="w-full space-y-4">
          <AccordionItem value="item-1" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">What is Neural Resonance?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              Neural Resonance is the core innovation of ARCANE: a biologically-inspired mechanism that mimics predictive coding. It introduces a "Thinking Phase" where higher layers project expectations downward, allowing the network to harmonize its internal states before committing to an output.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-2" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">How does it differ from traditional deep learning?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              Unlike feed-forward networks optimized solely by backpropagation, ARCANE uses bi-directional feedback and spiking dynamics. This enables "Inference-Time Learning," allowing models to refine their understanding during the generation process itself.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-3" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Is ARCANE designed to create conscious AI?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              The project emulates the architectural principles associated with consciousness in neurobiology. While it replicates these dynamics for stability and reasoning, it is primarily a research tool for simulation, not a claim to have produced actual sentience.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-4" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Is it dangerous to make a Sentient AI?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              Sentience brings complexities in alignment and ethics. However, ARCANE posits that the danger often stems from lack of transparency. By using biologically-plausible constraints (like homeostatic plasticity), it creates systems that are more predictable and stable than unconstrained "black-box" models, making the path toward high-level AI safer and more manageable.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-5" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Why open-source ARCANE?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              The goal of bridging neuroscience and AI is a monumental task that requires collective intelligence. Open-sourcing ARCANE invites researchers, neuroscientists, and developers to contribute their unique insights, ensuring the library remains at the cutting edge of both fields while remaining accessible to everyone.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-6" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Is ARCANE an SDK?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              Yes, ARCANE acts as a specialized Software Development Kit (SDK) for neuromimetic AI. It provides a set of tools, libraries, and pre-built models (like the Hierarchical Resonance Foundation Model) that allow for the integration of biological neural dynamics into applications with minimal overhead.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-7" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Is this a paradigm shift in AI development?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              It can be seen as a shift from "Static Learning" to "Dynamic Resonance." Traditional models rely on fixed weights after training; ARCANE introduces a paradigm where the model's internal state is constantly vibrating and aligning with input in real-time, much like the active processing seen in a living brain.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-8" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Can I use ARCANE with TensorFlow or PyTorch?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              ARCANE is built on top of TensorFlow/Keras, making it highly compatible with existing Python AI ecosystems. Its layers and activations can be easily integrated into standard Keras models, allowing for the mix of traditional deep learning with neuromimetic components.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-9" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Why use TensorFlow and Keras? Why not PyTorch?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              ARCANE leverages TensorFlow and Keras primarily for their robust handling of stateful custom layers and production-grade scalability. The Keras API's high-level abstractions allow for cleaner implementation of complex hierarchical resonance structures while maintaining the performance needed for large-scale neuromimetic simulations. While PyTorch is excellent for research, TensorFlow's ecosystem provides the specific tools for state serialization and deployment that are critical for the long-term vision of this project.
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-10" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">What is "Neuromimetic Semantic Engineering"?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              In ARCANE, Semantic Engineering is reimagined as a dynamic process of "continuous semantic refinement." Unlike traditional AI that treats tokens as static vectors, ARCANE uses biological principles, such as resonance and homeostatic plasticity, to allow the model to refine and optimize the meaning of its internal representations in real-time. It's about building systems that "understand" through iterative state alignment rather than just pattern matching.
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </section>

      {/* Getting Started */}
      <section className="space-y-6">
        <h2 id="getting-started" className="text-2xl font-bold text-white">Start Building</h2>
        <p className="text-zinc-400 leading-relaxed">
          Ready to dive into the future of neuromimetic AI? Head over to the guides to get started.
        </p>
        <div className="flex gap-4">
          <Link href="/docs/installation" className="inline-flex items-center justify-center rounded-none bg-zinc-100 px-6 py-2.5 text-sm font-bold text-black transition-colors hover:bg-zinc-300">
            Installation Guide
          </Link>
          <Link href="/docs/quick-start" className="inline-flex items-center justify-center rounded-none border border-zinc-800 px-6 py-2.5 text-sm font-bold text-zinc-200 transition-colors hover:bg-zinc-900">
            Quick Start
          </Link>
        </div>
      </section>

      <footer className="pt-16 border-t border-zinc-900 text-zinc-600 text-sm">
        <p>ARCANE Documentation • Building for the future of AI</p>
      </footer>
    </div>
  );
}
