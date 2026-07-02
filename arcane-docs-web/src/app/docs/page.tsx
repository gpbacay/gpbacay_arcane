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

      {/* When to use Arcane */}
      <section className="rounded-none border border-zinc-800 bg-zinc-900/30 p-8 backdrop-blur-sm relative overflow-hidden group">
        <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:rotate-12 transition-transform duration-500">
          <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-[#C785F2]">
            <path d="M9 12l2 2 4-4M21 12c0 4.97-4.03 9-9 9s-9-4.03-9-9 4.03-9 9-9 9 4.03 9 9z" />
          </svg>
        </div>
        <h2 id="when-to-use-arcane" className="text-2xl font-bold text-white mb-4">When to use ARCANE?</h2>
        <div className="space-y-4 text-zinc-300 leading-relaxed">
          <p>
            ARCANE is designed for researchers, developers, and organizations who want to push the boundaries of AI through biological inspiration rather than brute computational force.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h3 className="text-[#C785F2] font-semibold text-lg">Perfect For:</h3>
              <ul className="space-y-2 list-disc list-inside text-sm">
                <li><strong>Research Applications</strong>: Exploring biologically-plausible neural mechanisms</li>
                <li><strong>Stable Training</strong>: When you need more predictable and interpretable AI behavior</li>
                <li><strong>Energy Efficiency</strong>: Applications where power consumption matters</li>
                <li><strong>Complex Reasoning</strong>: Tasks requiring hierarchical information processing</li>
                <li><strong>Neuroscience Integration</strong>: Bridging computational neuroscience and AI</li>
              </ul>
            </div>
            <div className="space-y-3">
              <h3 className="text-[#B9DFE0] font-semibold text-lg">Long-Term Goals:</h3>
              <p className="text-xs text-zinc-500 leading-relaxed">
                ARCANE is built toward these applications. The library is a proof of concept today—early adopters can explore the architecture while production-scale capabilities are still maturing.
              </p>
              <ul className="space-y-2 list-disc list-inside text-sm">
                <li><strong>Real-time Applications</strong>: Low-latency inference for speed-critical workloads</li>
                <li><strong>Large-scale Production</strong>: High-throughput deployment at scale</li>
                <li><strong>Efficient Classification</strong>: Fast, lightweight pattern recognition without brute-force compute</li>
                <li><strong>Resource-Constrained Edge</strong>: Running on limited hardware with brain-like efficiency</li>
              </ul>
            </div>
          </div>
          <p className="text-zinc-400 italic">
            Choose ARCANE when you want to help shape biologically inspired AI—from research and prototyping today toward real-time, efficient, production-ready systems tomorrow.
          </p>
        </div>
      </section>

      {/* Developer and Origin */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-4">
          <h2 id="development-team" className="text-2xl font-bold text-white">Who developed it?</h2>
          <p className="text-zinc-400 leading-relaxed">
            ARCANE is led by <Link href="https://personal-portfolio-2025-delta.vercel.app/" target="_blank" rel="noopener noreferrer" className="text-[#C785F2] hover:underline font-bold">Gianne P. Bacay</Link>, a developer and researcher focused on neuromimetic AI and human–computer interaction. The project began in 2024 as independent research and is currently maintained as a solo-led open-source effort, with contributions welcome from the broader community.
          </p>
        </div>
        <div className="space-y-4">
          <h2 id="why-developed" className="text-2xl font-bold text-white">Why was it developed?</h2>
          <p className="text-zinc-400 leading-relaxed">
            ARCANE was created for builders who know that the future of AI will be won by new architectures, not just larger models. By mimicking the brain&apos;s internal feedback loops and adaptive mechanisms, it provides a framework for intelligence that is resilient, efficient, and aligned with the principles of natural cognition.
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
            Why Arcane? Where did it all began? Guided by the natural rules of thought, I see that the current world of huge and hidden AI models has become a territory of secrets understood by only a few. I chose the name Arcane to define a move away from this state of mystery, building a path toward clear and smart systems through <strong>Augmented Reconstruction of Consciousness through Artificial Neural Evolution</strong>. This framework seeks to improve machine logic by using real ideas from neural science, treating intelligence as an active process rather than just a way to match data.
          </p>
          <p>
            Drawing from the principles of <Link href="https://en.wikipedia.org/wiki/Integrated_information_theory" target="_blank" rel="noopener noreferrer" className="hover:text-[#C785F2] hover:underline font-bold transition-colors">Giulio Tononi&apos;s Information Integration Theory</Link>, I believe the depth of intelligence is found in how well a system fits information together. I used these ideas to build the foundation of <strong>Neuromimetic Semantic Engineering</strong>, ensuring that every internal state exists as a single, unified whole. This approach moves the architecture beyond looking for simple patterns and toward a state where all parts work together, mirroring the way biological minds turn sensory data into a clear picture of the world.
          </p>
          <p>
            The layered theory of consciousness described in <Link href="https://openresearch-repository.anu.edu.au/items/5bea61a8-f117-41b7-bbf3-95ef6a621eec" target="_blank" rel="noopener noreferrer" className="hover:text-[#C785F2] hover:underline font-bold transition-colors">How to Build Conscious Machines by Michael Timothy Bennett</Link> led me to design the architecture to reach a state of <Link href="/docs/neural-resonance" className="hover:text-[#C785F2] hover:underline font-bold transition-colors">Neural Resonance</Link>. This is a perfect balance where what the model expects and what it sees from the world are in bi-directional alignment. By closing the &quot;Alignment Gap&quot; through active feedback, ARCANE creates a framework where models grow and adapt through <strong>Artificial Neural Evolution</strong>. This represents the final goal of my work, transforming artificial intelligence into a clear and strong extension of natural human thought.
          </p>
        </div>
      </section>

      {/* Research & Project Questions */}
      <section id="research-questions" className="space-y-8">
        <div className="space-y-2">
          <h2 className="text-2xl font-bold text-white">Research &amp; Project Questions</h2>
          <p className="text-zinc-400 leading-relaxed max-w-3xl">
            Answers to common questions about the people, science, development history, and real-world direction of ARCANE.
          </p>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-bold uppercase tracking-widest text-[#C785F2]">Part I — Origins &amp; Ownership</h3>
          <Accordion type="single" collapsible className="w-full space-y-4">
            <AccordionItem value="rq-1" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
              <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Who is on the development team, and what are their qualifications?</AccordionTrigger>
              <AccordionContent className="text-zinc-400 leading-relaxed space-y-3">
                <p>
                  ARCANE is currently led by a solo research and engineering team headed by <Link href="https://personal-portfolio-2025-delta.vercel.app/" target="_blank" rel="noopener noreferrer" className="text-[#C785F2] hover:underline font-bold">Gianne P. Bacay</Link>, who serves as the primary inventor, developer, and researcher behind the architecture.
                </p>
                <p>
                  Gianne brings hands-on experience in AI systems development, computational neuroscience-inspired modeling, and open-source software engineering. The project is built as both a research framework and a production-oriented Python SDK, combining theoretical work (such as the Resonant State Alignment Algorithm) with practical library implementation in TensorFlow/Keras.
                </p>
                <p>
                  External collaborators and open-source contributors are welcome through the <Link href="https://github.com/gpbacay/gpbacay_arcane" target="_blank" rel="noopener noreferrer" className="text-[#C785F2] hover:underline font-bold">GitHub repository</Link>, but the core invention and direction remain researcher-led at this stage.
                </p>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="rq-2" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
              <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Where was the invention developed?</AccordionTrigger>
              <AccordionContent className="text-zinc-400 leading-relaxed space-y-3">
                <p>
                  ARCANE was developed as an independent research project, built and iterated outside of a formal corporate or university laboratory. Development took place in a personal research and engineering environment, with design, implementation, benchmarking, and documentation carried out directly by the project lead.
                </p>
                <p>
                  The library, examples, tests, and this documentation portal are maintained in the open-source repository at <Link href="https://github.com/gpbacay/gpbacay_arcane" target="_blank" rel="noopener noreferrer" className="text-[#C785F2] hover:underline font-bold">github.com/gpbacay/gpbacay_arcane</Link>, making the full development history publicly auditable.
                </p>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="rq-3" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
              <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Who funded this research, and who owns the intellectual property?</AccordionTrigger>
              <AccordionContent className="text-zinc-400 leading-relaxed space-y-3">
                <p>
                  ARCANE was developed through independent, self-directed research. There is no external grant funding, corporate sponsorship, or institutional backing disclosed for the project at this time.
                </p>
                <p>
                  Intellectual property is held by <strong className="text-zinc-300">Gianne P. Bacay</strong> (copyright holder: gpbacay) and is released under the <Link href="https://github.com/gpbacay/gpbacay_arcane/blob/main/LICENSE" target="_blank" rel="noopener noreferrer" className="text-[#C785F2] hover:underline font-bold">MIT License</Link>. This means anyone may use, modify, and distribute the software freely, provided the license and copyright notice are preserved.
                </p>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-bold uppercase tracking-widest text-[#C785F2]">Part II — Problem &amp; Technology</h3>
          <Accordion type="single" collapsible className="w-full space-y-4">
            <AccordionItem value="rq-4" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
              <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">What motivated you to invent ARCANE?</AccordionTrigger>
              <AccordionContent className="text-zinc-400 leading-relaxed space-y-3">
                <p>
                  Modern AI has become dominated by ever-larger models, opaque black boxes, and data-center-scale energy consumption. ARCANE was motivated by the belief that the next leap in intelligence will come from <strong className="text-zinc-300">architectural innovation</strong>—systems that think, adapt, and align the way biological brains do—not from brute-force scaling alone.
                </p>
                <p>
                  The name &quot;Arcane&quot; was chosen deliberately: to move away from hidden, inscrutable AI and toward transparent, biologically grounded systems built on principles of consciousness research and neural evolution.
                </p>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="rq-5" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
              <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">What specific problem does ARCANE solve?</AccordionTrigger>
              <AccordionContent className="text-zinc-400 leading-relaxed space-y-3">
                <p>
                  ARCANE addresses the <strong className="text-zinc-300">Alignment Gap</strong>: in traditional feed-forward networks, information flows in one direction, so lower layers cannot revise their understanding based on higher-level context before an output is committed.
                </p>
                <p>
                  Through the <Link href="/docs/research" className="text-[#C785F2] hover:underline font-bold">Resonant State Alignment Algorithm (RSAA)</Link> and <Link href="/docs/neural-resonance" className="text-[#C785F2] hover:underline font-bold">Neural Resonance</Link>, ARCANE lets hierarchical layers reach internal agreement <em>before</em> responding—enabling deliberative, inference-time state adaptation rather than reactive pattern matching alone.
                </p>
                <p>
                  It also targets the sustainability problem: building AI that can reason efficiently with brain-like sparsity and local learning rules, reducing reliance on massive compute for every task.
                </p>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="rq-6" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
              <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">What foundational scientific principles or technologies does it rely on?</AccordionTrigger>
              <AccordionContent className="text-zinc-400 leading-relaxed">
                <ul className="space-y-2 list-disc list-inside">
                  <li><strong className="text-zinc-300">Predictive Coding</strong> — top-down expectations matched against bottom-up sensory input</li>
                  <li><strong className="text-zinc-300">Hebbian Learning</strong> — synapses strengthen when neurons fire together</li>
                  <li><strong className="text-zinc-300">Homeostatic Plasticity</strong> — self-regulation to keep neural activity stable</li>
                  <li><strong className="text-zinc-300">Reservoir Computing</strong> — dynamic temporal processing via spiking reservoirs</li>
                  <li><strong className="text-zinc-300">Integrated Information Theory (IIT)</strong> — treating intelligence as unified information integration</li>
                  <li><strong className="text-zinc-300">Layered Consciousness Theory</strong> — hierarchical models of mind from consciousness research</li>
                  <li><strong className="text-zinc-300">TensorFlow / Keras</strong> — the underlying deep learning runtime and API</li>
                </ul>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="rq-7" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
              <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">What makes ARCANE unique compared to other existing solutions?</AccordionTrigger>
              <AccordionContent className="text-zinc-400 leading-relaxed space-y-3">
                <p>
                  Most AI frameworks optimize static weight matrices through backpropagation alone. ARCANE combines backpropagation with biologically inspired mechanisms that most libraries do not offer out of the box:
                </p>
                <ul className="space-y-2 list-disc list-inside">
                  <li>Bi-directional <strong className="text-zinc-300">Neural Resonance</strong> and a formal &quot;Thinking Phase&quot;</li>
                  <li><strong className="text-zinc-300">Inference-Time Learning</strong> — refining internal states during generation, not only during training</li>
                  <li><strong className="text-zinc-300">Spiking dynamics</strong>, Hebbian plasticity, and homeostatic self-regulation in standard Keras layers</li>
                  <li>A substrate-agnostic <strong className="text-zinc-300">RSAA</strong> designed to close the Alignment Gap mathematically</li>
                </ul>
                <p>
                  See the <Link href="/docs/benchmarks" className="text-[#C785F2] hover:underline font-bold">benchmarks</Link> for early evidence of improved stability and accuracy over traditional LSTM baselines on semantic tasks.
                </p>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="rq-8" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
              <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">What are the technical limitations or constraints of the current model?</AccordionTrigger>
              <AccordionContent className="text-zinc-400 leading-relaxed space-y-3">
                <p>
                  ARCANE is a <strong className="text-zinc-300">proof of concept</strong> today. The architecture and research direction are mature, but production-scale performance is still being hardened.
                </p>
                <ul className="space-y-2 list-disc list-inside">
                  <li>Built on <strong className="text-zinc-300">TensorFlow/Keras only</strong> — no native PyTorch support yet</li>
                  <li>Resonance cycles add compute during the Thinking Phase, so training can take longer than simpler baselines</li>
                  <li>Large-scale deployment, edge optimization, and real-time latency targets are <strong className="text-zinc-300">goals in progress</strong>, not fully realized capabilities</li>
                  <li>Benchmarks so far focus on research-scale datasets (e.g., Tiny Shakespeare, MNIST demos) rather than industry-scale workloads</li>
                  <li>The project emulates consciousness-related dynamics for research—it does not claim sentience</li>
                </ul>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-bold uppercase tracking-widest text-[#C785F2]">Part III — Development Journey &amp; Future</h3>
          <Accordion type="single" collapsible className="w-full space-y-4">
            <AccordionItem value="rq-9" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
              <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Were there breakthroughs, unexpected failures, or false paths during development?</AccordionTrigger>
              <AccordionContent className="text-zinc-400 leading-relaxed space-y-3">
                <p><strong className="text-zinc-300">Breakthroughs:</strong></p>
                <ul className="space-y-2 list-disc list-inside mb-3">
                  <li>Formalizing the <strong className="text-zinc-300">Resonant State Alignment Algorithm (RSAA)</strong> to close the Alignment Gap</li>
                  <li>Decoupling <strong className="text-zinc-300">state alignment from weight updates</strong>, enabling deliberation before output</li>
                  <li>Developing the <Link href="/docs/predictive-resonant-layer" className="text-[#C785F2] hover:underline font-bold">PredictiveResonantLayer</Link> for local, autonomous predictive coding</li>
                  <li>Demonstrating measurable gains in validation accuracy and training stability over traditional LSTMs</li>
                </ul>
                <p><strong className="text-zinc-300">False paths &amp; lessons:</strong></p>
                <ul className="space-y-2 list-disc list-inside">
                  <li>Early reliance on pure backpropagation alone could not fix the Alignment Gap—layers needed bi-directional feedback</li>
                  <li>Deep gradient chains caused instability; shifting toward <strong className="text-zinc-300">local Hebbian and homeostatic learning</strong> proved more stable and biologically faithful</li>
                  <li>Uniform activation patterns lacked the sparsity of real neurons; introducing <strong className="text-zinc-300">spiking dynamics</strong> improved efficiency and interpretability</li>
                </ul>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="rq-10" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
              <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">Are there broader societal or environmental applications for ARCANE?</AccordionTrigger>
              <AccordionContent className="text-zinc-400 leading-relaxed space-y-3">
                <p>Yes. Because ARCANE prioritizes brain-like efficiency and interpretability, it has implications beyond pure NLP research:</p>
                <ul className="space-y-2 list-disc list-inside">
                  <li><strong className="text-zinc-300">Environmental</strong> — lower power consumption through spiking, sparse activation compared to always-on dense models</li>
                  <li><strong className="text-zinc-300">Sustainable AI</strong> — an alternative to scaling intelligence only through larger data centers</li>
                  <li><strong className="text-zinc-300">Neuroscience &amp; education</strong> — a simulation platform for testing theories of predictive coding and consciousness</li>
                  <li><strong className="text-zinc-300">Robotics &amp; edge AI</strong> — long-term target for adaptive systems on resource-limited hardware</li>
                  <li><strong className="text-zinc-300">Safer AI alignment</strong> — more transparent, biologically constrained systems that behave predictably</li>
                </ul>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="rq-11" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
              <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">What are the next steps in the development or testing timeline?</AccordionTrigger>
              <AccordionContent className="text-zinc-400 leading-relaxed">
                <ul className="space-y-2 list-disc list-inside">
                  <li>Expand <Link href="/docs/benchmarks" className="text-[#C785F2] hover:underline font-bold">benchmarks</Link> across more datasets and task types</li>
                  <li>Harden real-time inference and reduce Thinking Phase latency for production workloads</li>
                  <li>Optimize for edge and resource-constrained deployment</li>
                  <li>Grow the open-source contributor base and publish further research (RSAA, resonance mechanisms)</li>
                  <li>Extend foundation models and integration paths (e.g., Ollama hybrid workflows)</li>
                  <li>Continue validating biological plausibility against neuroscience literature</li>
                </ul>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="rq-12" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
              <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">When can we realistically expect ARCANE on the market or in practical use?</AccordionTrigger>
              <AccordionContent className="text-zinc-400 leading-relaxed space-y-3">
                <p>
                  <strong className="text-zinc-300">Now (research &amp; prototyping):</strong> ARCANE is already available as an open-source Python library via <code className="text-[#B9DFE0] bg-zinc-800 px-1.5 py-0.5 text-sm">pip install gpbacay-arcane</code>. Researchers and developers can build, train, and experiment with neuromimetic models today.
                </p>
                <p>
                  <strong className="text-zinc-300">Near term (1–2 years):</strong> Expect continued maturation of the SDK, broader benchmarks, improved documentation, and early adopters integrating ARCANE layers into hybrid pipelines.
                </p>
                <p>
                  <strong className="text-zinc-300">Longer term (2–5+ years):</strong> Production deployments targeting real-time inference, large-scale throughput, and edge efficiency—as outlined in the Long-Term Goals—will depend on ongoing optimization, community contributions, and validation at industry scale.
                </p>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-bold uppercase tracking-widest text-[#B9DFE0]">Part IV — The Highlight</h3>
          <Accordion type="single" collapsible className="w-full space-y-4">
            <AccordionItem value="rq-13" className="border border-[#B9DFE0]/30 bg-zinc-900/30 px-4 rounded-none">
              <AccordionTrigger className="text-zinc-100 hover:text-white transition-colors font-semibold">
                In simple terms, how does ARCANE work?
              </AccordionTrigger>
              <AccordionContent className="text-zinc-400 leading-relaxed space-y-4">
                <p>
                  Imagine a group of musicians trying to play in harmony. In a normal AI, each musician plays their part once, in order, without listening to the others—and only afterward does a teacher tell them what went wrong. ARCANE works more like a <strong className="text-zinc-300">rehearsal</strong>: before performing, the musicians listen to each other, adjust, and tune until they are aligned. Only then do they play the final piece.
                </p>
                <p>
                  That rehearsal is the <strong className="text-zinc-300">Thinking Phase</strong> (Neural Resonance). Higher layers in the network send down what they <em>expect</em> to see; lower layers compare those expectations to what they actually received and adjust. This back-and-forth repeats until the whole system agrees on a coherent understanding—then it produces an answer.
                </p>
                <p>
                  Another analogy: think of <strong className="text-zinc-300">pausing before you speak</strong>. When someone asks a hard question, you don&apos;t blurt out the first word that comes to mind—you briefly reconcile what you know with what was asked, then respond. ARCANE gives AI that same moment of internal alignment.
                </p>
                <p>
                  Under the hood, it also borrows ideas from how real brains learn: neurons that fire together strengthen their connections (Hebbian learning), activity stays balanced through self-regulation (homeostatic plasticity), and signals are sent as sparse spikes rather than constant noise—making the system more efficient, like a brain that only uses energy when it is actively thinking.
                </p>
                <p className="text-zinc-300">
                  In one sentence: <strong>ARCANE is AI that thinks before it answers—using brain-inspired feedback, not just bigger datasets.</strong>
                </p>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      </section>

      {/* FAQs */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-white">Technical FAQ</h2>
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
          <AccordionItem value="item-11" className="border-zinc-800 bg-zinc-900/20 px-4 rounded-none">
            <AccordionTrigger className="text-zinc-200 hover:text-white transition-colors">What are the benefits of using Arcane technology?</AccordionTrigger>
            <AccordionContent className="text-zinc-400 leading-relaxed">
              ARCANE offers a more efficient and resilient alternative to traditional AI by focusing on architectural innovation. Key benefits include lower power consumption closer to a biological brain and inference-time learning through real-time semantic state alignment. This leads to better reasoning and a more transparent system grounded in neuroscientific principles, making it a powerful tool for research and applications that require high-level adaptability and stable, predictable AI behavior.
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
