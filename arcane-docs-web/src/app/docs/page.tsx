export default function DocsPage() {
  return (
    <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="space-y-4">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl text-white mb-2">Introduction</h1>
        <p className="text-xl text-zinc-400 font-medium leading-relaxed">
          Welcome to the official documentation for A.R.C.A.N.E.
        </p>
      </div>

      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-8 backdrop-blur-sm relative overflow-hidden group">
        <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:rotate-12 transition-transform duration-500">
          <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-[#C785F2]">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
        </div>
        <h2 id="what-is-arcane" className="text-2xl font-bold text-white mb-4">What is A.R.C.A.N.E.?</h2>
        <p className="text-zinc-300 leading-relaxed mb-6">
          <strong>Augmented Reconstruction of Consciousness through Artificial Neural Evolution</strong> is a comprehensive Python library that enables you to build, train, and deploy neuromimetic AI models.
        </p>
        <p className="text-zinc-300 leading-relaxed">
          Unlike traditional deep learning frameworks, A.R.C.A.N.E. incorporates biological neural principles such as bi-directional state alignment, spiking dynamics, and homeostatic plasticity.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-xl border border-zinc-800 hover:border-[#835BD9]/50 transition-colors bg-zinc-950">
          <h3 id="neural-resonance" className="text-lg font-bold text-zinc-100 mb-2">Inference-Time Learning</h3>
          <p className="text-sm text-zinc-400">Iterative state alignment through Neural Resonance, enabling the model to refine its semantic understanding during inference.</p>
        </div>
        <div className="p-6 rounded-xl border border-zinc-800 hover:border-[#C785F2]/50 transition-colors bg-zinc-950">
          <h3 id="biological-plausibility" className="text-lg font-bold text-zinc-100 mb-2">Inference-Time State Adaptation</h3>
          <p className="text-sm text-zinc-400">Dynamic adjustment of neural states to achieve hierarchical coherence before a final output is committed.</p>
        </div>
        <div className="p-6 rounded-xl border border-zinc-800 hover:border-[#A855F7]/50 transition-colors bg-zinc-950">
          <h3 id="neuromimetic-activations" className="text-lg font-bold text-zinc-100 mb-2">Neuromimetic Activations</h3>
          <p className="text-sm text-zinc-400">Stateful, event-driven units including Resonant Spiking and Homeostatic GELU.</p>
        </div>
      </div>

      <section className="space-y-6">
        <h2 id="getting-started" className="text-3xl font-bold text-white">Getting Started</h2>
        <p className="text-zinc-400 leading-relaxed">
          Start building bio-inspired models in minutes. Head over to our installation guide to set up the library on your system.
        </p>
        <div className="flex gap-4">
          <a href="/docs/installation" className="inline-flex items-center justify-center rounded-lg bg-zinc-100 px-6 py-2.5 text-sm font-bold text-black transition-colors hover:bg-zinc-300">
            Installation Guide
          </a>
          <a href="/docs/quick-start" className="inline-flex items-center justify-center rounded-lg border border-zinc-800 px-6 py-2.5 text-sm font-bold text-zinc-200 transition-colors hover:bg-zinc-900">
            Quick Start
          </a>
        </div>
      </section>

      <footer className="pt-16 border-t border-zinc-900 text-zinc-600 text-sm">
        <p>A.R.C.A.N.E. Documentation • Built for the future of AI</p>
      </footer>
    </div>
  );
}
