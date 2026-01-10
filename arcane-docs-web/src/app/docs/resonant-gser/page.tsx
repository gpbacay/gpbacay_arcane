
export default function ResonantGSERPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Resonant GSER
        </h1>
        <p className="text-xl text-zinc-400">
          Hierarchical Neural Resonance in A.R.C.A.N.E.
        </p>
      </div>

      <div className="space-y-8 text-zinc-300 leading-7">
        <p>
          <strong>ResonantGSER</strong> (Resonant Gated Spiking Elastic Reservoir) is the core mechanism implementing
          <strong>Hierarchical Neural Resonance</strong> in the A.R.C.A.N.E. framework.
        </p>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Core Concept
        </h2>
        <p>
          ResonantGSER implements a <strong>Thinking Phase</strong> where neural representations are iteratively refined before final outputs are produced.
        </p>
        <ol className="list-decimal pl-6 space-y-2 marker:text-zinc-500">
          <li><strong>Feedforward Pass</strong>: Initial processing through hierarchical layers.</li>
          <li><strong>Resonance Cycles</strong>: Iterative feedback and harmonization.</li>
          <li><strong>State Alignment</strong>: Layers synchronize representations.</li>
          <li><strong>Semantic Optimization</strong>: Direct optimization of latent space meanings.</li>
        </ol>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Mathematical Formulation
        </h2>
        <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-6 overflow-x-auto">
          <p className="font-mono text-sm text-zinc-400 mb-2">Resonance Cycle n:</p>
          <ul className="space-y-2 text-sm text-zinc-300 font-mono">
            <li>1. Projection: P = f(S_i)</li>
            <li>2. Divergence: Δ = S_lower - P</li>
            <li>3. Harmonization: S_lower_new = S_lower - (γ + β) · Δ</li>
          </ul>
        </div>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Implementation
        </h2>
        <pre className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950 p-4 text-sm text-zinc-300">
          <code>{`from gpbacay_arcane.layers import ResonantGSER

# Create a resonant layer
layer = ResonantGSER(
    units=128,
    resonance_factor=0.2, # Strength of top-down feedback
    resonance_cycles=5,   # Number of thinking iterations
    return_sequences=False
)`}</code>
        </pre>

        <div className="mt-8 rounded-lg border border-blue-900/50 bg-blue-900/10 p-4">
          <h4 className="font-semibold text-blue-200 mb-2">Key Parameters</h4>
          <ul className="list-disc pl-4 text-sm text-blue-300 space-y-1">
            <li><code>resonance_factor (0.15 - 0.30)</code>: Balances convergence speed vs stability.</li>
            <li><code>resonance_cycles (3 - 8)</code>: Cost vs accuracy trade-off.</li>
            <li><code>spike_threshold (0.3 - 0.7)</code>: Activity regularization.</li>
          </ul>
        </div>

      </div>
    </div>
  );
}
