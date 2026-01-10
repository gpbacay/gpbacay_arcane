
export default function NeuralResonancePage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Hierarchical Neural Resonance
        </h1>
        <p className="text-xl text-zinc-400">
          A bi-directional and self-modeling computational mechanism for deep state alignment.
        </p>
      </div>

      <div className="space-y-8 text-zinc-300 leading-7">
        <p>
          Hierarchical Neural Resonance is a core architectural innovation of the A.R.C.A.N.E. framework.
          It enables multi-layered neural systems to iteratively synchronize their internal semantic representations
          through continuous feedback loops and state alignment, bridging the gap between connectionist machine learning,
          Adaptive Resonance Theory (ART), and biological neurodynamics.
        </p>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          The "Thinking Phase"
        </h2>
        <p>
          Unlike traditional feed-forward networks (System 1) that map input to output in a single pass,
          A.R.C.A.N.E. introducing a <strong>Thinking Phase</strong> (System 2).
        </p>
        <p>
          During this phase, higher layers project expectations downward (<strong>Feedback Projection</strong>),
          and lower layers adjust their states to match (<strong>Harmonization</strong>).
          This minimizes <strong>Prediction Divergence</strong> locally before any weight updates occur.
        </p>

        <div className="my-8 rounded-lg bg-zinc-900 border border-zinc-800 p-6">
          <h3 className="text-lg font-semibold text-zinc-100 mb-4">The Resonance Cycle</h3>
          <div className="flex flex-col gap-4 text-sm font-mono text-zinc-400">
            <div className="flex items-center gap-4">
              <span className="text-purple-400">Higher Layer</span>
              <span className="flex-1 border-b border-dashed border-zinc-700 relative">
                <span className="absolute top-[-10px] left-1/2 -translate-x-1/2 text-xs">Project Feedback</span>
                <span className="absolute right-0 top-[-6px] transform rotate-45 border-r-2 border-b-2 border-zinc-700 w-2 h-2"></span>
              </span>
              <span className="text-blue-400">Lower Layer</span>
            </div>
            <div className="text-center text-xs text-zinc-500 py-2">
              Harmonization (Minimizing Divergence)
            </div>
            <div className="flex items-center gap-4">
              <span className="text-purple-400">Higher Layer</span>
              <span className="flex-1 border-b border-dashed border-zinc-700 relative">
                <span className="absolute top-[-2px] left-1/2 -translate-x-1/2 text-xs">Updated State</span>
                <span className="absolute left-0 top-[-6px] transform rotate-[225deg] border-r-2 border-b-2 border-zinc-700 w-2 h-2"></span>
              </span>
              <span className="text-blue-400">Lower Layer</span>
            </div>
          </div>
        </div>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Key Terminologies
        </h2>
        <dl className="grid gap-6 sm:grid-cols-2">
          <div className="rounded border border-zinc-800 bg-zinc-900/50 p-4">
            <dt className="font-semibold text-zinc-100 mb-2">Prospective Configuration</dt>
            <dd className="text-sm text-zinc-400">Neural activities are optimized to align with expectations <em>before</em> synaptic weight updates.</dd>
          </div>
          <div className="rounded border border-zinc-800 bg-zinc-900/50 p-4">
            <dt className="font-semibold text-zinc-100 mb-2">Feedback Projection</dt>
            <dd className="text-sm text-zinc-400">Top-down signal from a higher layer representing its expectation of the lower layer's activity.</dd>
          </div>
          <div className="rounded border border-zinc-800 bg-zinc-900/50 p-4">
            <dt className="font-semibold text-zinc-100 mb-2">Harmonization</dt>
            <dd className="text-sm text-zinc-400">The mechanism by which a layer adjusts its internal semantic representation to minimize divergence.</dd>
          </div>
          <div className="rounded border border-zinc-800 bg-zinc-900/50 p-4">
            <dt className="font-semibold text-zinc-100 mb-2">Validation Accuracy</dt>
            <dd className="text-sm text-zinc-400">Hierarchical Resonance achieves <strong>11.25%</strong> vs 9.50% for LSTM on shakespeare benchmark.</dd>
          </div>
        </dl>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Code Analysis
        </h2>
        <pre className="bg-zinc-950 p-4 rounded overflow-x-auto text-sm text-zinc-300 border border-zinc-800">
          {`# Example of Neural Resonance in Action
resonance_cb = NeuralResonanceCallback(resonance_cycles=10)

# The model will perform 10 internal alignment cycles
# for every batch of data before updating weights.
model.fit(X_train, y_train, callbacks=[resonance_cb])`}
        </pre>
      </div>
    </div>
  );
}
