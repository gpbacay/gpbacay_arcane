
export default function BiologicalLayersPage() {
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
          A.R.C.A.N.E. provides a suite of custom Keras layers that mimic the dynamics of biological neurons.
        </p>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Available Layers
        </h2>

        <div className="grid gap-6 md:grid-cols-2">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-6">
            <h3 className="text-lg font-bold text-zinc-100 mb-2">ResonantGSER</h3>
            <p className="text-sm text-zinc-400 mb-4">
              Hierarchical resonant layer with bi-directional feedback and spiking dynamics.
            </p>
            <a href="/docs/resonant-gser" className="text-sm font-medium text-purple-400 hover:underline">Learn more &rarr;</a>
          </div>

          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-6">
            <h3 className="text-lg font-bold text-zinc-100 mb-2">BioplasticDenseLayer</h3>
            <p className="text-sm text-zinc-400 mb-4">
              Implements Hebbian learning ("neurons that fire together, wire together") and homeostatic plasticity.
            </p>
          </div>

          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-6">
            <h3 className="text-lg font-bold text-zinc-100 mb-2">GSER</h3>
            <p className="text-sm text-zinc-400 mb-4">
              Gated Spiking Elastic Reservoir with dynamic reservoir sizing.
            </p>
          </div>

          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-6">
            <h3 className="text-lg font-bold text-zinc-100 mb-2">LatentTemporalCoherence</h3>
            <p className="text-sm text-zinc-400 mb-4">
              Distills temporal dynamics into coherence vectors for stable sequence processing.
            </p>
          </div>
        </div>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Usage Example
        </h2>
        <pre className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950 p-4 text-sm text-zinc-300">
          <code>{`from gpbacay_arcane.layers import BioplasticDenseLayer

# Create a Hebbian learning layer
layer = BioplasticDenseLayer(
    units=128,
    learning_rate=1e-3,
    target_avg=0.11,    # Homeostatic target activity
    homeostatic_rate=8e-5,
    activation='gelu'
)`}</code>
        </pre>
      </div>
    </div>
  );
}
