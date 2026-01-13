
export default function FoundationModelPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Hierarchical Resonance Foundation Model
        </h1>
        <p className="text-xl text-zinc-400">
          Advanced model with multi-level neural resonance, temporal coherence, and attention fusion.
        </p>
      </div>

      <div className="space-y-8 text-zinc-300 leading-7">
        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Overview
        </h2>
        <p>
          The <strong>HierarchicalResonanceFoundationModel</strong> is designed for complex reasoning tasks, research applications, and scenarios where biologically-plausible learning dynamics are prioritized. It features a deep hierarchy of <strong>ResonantGSER</strong> layers that engage in bi-directional state alignment.
        </p>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Usage
        </h2>
        <pre className="overflow-x-auto rounded-none border border-zinc-800 bg-zinc-950 p-4 text-sm text-zinc-300">
          <code>{`from gpbacay_arcane import HierarchicalResonanceFoundationModel, NeuralResonanceCallback

model = HierarchicalResonanceFoundationModel(
    vocab_size=5000,
    seq_len=32,
    hidden_dim=128,
    num_resonance_levels=4
)
model.build_model()
model.compile_model()

# Use neural resonance training
resonance_cb = NeuralResonanceCallback(resonance_cycles=10)
model.model.fit(X_train, y_train, callbacks=[resonance_cb])`}</code>
        </pre>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Key Features
        </h2>
        <ul className="list-disc pl-6 space-y-2 marker:text-zinc-500">
          <li><strong>Multi-level Resonance</strong>: synchronize state across 4+ hierarchical levels, facilitating <strong>Inference-Time Learning</strong> and <strong>Inference-Time State Adaptation</strong>.</li>
          <li><strong>Temporal Coherence</strong>: Distills temporal dynamics into coherence vectors.</li>
          <li><strong>Attention Fusion</strong>: Aggregates multi-pathway information with self-attention.</li>
        </ul>
      </div>
    </div>
  );
}
