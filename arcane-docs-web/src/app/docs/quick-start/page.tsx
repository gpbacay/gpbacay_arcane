
export default function QuickStartPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Quick Start
        </h1>
        <p className="text-xl text-zinc-400">
          Build your first neuromimetic model in minutes.
        </p>
      </div>

      <div className="space-y-8 text-zinc-300 leading-7">

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Basic Usage
        </h2>
        <p>
          Create a simple neuromimetic model for semantic tasks.
        </p>
        <pre className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950 p-4 text-sm text-zinc-300">
          <code>{`from gpbacay_arcane import NeuromimeticSemanticModel

# Create a simple neuromimetic model
model = NeuromimeticSemanticModel(vocab_size=1000)
model.build_model()
model.compile_model()

# Generate text (requires a trained tokenizer)
generated = model.generate_text(
    seed_text="artificial intelligence",
    tokenizer=your_tokenizer,
    max_length=50,
    temperature=0.8
)`}</code>
        </pre>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Advanced Usage with Resonance
        </h2>
        <p>
          Implement the biological "Thinking Phase" using hierarchical resonance.
        </p>
        <pre className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950 p-4 text-sm text-zinc-300">
          <code>{`from gpbacay_arcane import HierarchicalResonanceFoundationModel, NeuralResonanceCallback

# Create an advanced model with biological neural principles
model = HierarchicalResonanceFoundationModel(
    vocab_size=3000,
    seq_len=32,
    hidden_dim=128,
    num_resonance_levels=4
)

model.build_model()
model.compile_model(learning_rate=3e-4)

# Train with neural resonance (biological "thinking phase")
resonance_callback = NeuralResonanceCallback(resonance_cycles=10)
model.model.fit(X_train, y_train, callbacks=[resonance_callback])`}</code>
        </pre>

        <div className="mt-8 rounded-lg border border-yellow-900/50 bg-yellow-900/10 p-4">
          <p className="text-sm text-yellow-200">
            <strong>Note:</strong> Resonance cycles increase training time (~2x) but significantly improve stability and semantic alignment.
          </p>
        </div>
      </div>
    </div>
  );
}
