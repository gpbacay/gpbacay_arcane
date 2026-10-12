
import { CodeSnippet } from "@/components/CodeSnippet";

export default function QuickStartPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Quick Start
        </h1>
        <p className="text-xl text-zinc-400">
          Build an adaptive System 1 model, then choose how it decides, remembers, and adapts.
        </p>
      </div>

      <div className="space-y-8 text-zinc-300 leading-7">

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Basic Usage
        </h2>
        <p>
          Create a simple neuromimetic model for semantic tasks.
        </p>
        <CodeSnippet filename="quick-start.py" language="python" code={`from gpbacay_arcane import NeuromimeticSemanticModel

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
)`} />

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Advanced Usage with Resonance
        </h2>
        <p>
          Refine a current batch with isolated per-example targets, then optionally evaluate bounded adaptation.
        </p>
        <CodeSnippet filename="adaptive-inference.py" language="python" code={`from gpbacay_arcane import HierarchicalResonanceFoundationModel

# Create an advanced model with biological neural principles
model = HierarchicalResonanceFoundationModel(
    vocab_size=3000,
    seq_len=32,
    hidden_dim=128,
    num_resonance_levels=4
)

model.build_model()
model.compile_model(learning_rate=3e-4)

# Train normally, including an explicit feedback-alignment step when desired
model.model.fit(X_train, y_train)
model.train_feedback_step(X_train[:32], labels=y_train[:32])

# Resonance changes latent state for this batch without changing learned weights
predictions = model.predict_with_resonance(X_test, resonance_cycles=4)

# Optional test-time learning uses an objective, bounded updates, and rollback
adapted = model.adapt_at_inference(
    X_test,
    variables="all",
    objective="entropy",
    steps=2,
    episodic=True,
)`} />

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Sequence Models with PredictiveResonantLayer
        </h2>
        <p>
          For sequence or image-as-sequence tasks (e.g. MNIST), use <code className="bg-zinc-900 px-1.5 py-0.5 rounded">PredictiveResonantLayer</code> for local predictive resonance without hierarchical wiring.
        </p>
        <CodeSnippet filename="predictive-resonance.py" language="python" code={`from gpbacay_arcane import PredictiveResonantLayer, SpatioTemporalSummarization, BioplasticDenseLayer

# Sequence of 28 steps (e.g. MNIST rows), 128 units
x = PredictiveResonantLayer(
    units=128,
    resonance_cycles=3,
    resonance_step_size=0.2,
    return_sequences=True,
    persist_alignment=False
)(x)
x = SpatioTemporalSummarization(d_model=128)(x)
x = BioplasticDenseLayer(units=128, enable_inference_plasticity=True)(x)
# then pool and classify`} />

        <div className="mt-8 rounded-none border border-yellow-900/50 bg-yellow-900/10 p-4">
          <p className="text-sm text-yellow-200">
            <strong>Tune resonance for the task:</strong> each cycle adds compute. Compare the same checkpoint at 0/1/2/4/8 cycles, then select the best balance of accuracy, calibration, latency, forgetting, order sensitivity, batch-size sensitivity, and distribution shift. See <a href="/docs/inference-time-learning" className="underline hover:text-purple-300">Inference-Time Adaptation</a> for the full protocol.
          </p>
        </div>
      </div>
    </div>
  );
}
