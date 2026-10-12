
import { CodeSnippet } from "@/components/CodeSnippet";

export default function FoundationModelPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Hierarchical Resonance Foundation Model
        </h1>
        <p className="text-xl text-zinc-400">
          A hierarchical System 1 model with per-example resonance, trained feedback, and bounded adaptation.
        </p>
      </div>

      <div className="space-y-8 text-zinc-300 leading-7">
        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Overview
        </h2>
        <p>
          The <strong>HierarchicalResonanceFoundationModel</strong> organizes <strong>ResonantGSER</strong> layers into a feedback-driven sequence model. During resonance prediction, every example carries its own top-down target through the hierarchy. Feedback training and test-time adaptation remain separate, measurable operations with clear reset semantics.
        </p>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Usage
        </h2>
        <CodeSnippet filename="foundation-model.py" language="python" code={`from gpbacay_arcane import HierarchicalResonanceFoundationModel

model = HierarchicalResonanceFoundationModel(
    vocab_size=5000,
    seq_len=32,
    hidden_dim=128,
    num_resonance_levels=4,
    enable_inference_plasticity=True,
    use_hierarchical_homeostasis=True,
    homeostasis_config={
        "homeostasis_mode": "hybrid",
        "homeostatic_window_size": 32,
        "slow_bcm_tau": 4000.0,
        "adaptive_target_rate": 0.05,
        "resonance_coupling": 0.5,
        "hierarchy_target_slope": 0.1,
    },
)
model.build_model()
model.compile_model()

# Ordinary supervised training
model.model.fit(X_train, y_train)

# Train task weights and feedback projections end to end
metrics = model.train_feedback_step(
    X_train[:32], labels=y_train[:32], alignment_weight=0.1
)

# Refine the current batch with isolated per-example resonance state
predictions = model.predict_with_resonance(X_test, resonance_cycles=4)

# Optional bounded adaptation; episodic=True restores the checkpoint afterward
result = model.adapt_at_inference(
    X_test,
    variables="all",
    objective="entropy",
    steps=2,
    max_update_norm=1e-3,
    episodic=True,
)`} />

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Key Features
        </h2>
        <ul className="list-disc pl-6 space-y-2 marker:text-zinc-500">
          <li><strong>Per-example resonance</strong>: explicit batch-shaped targets avoid co-batching contamination during current-input refinement.</li>
          <li><strong>Trainable feedback</strong>: alignment and reconstruction losses provide a gradient path to feedback projections.</li>
          <li><strong>Bounded adaptation</strong>: optional entropy or consistency objectives support snapshots, update limits, rollback, and episodic reset.</li>
          <li><strong>Temporal Coherence</strong>: Distills temporal dynamics into coherence vectors.</li>
          <li><strong>Attention Fusion</strong>: Aggregates multi-pathway information with self-attention.</li>
        </ul>
        <p>
          Choose the resonance depth with a controlled ablation of the same checkpoint at 0, 1, 2, 4, and 8 cycles. This reveals the best accuracy, calibration, and latency trade-off for your task.
        </p>
      </div>
    </div>
  );
}
