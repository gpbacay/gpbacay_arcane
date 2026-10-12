
import { CodeSnippet } from "@/components/CodeSnippet";

export default function NeuromimeticModelPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Neuromimetic Semantic Model
        </h1>
        <p className="text-xl text-zinc-400">
          A compact System 1 sequence model with resonance, activity regulation, and optional fast-weight plasticity.
        </p>
      </div>

      <div className="space-y-8 text-zinc-300 leading-7">
        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Overview
        </h2>
        <p>
          The <strong>NeuromimeticSemanticModel</strong> packages ARCANE&apos;s core adaptive mechanisms into a smaller architecture for semantic sequence tasks. Use it when you want a shorter training path, lower default depth, and direct access to resonance and activity regulation.
        </p>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Usage
        </h2>
        <CodeSnippet filename="neuromimetic-model.py" language="python" code={`from gpbacay_arcane import NeuromimeticSemanticModel

model = NeuromimeticSemanticModel(
    vocab_size=1000,
    enable_inference_plasticity=True,
    use_hierarchical_homeostasis=True,
    homeostasis_config={"homeostasis_mode": "hybrid"},
)
model.build_model()
model.compile_model()

# Standard training
model.model.fit(X_train, y_train)

# Current-input refinement with fixed learned weights
predictions = model.predict_with_resonance(X_test, resonance_cycles=2)`} />

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Comparison vs Foundation Model
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-zinc-400">
            <thead className="border-b border-zinc-800 text-zinc-200">
              <tr>
                <th className="py-2 px-4">Feature</th>
                <th className="py-2 px-4">NeuromimeticSemanticModel</th>
                <th className="py-2 px-4">FoundationModel</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              <tr>
                <td className="py-2 px-4">Resonance Levels</td>
                <td className="py-2 px-4">Compact hierarchy</td>
                <td className="py-2 px-4">Configurable deeper hierarchy</td>
              </tr>
              <tr>
                <td className="py-2 px-4">Training Speed</td>
                <td className="py-2 px-4">Lower default depth</td>
                <td className="py-2 px-4">More recurrent levels</td>
              </tr>
              <tr>
                <td className="py-2 px-4">Best For</td>
                <td className="py-2 px-4">Compact semantic workloads</td>
                <td className="py-2 px-4">Deep hierarchy and adaptation workflows</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
