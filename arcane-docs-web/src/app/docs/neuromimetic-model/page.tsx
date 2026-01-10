
export default function NeuromimeticModelPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Neuromimetic Semantic Model
        </h1>
        <p className="text-xl text-zinc-400">
          Standard neuromimetic model with biological learning rules.
        </p>
      </div>

      <div className="space-y-8 text-zinc-300 leading-7">
        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Overview
        </h2>
        <p>
          The <strong>NeuromimeticSemanticModel</strong> provides a balance between performance and biological plausibility. It is optimized for general NLP tasks, faster training, and prototyping.
        </p>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Usage
        </h2>
        <pre className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950 p-4 text-sm text-zinc-300">
          <code>{`from gpbacay_arcane import NeuromimeticSemanticModel

model = NeuromimeticSemanticModel(vocab_size=1000)
model.build_model()
model.compile_model()

# Standard training
model.fit(X_train, y_train)`}</code>
        </pre>

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
                <td className="py-2 px-4">2 (Basic)</td>
                <td className="py-2 px-4">Multi-level (Advanced)</td>
              </tr>
              <tr>
                <td className="py-2 px-4">Training Speed</td>
                <td className="py-2 px-4">Faster</td>
                <td className="py-2 px-4">Slower (due to cycles)</td>
              </tr>
              <tr>
                <td className="py-2 px-4">Best For</td>
                <td className="py-2 px-4">General NLP, Prototyping</td>
                <td className="py-2 px-4">Complex Reasoning, Research</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
