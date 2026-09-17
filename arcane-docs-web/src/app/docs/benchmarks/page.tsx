
export default function BenchmarksPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Performance & Benchmarks
        </h1>
        <p className="text-xl text-zinc-400">
          Comparative analysis of ARCANE models vs traditional architectures.
        </p>
      </div>

      <div className="space-y-8 text-zinc-300 leading-7">
        <p>
          Exploratory Tiny Shakespeare run (15k chars, 10 epochs, unequal parameter counts). Treat as a smoke comparison, not a Transformer result.
        </p>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Tiny Shakespeare Results
        </h2>
        <div className="overflow-x-auto rounded-none border border-zinc-800">
          <table className="w-full text-left text-sm text-zinc-400">
            <thead className="bg-zinc-900 text-zinc-200">
              <tr>
                <th className="py-3 px-4">Model</th>
                <th className="py-3 px-4">Val Accuracy</th>
                <th className="py-3 px-4">Val Loss</th>
                <th className="py-3 px-4">Train Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800 bg-zinc-950">
              <tr>
                <td className="py-3 px-4 font-medium">Traditional Deep LSTM</td>
                <td className="py-3 px-4">9.50%</td>
                <td className="py-3 px-4">6.85</td>
                <td className="py-3 px-4">~45s</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-medium">Neuromimetic (Standard)</td>
                <td className="py-3 px-4">10.20%</td>
                <td className="py-3 px-4">6.42</td>
                <td className="py-3 px-4">~58s</td>
              </tr>
              <tr className="bg-purple-900/10">
                <td className="py-3 px-4 font-medium text-purple-200">Hierarchical Resonance</td>
                <td className="py-3 px-4 text-purple-200">11.25%</td>
                <td className="py-3 px-4 text-purple-200">6.15</td>
                <td className="py-3 px-4 text-purple-200">~95s</td>
              </tr>
            </tbody>
          </table>
        </div>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Key Advantages
        </h2>
        <ul className="list-disc pl-6 space-y-2 marker:text-zinc-500">
          <li>Hierarchical Resonance used ~385K params vs ~195K for the LSTM baseline.</li>
          <li>Accuracy moved 9.50% → 11.25% on this small character LM; MNIST gaps in the markdown docs are within noise.</li>
          <li>Mechanism unit tests (STE spikes, closed-form resonance, BCM plastic kernel, GSER elastic gate) live in <code className="bg-zinc-900 px-1.5 py-0.5 rounded">tests/test_mechanism_correctness.py</code>.</li>
        </ul>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Run the Benchmark
        </h2>
        <pre className="overflow-x-auto rounded-none border border-zinc-800 bg-zinc-950 p-4 text-sm text-zinc-300">
          <code>python examples/test_hierarchical_resonance_comparison.py</code>
        </pre>
      </div>
    </div>
  );
}
