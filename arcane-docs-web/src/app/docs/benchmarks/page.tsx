
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
          Comprehensive testing on the Tiny Shakespeare dataset demonstrates the advantages of biological neural mechanisms.
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
          <li><strong>18.4% relative improvement</strong> in validation accuracy over traditional LSTM.</li>
          <li><strong>Lowest loss variance (0.0142)</strong> indicating superior training stability.</li>
          <li><strong>Smallest train/val gap (0.048)</strong> indicating reduced overfitting.</li>
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
