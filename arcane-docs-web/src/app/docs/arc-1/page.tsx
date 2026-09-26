import Link from "next/link";
import { Download } from "lucide-react";
import { Arc1Demo } from "@/components/Arc1Demo";
import { BarChart } from "@/components/BarChart";
import { Mermaid } from "@/components/markdown";
import { ARC1_METRICS } from "@/config/arc1-metrics";

const h2 = "text-2xl font-bold tracking-tight text-zinc-100 mt-14 mb-4 border-b border-zinc-800 pb-2";
const h3 = "text-lg font-semibold text-zinc-100 mt-8 mb-2";
const code = "bg-zinc-900 px-1.5 py-0.5 text-zinc-100";
const p = "max-w-2xl leading-relaxed";

const ARCHITECTURE = `flowchart TB
  subgraph inputs[" "]
    direction LR
    U["Utterance<br/><i>convert 100 USD to PHP</i>"]:::req
    S["Schema texts<br/>tools, arguments, options, labels"]:::tool
  end
  S --> M{"SchemaMemory<br/>cached?"}:::tool
  U --> P["Perception blocks x3, one shared batch<br/>FieldAttention, ResonantChannelMixer<br/>ConceptEngram, FieldResonance"]
  M -- new --> P
  P --> F["Utterance field U"]:::req
  P --> E["Schema engrams"]:::tool
  M -- cached --> E
  E --> C["Probes<br/>engram + context + role"]:::tool
  F --> B["ResonantBinding x K cycles<br/>attend, hear, GSER gate, settle"]:::bind
  C --> B
  B --> R1["fire"]:::out
  B --> R2["anchor"]:::out
  B --> R3["select"]:::out
  F --> R4["embedding"]:::out
  style inputs fill:transparent,stroke:transparent
  classDef req fill:#0d1718,stroke:#B9DFE0,color:#f4f4f5
  classDef tool fill:#1a0f15,stroke:#F294C0,color:#f4f4f5
  classDef bind fill:#170f22,stroke:#C785F2,color:#f4f4f5,stroke-width:2px
  classDef out fill:#18181b,stroke:#9B6BE6,color:#f4f4f5`;

const REQUEST_FLOW = `sequenceDiagram
  participant App as Your app
  participant Agent as Arc1Agent
  participant Mem as SchemaMemory
  participant Model as ARC 1
  App->>Agent: run(text, tools) / extract / classify
  Agent->>Mem: look up schema engrams
  Mem-->>Agent: cached engrams, uncached texts
  Agent->>Model: utterance + uncached texts + probe plan
  Note over Model: one forward pass:<br/>perceive, compose, bind, read out
  Model-->>Agent: fire, anchor, select logits + new engrams
  Agent->>Mem: store new engrams
  Note over Agent: calibrate, resolve spans,<br/>validate against the schema
  Agent-->>App: typed result, confidence, latency_ms`;

const COMPONENTS = [
  ["FieldAttention", "Bidirectional, pad-masked multi-head attention with rotary positions."],
  ["ResonantChannelMixer", "Low-rank channel mixing behind a GSER spiking gate."],
  ["ConceptEngram", "Hashed n-gram memory in the first block; grounds names, codes, and numbers."],
  ["FieldResonance", "Pulls each token toward the sentence prototype, then applies a spike reset."],
  ["ResonantBinding", "Shared-weight cycles in which each probe attends, gates, and settles."],
];

const READOUTS = [
  ["fire", "Tool applies, optional argument present, boolean value", "Calibrated sigmoid"],
  ["anchor", "String and number arguments", "Start and end pointers over the input tokens"],
  ["select", "Enum arguments and classification labels", "Parameter-to-option resonance, softmax"],
  ["embedding", "Search, routing, deduplication", "Attention-pooled field, L2-normalised"],
];

const RCN_LAYOUT = [
  ["Header", "128 bytes", "Magic, version, weight format, baked cycles, dimensions, offsets, CRC32"],
  ["Metadata", "about 14 KB", "Architecture, readout calibration, tokenizer, tensor directory"],
  ["Tensor data", "rest of file", "Every tensor 64-byte aligned, in its final layout"],
];

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

function mb(bytes: number) {
  return `${(bytes / 1e6).toFixed(2)} MB`;
}

export default function Arc1Page() {
  const m = ARC1_METRICS;
  const full = m.byCycles[m.byCycles.length - 1];
  const accuracy = [
    { label: "Tool selection", value: full.toolSelection },
    { label: "Exact call", value: full.exactCall },
    { label: "No-tool rejection", value: full.noToolAcc },
    { label: "Extraction F1", value: full.extractionF1 },
    { label: "Intent category", value: full.classifyIntent },
    { label: "Sentiment", value: full.classifySentiment },
    { label: "Support routing", value: full.classifySupport },
    { label: "Unseen-tool intents", value: full.classifyUnseen },
    { label: "Unseen-tool calls", value: full.exactCallUnseen },
  ]
    .sort((a, b) => b.value - a.value)
    .map((d) => ({ ...d, detail: `${pct(d.value)} on held-out data, ${full.cycles} cycles` }));
  const sizes = m.formats.map((f) => ({
    label: f.label,
    value: f.bytes,
    detail:
      f.exactCall != null
        ? `${mb(f.bytes)}, exact call ${pct(f.exactCall)}, extraction F1 ${pct(f.extractF1 ?? 0)}`
        : mb(f.bytes),
  }));

  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <header className="not-prose mb-10">
        <h1 className="max-w-3xl text-4xl font-extrabold leading-[1.05] tracking-[-0.03em] text-zinc-50 sm:text-5xl">
          ARC 1 turns a sentence into a decision in one pass.
        </h1>
        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-zinc-400">
          ARC 1 is a compact, non-autoregressive decision model built on ARCANE resonance. It maps text to calibrated,
          typed output: tool calls with grounded arguments, extracted records, or classification labels. It does not
          generate prose, so every result is schema-valid and returned in milliseconds.
        </p>
        <dl className="mt-8 grid max-w-2xl grid-cols-3 border-y border-zinc-800">
          <div className="py-4 pr-4">
            <dt className="text-sm text-zinc-500">Parameters</dt>
            <dd className="mt-1 text-2xl font-semibold text-zinc-50">{(m.parameters / 1e6).toFixed(2)}M</dd>
          </div>
          <div className="border-l border-zinc-800 py-4 pl-4 pr-4">
            <dt className="text-sm text-zinc-500">Median latency</dt>
            <dd className="mt-1 text-2xl font-semibold text-zinc-50">{m.latencyP50Ms.toFixed(0)} ms</dd>
          </div>
          <div className="border-l border-zinc-800 py-4 pl-4">
            <dt className="text-sm text-zinc-500">Model file</dt>
            <dd className="mt-1 text-2xl font-semibold text-zinc-50">{(m.rcn.bytes / 1e6).toFixed(2)} MB</dd>
          </div>
        </dl>
        <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-3">
          <a
            href={m.rcn.file}
            download
            className="inline-flex items-center gap-2 bg-white px-4 py-2.5 text-sm font-medium text-black transition-colors hover:bg-zinc-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#C785F2]"
          >
            <Download className="h-4 w-4" aria-hidden />
            Download arc1-tiny.rcn
          </a>
          <a href="#rcn-format" className="text-sm text-zinc-400 underline hover:text-zinc-100">
            About the .rcn format
          </a>
        </div>
        <p className="mt-3 text-xs text-zinc-500">
          Latency is per request on a laptop CPU, with {m.cycles} binding cycles and schemas cached.
        </p>
      </header>

      <Arc1Demo />

      <div className="text-zinc-300">
        <h2 id="architecture" className={h2}>
          Architecture
        </h2>
        <p className={p}>
          ARC 1 uses <strong className="text-zinc-100">Resonant Schema Binding</strong>. The input is perceived once.
          Every tool, argument, option, and label is represented as a schema engram, and all engrams bind to the input
          in parallel. Readouts then produce typed decisions directly, with no decoding loop.
        </p>
        <Mermaid chart={ARCHITECTURE} minWidth={520} />
        <div className="not-prose overflow-x-auto">
          <table className="w-full border-collapse text-left text-sm">
            <tbody>
              {COMPONENTS.map(([name, role]) => (
                <tr key={name} className="border-b border-zinc-900 align-top">
                  <td className="w-52 py-2.5 pr-4">
                    <code className="text-zinc-100">{name}</code>
                  </td>
                  <td className="py-2.5 text-zinc-400">{role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h3 className={h3}>Request flow</h3>
        <p className={p}>
          A request costs one forward pass. Schema texts not yet cached are perceived in the same batch as the input,
          so first-time tools add no extra pass. The agent applies calibration and schema validation before returning.
        </p>
        <Mermaid chart={REQUEST_FLOW} minWidth={640} />

        <h3 className={h3}>Readouts</h3>
        <div className="not-prose overflow-x-auto">
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className="py-2 pr-4 font-medium">Readout</th>
                <th className="py-2 pr-4 font-medium">Decides</th>
                <th className="py-2 font-medium">Mechanism</th>
              </tr>
            </thead>
            <tbody>
              {READOUTS.map(([name, decides, how]) => (
                <tr key={name} className="border-b border-zinc-900 align-top">
                  <td className="py-2.5 pr-4">
                    <code className="text-zinc-100">{name}</code>
                  </td>
                  <td className="py-2.5 pr-4 text-zinc-300">{decides}</td>
                  <td className="py-2.5 text-zinc-500">{how}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={`${p} mt-4`}>
          Each readout carries a temperature fitted on held-out data, so confidence values are calibrated: fire
          decisions have an expected calibration error of {pct(m.fireEceAfter)}. When no tool fires, ARC 1 returns no
          call.
        </p>

        <h2 id="results" className={h2}>
          Performance
        </h2>
        <p className={p}>
          All figures come from synthetic data held out from training. Tool figures use unseen argument values;
          unseen-tool rows use tools excluded from training entirely.
        </p>
        <BarChart
          title="Held-out accuracy by task"
          subtitle={`${full.cycles} binding cycles; hover a bar for details`}
          data={accuracy}
          max={1}
          ticks={[0, 0.25, 0.5, 0.75, 1]}
          unit="percent"
        />
        <p className={p}>
          ARC 1 has no pretrained language knowledge. It is strongest when inputs share vocabulary with its schemas and
          weakest on paraphrases unlike its training data, as the support-routing and unseen-tool results show.
        </p>
        <div className="not-prose mt-6 overflow-x-auto">
          <table className="w-full border-collapse text-left text-sm tabular-nums">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className="py-2 pr-4 font-medium">Metric</th>
                {m.byCycles.map((c) => (
                  <th key={c.cycles} className="py-2 pr-4 text-right font-medium">
                    {c.cycles} {c.cycles === 1 ? "cycle" : "cycles"}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="text-zinc-300">
              {(
                [
                  ["Tool selection", "toolSelection"],
                  ["Exact call", "exactCall"],
                  ["Argument accuracy", "argumentAcc"],
                  ["No-tool rejection", "noToolAcc"],
                  ["Exact call, unseen tools", "exactCallUnseen"],
                  ["Extraction field F1", "extractionF1"],
                  ["Classification, intent category", "classifyIntent"],
                  ["Classification, sentiment", "classifySentiment"],
                  ["Classification, support routing", "classifySupport"],
                  ["Classification, unseen-tool intents", "classifyUnseen"],
                ] as const
              ).map(([label, key]) => (
                <tr key={key} className="border-b border-zinc-900">
                  <td className="py-2.5 pr-4">{label}</td>
                  {m.byCycles.map((c) => (
                    <td key={c.cycles} className="py-2.5 pr-4 text-right">
                      {pct(c[key])}
                    </td>
                  ))}
                </tr>
              ))}
              <tr className="border-b border-zinc-900">
                <td className="py-2.5 pr-4">Median latency</td>
                {m.byCycles.map((c) => (
                  <td key={c.cycles} className="py-2.5 pr-4 text-right">
                    {c.latencyP50Ms.toFixed(1)} ms
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>

        <h2 id="rcn-format" className={h2}>
          The .rcn model format
        </h2>
        <p className={p}>
          ARC 1 ships as a single <code className={code}>.rcn</code> file: a self-contained container designed for
          small devices. One file carries the architecture, calibration, tokenizer, a baked device profile, and the
          weights. Nothing else is needed to run the model.
        </p>
        <ul className="max-w-2xl space-y-2">
          <li>
            <strong className="text-zinc-100">Read in place.</strong> Tensors are stored 64-byte aligned in their final
            layout. The runtime memory-maps the file and reads them directly, with no parsing step.
          </li>
          <li>
            <strong className="text-zinc-100">Header-first.</strong> A fixed 128-byte header describes the model, so
            tools can inspect a file without reading the weights.
          </li>
          <li>
            <strong className="text-zinc-100">Native quantization.</strong> Weights are stored as{" "}
            <code className={code}>f16</code>, <code className={code}>rq8</code> (8.5 bits per weight), or{" "}
            <code className={code}>rq4</code> (4.5 bits per weight), using block-wise scales over 32 values. Small
            tensors stay f16.
          </li>
          <li>
            <strong className="text-zinc-100">Baked device profile.</strong> The binding-cycle count is fixed at export,
            so a file built for a constrained device runs its fastest setting by default.
          </li>
          <li>
            <strong className="text-zinc-100">Verified.</strong> A CRC32 of the tensor data is checked at load.
          </li>
        </ul>
        <div className="not-prose mt-6 overflow-x-auto">
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className="py-2 pr-4 font-medium">Region</th>
                <th className="py-2 pr-4 font-medium">Size</th>
                <th className="py-2 font-medium">Contents</th>
              </tr>
            </thead>
            <tbody>
              {RCN_LAYOUT.map(([region, size, contents]) => (
                <tr key={region} className="border-b border-zinc-900 align-top">
                  <td className="py-2.5 pr-4 text-zinc-100">{region}</td>
                  <td className="whitespace-nowrap py-2.5 pr-4 text-zinc-400">{size}</td>
                  <td className="py-2.5 text-zinc-400">{contents}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <BarChart
          title="Model size by format"
          subtitle="Same ARC 1 weights; hover a bar for held-out accuracy"
          data={sizes}
          max={6e6}
          ticks={[0, 2e6, 4e6, 6e6]}
          unit="mb"
          highlight=".rcn rq4"
        />
        <p className={p}>
          The <code className={code}>rq4</code> file is {(m.formats[0].bytes / m.rcn.bytes).toFixed(1)}x smaller than
          the float32 weights, with held-out accuracy unchanged within measurement noise.
        </p>

        <h3 className={h3}>Download and run</h3>
        <div className="not-prose flex flex-col gap-3 border border-zinc-800 bg-zinc-950 p-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0">
            <div className="text-sm font-medium text-zinc-100">arc1-tiny.rcn</div>
            <div className="mt-0.5 text-xs text-zinc-500">
              {mb(m.rcn.bytes)}, {m.rcn.quant}, tokenizer included
            </div>
            <div className="mt-1 break-all font-mono text-[11px] text-zinc-500">sha256 {m.rcn.sha256}</div>
          </div>
          <a
            href={m.rcn.file}
            download
            className="inline-flex shrink-0 items-center justify-center gap-2 bg-white px-4 py-2 text-sm font-medium text-black transition-colors hover:bg-zinc-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#C785F2]"
          >
            <Download className="h-4 w-4" aria-hidden />
            Download
          </a>
        </div>
        <pre className="not-prose mt-4 overflow-x-auto border border-zinc-800 bg-zinc-950 p-4 text-[13px] leading-relaxed text-zinc-300">
          {`# from a clone of the ARCANE repository: pip install -e .
from gpbacay_arcane import Arc1Agent, ToolParam, ToolSpec, load_rcn

model, tokenizer, header = load_rcn("arc1-tiny.rcn")
agent = Arc1Agent(model, tokenizer)

weather = ToolSpec("get_weather", "Get the current weather for a city.",
                   [ToolParam("city", description="City name")])
agent.run("is it raining in Tokyo?", tools=[weather])
# {"function_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}], ...}

agent.classify("the headphones sound amazing", ["positive", "negative", "neutral"])
# {"label": "positive", "confidence": ..., "distribution": {...}}`}
        </pre>
        <p className={`${p} mt-4 text-zinc-400`}>
          To build your own file, run{" "}
          <code className={code}>
            python examples/export_arc1.py --rcn arc1.rcn --quant rq4 --cycles 2 --tokenizer Models/arc1_arc1_tiny_tokenizer.json
          </code>
          . Inspect any file with <code className={code}>python -m gpbacay_arcane.rcn arc1.rcn</code>.
        </p>

        <h2 id="run-locally" className={h2}>
          Run the sandbox locally
        </h2>
        <p className={p}>
          From <code className={code}>arcane-docs-web</code>, run <code className={code}>npm run dev:with-arc1</code>.
          This starts the site and the ARC 1 API on port 8002, with <code className={code}>/run</code>,{" "}
          <code className={code}>/extract</code>, <code className={code}>/classify</code>, and{" "}
          <code className={code}>/embed</code> endpoints. Train new weights with{" "}
          <code className={code}>python examples/train_arc1.py --preset arc1-tiny</code>.
        </p>
        <p className={`${p} mt-4 text-zinc-400`}>
          Limits: one call per tool per request, and input is truncated to {m.seqLen} tokens. For open-ended answers,
          pair ARC 1 with the{" "}
          <Link href="/docs/chat" className="font-medium text-[#C785F2] underline hover:text-[#d49cf5]">
            ARCANE small language model
          </Link>
          .
        </p>
      </div>
    </div>
  );
}
