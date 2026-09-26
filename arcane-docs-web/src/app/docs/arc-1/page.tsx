import Link from "next/link";
import { Arc1Demo } from "@/components/Arc1Demo";
import { Arc1ArchitectureDiagram } from "@/components/Arc1ArchitectureDiagram";
import { ARC1_METRICS } from "@/config/arc1-metrics";

const h2 = "text-2xl font-bold tracking-tight text-zinc-100 mt-14 mb-4 border-b border-zinc-800 pb-2";
const code = "bg-zinc-900 px-1.5 py-0.5 text-zinc-100";

const READOUTS = [
  {
    name: "fire",
    decides: "Whether a tool applies, an optional argument is present, or a boolean is true",
    how: "Calibrated sigmoid on the settled probe, what it heard, and its peak resonance",
  },
  {
    name: "anchor",
    decides: "String and number arguments",
    how: "Start and end pointers over the user's own tokens, so values are always grounded",
  },
  {
    name: "select",
    decides: "Enum arguments and classification labels",
    how: "The argument probe picks the option probe it resonates with most; for classify, the options are your labels",
  },
  {
    name: "embedding",
    decides: "Intent search and routing",
    how: "Attention-pooled utterance field, L2-normalised",
  },
];

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

export default function Arc1Page() {
  const m = ARC1_METRICS;
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <header className="not-prose mb-10">
        <h1 className="max-w-3xl text-4xl font-extrabold leading-[1.05] tracking-[-0.03em] text-zinc-50 sm:text-5xl">
          ARC 1 turns a sentence into a decision in one pass.
        </h1>
        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-zinc-400">
          An ultra-compact, non-autoregressive System 1 model built on ARCANE resonance. In a single forward pass it
          turns text into calibrated, typed output: tool calls with grounded arguments, extracted records, or a
          classification label, ready to drop into your software. It never writes prose, so there is no filler to
          wait for and nothing to hallucinate.
        </p>
        <dl className="mt-8 grid max-w-2xl grid-cols-3 border-y border-zinc-800">
          <div className="py-4 pr-4">
            <dt className="text-sm text-zinc-500">Parameters</dt>
            <dd className="mt-1 text-2xl font-semibold tabular-nums text-zinc-50">{(m.parameters / 1e6).toFixed(2)}M</dd>
          </div>
          <div className="border-l border-zinc-800 py-4 pl-4 pr-4">
            <dt className="text-sm text-zinc-500">Median latency</dt>
            <dd className="mt-1 text-2xl font-semibold tabular-nums text-zinc-50">{m.latencyP50Ms.toFixed(0)} ms</dd>
          </div>
          <div className="border-l border-zinc-800 py-4 pl-4">
            <dt className="text-sm text-zinc-500">Forward passes</dt>
            <dd className="mt-1 text-2xl font-semibold tabular-nums text-zinc-50">1</dd>
          </div>
        </dl>
        <p className="mt-3 text-xs text-zinc-500">
          Latency is measured per request on a laptop CPU, with {m.cycles} binding cycles and the schemas already cached.
        </p>
      </header>

      <Arc1Demo />

      <div className="text-zinc-300">
        <h2 id="architecture" className={h2}>
          Architecture
        </h2>
        <p className="max-w-2xl leading-relaxed">
          ARC 1 uses Resonant Schema Binding. Most tool-calling models re-read the sentence once for every tool
          and argument, or write the call out token by token. ARC 1 reads the sentence once, keeps every tool
          schema as a remembered vector, and lets all of them resonate against the sentence at the same time.
        </p>
        <Arc1ArchitectureDiagram />
        <div className="not-prose grid gap-px bg-zinc-800 sm:grid-cols-2">
          {[
            [
              "Perceive once",
              "The request lane runs a single time per request. Attention is bidirectional and pad-masked, so a word early in the sentence already knows what comes after it.",
            ],
            [
              "Remember schemas",
              "Tool, argument, and option texts go through the same blocks and are pooled into engrams. They rarely change, so they are cached and reused.",
            ],
            [
              "Bind by resonance",
              "Each probe queries the utterance field, and a spiking GSER gate controls how much it integrates. More cycles let probes settle; one cycle is fastest.",
            ],
            [
              "Read out, never generate",
              "Arguments are copied from the user's tokens or selected from the schema, and labels are selected from your label set, so every output is schema-valid and grounded in what was said.",
            ],
          ].map(([t, d]) => (
            <div key={t} className="bg-black p-4">
              <h3 className="text-sm font-semibold text-zinc-100">{t}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-zinc-400">{d}</p>
            </div>
          ))}
        </div>
        <p className="mt-4 max-w-2xl leading-relaxed text-zinc-400">
          Cost per request is one perception pass over the sentence plus a light binding step for each probe, which
          is why a {(m.parameters / 1e6).toFixed(1)}M-parameter model answers in milliseconds on a laptop CPU.
        </p>

        <h2 id="readouts" className={h2}>
          Readouts
        </h2>
        <dl className="not-prose divide-y divide-zinc-900 border-y border-zinc-800">
          {READOUTS.map((r) => (
            <div key={r.name} className="grid gap-1 py-3 md:grid-cols-[8rem_1fr_1fr] md:gap-4">
              <dt>
                <code className="text-zinc-100">{r.name}</code>
              </dt>
              <dd className="text-sm text-zinc-300">{r.decides}</dd>
              <dd className="text-sm text-zinc-500">{r.how}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 max-w-2xl leading-relaxed">
          Each readout has one temperature fitted on held-out data, so a reported 90% means right about nine times
          in ten. When no tool fires, ARC 1 returns no call rather than guessing.
        </p>

        <h2 id="results" className={h2}>
          Measured results
        </h2>
        <p className="max-w-2xl leading-relaxed">
          Evaluated on synthetic requests the model never trained on. Held-out values use unseen cities, names, and
          codes. Unseen tools are whole tools left out of training. Classification uses held-out sentences with
          label wordings the model has learned; intents of unseen tools are the zero-shot case, where the labels
          themselves are new. ARC 1 has no pretrained language knowledge, so accuracy drops when a sentence shares
          few words with anything it learned, as the support-routing row shows.
        </p>
        <div className="not-prose mt-4 overflow-x-auto">
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
                  ["Tool selection, held-out values", "toolSelection"],
                  ["Exact call, held-out values", "exactCall"],
                  ["Argument accuracy, held-out values", "argumentAcc"],
                  ["Correctly silent when no tool applies", "noToolAcc"],
                  ["Exact call, unseen tools", "exactCallUnseen"],
                  ["Extraction field F1", "extractionF1"],
                  ["Classification: intent category, held-out texts", "classifyIntent"],
                  ["Classification: sentiment, held-out sentences", "classifySentiment"],
                  ["Classification: support routing, held-out sentences", "classifySupport"],
                  ["Classification: intents of unseen tools", "classifyUnseen"],
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
                <td className="py-2.5 pr-4">Median latency per request</td>
                {m.byCycles.map((c) => (
                  <td key={c.cycles} className="py-2.5 pr-4 text-right">
                    {c.latencyP50Ms.toFixed(1)} ms
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-sm text-zinc-500">
          Calibration error for fire decisions is {pct(m.fireEceAfter)} after temperature fitting (
          {pct(m.fireEceBefore)} before). Intent retrieval@1 with embeddings is {pct(m.retrievalAt1)}.
        </p>

        <h2 id="use-it" className={h2}>
          Use it from Python
        </h2>
        <pre className="not-prose overflow-x-auto border border-zinc-800 bg-zinc-950 p-4 text-[13px] leading-relaxed text-zinc-300">
          {`from gpbacay_arcane import Arc1Agent, Arc1Config, Arc1Model, BytePairTokenizer, ToolParam, ToolSpec
import json

config = Arc1Config.from_dict(json.load(open("Models/arc1_arc1_tiny.config.json")))
model = Arc1Model(config).build_model()
model.load_weights("Models/arc1_arc1_tiny.weights.h5")
agent = Arc1Agent(model, BytePairTokenizer.load("Models/arc1_arc1_tiny_tokenizer.json"))

weather = ToolSpec("get_weather", "Get the current weather for a city.",
                   [ToolParam("city", description="City name")])
agent.run("is it raining in Tokyo?", tools=[weather], cycles=2)
# {"function_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
#  "confidence": ..., "latency_ms": ..., ...}

agent.classify("my card was charged twice", ["billing", "technical support", "sales"])
# {"label": "billing", "confidence": ..., "distribution": {...}, ...}`}
        </pre>

        <h2 id="run-locally" className={h2}>
          Run the sandbox locally
        </h2>
        <p className="max-w-2xl leading-relaxed">
          From <code className={code}>arcane-docs-web</code>, run <code className={code}>npm run dev:with-arc1</code>.
          It starts this site and the ARC 1 API (<code className={code}>examples/serve_arc1_api.py</code>) on port
          8002. To train your own weights, run{" "}
          <code className={code}>python examples/train_arc1.py --preset arc1-tiny</code>. To export for devices, run{" "}
          <code className={code}>python examples/export_arc1.py --tflite</code>.
        </p>
        <p className="mt-4 max-w-2xl leading-relaxed text-zinc-400">
          Limits: one call per tool per request, and very long text is truncated to {m.seqLen} tokens. ARC 1 decides,
          extracts, and classifies. For open-ended answers, pair it with the{" "}
          <Link href="/docs/chat" className="font-medium text-[#C785F2] underline hover:text-[#d49cf5]">
            ARCANE small language model
          </Link>
          .
        </p>
      </div>
    </div>
  );
}
