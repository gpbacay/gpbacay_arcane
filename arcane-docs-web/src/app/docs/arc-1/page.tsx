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

/* Rows: what the job needs, then how a generative LLM, a fixed classifier, and ARC 1 handle it. */
const COMPARISON = [
  ["Output", "Free text you parse and hope is valid JSON", "One label from a fixed set", "Typed calls, records, or labels, valid by construction"],
  ["Tools and labels", "Described in the prompt", "Fixed at training time", "Passed with each request, cached as engrams"],
  ["Argument values", "Generated, can be invented", "Not supported", "Copied from the input text"],
  ["Cost per decision", "One decoding step per output token", "One pass", "One pass, whatever the output size"],
  ["When nothing fits", "Often calls something anyway", "Always picks a label", "Returns no call, with calibrated confidence"],
  ["Where it runs", "GPU or a hosted API", "Anywhere", "A laptop CPU or small device, from a sub-megabyte file"],
];

/* The ideas ARC 1 introduces; each ties to a readout or component described further down. */
const NOVELTY = [
  [
    "Schemas are inputs, not memorised outputs",
    "Every tool, argument, option, and label is encoded as a schema engram and bound to the sentence in parallel. You change the tool set per request, with no retraining and no prompt engineering.",
  ],
  [
    "Arguments are pointed to, not written",
    "The anchor readout picks a start and end position in the input. A city or a name is always a span of what the user wrote, so ARC 1 cannot invent a value that isn't there.",
  ],
  [
    "Options are chosen from your list",
    "Enum arguments and labels are selected by resonance between the parameter and each option. The answer is always one of the options you offered.",
  ],
  [
    "It knows when to stay silent",
    "Each tool gets a calibrated fire probability. When no tool fits, nothing fires, so a request like “tell me a joke” produces no call instead of a wrong one.",
  ],
];

/* What a decoder emits one step at a time for the hero's call. */
const DECODE_STEPS = ['{"name"', ': "convert', '_currency"', ', "arguments"', ': {"amount"', ": 100", ', "from', '_currency"', ': "USD"', ", ..."];

/* Illustrative gate openness per cycle (0-1) and the word each probe attends to. Not measured values. */
const PROBES = [
  { name: "convert_currency", kind: "tool", word: "convert", gates: [0.55, 0.85, 0.96] },
  { name: "amount", kind: "anchor", word: "100", gates: [0.45, 0.8, 0.94] },
  { name: "from_currency", kind: "anchor", word: "USD", gates: [0.4, 0.78, 0.93] },
  { name: "to_currency", kind: "anchor", word: "PHP", gates: [0.35, 0.72, 0.92] },
  { name: "get_weather", kind: "tool", word: "to", gates: [0.22, 0.09, 0.03] },
];

/* The four updates inside ResonantBinding, in order, from gpbacay_arcane/mechanisms.py. */
const CYCLE_STEPS = [
  ["Attend", "Where does it resonate?", "a = softmax(Wq·p · Wk·U)", "The probe scores every token of the utterance and focuses on the ones that match it."],
  ["Hear", "What does it hear?", "r = Wl · Σ a·Wv·U", "It pulls in a summary of the tokens it focused on."],
  ["Gate", "Is it a real match?", "g = σ((Wg[p; r] − θ) / leak)", "A GSER spiking gate opens only when what it heard fits the probe, and stays shut otherwise."],
  ["Settle", "Update and repeat", "p ← p + g·r + Mixer(p)", "The probe absorbs what passed the gate, then runs the next cycle with the same weights."],
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
    { label: "Feed ranking", value: full.classifyFeed },
    { label: "Search intent", value: full.classifySearch },
    { label: "Product recommendation", value: full.classifyProducts },
    { label: "Unseen-tool intents", value: full.classifyUnseen },
    { label: "Unseen-tool calls", value: full.exactCallUnseen },
  ]
    .sort((a, b) => b.value - a.value)
    .map((d) => ({ ...d, detail: `${pct(d.value)} on held-out data, ${full.cycles} cycles` }));
  // ponytail: fixed 85% cut between "strong" and "still weak"; tune if the metrics shift.
  const strong = accuracy.filter((d) => d.value >= 0.85);
  const weak = accuracy.filter((d) => d.value < 0.85);
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
          ARC 1 is a small model that reads a request and decides what to do: which tool to call and with which
          arguments, which fields to pull out, or which label applies. Instead of writing an answer, it uses{" "}
          <strong className="font-semibold text-zinc-200">Resonant Schema Binding</strong>: every possible answer
          listens to the sentence at once, and the ones that match settle into place. Every result fits the schema you
          gave it.
        </p>
        <figure className="mt-8 max-w-2xl border border-zinc-800 bg-zinc-950">
          <div className="border-b border-zinc-800 px-4 py-4 sm:px-5">
            <p className="flex flex-wrap items-start gap-x-[0.3em] gap-y-2 text-xl leading-snug text-zinc-300">
              <span>convert</span>
              {(
                [
                  ["100", "amount", "#B9DFE0"],
                  ["USD", "from_currency", "#F294C0"],
                ] as const
              ).map(([word, arg, color]) => (
                <span key={arg} className="inline-flex flex-col items-center">
                  <mark className="px-1 text-zinc-50" style={{ boxShadow: `inset 0 -2px 0 ${color}`, background: `${color}24` }}>
                    {word}
                  </mark>
                  <span className="mt-1 whitespace-nowrap px-1 text-[11px] font-medium leading-none" style={{ color }}>
                    {arg}
                  </span>
                </span>
              ))}
              <span>to</span>
              <span className="inline-flex flex-col items-center">
                <mark className="px-1 text-zinc-50" style={{ boxShadow: "inset 0 -2px 0 #9DE4FA", background: "#9DE4FA24" }}>
                  PHP
                </mark>
                <span className="mt-1 whitespace-nowrap px-1 text-[11px] font-medium leading-none text-[#9DE4FA]">
                  to_currency
                </span>
              </span>
            </p>
          </div>
          <pre className="overflow-x-auto px-4 py-3 text-[13px] leading-relaxed text-zinc-300 sm:px-5">
            {`{"name": "convert_currency",
 "arguments": {"amount": 100, "from_currency": "USD", "to_currency": "PHP"}}`}
          </pre>
          <figcaption className="border-t border-zinc-800 px-4 py-2.5 text-xs text-zinc-500 sm:px-5">
            One forward pass, no decoding. Each argument is a span ARC 1 resonated with in the sentence, not text it
            generated.{" "}
            <a href="#resonant-schema-binding" className="text-[#C785F2] underline hover:text-[#d49cf5]">
              How Resonant Schema Binding works
            </a>
          </figcaption>
        </figure>
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

      <p className="not-prose mb-3 text-sm text-zinc-400">
        Try it below. Pick an example scene, or type your own request.
      </p>
      <Arc1Demo />

      <div className="text-zinc-300">
        <h2 id="problem" className={h2}>
          The problem it solves
        </h2>
        <p className={p}>
          Most automation comes down to small decisions: which action a message asks for, which values to fill in, which
          queue a ticket belongs to. Today these are usually handed to a large language model that writes its answer
          token by token. That works, but it is slow and expensive for a yes-or-no choice, needs a GPU or a hosted API,
          and can return malformed JSON, a tool that doesn&apos;t exist, or an argument value the user never said.
        </p>
        <p className={`${p} mt-4`}>
          ARC 1 is built only for the decision. It reads the request once and returns a typed answer, so it can run
          next to the app that needs it, on every message, in a few milliseconds.
        </p>
        <div className="not-prose mt-6 overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className="w-36 py-2 pr-4 font-medium" />
                <th className="py-2 pr-4 font-medium">Generative LLM</th>
                <th className="py-2 pr-4 font-medium">Fixed classifier</th>
                <th className="py-2 font-medium text-zinc-100">ARC 1</th>
              </tr>
            </thead>
            <tbody>
              {COMPARISON.map(([row, llm, clf, arc]) => (
                <tr key={row} className="border-b border-zinc-900 align-top">
                  <th scope="row" className="py-2.5 pr-4 font-medium text-zinc-300">
                    {row}
                  </th>
                  <td className="py-2.5 pr-4 text-zinc-500">{llm}</td>
                  <td className="py-2.5 pr-4 text-zinc-500">{clf}</td>
                  <td className="border-l-2 border-[#C785F2] bg-[#C785F2]/[0.06] py-2.5 pl-3 text-zinc-100">{arc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={`${p} mt-4 text-sm text-zinc-500`}>
          ARC 1 does not replace a language model for open-ended answers. It replaces one for the routing decision in
          front of it.
        </p>

        <h2 id="resonant-schema-binding" className={h2}>
          Resonant Schema Binding
        </h2>
        <p className={`${p} text-lg text-zinc-200`}>
          A language model decides by writing. ARC 1 decides by resonating: every possible answer listens to the sentence
          at the same time, and the ones that match it settle into place.
        </p>
        <p className={`${p} mt-4`}>
          This is the mechanism behind ARC 1, and the reason it can be this small and this fast. Each tool, argument,
          option, and label becomes a <em>probe</em>. The sentence is read once into a field of tokens. Then all probes
          bind to that field in parallel over a few short cycles. No output is generated, so there is nothing to decode.
        </p>

        <h3 id="decoding-vs-resonance" className={h3}>
          Decoding versus resonance
        </h3>
        <div className="not-prose mt-4 grid gap-px border border-zinc-800 bg-zinc-800 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
          <div className="bg-zinc-950 p-4 sm:p-5">
            <p className="font-semibold text-zinc-300">Decoding</p>
            <p className="mt-0.5 text-xs text-zinc-500">A generative model writes the call one token at a time.</p>
            <ol className="mt-4 flex flex-wrap gap-1.5" aria-label="Decoding steps">
              {DECODE_STEPS.map((t, i) => (
                <li key={i} className="flex flex-col border border-zinc-800 bg-black px-2 py-1">
                  <span className="text-[10px] tabular-nums text-zinc-600">step {i + 1}</span>
                  <code className="whitespace-pre text-xs text-zinc-400">{t}</code>
                </li>
              ))}
            </ol>
            <p className="mt-4 text-sm text-zinc-400">
              Cost grows with the length of the answer, each step waits for the last, and any token can go wrong.
            </p>
          </div>
          <div className="bg-zinc-950 p-4 sm:p-5">
            <p className="font-semibold text-zinc-100">Resonance</p>
            <p className="mt-0.5 text-xs text-zinc-500">
              ARC 1 runs every probe at once. Bars show how far each probe&apos;s gate opens per cycle.
            </p>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[380px] border-collapse text-left text-xs">
                <thead>
                  <tr className="text-zinc-500">
                    <th className="pb-2 pr-3 font-medium">Probe</th>
                    {[1, 2, 3].map((c) => (
                      <th key={c} className="pb-2 pr-3 font-medium">
                        Cycle {c}
                      </th>
                    ))}
                    <th className="pb-2 font-medium">Reads</th>
                  </tr>
                </thead>
                <tbody>
                  {PROBES.map((pr) => {
                    const open = pr.gates[pr.gates.length - 1] >= 0.5;
                    return (
                      <tr key={pr.name} className="border-t border-zinc-900">
                        <th scope="row" className="py-2 pr-3 font-normal">
                          <code className={open ? "text-zinc-100" : "text-zinc-500"}>{pr.name}</code>
                        </th>
                        {pr.gates.map((g, i) => (
                          <td key={i} className="py-2 pr-3">
                            <div className="h-1.5 w-full min-w-[3rem] bg-zinc-900" aria-label={`gate ${Math.round(g * 100)}% open`}>
                              <div className="h-full" style={{ width: `${g * 100}%`, background: open ? "#C785F2" : "#3f3f46" }} />
                            </div>
                          </td>
                        ))}
                        <td className="py-2">
                          {open ? (
                            <mark className="bg-[#C785F2]/15 px-1 text-zinc-50">{pr.kind === "tool" ? "fires" : pr.word}</mark>
                          ) : (
                            <span className="text-zinc-500">stays silent</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-4 text-sm text-zinc-300">
              Cost is fixed: one pass over the sentence plus {m.cycles} short cycles, whatever the answer&apos;s length.
              A tool that doesn&apos;t match simply never opens its gate.
            </p>
            <p className="mt-2 text-[11px] text-zinc-600">Gate values are illustrative, not measured.</p>
          </div>
        </div>

        <h3 id="binding-cycle" className={h3}>
          One binding cycle
        </h3>
        <p className={p}>
          Each probe <code className={code}>p</code> runs the same four steps against the utterance field{" "}
          <code className={code}>U</code>, with weights shared across cycles.
        </p>
        <ol className="not-prose mt-5 grid gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-2 xl:grid-cols-4">
          {CYCLE_STEPS.map(([name, question, formula, body], i) => (
            <li key={name} className="flex flex-col bg-zinc-950 p-4">
              <div className="flex items-baseline gap-2">
                <span className="text-sm tabular-nums text-[#C785F2]">{i + 1}</span>
                <span className="font-semibold text-zinc-100">{name}</span>
              </div>
              <p className="mt-0.5 text-xs text-zinc-500">{question}</p>
              <code className="mt-3 block overflow-x-auto whitespace-nowrap bg-black px-2 py-1.5 text-[12px] text-zinc-300">
                {formula}
              </code>
              <p className="mt-3 text-sm leading-relaxed text-zinc-400">{body}</p>
            </li>
          ))}
        </ol>

        <h3 id="why-resonance" className={h3}>
          Why it matters
        </h3>
        <ul className="max-w-2xl space-y-2">
          <li>
            <strong className="text-zinc-100">One read, many decisions.</strong> The sentence&apos;s keys and values are
            computed once and shared by every probe, so adding a tool adds a few probes, not another pass.
          </li>
          <li>
            <strong className="text-zinc-100">Silence is built in.</strong> The spiking gate gives a non-matching probe
            nothing to absorb, which is how ARC 1 rejects {pct(full.noToolAcc)} of requests no tool fits.
          </li>
          <li>
            <strong className="text-zinc-100">Speed on a dial.</strong> Every cycle count is trained, so you can run 1
            cycle ({m.byCycles[0].latencyP50Ms.toFixed(1)} ms, no-tool rejection {pct(m.byCycles[0].noToolAcc)}) or{" "}
            {full.cycles} ({full.latencyP50Ms.toFixed(1)} ms, {pct(full.noToolAcc)}) from the same weights.
          </li>
          <li>
            <strong className="text-zinc-100">Neuromorphic, not a decoder.</strong> Perception uses ARCANE&apos;s field
            resonance and spiking channel mixers; binding uses GSER gates. There is no token-by-token decoder anywhere
            in the model.
          </li>
        </ul>

        <h2 id="whats-new" className={h2}>
          What&apos;s new about it
        </h2>
        <p className={p}>
          Resonant Schema Binding binds a description of every possible answer to the input and reads the result
          directly. That design gives ARC 1 properties a generative model doesn&apos;t have.
        </p>
        <dl className="not-prose mt-6 grid gap-x-10 gap-y-6 sm:grid-cols-2">
          {NOVELTY.map(([title, body]) => (
            <div key={title} className="border-t border-zinc-800 pt-4">
              <dt className="font-semibold text-zinc-100">{title}</dt>
              <dd className="mt-1.5 text-sm leading-relaxed text-zinc-400">{body}</dd>
            </div>
          ))}
          <div className="border-t border-[#C785F2] pt-4">
            <dt className="font-semibold text-zinc-100">Small enough to ship inside an app</dt>
            <dd className="mt-1.5 text-sm leading-relaxed text-zinc-400">
              {(m.parameters / 1e6).toFixed(2)}M parameters in a {mb(m.rcn.bytes)} file, with a median latency of{" "}
              {m.latencyP50Ms.toFixed(1)} ms on a laptop CPU. The model, tokenizer, and calibration travel together in
              one <code className={code}>.rcn</code> file.
            </dd>
          </div>
        </dl>

        <h2 id="results" className={h2}>
          Accuracy
        </h2>
        <p className={p}>
          All figures come from synthetic data held out from training. Tool figures use argument values ARC 1 never saw;
          the unseen-tool rows use tools excluded from training entirely.
        </p>
        <div className="not-prose mt-6 grid max-w-3xl gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-2">
          {(
            [
              ["Strong", "Decisions over the vocabulary of its schemas", strong, "#C785F2"],
              ["Still weak", "Paraphrases and tools unlike its training data", weak, "#52525b"],
            ] as const
          ).map(([title, hint, rows, color]) => (
            <div key={title} className="bg-zinc-950 p-4 sm:p-5">
              <h3 className="font-semibold text-zinc-100">{title}</h3>
              <p className="mt-0.5 text-xs text-zinc-500">{hint}</p>
              <ul className="mt-4 space-y-2.5">
                {rows.map((r) => (
                  <li key={r.label} className="flex items-baseline justify-between gap-4 text-sm">
                    <span className="text-zinc-300">{r.label}</span>
                    <span className="font-semibold tabular-nums" style={{ color: color === "#52525b" ? "#a1a1aa" : color }}>
                      {pct(r.value)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <p className={`${p} mt-4`}>
          Confidence values are calibrated: tool fire decisions have an expected calibration error of{" "}
          {pct(m.fireEceAfter)}, so a 90% confidence means about 90% of such calls are right. That makes a simple
          threshold a reliable way to hand uncertain requests to a person or a larger model.
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
          ARC 1 has no pretrained language knowledge, so it relies on the words in a request matching the words in its
          schemas. Label hints help: <code className={code}>billing: charges, refunds</code> gives ARC 1 more words to
          match than <code className={code}>billing</code> alone.
        </p>
        <details className="not-prose mt-6 border-y border-zinc-800">
          <summary className="cursor-pointer py-3 text-sm text-zinc-300 hover:text-zinc-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-[#C785F2]">
            All metrics by binding-cycle count
          </summary>
          <div className="overflow-x-auto pb-2">
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
                    ["Classification, feed ranking", "classifyFeed"],
                    ["Classification, search intent", "classifySearch"],
                    ["Classification, product recommendation", "classifyProducts"],
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
                <tr>
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
        </details>

        <h2 id="architecture" className={h2}>
          How it works
        </h2>
        <p className={p}>
          Under <strong className="text-zinc-100">Resonant Schema Binding</strong>, the input is perceived once.
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
