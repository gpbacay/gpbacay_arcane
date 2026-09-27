import Link from "next/link";
import type { ReactNode } from "react";
import { ArrowDown, ArrowRight, Ban, Crosshair, Download, Gauge, Layers, Radar } from "lucide-react";
import { Arc1Demo } from "@/components/Arc1Demo";
import { BarChart } from "@/components/BarChart";
import { Mermaid } from "@/components/markdown";
import { ARC1_METRICS } from "@/config/arc1-metrics";

const h2Base = "scroll-mt-28 text-2xl font-bold tracking-tight text-zinc-100 mb-4 border-b border-zinc-800 pb-2";
const h2 = `${h2Base} mt-16`;
const h3 = "scroll-mt-28 text-lg font-semibold text-zinc-100 mt-10 mb-2";
const code = "bg-zinc-900 px-1.5 py-0.5 text-zinc-100";
const p = "max-w-2xl leading-relaxed";
const link = "font-medium text-[#C785F2] underline hover:text-[#d49cf5]";

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

/* What sets ARC 1 apart; each ties to a mechanism described further down the page. */
const CAPABILITIES = [
  {
    icon: Layers,
    title: "Your tools, labels, and fields are inputs",
    body: "You describe the options with each request. ARC 1 reads the descriptions alongside the sentence, so changing the list doesn't mean building a new model.",
  },
  {
    icon: Crosshair,
    title: "It points instead of writing",
    body: "Names, places, and amounts are marked in your own text, never typed out from memory. Text values are always copied from your words, and labels always come from your list.",
  },
  {
    icon: Ban,
    title: "It knows when to do nothing",
    body: "Each tool has to earn its place. When nothing fits, like “tell me a joke” in a smart-home app, it returns no call instead of guessing one.",
  },
  {
    icon: Gauge,
    title: "Its confidence means something",
    body: "On the test set, a tool it was 90% sure of was the right tool about 90% of the time, so a simple threshold can decide when to ask a person.",
  },
  {
    icon: Radar,
    title: "It understands similarity",
    body: "Every request also becomes a point on a map of meaning, useful for search, routing, and spotting duplicates.",
  },
  {
    icon: ArrowRight,
    title: "One read, one small file",
    body: "Every option is weighed in the same single pass, on an ordinary CPU, from a file you can copy anywhere.",
  },
];

const GLOSSARY = [
  ["Tool", "An action your app can take, like sending a message or converting currency."],
  ["Argument", "A detail a tool needs, like the city for a weather lookup."],
  ["Schema", "The list of tools, fields, or labels you give ARC 1 for a request."],
  ["Binding cycle", "One round in which every possible answer checks itself against the sentence."],
  ["Held-out data", "Test examples kept apart during training, so scores reflect new inputs."],
];

/* ARCANE parts ARC 1 is assembled from: [class, plain-language role, module, docs link]. */
const ARCANE_PARTS: [string, string, string, string | null][] = [
  ["FieldAttention", "Reads the whole sentence at once, so every word sees every other word.", "mechanisms", null],
  ["ConceptEngram", "A memory of short word patterns that helps it recognise names, codes, and numbers.", "mechanisms", null],
  ["ResonantChannelMixer", "Mixes what it has read through a spiking gate, using very few weights.", "layers", "/docs/layers"],
  ["FieldResonance", "Pulls every word toward the overall meaning of the sentence, then resets, like a neuron firing.", "mechanisms", "/docs/neural-resonance"],
  ["GSER gate", "The spiking gate that opens when a candidate matches the sentence, taken from ARCANE's Gated Spiking Elastic Reservoir.", "mechanisms", "/docs/resonant-gser"],
  ["ResonantBinding", "Lets every tool, field, and label listen to the sentence over a few rounds and settle.", "mechanisms", null],
  ["Arc1Agent", "The helper your app talks to. It drops tool calls with unknown tools, unknown arguments, or options outside your list.", "tools", null],
  ["load_rcn", "Opens the single-file .rcn model, with the tokenizer and calibration inside.", "rcn", null],
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
  ["Attend", "Where does it resonate?", "a = softmax(Wq·p · Wk·U)", "The probe scores every word of the sentence and focuses on the ones that match it."],
  ["Hear", "What does it hear?", "r = Wl · Σ a·Wv·U", "It pulls in a summary of the words it focused on."],
  ["Gate", "Is it a real match?", "g = σ((Wg[p; r] − θ) / leak)", "A GSER spiking gate opens only when what it heard fits the probe, and stays shut otherwise."],
  ["Settle", "Update and repeat", "p ← p + g·r + Mixer(p)", "The probe absorbs what passed the gate, then runs the next cycle with the same weights."],
];

/*
 * Published figures from https://laya.convaiinnovations.com/ (checked September 2026): English checkpoint,
 * ModernBERT-large. Update these if that page changes.
 */
const LAYA = { params: 421e6, downloadBytes: 808e6, gpuLatencyMs: 32.8, ece: 0.081 };

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

function mb(bytes: number) {
  return `${(bytes / 1e6).toFixed(2)} MB`;
}

function times(ratio: number) {
  return `${Number(ratio.toPrecision(3)).toLocaleString("en-US")}x`;
}

function PlainWords({ children }: { children: ReactNode }) {
  return (
    <aside className="not-prose my-6 max-w-2xl border-l-2 border-[#B9DFE0] bg-[#B9DFE0]/[0.05] px-4 py-3">
      <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#B9DFE0]">In plain words</p>
      <div className="mt-1.5 text-[15px] leading-relaxed text-zinc-200">{children}</div>
    </aside>
  );
}

export default function Arc1Page() {
  const m = ARC1_METRICS;
  const full = m.byCycles[m.byCycles.length - 1];
  const fast = m.byCycles[0];
  const of = (v: number, n: number) => `${Math.round(v * n)} of ${n}`;



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

  const COMPARE: [string, string, string, string][] = [
    ["Size", "Usually billions of parameters", `${(LAYA.params / 1e6).toFixed(0)}M parameters`, `${(m.parameters / 1e6).toFixed(2)}M parameters`],
    ["Download", "Usually gigabytes", `About ${(LAYA.downloadBytes / 1e6).toFixed(0)} MB`, mb(m.rcn.bytes)],
    ["Hardware", "Usually a GPU or a paid API", `A GPU, ${LAYA.gpuLatencyMs} ms per decision`, `A laptop CPU, ${full.latencyP50Ms.toFixed(1)} ms median`],
    ["Tool calls with details", "Written word by word, values can be invented", "Not offered: choose, score, or yes/no", "Built in, text values copied from the input"],
    ["Pulling out fields", "Generated, can be invented", "Not offered", `Built in, copied from the text, field F1 ${pct(full.extractionF1)}`],
    ["Calibration error", "Varies by model", `${pct(LAYA.ece)} (its own tasks)`, `${pct(m.fireEceAfter)} on tool decisions`],
  ];

  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <header className="not-prose mb-12">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#C785F2]">ARCANE · Technical report</p>
        <h1 className="mt-3 max-w-3xl text-4xl font-extrabold leading-[1.08] tracking-[-0.03em] text-zinc-50 sm:text-5xl">
          ARC 1: Grounded Structured Decisions via Resonant Schema Binding
        </h1>
        <section aria-label="Abstract" className="mt-6 max-w-2xl border-l-2 border-zinc-800 pl-5">
          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-zinc-500">Abstract</p>
          <p className="mt-2 text-[17px] leading-relaxed text-zinc-300">
            Applications increasingly delegate small structured decisions, such as which tool to call, which fields to
            extract, and which label applies, to large generative language models. These models decode token by token,
            typically require a GPU or a hosted API, and can emit malformed output or argument values absent from the
            input. I introduce ARC 1, a {(m.parameters / 1e6).toFixed(2)}M-parameter neuromimetic model built for these
            decisions alone: tool calling, field extraction, classification, and semantic retrieval. In place of
            generation, ARC 1 uses Resonant Schema Binding: each tool, argument, option, and label is encoded as a schema
            engram, and all of them bind to the input in parallel through a few cycles of spiking, gated resonance built
            from the ARCANE library. Argument
            values are selected as spans of the input and enum values are selected from the schema, so the model cannot
            invent text that is in neither, and it abstains when no tool applies. Grounded is not the same as correct:
            it can still pick the wrong span, option, or tool. On held-out synthetic data, ARC 1 reaches{" "}
            {pct(full.toolSelection)} tool selection, {pct(full.exactCall)} exact-call accuracy,{" "}
            {pct(full.extractionF1)} extraction F1, and {pct(full.noToolAcc)} correct abstention, with an expected
            calibration error of {pct(m.fireEceAfter)}. It runs in {full.latencyP50Ms.toFixed(1)} ms median on a laptop
            CPU from a single {mb(m.rcn.bytes)} file. Generalization to tools absent from training remains limited (
            {pct(full.exactCallUnseen)} exact calls).
          </p>
        </section>

        <div className="mt-8 flex flex-wrap items-center gap-3">
          <a
            href="#try-it"
            className="inline-flex items-center gap-2 bg-white px-4 py-2.5 text-sm font-medium text-black transition-colors hover:bg-zinc-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#C785F2]"
          >
            <ArrowDown className="h-4 w-4" aria-hidden />
            Try your own request
          </a>
          <a
            href={m.rcn.file}
            download
            className="inline-flex items-center gap-2 border border-zinc-700 px-4 py-2.5 text-sm font-medium text-zinc-100 transition-colors hover:border-zinc-500 hover:bg-zinc-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#C785F2]"
          >
            <Download className="h-4 w-4" aria-hidden />
            Download arc1-tiny.rcn
          </a>
        </div>
      </header>

      <div className="text-zinc-300">
        <h2 id="try-it" className={h2Base}>
          Try it live
        </h2>
        <p className={`${p} mb-5`}>
          This runs the real model on a live server. Pick an example scene, or type your own request and press Enter.
          The panel shows what ARC 1 found in your sentence and how sure it was.
        </p>
        <Arc1Demo />

        <h2 id="what-it-does" className={h2}>
          What ARC 1 does
        </h2>
        <p className={p}>
          Most automation comes down to small decisions: what a message is asking for, which details to fill in, and
          which queue it belongs to. ARC 1 is built only for those decisions, and that focus gives it abilities a
          general chat model doesn&apos;t have.
        </p>
        <ul className="not-prose mt-6 grid gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-2 lg:grid-cols-3">
          {CAPABILITIES.map(({ icon: Icon, title, body }) => (
            <li key={title} className="bg-zinc-950 p-4 sm:p-5">
              <Icon className="h-5 w-5 text-[#C785F2]" aria-hidden />
              <p className="mt-3 font-semibold text-zinc-100">{title}</p>
              <p className="mt-1.5 text-sm leading-relaxed text-zinc-400">{body}</p>
            </li>
          ))}
        </ul>
        <PlainWords>
          ARC 1 is not a chatbot. It doesn&apos;t write sentences. It reads a request, then fills in a form you designed:
          which button to press and what to type in each box. That&apos;s why its answers always fit your app, and why
          it can be thousands of times smaller than a typical chat model.
        </PlainWords>
        <details className="not-prose max-w-2xl border-y border-zinc-800">
          <summary className="cursor-pointer py-3 text-sm text-zinc-300 hover:text-zinc-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-[#C785F2]">
            Words used on this page
          </summary>
          <dl className="grid gap-x-6 gap-y-3 pb-4 text-sm sm:grid-cols-[9rem_1fr]">
            {GLOSSARY.map(([term, def]) => (
              <div key={term} className="contents">
                <dt className="font-medium text-zinc-100">{term}</dt>
                <dd className="text-zinc-400">{def}</dd>
              </div>
            ))}
          </dl>
        </details>

        <h2 id="arcane" className={h2}>
          Built on the ARCANE library
        </h2>
        <p className={p}>
          ARCANE is a Python library of neuromimetic layers: pieces of neural network that borrow ideas from how real
          neurons work, such as resonance, spiking, and gating. ARC 1 is assembled entirely from those
          parts, which shows that brain-inspired layers can do useful, everyday work, not only research
          experiments.
        </p>
        <PlainWords>
          Think of ARCANE as a box of specialised building blocks, and ARC 1 as a machine built from that box. Every
          part listed below ships in the <code className={code}>gpbacay_arcane</code> package, so you can use the same
          blocks to build your own models.
        </PlainWords>
        <div className="not-prose overflow-x-auto">
          <table className="w-full min-w-[560px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className="py-2 pr-4 font-medium">ARCANE part</th>
                <th className="py-2 pr-4 font-medium">What it does in ARC 1</th>
                <th className="py-2 font-medium">Module</th>
              </tr>
            </thead>
            <tbody>
              {ARCANE_PARTS.map(([name, role, mod, href]) => (
                <tr key={name} className="border-b border-zinc-900 align-top">
                  <td className="whitespace-nowrap py-2.5 pr-4">
                    {href ? (
                      <Link href={href} className="text-zinc-100 underline decoration-zinc-600 hover:decoration-[#C785F2]">
                        <code>{name}</code>
                      </Link>
                    ) : (
                      <code className="text-zinc-100">{name}</code>
                    )}
                  </td>
                  <td className="py-2.5 pr-4 text-zinc-300">{role}</td>
                  <td className="whitespace-nowrap py-2.5 font-mono text-xs text-zinc-500">{mod}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          There is no token-by-token decoder anywhere in the model. For the ideas behind these parts, see{" "}
          <Link href="/docs/neural-resonance" className={link}>
            Neural Resonance
          </Link>{" "}
          and{" "}
          <Link href="/docs/resonant-gser" className={link}>
            Resonant GSER
          </Link>
          .
        </p>

        <h2 id="how-it-works" className={h2}>
          How it decides
        </h2>
        <p className={`${p} text-lg text-zinc-200`}>
          A chat model decides by writing. ARC 1 decides by resonating: every possible answer listens to the sentence at
          the same time, and the ones that match it settle into place.
        </p>
        <PlainWords>
          Picture a roll call. Each tool, each detail, and each label is a person in the room, and ARC 1 reads your
          sentence out loud once. Whoever hears their name speaks up, and everyone else stays quiet. A few quick rounds
          later the answer is settled. I call this <strong>Resonant Schema Binding</strong>.
        </PlainWords>

        <h3 id="decoding-vs-resonance" className={h3}>
          Writing versus resonating
        </h3>
        <div className="not-prose mt-4 grid grid-cols-1 gap-px border border-zinc-800 bg-zinc-800 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
          <div className="bg-zinc-950 p-4 sm:p-5">
            <p className="font-semibold text-zinc-300">Writing (a chat model)</p>
            <p className="mt-0.5 text-xs text-zinc-500">The call is produced one piece at a time.</p>
            <ol className="mt-4 flex flex-wrap gap-1.5" aria-label="Decoding steps">
              {DECODE_STEPS.map((t, i) => (
                <li key={i} className="flex flex-col border border-zinc-800 bg-black px-2 py-1">
                  <span className="text-[10px] tabular-nums text-zinc-600">step {i + 1}</span>
                  <code className="whitespace-pre text-xs text-zinc-400">{t}</code>
                </li>
              ))}
            </ol>
            <p className="mt-4 text-sm text-zinc-400">
              Longer answers take longer, each step waits for the last, and any piece can go wrong.
            </p>
          </div>
          <div className="bg-zinc-950 p-4 sm:p-5">
            <p className="font-semibold text-zinc-100">Resonating (ARC 1)</p>
            <p className="mt-0.5 text-xs text-zinc-500">
              Every candidate listens at once. Bars show how far each one&apos;s gate opens per round.
            </p>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[380px] border-collapse text-left text-xs">
                <thead>
                  <tr className="text-zinc-500">
                    <th className="pb-2 pr-3 font-medium">Candidate</th>
                    {[1, 2, 3].map((c) => (
                      <th key={c} className="pb-2 pr-3 font-medium">
                        Round {c}
                      </th>
                    ))}
                    <th className="pb-2 font-medium">Result</th>
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
              The cost is fixed: one read of the sentence plus {m.cycles} short rounds, however long the answer is. A tool
              that doesn&apos;t match keeps its gate shut.
            </p>
            <p className="mt-2 text-[11px] text-zinc-600">Gate values are illustrative, not measured.</p>
          </div>
        </div>

        <h3 id="binding-cycle" className={h3}>
          What happens in one round
        </h3>
        <p className={p}>
          Each candidate runs the same four steps against the sentence, reusing the same weights every round.
        </p>
        <ol className="not-prose mt-5 grid gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-2 xl:grid-cols-4">
          {CYCLE_STEPS.map(([name, question, formula, body], i) => (
            <li key={name} className="flex flex-col bg-zinc-950 p-4">
              <div className="flex items-baseline gap-2">
                <span className="text-sm tabular-nums text-[#C785F2]">{i + 1}</span>
                <span className="font-semibold text-zinc-100">{name}</span>
              </div>
              <p className="mt-0.5 text-xs text-zinc-500">{question}</p>
              <p className="mt-3 text-sm leading-relaxed text-zinc-400">{body}</p>
              <code className="mt-auto block overflow-x-auto whitespace-nowrap bg-black px-2 py-1.5 pt-1.5 text-[12px] text-zinc-500">
                {formula}
              </code>
            </li>
          ))}
        </ol>
        <ul className="mt-6 max-w-2xl space-y-2">
          <li>
            <strong className="text-zinc-100">It copies instead of inventing.</strong> Details such as a city or an
            amount are pointed to in your sentence, so a text value is always words you wrote. It can still point to
            the wrong words, and numbers are converted to their type, so &ldquo;three&rdquo; comes back as 3.
          </li>
          <li>
            <strong className="text-zinc-100">It knows when to stay quiet.</strong> When no tool fits, like &ldquo;tell
            me a joke&rdquo;, no gate opens and ARC 1 returns no call. It did this for {pct(full.noToolAcc)} of such
            test requests.
          </li>
          <li>
            <strong className="text-zinc-100">One read, many decisions.</strong> The sentence is read once and shared
            by every candidate, so adding a tool adds a few candidates, not another pass. Even a single round gets{" "}
            {pct(fast.exactCall)} of calls exactly right, against {pct(full.exactCall)} with {full.cycles}.
          </li>
        </ul>

        <h2 id="accuracy" className={h2}>
          How accurate it is
        </h2>
        <p className={p}>
          I test ARC 1 on examples it never saw during training. The tool tests even use values, such as city names and
          amounts, that never appeared in training. Here is what that looks like in everyday numbers.
        </p>
        <ul className="not-prose mt-6 grid gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-2">
          {[
            [pct(full.toolSelection), "picked the right tool", `${of(full.toolSelection, m.n.tools)} requests`],
            [pct(full.exactCall), "got the whole call exactly right, tool and every detail", `${of(full.exactCall, m.n.tools)} requests`],
            [pct(full.exactRecord), "pulled out every field of a record perfectly", `${of(full.exactRecord, m.n.extraction)} texts`],
            [pct(full.noToolAcc), "stayed quiet when no tool applied", "every such test request"],
          ].map(([value, what, n]) => (
            <li key={what} className="bg-zinc-950 p-4 sm:p-5">
              <p className="text-3xl font-semibold tabular-nums text-[#C785F2]">{value}</p>
              <p className="mt-1 text-sm text-zinc-200">{what}</p>
              <p className="mt-0.5 text-xs text-zinc-500">{n}</p>
            </li>
          ))}
        </ul>

        <div className="not-prose mt-8 grid max-w-3xl gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-2">
          {(
            [
              ["Strong", "Decisions that match the words of its schemas", strong, "#C785F2"],
              ["Still weak", "Paraphrases and tools unlike its training data", weak, "#a1a1aa"],
            ] as const
          ).map(([title, hint, rows, color]) => (
            <div key={title} className="bg-zinc-950 p-4 sm:p-5">
              <p className="font-semibold text-zinc-100">{title}</p>
              <p className="mt-0.5 text-xs text-zinc-500">{hint}</p>
              <ul className="mt-4 space-y-2.5">
                {rows.map((r) => (
                  <li key={r.label} className="flex items-baseline justify-between gap-4 text-sm">
                    <span className="text-zinc-300">{r.label}</span>
                    <span className="font-semibold tabular-nums" style={{ color }}>
                      {pct(r.value)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <BarChart
          title="Held-out accuracy by task"
          subtitle={`${full.cycles} binding cycles; hover a bar for details`}
          data={accuracy}
          max={1}
          ticks={[0, 0.25, 0.5, 0.75, 1]}
          unit="percent"
        />

        <h3 id="confidence" className={h3}>
          Confidence you can act on
        </h3>
        <p className={p}>
          Every answer comes with a confidence score. For the yes/no decisions (does this tool apply, is this detail
          present), across {m.n.calibrationFire} test decisions the gap between stated confidence and real accuracy
          averaged {pct(m.fireEceAfter)}. Calibration of the full-call and label scores has not been measured yet.
        </p>
        <PlainWords>
          When ARC 1 says it is 90% sure a tool applies, it was right about 90% of the time on test data. So a simple
          rule, such as &ldquo;below 80%, ask a person&rdquo;, is a reasonable start. Check it on your own traffic.
        </PlainWords>
        <details className="not-prose mt-2 border-y border-zinc-800">
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
                    [`Tool selection (n=${m.n.tools})`, "toolSelection"],
                    ["Exact call", "exactCall"],
                    ["Argument accuracy", "argumentAcc"],
                    ["No-tool rejection", "noToolAcc"],
                    [`Tool selection, unseen tools (n=${m.n.unseenTools})`, "toolSelectionUnseen"],
                    ["Exact call, unseen tools", "exactCallUnseen"],
                    [`Extraction field F1 (n=${m.n.extraction})`, "extractionF1"],
                    ["Extraction, whole record exact", "exactRecord"],
                    [`Classification, overall (n=${m.n.classify})`, "classifyOverall"],
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

        <h2 id="limitations" className={h2}>
          Limitations
        </h2>
        <p className={p}>
          ARC 1 is small on purpose, and that comes with real trade-offs. Knowing them helps you decide where it fits.
        </p>
        <ul className="not-prose mt-6 max-w-2xl divide-y divide-zinc-900 border-y border-zinc-800">
          {[
            [
              "Tools unlike anything it trained on",
              `For tools left out of training entirely, it picked the right one ${pct(full.toolSelectionUnseen)} of the time and got the full call right ${pct(full.exactCallUnseen)} of the time (${m.n.unseenTools} requests). New kinds of tools need more training examples.`,
            ],
            [
              "Loosely worded categories",
              `Classification is ${pct(full.classifyOverall)} overall (${m.n.classify} texts). It is perfect on intent, but support routing (${pct(full.classifySupport)}) and product topics (${pct(full.classifyProducts)}) are weak. Each of these subsets has only about 10 to 30 examples, so treat those numbers as rough.`,
            ],
            [
              "No general world knowledge",
              "It hasn't read the internet. It matches the words in a request to the words in your schema, so “billing: charges, refunds” works better than “billing” alone.",
            ],
            [
              "Synthetic test data",
              "All scores come from generated examples held out from training, not from real customer traffic. Expect lower numbers on messy real-world text until it is tuned on your data.",
            ],
            [
              "Casually worded details",
              "Extraction is most reliable when details are stated plainly, as in “My name is Maria Santos and I live in Cebu”, a form, or an email signature. A loose phrase like “ship it to Maria Santos in Cebu” can come back with fields missing.",
            ],
            [
              "Grounded is not the same as correct",
              "Text values are always copied from the request, but they can be copied into the wrong field. A required detail the user never gave is still filled with some words from the request, a yes/no option is always answered even when the request doesn't mention it, and an option list always returns its closest choice. Numbers are converted, so “1,5” reads as 15. Check required details before acting on a call.",
            ],
            ["English only", "It was trained and tested only on English."],
            [
              "Short inputs, one call per tool",
              `Requests are cut at ${m.seqLen} tokens, about a short paragraph, and each tool can be called at most once per request.`,
            ],
            ["Not a chatbot", "It cannot hold a conversation or write replies. Pair it with a language model for that."],
          ].map(([title, body]) => (
            <li key={title} className="py-3.5">
              <p className="font-medium text-zinc-100">{title}</p>
              <p className="mt-1 text-sm leading-relaxed text-zinc-400">{body}</p>
            </li>
          ))}
        </ul>

        <h2 id="comparison" className={h2}>
          How it compares
        </h2>
        <p className={p}>
          Teams usually handle these decisions in one of two ways: a chat-style language model, or a large decision
          model such as Laya AI. Here is where ARC 1 stands apart.
        </p>
        <ul className="not-prose mt-6 grid gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-3">
          {[
            [times(LAYA.params / m.parameters), "fewer parameters", "than a 421M-parameter decision model"],
            [times(LAYA.downloadBytes / m.rcn.bytes), "smaller download", `${mb(m.rcn.bytes)} instead of about 808 MB`],
            ["No GPU", "needed", `${full.latencyP50Ms.toFixed(1)} ms median on a laptop CPU`],
          ].map(([value, what, note]) => (
            <li key={what} className="bg-zinc-950 p-4 sm:p-5">
              <p className="text-3xl font-semibold tabular-nums text-[#C785F2]">{value}</p>
              <p className="mt-1 text-sm text-zinc-200">{what}</p>
              <p className="mt-0.5 text-xs text-zinc-500">{note}</p>
            </li>
          ))}
        </ul>
        <div className="not-prose mt-6 overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className="w-40 py-2 pr-4 font-medium" />
                <th className="py-2 pr-4 font-medium">Chat-style LLM</th>
                <th className="py-2 pr-4 font-medium">Laya AI</th>
                <th className="py-2 font-medium text-zinc-100">ARC 1</th>
              </tr>
            </thead>
            <tbody>
              {COMPARE.map(([row, llm, laya, arc]) => (
                <tr key={row} className="border-b border-zinc-900 align-top">
                  <th scope="row" className="py-2.5 pr-4 font-medium text-zinc-300">
                    {row}
                  </th>
                  <td className="py-2.5 pr-4 text-zinc-500">{llm}</td>
                  <td className="py-2.5 pr-4 text-zinc-500">{laya}</td>
                  <td className="border-l-2 border-[#C785F2] bg-[#C785F2]/[0.06] py-2.5 pl-3 text-zinc-100">{arc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={`${p} mt-4 text-xs text-zinc-500`}>
          Laya AI figures are from its{" "}
          <a href="https://laya.convaiinnovations.com/" target="_blank" rel="noreferrer" className="underline hover:text-zinc-300">
            public page
          </a>{" "}
          (English checkpoint, checked September 2026), measured on its own tasks and a GPU. ARC 1 figures come from its
          held-out tests on a laptop CPU. They compare size and capability, not accuracy on a shared dataset.
        </p>

        <h2 id="architecture" className={h2}>
          Under the hood
        </h2>
        <p className={p}>
          For technical readers: the input is perceived once. Every tool, argument, option, and label is encoded as a
          schema engram, and all engrams bind to the input in parallel. Readouts then produce typed decisions directly,
          with no decoding loop.
        </p>
        <Mermaid chart={ARCHITECTURE} minWidth={520} />

        <h3 id="request-flow" className={h3}>
          Request flow
        </h3>
        <p className={p}>
          A request costs one forward pass. Schema texts not yet cached are perceived in the same batch as the input,
          so first-time tools add no extra pass. The agent applies calibration and schema validation before returning.
        </p>
        <Mermaid chart={REQUEST_FLOW} minWidth={640} />

        <h3 id="readouts" className={h3}>
          Readouts
        </h3>
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
          Each readout carries a temperature fitted on held-out data. When no tool fires, ARC 1 returns no call.
        </p>

        <h2 id="rcn-format" className={h2}>
          The .rcn model file
        </h2>
        <p className={p}>
          ARC 1 ships as one <code className={code}>.rcn</code> file that holds everything needed to run it: the
          architecture, the tokenizer, the calibration, and the weights. Copy the file to a device and it runs.
        </p>
        <ul className="max-w-2xl space-y-2">
          <li>
            <strong className="text-zinc-100">Opens fast.</strong> The file is memory-mapped and each tensor is stored
            in its final layout; quantized weights are expanded to full precision as they load.
          </li>
          <li>
            <strong className="text-zinc-100">Compressed with no measurable accuracy loss.</strong> Weights can be
            stored at 16, 8.5, or 4.5 bits each. The 4.5-bit file is{" "}
            {(m.formats[0].bytes / m.rcn.bytes).toFixed(1)}x smaller than the original and scored the same within
            measurement noise on the held-out tests.
          </li>
          <li>
            <strong className="text-zinc-100">Checked on load.</strong> A checksum catches a corrupted download before
            the model runs.
          </li>
        </ul>
        <BarChart
          title="Model size by format"
          subtitle="Same ARC 1 weights; hover a bar for held-out accuracy"
          data={sizes}
          max={6e6}
          ticks={[0, 2e6, 4e6, 6e6]}
          unit="mb"
          highlight=".rcn rq4"
        />
        <details className="not-prose border-y border-zinc-800">
          <summary className="cursor-pointer py-3 text-sm text-zinc-300 hover:text-zinc-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-[#C785F2]">
            File layout
          </summary>
          <div className="overflow-x-auto pb-3">
            <table className="w-full border-collapse text-left text-sm">
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
            <p className="mt-3 text-xs text-zinc-500">
              Quantized formats use block-wise scales over 32 values (<code>rq8</code>, <code>rq4</code>); small tensors
              stay <code>f16</code>. The binding-cycle count is baked in at export, and a CRC32 of the tensor data is
              checked at load.
            </p>
          </div>
        </details>

        <h2 id="get-started" className={h2}>
          Use it in your app
        </h2>
        <div className="not-prose mt-4 flex flex-col gap-3 border border-zinc-800 bg-zinc-950 p-4 sm:flex-row sm:items-center sm:justify-between">
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
        <h3 id="python" className={h3}>
          Python
        </h3>
        <p className={p}>
          The package includes the model file, so nothing else needs downloading.
        </p>
        <pre className="not-prose mt-4 overflow-x-auto border border-zinc-800 bg-zinc-950 p-4 text-[13px] leading-relaxed text-zinc-300">
          {`pip install gpbacay-arcane

import gpbacay_arcane as arcane
from gpbacay_arcane import ToolParam, ToolSpec

agent = arcane.load_arc1()  # bundled arc1-tiny.rcn; load_arc1("my.rcn") for your own

# Call a tool
weather = ToolSpec("get_weather", "Get the current weather for a city.",
                   [ToolParam("city", description="City name")])
agent.run("is it raining in Tokyo?", tools=[weather])
# {"function_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}], ...}

# Pull out details
agent.extract("My name is Maria Santos and I live in Cebu", {"name": "Person name", "city": "City"})
# {"record": {"name": "Maria Santos", "city": "Cebu"}, ...}

# Sort into labels
agent.classify("the headphones sound amazing", ["positive", "negative", "neutral"])
# {"label": "positive", "confidence": ..., "distribution": {...}}`}
        </pre>

        <h3 id="nodejs" className={h3}>
          Node.js
        </h3>
        <p className={p}>
          The Node package runs the same model in a local Python process, so it needs the Python package installed
          too. Results match Python exactly. The first <code className={code}>load()</code> takes a while as
          TensorFlow starts; later calls take milliseconds.
        </p>
        <pre className="not-prose mt-4 overflow-x-auto border border-zinc-800 bg-zinc-950 p-4 text-[13px] leading-relaxed text-zinc-300">
          {`pip install gpbacay-arcane
npm install gpbacay-arcane

const { load } = require("gpbacay-arcane");

const agent = await load();  // load({ python: "path/to/python", model: "my.rcn" }) to override

await agent.run("is it raining in Tokyo?", [
  { name: "get_weather", description: "Get the current weather for a city.",
    parameters: [{ name: "city", description: "City name" }] },
]);
// { function_calls: [{ name: "get_weather", arguments: { city: "Tokyo" } }], ... }

await agent.extract("My name is Maria Santos and I live in Cebu", { name: "Person name", city: "City" });
await agent.classify("the headphones sound amazing", ["positive", "negative", "neutral"]);
await agent.embed("hello");

agent.close();`}
        </pre>
        <p className={`${p} mt-4 text-zinc-400`}>
          Every run, extract, and classify result includes a confidence score and the time it took. To build your own file, run{" "}
          <code className={code}>python examples/export_arc1.py --rcn arc1.rcn --quant rq4 --cycles 2</code>, and
          inspect any file with <code className={code}>python -m gpbacay_arcane.rcn arc1.rcn</code>.
        </p>

        <h3 id="run-locally" className={h3}>
          Run the demo server yourself
        </h3>
        <p className={p}>
          The live demo on this page is <code className={code}>examples/serve_arc1_api.py</code>, with{" "}
          <code className={code}>/run</code>, <code className={code}>/extract</code>,{" "}
          <code className={code}>/classify</code>, and <code className={code}>/embed</code> endpoints. From{" "}
          <code className={code}>arcane-docs-web</code>, <code className={code}>npm run dev:with-arc1</code> starts it
          next to this site on port 8002. Train new weights with{" "}
          <code className={code}>python examples/train_arc1.py --preset arc1-tiny</code>.
        </p>
        <p className={`${p} mt-4 text-zinc-400`}>
          Need open-ended replies as well? Pair ARC 1 with the{" "}
          <Link href="/docs/chat" className={link}>
            ARCANE small language model
          </Link>
          : ARC 1 makes the decision, and the language model writes the answer.
        </p>
      </div>
    </div>
  );
}
