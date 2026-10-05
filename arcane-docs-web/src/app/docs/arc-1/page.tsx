import Link from "next/link";
import { ArrowDown, Ban, Crosshair, Download, Gauge, Layers, Star } from "lucide-react";
import { Arc1Demo } from "@/components/Arc1Demo";
import { BarChart } from "@/components/BarChart";
import { CodeSnippet } from "@/components/CodeSnippet";
import { Mermaid } from "@/components/markdown";
import { Tabs } from "@/components/Tabs";
import { ARC1_METRICS } from "@/config/arc1-metrics";

const h2Base = "scroll-mt-28 text-2xl font-bold tracking-tight text-zinc-100 mb-4 border-b border-zinc-800 pb-2";
const h2 = `${h2Base} mt-16`;
const h3 = "scroll-mt-28 text-lg font-semibold text-zinc-100 mt-8 mb-2";
const code = "bg-zinc-900 px-1.5 py-0.5 text-zinc-100";
const p = "max-w-2xl leading-relaxed";
const link = "font-medium text-[#C785F2] underline hover:text-[#d49cf5]";

const CAPABILITIES = [
  {
    icon: Layers,
    title: "Schema as input",
    body: "Tools, fields, and labels are passed with each request. Changing them needs no retraining.",
  },
  {
    icon: Crosshair,
    title: "Copies, never invents",
    body: "Text values are spans of your input; labels come from your list.",
  },
  {
    icon: Ban,
    title: "Abstains",
    body: "When no tool fits, it returns no call instead of guessing.",
  },
  {
    icon: Gauge,
    title: "Calibrated confidence",
    body: "Stated confidence tracks real accuracy, so a threshold can decide when to ask a person.",
  },
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

const REPO_URL = "https://github.com/gpbacay/gpbacay_arcane";

const DOWNLOAD_NOTICES: Record<string, string> = {
  star: "We couldn't find a star from that GitHub account. Star the repository, then press Download again.",
  denied: "GitHub sign-in was cancelled, so the download didn't start. Press Download to try again.",
  error: "Something went wrong while checking your star. Press Download to try again.",
};

export default async function Arc1Page({ searchParams }: { searchParams: Promise<{ download?: string }> }) {
  const notice = DOWNLOAD_NOTICES[(await searchParams).download ?? ""];
  const m = ARC1_METRICS;
  const full = m.byCycles[m.byCycles.length - 1];
  const of = (v: number, n: number) => `${Math.round(v * n)} of ${n}`;

  const accuracy = [
    { label: "Tool selection", value: full.toolSelection },
    { label: "Exact call", value: full.exactCall },
    { label: "No-tool rejection", value: full.noToolAcc },
    { label: "Extraction F1", value: full.extractionF1 },
    { label: "Classification", value: full.classifyOverall },
    { label: "Unseen-tool calls", value: full.exactCallUnseen },
  ]
    .sort((a, b) => b.value - a.value)
    .map((d) => ({ ...d, detail: `${pct(d.value)} on held-out data, ${full.cycles} cycles` }));

  const COMPARE: [string, string, string, string][] = [
    ["Size", "Billions of parameters", `${(LAYA.params / 1e6).toFixed(0)}M parameters`, `${(m.parameters / 1e6).toFixed(2)}M parameters`],
    ["Download", "Gigabytes", `About ${(LAYA.downloadBytes / 1e6).toFixed(0)} MB`, mb(m.rcn.bytes)],
    ["Hardware", "GPU or paid API", `GPU, ${LAYA.gpuLatencyMs} ms`, `CPU,${full.latencyP50Ms.toFixed(1)} ms median`],
    ["Tool calls and extraction", "Generated, values can be invented", "Not offered", "Built in, values copied from input"],
    ["Calibration error", "Varies", `${pct(LAYA.ece)} (its own tasks)`, `${pct(m.fireEceAfter)} on tool decisions`],
  ];

  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <header className="not-prose mb-12">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#C785F2]">ARCANE · Technical report</p>
        <h1 className="mt-3 max-w-3xl text-4xl font-extrabold leading-[1.08] tracking-[-0.03em] text-zinc-50 sm:text-5xl">
          Stop Using LLMs for If-Else Decisions: Meet ARC 1
        </h1>
        <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-zinc-300">
          A {(m.parameters / 1e6).toFixed(1)}M-parameter, open-source, neuromimetic model that makes grounded, typed
          decisions in {full.latencyP50Ms.toFixed(1)} milliseconds on a standard CPU.
        </p>
        <p className="mt-4 max-w-2xl text-[17px] leading-relaxed text-zinc-300">
          We keep asking multi-billion-parameter language models to pick a tool, route a ticket, or pull a date and an
          amount out of an invoice, then run a fragile parser over the prose they generate. It is slow, expensive, and
          it can invent values that were never in the input. Most automation does not need eloquence. It needs a
          correct, typed decision, delivered instantly, with an honest confidence score. ARC 1 makes that decision
          without generating text: learned probes for each tool, argument, and label resonate with the input through
          spiking dynamics until they settle, in a single forward pass. It ships as one {mb(m.rcn.bytes)} .rcn file
          under the MIT license.
        </p>

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
            title="Free, no sign-in. Needs fine-tuning before use."
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
          Runs the real model on a live server. Pick an example or type your own request. Switch the harness to{" "}
          <strong className="text-zinc-100">+ Hippocampus</strong> to give ARC 1 a memory of examples: the{" "}
          <em>New tools</em> scene uses tools it never trained on, taught by a few examples instead of fine-tuning.{" "}
          <Link href="/docs/hippocampus" className="font-medium text-[#C785F2] underline hover:text-[#d49cf5]">
            How Hippocampus works
          </Link>
        </p>
        <Arc1Demo />

        <h2 id="what-it-does" className={h2}>
          What it does
        </h2>
        <p className={p}>
          The input is read once. Every tool, argument, option, and label then listens to it in {m.cycles} short
          cycles, and the ones that match open a spiking gate. Readouts return typed results directly, with no decoding
          loop.
        </p>
        <ul className="not-prose mt-6 grid gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-2 lg:grid-cols-4">
          {CAPABILITIES.map(({ icon: Icon, title, body }) => (
            <li key={title} className="bg-zinc-950 p-4 sm:p-5">
              <Icon className="h-5 w-5 text-[#C785F2]" aria-hidden />
              <p className="mt-3 font-semibold text-zinc-100">{title}</p>
              <p className="mt-1.5 text-sm leading-relaxed text-zinc-400">{body}</p>
            </li>
          ))}
        </ul>
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          Built from ARCANE parts:{" "}
          <Link href="/docs/layers#resonant-channel-mixer" className={link}>
            ResonantChannelMixer
          </Link>
          ,{" "}
          <Link href="/docs/layers#field-resonance" className={link}>
            FieldResonance
          </Link>
          , and the{" "}
          <Link href="/docs/layers#dense-gser" className={link}>
            GSER gate
          </Link>
          .
        </p>

        <h2 id="architecture" className={h2}>
          Architecture
        </h2>
        <p className={p}>
          ARC 1 uses Resonant Schema Binding. The request and every schema element go through the same perception
          blocks once. Each schema element becomes a probe that binds to the request in {m.cycles} shared-weight cycles,
          and four readouts are taken from the settled probes. Nothing is decoded token by token.
        </p>
        <Mermaid
          minWidth={560}
          chart={`flowchart TD
  REQ["Request text"] --> PER
  SCH["Schema<br/>tools, parameters, options, labels"] --> PER
  PER["Perception blocks (bidirectional, read once)<br/>FieldAttention, ResonantChannelMixer,<br/>ConceptEngram memory, FieldResonance"]
  PER --> FIELD["Utterance field"]
  PER --> ENG["Schema engrams<br/>(cached after first use)"]
  ENG --> PROBE["Probes<br/>engram + context engram + role"]
  FIELD --> BIND
  PROBE --> BIND
  BIND["Resonant binding<br/>GSER spiking gate, ${m.cycles} cycles, all probes in parallel"]
  BIND --> FIRE["Fire<br/>does a tool or optional argument apply?"]
  BIND --> ANCHOR["Anchor<br/>start and end span copied from the request"]
  BIND --> SELECT["Select<br/>option or label choice"]
  FIELD --> EMB["Embedding<br/>semantic retrieval"]`}
        />
        <p className={`${p} text-sm text-zinc-400`}>
          Fire, anchor and select each have a temperature fitted on held-out data, which is what makes the reported
          confidence meaningful. Changing the number of cycles trades accuracy for compute without new weights.
        </p>

        <h2 id="accuracy" className={h2}>
          Accuracy
        </h2>
        <p className={p}>
          Measured mostly on held-out synthetic examples, including tool values never seen in training. One check uses
          real text, below.
        </p>
        <ul className="not-prose mt-6 grid gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-2 lg:grid-cols-4">
          {[
            [pct(full.toolSelection), "right tool", `${of(full.toolSelection, m.n.tools)} requests`],
            [pct(full.exactCall), "exact call, tool and arguments", `${of(full.exactCall, m.n.tools)} requests`],
            [pct(full.extractionF1), "extraction F1", `${m.n.extraction} texts`],
            [pct(full.noToolAcc), "correct abstention", "no-tool requests"],
          ].map(([value, what, n]) => (
            <li key={what} className="bg-zinc-950 p-4 sm:p-5">
              <p className="text-3xl font-semibold tabular-nums text-[#C785F2]">{value}</p>
              <p className="mt-1 text-sm text-zinc-200">{what}</p>
              <p className="mt-0.5 text-xs text-zinc-500">{n}</p>
            </li>
          ))}
        </ul>
        <BarChart
          title="Held-out accuracy by task"
          subtitle={`${full.cycles} binding cycles; hover a bar for details`}
          data={accuracy}
          max={1}
          ticks={[0, 0.25, 0.5, 0.75, 1]}
          unit="percent"
        />
        <p className={`${p} mt-4`}>
          Expected calibration error on tool decisions is {pct(m.fireEceAfter)} over {m.n.calibrationFire} test
          decisions. Full-call and label calibration has not been measured.
        </p>
        <p className={`${p} mt-4`}>
          On {m.realIntents.n} real utterances from CLINC150, for intents that were left out of every training set, it
          picked the right one of {m.realIntents.labels} labels {pct(m.realIntents.accuracy)} of the time (chance is{" "}
          {pct(m.realIntents.chance)}). Before the language distillation described below, the same check scored about 38%.
        </p>

        <h2 id="training" className={h2}>
          How it was trained
        </h2>
        <p className={p}>
          ARC 1 starts with no language knowledge, so a frozen Qwen3-Embedding-0.6B teaches it offline which texts mean the
          same thing. The teacher is not in the download and is never run at inference, so the size and speed are unchanged.
        </p>
        <ol className="mt-4 max-w-2xl list-decimal space-y-2 pl-5 text-sm leading-relaxed text-zinc-300">
          <li>
            <strong className="text-zinc-100">Contrastive pre-training.</strong> Each request is pulled toward its tool or
            intent and pushed away from the others (35k pairs from the synthetic tools, CLINC150 and Banking77), while its
            vectors learn to match the teacher&apos;s sense of similarity.
          </li>
          <li>
            <strong className="text-zinc-100">Hard negatives.</strong> The same, with the teacher&apos;s nearest wrong
            answers added as negatives.
          </li>
          <li>
            <strong className="text-zinc-100">Multi-task training with replay.</strong> The usual tool, extraction and
            label losses, with 40% of each step replaying stage 1 so the new knowledge is not forgotten.
          </li>
        </ol>
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          The trade-off: against the previous checkpoint this gains about 19 points on real phrasings and 31 points on
          classifying requests for unseen tools, and gives up about 6 points on exact tool calls and fully correct
          extracted records. Recipe and commands are in the{" "}
          <a href={`${REPO_URL}#arc-1--automation-foundation-model`} target="_blank" rel="noreferrer" className={link}>
            repository README
          </a>
          .
        </p>

        <h2 id="limitations" className={h2}>
          Limitations
        </h2>
        <ul className="not-prose mt-2 max-w-2xl divide-y divide-zinc-900 border-y border-zinc-800">
          {[
            [
              "Unseen tools",
              `Right tool ${pct(full.toolSelectionUnseen)}, exact call ${pct(full.exactCallUnseen)} (${m.n.unseenTools} requests). New kinds of tools need training examples.`,
            ],
            [
              "Weak categories",
              `Classification is ${pct(full.classifyOverall)} overall; feed ranking (${pct(full.classifyFeed)}), product topics (${pct(full.classifyProducts)}) and search intent (${pct(full.classifySearch)}) are weakest, on small subsets.`,
            ],
            [
              "Grounded is not correct",
              "Values are copied from the request but can land in the wrong field. Required details the user never gave are still filled, and option lists always return a closest choice. Validate before acting.",
            ],
            [
              "Mostly synthetic test data",
              `Most scores come from generated held-out examples, not real traffic. The one real-text check is ${m.realIntents.n} CLINC150 utterances in a five-way choice.`,
            ],
            [
              "Scope",
              `English only, inputs cut at ${m.seqLen} tokens, each tool called at most once per request, only the light language knowledge distilled from a small text encoder and no world knowledge, and no chat replies.`,
            ],
          ].map(([title, body]) => (
            <li key={title} className="py-3.5">
              <p className="font-medium text-zinc-100">{title}</p>
              <p className="mt-1 text-sm leading-relaxed text-zinc-400">{body}</p>
            </li>
          ))}
        </ul>

        <h2 id="comparison" className={h2}>
          Comparison
        </h2>
        <div className="not-prose mt-4 overflow-x-auto">
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
          (English checkpoint, checked September 2026), measured on its own tasks and a GPU. This compares size and
          capability, not accuracy on a shared dataset.
        </p>

        <h2 id="get-started" className={h2}>
          Get started
        </h2>
        {notice && (
          <p role="alert" className="not-prose mt-4 border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
            {notice}
          </p>
        )}
        <p className="not-prose mt-4 border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
          <strong>Needs fine-tuning or Hippocampus.</strong> This download is a base checkpoint, not a finished model.
          Fine-tune it on your own data and tools, or give it a few decided examples of each through the{" "}
          <Link href="/docs/hippocampus" className="underline">
            Hippocampus
          </Link>{" "}
          harness, and test it on your own requests before relying on it in your application.
        </p>
        <div className="not-prose mt-4 flex flex-col gap-3 border border-zinc-800 bg-zinc-950 p-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0">
            <div className="text-sm font-medium text-zinc-100">arc1-tiny.rcn</div>
            <div className="mt-0.5 text-xs text-zinc-500">
              {mb(m.rcn.bytes)}, {m.rcn.quant}, tokenizer and calibration included
            </div>
            <div className="mt-1 break-all font-mono text-[11px] text-zinc-500">sha256 {m.rcn.sha256}</div>
          </div>
          <a
            href={m.rcn.file}
            className="inline-flex shrink-0 items-center justify-center gap-2 bg-white px-4 py-2 text-sm font-medium text-black transition-colors hover:bg-zinc-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#C785F2]"
          >
            <Download className="h-4 w-4" aria-hidden />
            Download
          </a>
        </div>
        <p className="not-prose mt-2 flex items-center gap-1.5 text-xs text-zinc-500">
          <Star className="h-3.5 w-3.5 shrink-0" aria-hidden />
          <span>
            Free, with no sign-in. If you find ARC 1 useful, a{" "}
            <a href={REPO_URL} target="_blank" rel="noreferrer" className="text-zinc-300 underline hover:text-zinc-100">
              star on GitHub
            </a>{" "}
            is appreciated.
          </span>
        </p>

        <Tabs
          name="sdk"
          tabs={[
            {
              id: "python",
              label: "Python",
              content: (
                <>
        <CodeSnippet filename="Terminal" language="bash" lineNumbers={false} code="pip install gpbacay-arcane" />
        <CodeSnippet
          filename="arc1.py"
          language="python"
          code={`import gpbacay_arcane as arcane
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
        />
                </>
              ),
            },
            {
              id: "nodejs",
              label: "Node.js",
              content: (
                <>
        <p className={p}>
          On a server, it runs the same model in a local Python process, so the Python package must be installed too.
          To run with no Python at all, use the browser build below.
        </p>
        <CodeSnippet
          filename="Terminal"
          language="bash"
          lineNumbers={false}
          code={`pip install gpbacay-arcane
npm install gpbacay-arcane`}
        />
        <CodeSnippet
          filename="index.js"
          language="javascript"
          code={`const { load } = require("gpbacay-arcane");

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
        />
                </>
              ),
            },
            {
              id: "browser",
              label: "Browser (Next.js, Vite, plain JS)",
              content: (
                <>
        <p className={p}>
          <code className={code}>gpbacay-arcane/web</code> runs ARC 1 client-side with onnxruntime-web (WebAssembly), with no server and no
          Python. The 5.5 MB model is fetched once and cached by the browser. It has the same API as the Node build and
          returns the same results, and calls take about 10 to 20 ms once loaded.
        </p>
        <CodeSnippet filename="Terminal" language="bash" lineNumbers={false} code="npm install gpbacay-arcane onnxruntime-web" />
        <CodeSnippet
          filename="Arc1.tsx"
          language="tsx"
          code={`"use client";  // Next.js: use it in a client component
import { load } from "gpbacay-arcane/web";

const agent = await load();  // once, e.g. in useEffect

const r = await agent.run("is it raining in Tokyo?", [
  { name: "get_weather", description: "Get the current weather for a city.",
    parameters: [{ name: "city", description: "City name" }] },
]);
// r.function_calls -> [{ name: "get_weather", arguments: { city: "Tokyo" } }]

await agent.extract("My name is Maria Santos and I live in Cebu", { name: "Person name", city: "City" });
await agent.classify("the headphones sound amazing", ["positive", "negative", "neutral"]);
await agent.embed("hello");`}
        />
        <p className={`${p} mt-4 text-zinc-400`}>
          To serve the files yourself, use <code className={code}>load({"{"} model: &quot;/arc1/arc1.onnx&quot;, config:
          &quot;/arc1/arc1.json&quot;, wasmPaths: &quot;/ort/&quot; {"}"})</code>. To export your own model, run{" "}
          <code className={code}>python examples/export_arc1_onnx.py --model your.rcn</code>. Binding cycles are fixed at export time.
        </p>
                </>
              ),
            },
          ]}
        />
        <p className={`${p} mt-4 text-zinc-400`}>
          Need open-ended replies too? Pair ARC 1 with the{" "}
          <Link href="/docs/chat" className={link}>
            ARCANE small language model
          </Link>
          .
        </p>

        <h2 id="finetuning" className={h2}>
          Finetuning
        </h2>
        <p className={p}>
          The downloaded <code className={code}>arc1-tiny.rcn</code> is a base checkpoint. Finetuning teaches it your own
          tools, fields and labels. Training runs from the source repository and learns from examples it generates out of
          tool definitions you write, so you describe each tool once and it produces the phrasings. It needs Python and TensorFlow.
        </p>

        <Tabs
          name="ft"
          tabs={[
            {
              id: "yourself",
              label: "Fine tune it yourself",
              content: (
            <>
        <h3 id="ft-setup" className={h3}>
          1. Set up the repository
        </h3>
        <CodeSnippet
          filename="Terminal"
          language="bash"
          lineNumbers={false}
          code={`git clone ${REPO_URL}
cd gpbacay_arcane
pip install -e .`}
        />

        <h3 id="ft-unpack" className={h3}>
          2. Unpack the download into trainable files
        </h3>
        <p className={p}>
          An <code className={code}>.rcn</code> file is for running the model. Convert it back to weights and a tokenizer so
          training can continue from it. The file names below are the ones the trainer looks for.
        </p>
        <CodeSnippet
          filename="unpack.py"
          language="python"
          code={`import os
from gpbacay_arcane.rcn import load_rcn

os.makedirs("Models", exist_ok=True)
model, tokenizer, _ = load_rcn("arc1-tiny.rcn")
model.save_weights("Models/arc1_arc1_tiny.weights.h5")
tokenizer.save("Models/arc1_arc1_tiny_tokenizer.json")`}
        />

        <h3 id="ft-tools" className={h3}>
          3. Describe your tools
        </h3>
        <p className={p}>
          Open <code className={code}>gpbacay_arcane/arc1_data.py</code> and add a <code className={code}>ToolDef</code> to
          the list in <code className={code}>build_tool_library()</code>. Give it a few descriptions, its parameters, and
          10 or more phrasings with <code className={code}>{"{param}"}</code> placeholders. A <code className={code}>SlotDef</code>{" "}
          says how each value is written in text.
        </p>
        <CodeSnippet
          filename="arc1_data.py"
          language="python"
          code={`ToolDef(
    "book_table",
    ["Reserve a table at a restaurant.", "Make a dinner reservation."],
    [P("restaurant", "string", "Restaurant name"), P("guests", "integer", "Party size")],
    ["book a table at {restaurant} for {guests}",
     "reserve {restaurant}, {guests} people",
     "table for {guests} at {restaurant} please"],
    {"restaurant": SlotDef(choice_of(["Luigi's", "Sushi Zen", "The Anchor", "Casa Verde"])),
     "guests": SlotDef(lambda r, s: _int_surface(r, 1, 12), _int)},
    domain="food",
),`}
        />
        <p className={`${p} mt-4 text-zinc-400`}>
          <code className={code}>choice_of</code> takes a fixed list of values. The file also has <code className={code}>pool</code>,
          which splits values into training and held-out sets so the evaluation in step 5 can test values the model has not seen.
        </p>

        <h3 id="ft-train" className={h3}>
          4. Train from the checkpoint
        </h3>
        <CodeSnippet
          filename="Terminal"
          language="bash"
          lineNumbers={false}
          code={`python examples/train_arc1.py --preset arc1-tiny --resume --steps 1500 --learning-rate 3e-4`}
        />
        <p className={`${p} mt-4 text-zinc-400`}>
          <code className={code}>--resume</code> starts from the weights and tokenizer from step 2. The default learning
          rate is 2e-3, which is for training from scratch. Use a smaller one so the model keeps what it already knows. Loss is
          printed every 50 steps and weights are saved every 500. Fine-tuning only on your own tools can wear away the
          distilled language knowledge, so check the real-text score in the metrics file afterwards.
        </p>

        <h3 id="ft-eval" className={h3}>
          5. Calibrate and evaluate
        </h3>
        <CodeSnippet
          filename="Terminal"
          language="bash"
          lineNumbers={false}
          code={`python examples/train_arc1.py --preset arc1-tiny --eval-only`}
        />
        <p className={`${p} mt-4 text-zinc-400`}>
          This refits the confidence temperatures on held-out values, prints accuracy, and writes{" "}
          <code className={code}>Models/arc1_arc1_tiny.metrics.json</code>. Run it before you export, and check that your
          new tools score well and the old ones did not drop.
        </p>

        <h3 id="ft-export" className={h3}>
          6. Export and use it
        </h3>
        <CodeSnippet
          filename="Terminal"
          language="bash"
          lineNumbers={false}
          code={`python examples/export_arc1.py --config Models/arc1_arc1_tiny.config.json \\
  --weights Models/arc1_arc1_tiny.weights.h5 \\
  --tokenizer Models/arc1_arc1_tiny_tokenizer.json \\
  --rcn my-arc1.rcn --quant rq8`}
        />
        <CodeSnippet
          filename="use.py"
          language="python"
          code={`import gpbacay_arcane as arcane

agent = arcane.load_arc1("my-arc1.rcn")`}
        />
        <p className={`${p} mt-4 text-zinc-400`}>
          For Node use <code className={code}>load({"{"} model: &quot;my-arc1.rcn&quot; {"}"})</code>. For the browser, convert
          it with <code className={code}>python examples/export_arc1_onnx.py --model my-arc1.rcn</code>.
        </p>
            </>
              ),
            },
            {
              id: "ai",
              label: "Finetune using AI",
              content: (
            <>
              <p className={`${p} mt-6`}>
                Open the cloned repository in a coding assistant (Claude Code, Cursor, Codex and the like), download{" "}
                <code className={code}>arc1-tiny.rcn</code> into it, and paste this prompt. Replace the bracketed part with
                your tools, fields or labels. The assistant does the same six steps as the other tab.
              </p>
              <CodeSnippet
                filename="Prompt"
                language="markdown"
                lineNumbers={false}
                code={`Finetune the ARC 1 model in this repository (gpbacay_arcane) for my own tools.

What I need it to do:
[List each tool: name, what it does, its parameters (name, type, description),
and 5 or more example ways a user would ask for it. Add any fields to extract
or labels to classify into.]

Do this in order, and stop and tell me if a step fails:
1. Run "pip install -e ." if the package is not installed.
2. Load ./arc1-tiny.rcn with gpbacay_arcane.rcn.load_rcn, then save the weights to
   Models/arc1_arc1_tiny.weights.h5 and the tokenizer to Models/arc1_arc1_tiny_tokenizer.json.
3. Add a ToolDef for each of my tools to build_tool_library() in
   gpbacay_arcane/arc1_data.py. Follow the existing entries: descriptions, params,
   templates with {param} placeholders (10 or more), and a SlotDef per parameter.
4. Run: python examples/train_arc1.py --preset arc1-tiny --resume --steps 1500 --learning-rate 3e-4
5. Run: python examples/train_arc1.py --preset arc1-tiny --eval-only
   Report the accuracy for my new tools and confirm the original tools did not get worse.
6. Only if the results look good, export with: python examples/export_arc1.py
   --config Models/arc1_arc1_tiny.config.json --weights Models/arc1_arc1_tiny.weights.h5
   --tokenizer Models/arc1_arc1_tiny_tokenizer.json --rcn my-arc1.rcn --quant rq8
   Then check that gpbacay_arcane.load_arc1("my-arc1.rcn") runs one of my tools.

Do not change any model code other than build_tool_library().`}
              />
              <p className={`${p} mt-4 text-zinc-400`}>
                Read the assistant&apos;s evaluation numbers yourself before you use the exported file.
              </p>
            </>
              ),
            },
          ]}
        />
      </div>
    </div>
  );
}
