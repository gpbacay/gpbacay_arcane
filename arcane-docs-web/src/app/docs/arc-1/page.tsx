import Link from "next/link";
import { ArrowDown, Ban, Crosshair, Download, Gauge, Layers } from "lucide-react";
import { Arc1Demo } from "@/components/Arc1Demo";
import { BarChart } from "@/components/BarChart";
import { CodeSnippet } from "@/components/CodeSnippet";
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

export default function Arc1Page() {
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
          ARC 1: Grounded Structured Decisions via Resonant Schema Binding
        </h1>
        <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-zinc-300">
          ARC 1 is a {(m.parameters / 1e6).toFixed(2)}M-parameter neuromimetic model for tool calling, field extraction,
          classification, and semantic retrieval. It makes these decisions without generating text: each tool,
          argument, and label binds to the input in parallel through a few cycles of gated resonance built from the
          ARCANE library. It runs in {full.latencyP50Ms.toFixed(1)} ms median on a CPU from a single{" "}
          {mb(m.rcn.bytes)} file.
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
        <p className={`${p} mb-5`}>Runs the real model on a live server. Pick an example or type your own request.</p>
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
          <Link href="/docs/layers" className={link}>
            ResonantChannelMixer
          </Link>
          ,{" "}
          <Link href="/docs/neural-resonance" className={link}>
            FieldResonance
          </Link>
          , and the{" "}
          <Link href="/docs/resonant-gser" className={link}>
            GSER gate
          </Link>
          .
        </p>

        <h2 id="accuracy" className={h2}>
          Accuracy
        </h2>
        <p className={p}>Measured on held-out synthetic examples, including tool values never seen in training.</p>
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
              `Classification is ${pct(full.classifyOverall)} overall; support routing (${pct(full.classifySupport)}) and product topics (${pct(full.classifyProducts)}) are weakest, on small subsets.`,
            ],
            [
              "Grounded is not correct",
              "Values are copied from the request but can land in the wrong field. Required details the user never gave are still filled, and option lists always return a closest choice. Validate before acting.",
            ],
            ["Synthetic test data", "All scores come from generated held-out examples, not real traffic."],
            [
              "Scope",
              `English only, inputs cut at ${m.seqLen} tokens, each tool called at most once per request, no world knowledge, and no chat replies.`,
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

        <h3 id="nodejs" className={h3}>
          Node.js
        </h3>
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

        <h3 id="browser" className={h3}>
          Browser (Next.js, Vite, plain JS)
        </h3>
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
        <p className={`${p} mt-4 text-zinc-400`}>
          Need open-ended replies too? Pair ARC 1 with the{" "}
          <Link href="/docs/chat" className={link}>
            ARCANE small language model
          </Link>
          .
        </p>
      </div>
    </div>
  );
}
