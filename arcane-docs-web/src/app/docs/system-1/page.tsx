import Link from "next/link";
import {
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  DatabaseZap,
  Gauge,
  Network,
  Route,
} from "lucide-react";

const pillars = [
  {
    icon: BrainCircuit,
    title: "Adaptive inference",
    body: "The model can align transient state or bounded fast weights while processing the current input.",
    href: "/docs/inference-time-learning",
  },
  {
    icon: Route,
    title: "Grounded decisions",
    body: "Outputs are typed actions, labels, fields, scores, or abstentions tied to the supplied input and schema.",
    href: "/docs/arc-1",
  },
  {
    icon: DatabaseZap,
    title: "Neural memory",
    body: "Relevant experiences can guide the next decision immediately, without being baked into model weights.",
    href: "/docs/hippocampus",
  },
  {
    icon: Network,
    title: "Biological mechanisms",
    body: "Feedback, resonance, spiking gates, local plasticity, and homeostasis are available as composable parts.",
    href: "/docs/layers",
  },
];

const criteria = [
  { title: "Direct output", body: "Return a typed decision instead of generated prose that needs parsing." },
  { title: "Honest uncertainty", body: "Report calibrated confidence or abstain when evidence is weak." },
  { title: "Controlled adaptation", body: "Keep every update bounded, measurable, and resettable." },
  { title: "Traceable evidence", body: "Show the input, schema, or memory behind the result." },
  { title: "Separate metrics", body: "Measure quality, latency, adaptation, and memory influence independently." },
];

const systemOneLoop = [
  { number: "01", title: "Perceive", body: "Encode the request, schema, and context." },
  { number: "02", title: "Recall", body: "Retrieve the most relevant experience." },
  { number: "03", title: "Adapt", body: "Align state within explicit limits." },
  { number: "04", title: "Act", body: "Return a grounded decision with confidence and evidence." },
];

const comparison = [
  ["Primary output", "Typed decision or state", "Open-ended token sequence"],
  ["Computation", "Direct prediction with optional bounded adaptation", "Autoregressive decoding across tokens"],
  ["Uncertainty", "Calibrated score and explicit abstention", "Usually inferred from generated output"],
  ["Memory", "Retrieved decisions and examples", "Retrieved text placed in context"],
  ["Best fit", "Routing, extraction, classification, control", "Conversation, synthesis, and open-ended generation"],
];

export default function SystemOnePage() {
  return (
    <div className="text-zinc-300">
      <header className="mb-12 border-b border-zinc-900 pb-10">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#C785F2]">ARCANE 5.0 architecture</p>
        <h1 className="mt-3 max-w-3xl text-4xl font-extrabold leading-tight tracking-[-0.035em] text-white sm:text-5xl">
          True System 1 AI, defined by behavior.
        </h1>
        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-zinc-300">
          ARCANE builds fast, automatic models that respond with grounded outcomes, learn from bounded experience,
          and expose the evidence and confidence behind each decision.
        </p>
      </header>

      <section className="mb-16" aria-labelledby="definition">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#B9DFE0]">Operational definition</p>
        <h2 id="definition" className="mt-3 text-2xl font-bold text-white">What makes a model System 1?</h2>
        <p className="mt-4 max-w-2xl leading-relaxed text-zinc-400">
          In ARCANE, System 1 describes an engineering profile: fast and automatic inference for repeatable decisions.
          It does not mean consciousness or human intuition. The model receives an input and a bounded decision space,
          then returns a direct outcome with measurable uncertainty and optional memory-guided adaptation.
        </p>

        <div className="mt-8 grid gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-2">
          {pillars.map(({ icon: Icon, title, body, href }) => (
            <Link key={title} href={href} className="group bg-zinc-950 p-6 transition-colors hover:bg-zinc-900">
              <Icon className="h-6 w-6 text-[#C785F2]" aria-hidden="true" />
              <h3 className="mt-4 font-semibold text-zinc-100">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-zinc-400">{body}</p>
              <span className="mt-4 inline-flex items-center gap-2 text-sm font-semibold text-zinc-300 group-hover:text-[#C785F2]">
                Explore <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </span>
            </Link>
          ))}
        </div>
      </section>

      <section className="mb-16 border-y border-zinc-900 py-10" aria-labelledby="criteria">
        <div className="max-w-2xl">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center border border-[#C785F2]/40 bg-[#C785F2]/10">
              <Gauge className="h-4 w-4 text-[#C785F2]" aria-hidden="true" />
            </span>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#C785F2]">System 1 contract</p>
          </div>
          <h2 id="criteria" className="mt-5 text-2xl font-bold text-white">Qualification criteria</h2>
          <p className="mt-3 text-sm leading-relaxed text-zinc-400">
            A qualifying model must satisfy all five behaviors on its target task.
          </p>
        </div>

        <ol className="mt-8 max-w-2xl divide-y divide-zinc-900 border-y border-zinc-900">
          {criteria.map(({ title, body }, index) => (
            <li key={title} className="group flex gap-4 py-5 sm:gap-5">
              <span className="mt-0.5 font-mono text-xs text-zinc-600 transition-colors group-hover:text-[#C785F2]">
                {String(index + 1).padStart(2, "0")}
              </span>
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-[#B9DFE0]" aria-hidden="true" />
              <div>
                <h3 className="font-semibold text-zinc-100">{title}</h3>
                <p className="mt-1 text-sm leading-relaxed text-zinc-400">{body}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="mb-16" aria-labelledby="flow">
        <h2 id="flow" className="text-2xl font-bold text-white">The ARCANE System 1 loop</h2>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-zinc-400">
          One continuous path from input to accountable action.
        </p>

        <ol className="relative mt-8 max-w-2xl border-l border-zinc-800 pl-8">
          {systemOneLoop.map(({ number, title, body }, index) => (
            <li key={number} className="relative pb-8 last:pb-0">
              <span
                className="absolute -left-[45px] top-0 flex h-7 w-7 items-center justify-center border border-zinc-700 bg-black font-mono text-[10px] text-[#C785F2] shadow-[0_0_0_6px_#000]"
                aria-hidden="true"
              >
                {number}
              </span>
              <div className="border border-zinc-900 bg-zinc-950/60 px-5 py-4 transition-colors hover:border-zinc-700 hover:bg-zinc-950">
                <div className="flex items-center justify-between gap-4">
                  <h3 className="font-semibold text-zinc-100">{title}</h3>
                  {index < systemOneLoop.length - 1 && (
                    <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-zinc-600">Then</span>
                  )}
                </div>
                <p className="mt-2 text-sm leading-relaxed text-zinc-400">{body}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="mb-16" aria-labelledby="comparison">
        <h2 id="comparison" className="text-2xl font-bold text-white">System 1 and generative AI solve different jobs</h2>
        <p className="mt-3 max-w-2xl leading-relaxed text-zinc-400">
          ARCANE does not replace generative models. It handles fast, repeatable decisions directly and can route work
          to a generative model when the task needs open-ended language or multi-step synthesis.
        </p>
        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[620px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className="py-3 pr-4 font-medium">Dimension</th>
                <th className="py-3 pr-4 font-medium text-zinc-200">ARCANE System 1</th>
                <th className="py-3 font-medium">Generative model</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map(([dimension, systemOne, generative]) => (
                <tr key={dimension} className="border-b border-zinc-900 align-top">
                  <th scope="row" className="py-3 pr-4 font-medium text-zinc-300">{dimension}</th>
                  <td className="border-l-2 border-[#C785F2] bg-[#C785F2]/[0.05] py-3 pl-3 pr-4 text-zinc-100">{systemOne}</td>
                  <td className="py-3 text-zinc-500">{generative}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="border border-zinc-800 bg-zinc-950 p-6 sm:p-8" aria-labelledby="start">
        <h2 id="start" className="text-xl font-bold text-white">Build the model, then prove the behavior</h2>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-zinc-400">
          Start with ARC 1 for grounded decisions, add Hippocampus for immediate memory, and use adaptive layers when
          the task benefits from resonance or plasticity. Validate the complete system on held-out data before deployment.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link href="/docs/quick-start" className="inline-flex items-center gap-2 bg-white px-5 py-2.5 text-sm font-bold text-black hover:bg-zinc-200">
            Quick start <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </Link>
          <Link href="/docs/arc-1" className="inline-flex items-center gap-2 border border-zinc-700 px-5 py-2.5 text-sm font-bold text-zinc-200 hover:border-zinc-500 hover:bg-zinc-900">
            Explore ARC 1
          </Link>
        </div>
      </section>
    </div>
  );
}
