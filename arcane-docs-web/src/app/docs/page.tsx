import Link from "next/link";
import {
  ArrowRight,
  BrainCircuit,
  Braces,
  DatabaseZap,
  Gauge,
  GitBranch,
  Layers3,
  Route,
} from "lucide-react";

const paths = [
  {
    icon: Route,
    title: "Automate decisions",
    body: "ARC 1 is the System 1 decision model: a typed tool call, copied fields, or a label in one pass, on your own CPU.",
    href: "/docs/arc-1",
    action: "Explore ARC 1",
  },
  {
    icon: Layers3,
    title: "Build adaptive models",
    body: "Compose recurrent, spiking, plastic, and homeostatic mechanisms around the behavior your system needs.",
    href: "/docs/layers",
    action: "Browse layers",
  },
  {
    icon: BrainCircuit,
    title: "Refine at inference time",
    body: "Use resonance and bounded adaptation to update state for the input currently being processed.",
    href: "/docs/inference-time-learning",
    action: "Learn the workflow",
  },
  {
    icon: DatabaseZap,
    title: "Add memory and grounding",
    body: "Retrieve decided examples with Hippocampus or build evidence-linked reasoning with Graph of Thought.",
    href: "/docs/hippocampus",
    action: "Explore memory",
  },
];

const benefits = [
  ["Direct outputs", "Return a tool call, label, field, score, embedding, or prediction instead of parsing generated prose."],
  ["Controlled adaptation", "Separate learned weights, transient resonance state, and optional bounded updates."],
  ["Memory with evidence", "Recall the examples behind a decision and add or remove experience without retraining."],
  ["Observable behavior", "Measure accuracy, calibration, latency, state change, and adaptation cost directly."],
];

export default function DocsPage() {
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header className="mb-12 border-b border-zinc-900 pb-10">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#C785F2]">ARCANE documentation</p>
        <h1 className="mt-3 max-w-3xl text-4xl font-extrabold leading-tight tracking-[-0.035em] text-white sm:text-5xl">
          Build adaptive System 1 AI.
        </h1>
        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-zinc-300">
          <strong className="text-zinc-100">Augmented Reconstruction of Consciousness through Artificial Neural Evolution</strong>{" "}
          is a library for models that make fast, grounded decisions using adaptive inference, neural memory, and
          biologically inspired computation.
        </p>
        <div className="mt-7 flex flex-col gap-3 sm:flex-row">
          <Link href="/docs/quick-start" className="inline-flex items-center justify-center gap-2 bg-white px-5 py-2.5 text-sm font-bold text-black transition-colors hover:bg-zinc-200">
            Quick start <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </Link>
          <Link href="/docs/installation" className="inline-flex items-center justify-center border border-zinc-700 px-5 py-2.5 text-sm font-bold text-zinc-200 transition-colors hover:border-zinc-500 hover:bg-zinc-900">
            Installation
          </Link>
        </div>
      </header>

      <section className="mb-16" aria-labelledby="problem">
        <div className="max-w-2xl">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#B9DFE0]">Purpose</p>
          <h2 id="problem" className="mt-3 text-2xl font-bold text-white">What System 1 means in ARCANE</h2>
          <p className="mt-4 leading-relaxed text-zinc-400">
            System 1 is an engineering profile for fast, automatic decisions, not a claim about human consciousness.
            An ARCANE System 1 model produces a bounded output directly, reports uncertainty, can recall relevant
            experience, and adapts through explicit state or fast-weight mechanisms.
          </p>
        </div>
        <div className="mt-7 grid gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-2">
          {[
            [Braces, "Direct", "Return a tool, label, score, field, or abstention without an autoregressive decoding loop."],
            [Gauge, "Adaptive", "Align state or update bounded fast weights while processing the current input."],
            [GitBranch, "Experience-guided", "Retrieve prior decisions and use them as evidence for the current task."],
            [Layers3, "Neuromimetic", "Compose feedback, spiking, plasticity, and homeostatic mechanisms around the task."],
          ].map(([Icon, title, body]) => {
            const FeatureIcon = Icon as typeof Braces;
            return (
              <article key={title as string} className="bg-zinc-950 p-5">
                <FeatureIcon className="h-5 w-5 text-[#C785F2]" aria-hidden="true" />
                <h3 className="mt-4 font-semibold text-zinc-100">{title as string}</h3>
                <p className="mt-2 text-sm leading-relaxed text-zinc-400">{body as string}</p>
              </article>
            );
          })}
        </div>
        <Link href="/docs/system-1" className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-[#C785F2] hover:text-[#d49cf5]">
          Read the System 1 architecture guide <ArrowRight className="h-4 w-4" aria-hidden="true" />
        </Link>
      </section>

      <section className="mb-16" aria-labelledby="choose-a-path">
        <h2 id="choose-a-path" className="text-2xl font-bold text-white">Choose your path</h2>
        <p className="mt-3 max-w-2xl leading-relaxed text-zinc-400">
          Start with the outcome you need. Each path includes the concepts, implementation details, and working examples for that job.
        </p>
        <div className="mt-7 grid gap-4 sm:grid-cols-2">
          {paths.map(({ icon: Icon, title, body, href, action }) => (
            <Link key={title} href={href} className="group border border-zinc-800 bg-zinc-950/60 p-6 transition-colors hover:border-[#C785F2]/60 hover:bg-zinc-900/60">
              <Icon className="h-6 w-6 text-[#C785F2]" aria-hidden="true" />
              <h3 className="mt-5 text-lg font-semibold text-zinc-100">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-zinc-400">{body}</p>
              <span className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-zinc-200 transition-colors group-hover:text-[#C785F2]">
                {action} <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" aria-hidden="true" />
              </span>
            </Link>
          ))}
        </div>
      </section>

      <section className="mb-16 grid gap-8 border-y border-zinc-900 py-10 md:grid-cols-[0.7fr_1.3fr]" aria-labelledby="benefits">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#B9DFE0]">Design principles</p>
          <h2 id="benefits" className="mt-3 text-2xl font-bold text-white">What ARCANE gives you</h2>
        </div>
        <dl className="grid gap-x-8 gap-y-6 sm:grid-cols-2">
          {benefits.map(([term, description]) => (
            <div key={term}>
              <dt className="font-semibold text-zinc-100">{term}</dt>
              <dd className="mt-1.5 text-sm leading-relaxed text-zinc-400">{description}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="mb-16" aria-labelledby="how-it-fits">
        <h2 id="how-it-fits" className="text-2xl font-bold text-white">Perceive. Recall. Adapt. Act.</h2>
        <div className="mt-6 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {[
            ["01", "Perceive", "Encode an input, schema, sequence, or retrieved example into a shared representation."],
            ["02", "Recall", "Retrieve relevant decisions and examples that can guide the current response."],
            ["03", "Adapt", "Align internal state or bounded fast weights according to the current input."],
            ["04", "Act", "Return a grounded decision or pass the refined representation to the next model stage."],
          ].map(([number, title, body]) => (
            <div key={number} className="border-l border-zinc-700 pl-5">
              <p className="font-mono text-xs text-[#C785F2]">{number}</p>
              <h3 className="mt-3 font-semibold text-zinc-100">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-zinc-400">{body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="border border-zinc-800 bg-zinc-950 p-6 sm:p-8" aria-labelledby="project">
        <div className="grid gap-6 md:grid-cols-[1fr_auto] md:items-center">
          <div>
            <h2 id="project" className="text-xl font-bold text-white">Open source and ready to extend</h2>
            <p className="mt-3 max-w-2xl text-sm leading-relaxed text-zinc-400">
              ARCANE is led by Gianne P. Bacay and released under the MIT License. The source, tests, models, and
              implementation history are available for inspection, reuse, and contribution.
            </p>
          </div>
          <a href="https://github.com/gpbacay/gpbacay_arcane" target="_blank" rel="noreferrer" className="inline-flex items-center justify-center gap-2 border border-zinc-700 px-5 py-2.5 text-sm font-semibold text-zinc-200 transition-colors hover:border-zinc-500 hover:bg-zinc-900">
            View on GitHub <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </a>
        </div>
      </section>
    </div>
  );
}
