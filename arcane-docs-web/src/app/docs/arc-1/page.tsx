import Link from "next/link";
import { Arc1Demo } from "@/components/Arc1Demo";

export default function Arc1Page() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-8">
        <p className="text-sm uppercase tracking-[0.18em] text-[#C785F2] mb-3">ARC 1</p>
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-3 text-zinc-100 leading-[1.15]">
          Automation foundation for tiny devices
        </h1>
        <p className="text-lg text-zinc-400 max-w-2xl leading-relaxed">
          Tool calls, structured extraction, confidence, and a depth ladder — try it in the sandbox
          below. Inspired by product playgrounds like{" "}
          <a
            href="https://cactuscompute.com/dashboard/playground"
            className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium"
            target="_blank"
            rel="noreferrer"
          >
            Cactus
          </a>
          , built on ARCANE mechanisms.
        </p>
      </div>

      <Arc1Demo />

      <div className="space-y-6 text-zinc-300 mt-12">
        <h2
          id="architecture"
          className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2"
        >
          Architecture
        </h2>
        <p className="leading-relaxed">
          Each <code className="bg-zinc-900 px-1.5 py-0.5">Arc1DecoderBlock</code> runs causal
          attention, a cheap <code className="bg-zinc-900 px-1.5 py-0.5">ResonantChannelMixer</code>{" "}
          (low-rank + DenseGSER gate), a{" "}
          <code className="bg-zinc-900 px-1.5 py-0.5">ConceptEngram</code> hashed n-gram memory,
          then <code className="bg-zinc-900 px-1.5 py-0.5">ResonantSequenceMixer</code>. One weight
          set exposes nested ladder depths so you can slice compute for the device; every depth is
          trained, because each training step samples one.
        </p>

        <h2
          id="decisions"
          className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2"
        >
          Typed decisions, not free-form JSON
        </h2>
        <p className="leading-relaxed">
          Following{" "}
          <a
            href="https://laya.convaiinnovations.com/"
            className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium"
            target="_blank"
            rel="noreferrer"
          >
            Laya
          </a>
          , ARC 1 never writes JSON token by token. Each request is two batched forward passes over
          short sequences, one per decision:
        </p>
        <ul className="list-disc pl-6 space-y-1">
          <li>
            <strong>noul</strong>: calibrated P(true) that a tool applies, that an optional
            argument is present, or that a boolean argument is true.
          </li>
          <li>
            <strong>span</strong>: start/end pointers that copy strings and numbers from the
            user&apos;s text, so values are never invented.
          </li>
          <li>
            <strong>choice</strong>: a softmax over enum options.
          </li>
        </ul>
        <p className="leading-relaxed">
          Output is schema-valid by construction. Each head has one temperature fitted on held-out
          data, so the reported confidence is calibrated. Because the model is causal, arguments are
          read from a repeated (&quot;echo&quot;) copy of the text: every pointed-at token has then
          already seen the whole sentence.
        </p>

        <h2
          id="how-to-run"
          className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2"
        >
          How to run locally
        </h2>
        <div className="rounded-xl border border-amber-900/40 bg-amber-950/20 p-4 not-prose">
          <p className="text-sm text-amber-100/90">
            From <code className="bg-zinc-900 px-1.5 py-0.5 text-amber-50">arcane-docs-web</code>{" "}
            run <code className="bg-zinc-900 px-1.5 py-0.5 text-amber-50">npm run dev:with-arc1</code>.
            That starts Next.js and{" "}
            <code className="bg-zinc-900 px-1.5 py-0.5">python examples/serve_arc1_api.py</code> on
            port 8002.
          </p>
        </div>

        <p className="leading-relaxed">
          Related:{" "}
          <Link href="/docs/chat" className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium">
            Chat with ARCANE SLM
          </Link>{" "}
          and{" "}
          <Link href="/docs/layers" className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium">
            biological layers
          </Link>
          .
        </p>
      </div>
    </div>
  );
}
