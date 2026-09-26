import Link from "next/link";
import { SlmChat } from "@/components/SlmChat";

export default function ChatPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-8">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Chat with ARCANE
        </h1>
        <p className="text-xl text-zinc-400">
          Talk to{" "}
          <code className="text-[#C785F2] bg-zinc-900 px-1.5 py-0.5">Arc1LanguageModel</code>
          — ARC 1&apos;s perception stack (FieldAttention, ResonantChannelMixer, ConceptEngram, FieldResonance)
          run causally as a ~10M-parameter (~40 MB) small language model, distilled from Qwen2.5-0.5B.
        </p>
      </div>

      <SlmChat />

      <div className="space-y-6 text-zinc-300 mt-10">
        <h2
          id="how-this-works"
          className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2"
        >
          How this works
        </h2>
        <p className="leading-relaxed">
          The page calls a same-origin proxy at <code className="bg-zinc-900 px-1.5 py-0.5">/api/slm-chat</code>,
          which forwards to the Python SLM server. That server loads the{" "}
          <Link href="/docs/arc-1" className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium">
            ARC 1
          </Link>{" "}
          language model and samples next tokens. It is the same architecture as the ARC 1 decision model, with
          two changes: attention gets a causal mask, and field resonance harmonizes each token toward the running
          mean of its prefix instead of the whole field. A tied embedding head turns the field into next-token logits.
        </p>

        <div className="rounded-none border border-amber-900/50 bg-amber-900/10 p-4 not-prose">
          <p className="text-sm text-amber-200">
            <strong>Local:</strong> from <code className="bg-zinc-900 px-1.5 py-0.5 text-amber-100">arcane-docs-web</code> run{" "}
            <code className="bg-zinc-900 px-1.5 py-0.5 text-amber-100">npm run dev:with-slm</code>. That starts Next.js
            and <code className="bg-zinc-900 px-1.5 py-0.5">python examples/serve_slm_api.py</code> together. Or start
            the API yourself on port 8001 and set{" "}
            <code className="bg-zinc-900 px-1.5 py-0.5">SLM_API_URL=http://127.0.0.1:8001</code> in{" "}
            <code className="bg-zinc-900 px-1.5 py-0.5">.env.local</code>.
          </p>
          <p className="text-sm text-amber-200 mt-2">
            Train it (minutes on CPU) with{" "}
            <code className="bg-zinc-900 px-1.5 py-0.5">python examples/distill_arcane_slm.py --arch arc1</code>.
            The server picks up <code className="bg-zinc-900 px-1.5 py-0.5">Models/arc1_lm.weights.h5</code> automatically.
          </p>
        </div>
      </div>
    </div>
  );
}
