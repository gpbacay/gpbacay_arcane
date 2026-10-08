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
          Try ARC 1&apos;s language models directly. ARC 1 LM v2 is a 12.6M-parameter hybrid decoder with
          gated short convolutions, grouped-query attention, selective causal resonance, and cached generation.
          Use the model selector to compare it with the original ARC 1 LM.
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
          which forwards the selected model ID to the Python SLM server. The server keeps the available{" "}
          <Link href="/docs/arc-1" className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium">
            ARC 1
          </Link>{" "}
          checkpoints loaded and samples from the model you select. Health metadata shows whether each checkpoint
          is loaded, trained, and ready before the page enables generation.
        </p>
        <p className="leading-relaxed">
          These are small next-token continuation models, not instruction-tuned assistants. Selecting a model proves
          which architecture produced a response, but a loaded checkpoint does not guarantee a factually correct
          answer. The current v2 checkpoint received only bounded smoke training and is exposed here as an experimental
          architecture preview.
        </p>

        <div className="rounded-none border border-amber-900/50 bg-amber-900/10 p-4 not-prose">
          <p className="text-sm text-amber-200">
            <strong>Local:</strong> from <code className="bg-zinc-900 px-1.5 py-0.5 text-amber-100">arcane-docs-web</code> run{" "}
            <code className="bg-zinc-900 px-1.5 py-0.5 text-amber-100">npm run dev:with-slm</code>. This starts Next.js
            and <code className="bg-zinc-900 px-1.5 py-0.5">python examples/serve_slm_api.py</code> together. Or start
            the API on port 8001 and set{" "}
            <code className="bg-zinc-900 px-1.5 py-0.5">SLM_API_URL=http://127.0.0.1:8001</code> in{" "}
            <code className="bg-zinc-900 px-1.5 py-0.5">.env.local</code>.
          </p>
          <p className="text-sm text-amber-200 mt-2">
            The server discovers <code className="bg-zinc-900 px-1.5 py-0.5">Models/arc1_lm_v2.*</code> and{" "}
            <code className="bg-zinc-900 px-1.5 py-0.5">Models/arc1_lm.*</code> automatically. Set{" "}
            <code className="bg-zinc-900 px-1.5 py-0.5">SLM_DEFAULT_MODEL=arc1-lm-v1</code> to change the initial selection.
          </p>
        </div>
      </div>
    </div>
  );
}
