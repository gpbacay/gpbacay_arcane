import { Markdown } from "@/components/markdown";
import fs from "fs";
import path from "path";

export default function PredictiveResonantLayerPage() {
  const docsPath = path.join(process.cwd(), "..", "docs", "PREDICTIVE_RESONANT_LAYER.md");
  let content = "";

  try {
    content = fs.readFileSync(docsPath, "utf8");
    // Remove the title from markdown to avoid duplication with the H1 below
    content = content.replace(/^# .*\n/, "");
  } catch (error) {
    content = "Could not load documentation content.";
  }

  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          PredictiveResonantLayer
        </h1>
        <p className="text-xl text-zinc-400">
          Autonomous local resonance for temporal coherence and self-contained alignment.
        </p>
      </div>

      <div className="my-10 p-10 border border-zinc-800 bg-zinc-950/30 rounded-none relative overflow-hidden group">
        <div className="absolute top-0 right-0 w-64 h-64 bg-[#C785F2]/5 rounded-full blur-3xl -mr-32 -mt-32" />
        <div className="relative z-10 flex flex-col md:flex-row gap-12 items-center">
          <div className="flex-1 space-y-6">
            <div className="flex items-center gap-3">
              <div className="h-[1px] w-8 bg-[#C785F2]" />
              <h3 id="design-philosophy" className="text-[#C785F2] font-bold tracking-[0.2em] uppercase text-xs scroll-mt-24">Design Philosophy</h3>
            </div>
            <p className="text-zinc-300 text-lg leading-relaxed font-light italic">
              "Self-Contained Intelligence"
            </p>
            <p className="text-zinc-400 leading-relaxed text-sm max-w-2xl">
              The PredictiveResonantLayer functions as an autonomous unit, generating its own internal expectations without requiring external guidance. This creates a resilient, high-fidelity neural component capable of stable integration into any architecture.
            </p>
          </div>
        </div>
      </div>

      <Markdown content={content} />
    </div>
  );
}
