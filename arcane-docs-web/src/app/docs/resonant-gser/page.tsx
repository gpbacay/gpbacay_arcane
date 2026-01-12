import { Markdown } from "@/components/markdown";
import fs from "fs";
import path from "path";
import Image from "next/image";

export default function ResonantGSERPage() {
  const docsPath = path.join(process.cwd(), "..", "docs", "RESONANT_GSER.md");
  let content = "";
  
  try {
    content = fs.readFileSync(docsPath, "utf8");
    content = content.replace(/^# .*\n/, "");
  } catch (error) {
    content = "Could not load documentation content.";
  }

  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          ResonantGSER Layer
        </h1>
        <p className="text-xl text-zinc-400">
          The core mechanism implementing Hierarchical Neural Resonance.
        </p>
      </div>

      <div className="my-10 flex justify-center overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950/50 p-6 shadow-2xl">
        <div className="text-center">
          <Image
            src="/ResonantGSER_Layer_Logic.png"
            alt="ResonantGSER Layer Logic"
            width={600}
            height={300}
            className="rounded-lg mx-auto h-auto shadow-lg"
          />
          <p className="mt-6 px-4 text-sm text-zinc-500 italic">
            Figure 1: Internal logic and state synchronization of the ResonantGSER layer.
          </p>
        </div>
      </div>

      <Markdown content={content} />
    </div>
  );
}
