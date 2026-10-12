import { Markdown } from "@/components/markdown";
import fs from "fs";
import path from "path";

export default function InferenceTimeLearningPage() {
  const docsPath = path.join(process.cwd(), "..", "docs", "INFERENCE_TIME_RESONANCE.md");
  let content = "";

  try {
    content = fs.readFileSync(docsPath, "utf8").replace(/^# .*\n/, "");
  } catch {
    content = "Could not load inference-time adaptation documentation.";
  }

  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Inference-Time Learning
        </h1>
        <p className="text-xl text-zinc-400">
          The adaptation layer of ARCANE System 1: fixed-weight alignment, bounded fast-weight learning, rollback, and controlled evaluation.
        </p>
      </div>
      <Markdown content={content} />
    </div>
  );
}
