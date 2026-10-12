import { Markdown } from "@/components/markdown";
import fs from "fs";
import path from "path";

export default function HomeostaticPlasticityPage() {
  const docsPath = path.join(process.cwd(), "..", "docs", "HOMEOSTATIC_PLASTICITY.md");
  let content = "";

  try {
    content = fs.readFileSync(docsPath, "utf8").replace(/^# .*\n/, "");
  } catch {
    content = "Could not load homeostatic plasticity documentation.";
  }

  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Homeostatic Plasticity
        </h1>
        <p className="text-xl text-zinc-400">
          Bounded activity regulation that lets adaptive System 1 models change without losing control.
        </p>
      </div>
      <Markdown content={content} />
    </div>
  );
}
