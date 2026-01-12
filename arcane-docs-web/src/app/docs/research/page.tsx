import { Markdown } from "@/components/markdown";
import fs from "fs";
import path from "path";

export default function ResearchPaperPage() {
  const docsPath = path.join(process.cwd(), "..", "docs", "Closing_the_Alignment_Gap_RSAA.md");
  let content = "";
  
  try {
    content = fs.readFileSync(docsPath, "utf8");
  } catch (error) {
    content = "Could not load research paper content.";
  }

  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Research Paper
        </h1>
        <p className="text-xl text-zinc-400">
          The formal mathematical foundation of the Resonant State Alignment Algorithm.
        </p>
      </div>

      <Markdown content={content} />
    </div>
  );
}
