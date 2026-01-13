import { Markdown } from "@/components/markdown";
import fs from "fs";
import path from "path";
import Image from "next/image";

export default function NeuralResonancePage() {
  const docsPath = path.join(process.cwd(), "..", "docs", "NEURAL_RESONANCE.md");
  let content = "";
  
  try {
    content = fs.readFileSync(docsPath, "utf8");
    // Remove the title from the markdown
    content = content.replace(/^# .*\n/, "");
  } catch (error) {
    content = "Could not load documentation content.";
  }

  // Split content by section headers
  const sections = content.split(/## (Glossary of Terminologies|Architecture and Information Flow|Advantages and Disadvantages|Technical Implementation Details|Performance & Comparison|Implementation Example|Scientific Context|Conclusion)/);

  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Hierarchical Neural Resonance
        </h1>
        <p className="text-xl text-zinc-400">
          A bi-directional and self-modeling computational mechanism for deep state alignment.
        </p>
      </div>

      {/* Overview and Problem/Solution (sections[0]) */}
      <Markdown content={sections[0]} />

      {/* Glossary (sections[1] is name, sections[2] is content) */}
      {sections[1] && <h2 id="glossary-of-terminologies">{sections[1]}</h2>}
      {sections[2] && <Markdown content={sections[2]} />}

      {/* Architecture (sections[3] is name, sections[4] is content) */}
      {sections[3] && <h2 id="architecture-and-information-flow">{sections[3]}</h2>}
      {sections[4] && (() => {
        const archParts = sections[4].split(/(### .*)/);
        // archParts[0] is intro
        // archParts[1] is "### Hierarchical Structure"
        // archParts[2] is its content
        // archParts[3] is "### The Resonance Cycle..."
        // archParts[4] is its content
        // archParts[5] is "### Internal Logic..."
        // archParts[6] is its content

        return (
          <>
            <Markdown content={archParts[0]} />
            
            {archParts[1] && (
              <div className="mt-8 mb-12">
                <Markdown content={archParts[1] + archParts[2].split("```mermaid")[0]} />
                <div className="mt-8 rounded-none border border-zinc-800 bg-zinc-950/50 p-8 shadow-xl">
                  <div className="flex justify-center">
                    <Image
                      src="/Heirarchical_Structure.png"
                      alt="ARCANE Hierarchical Structure"
                      width={216}
                      height={454}
                      className="rounded-none h-auto shadow-2xl border border-zinc-800 bg-black/20"
                    />
                  </div>
                  <p className="mt-6 text-center text-sm text-zinc-500 italic">
                    Figure 1: The multi-layered hierarchical organization of systemic neural resonance.
                  </p>
                </div>
              </div>
            )}
            
            {archParts[3] && (
              <div className="mt-8 mb-12">
                <Markdown content={archParts[3] + archParts[4].split("```mermaid")[0]} />
                <div className="mt-8 rounded-none border border-zinc-800 bg-zinc-950/50 p-8 shadow-xl">
                  <div className="flex justify-center">
                    <Image
                      src="/Resonance_Cycle.png"
                      alt="The Resonance Cycle"
                      width={600}
                      height={400}
                      className="rounded-none h-auto shadow-2xl border border-zinc-800"
                    />
                  </div>
                  <p className="mt-6 text-center text-sm text-zinc-500 italic">
                    Figure 2: Information flow during hierarchical state synchronization.
                  </p>
                </div>
              </div>
            )}

            {archParts[5] && (
              <div className="mt-8 mb-12">
                <Markdown content={archParts[5] + archParts[6].split("```mermaid")[0]} />
                <div className="mt-8 rounded-none border border-zinc-800 bg-zinc-950/50 p-8 shadow-xl">
                  <div className="flex justify-center">
                    <Image
                      src="/ResonantGSER_Layer_Logic.png"
                      alt="Internal Logic of a ResonantGSER Layer"
                      width={800}
                      height={300}
                      className="rounded-none h-auto shadow-2xl border border-zinc-800"
                    />
                  </div>
                  <p className="mt-6 text-center text-sm text-zinc-500 italic">
                    Figure 3: Local divergence calculation and state harmonization within the GSER cell.
                  </p>
                </div>
              </div>
            )}
          </>
        );
      })()}

      {/* Advantages/Disadvantages (sections[5], sections[6]) */}
      {sections[5] && <h2 id="advantages-and-disadvantages">{sections[5]}</h2>}
      {sections[6] && <Markdown content={sections[6]} />}

      {/* Technical Details (sections[7], sections[8]) */}
      {sections[7] && <h2 id="technical-implementation-details">{sections[7]}</h2>}
      {sections[8] && <Markdown content={sections[8]} />}

      {/* Performance & Comparison (sections[9], sections[10]) */}
      {sections[9] && <h2 id="performance-comparison">{sections[9]}</h2>}
      {sections[10] && <Markdown content={sections[10]} />}

      {/* Implementation Example (sections[11], sections[12]) */}
      {sections[11] && <h2 id="implementation-example">{sections[11]}</h2>}
      {sections[12] && <Markdown content={sections[12]} />}

      {/* Remaining sections */}
      {sections.slice(13).map((section, i) => {
        if (i % 2 === 0) {
          return <h2 key={i} id={section.toLowerCase().replace(/\s+/g, '-')}>{section}</h2>
        } else {
          return <Markdown key={i} content={section} />
        }
      })}
    </div>
  );
}
