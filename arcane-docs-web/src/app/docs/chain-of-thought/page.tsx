import { readFileSync } from "node:fs";
import { join } from "node:path";
import { CodeSnippet } from "@/components/CodeSnippet";
import { ChainOfThoughtTrace, type ThoughtStep } from "@/components/arcane/chain-of-thought";

const h2 = "scroll-mt-28 mt-16 text-2xl font-bold tracking-tight text-zinc-100 mb-4 border-b border-zinc-800 pb-2";
const code = "bg-zinc-900 px-1.5 py-0.5 text-zinc-100";
const p = "max-w-2xl leading-relaxed";
const th = "py-2 pr-4 font-medium";
const td = "py-2.5 pr-4 align-top text-zinc-300";

// Read at build time, so the code to copy is always the component this page renders.
const SOURCE = readFileSync(join(process.cwd(), "src/components/arcane/chain-of-thought.tsx"), "utf8");

const COMPOSED = `import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtHeader,
  ChainOfThoughtResult,
  ChainOfThoughtResults,
  ChainOfThoughtStep,
  ChainOfThoughtSubsteps,
} from "@/components/arcane/chain-of-thought";

<ChainOfThought defaultOpen>
  <ChainOfThoughtHeader>Thought · ARC 1 + Hippocampus</ChainOfThoughtHeader>
  <ChainOfThoughtContent>
    <ChainOfThoughtStep label="Recalled a similar request">
      <ChainOfThoughtResults>
        <ChainOfThoughtResult>go to the hippocampus page</ChainOfThoughtResult>
      </ChainOfThoughtResults>
    </ChainOfThoughtStep>
    <ChainOfThoughtStep label="Decided on 2 actions">
      <ChainOfThoughtSubsteps>
        <ChainOfThoughtStep label="Open Hippocampus" />
        <ChainOfThoughtStep label="Answer about Occipital" />
      </ChainOfThoughtSubsteps>
    </ChainOfThoughtStep>
    <ChainOfThoughtStep label="Opening the page" status="active" />
  </ChainOfThoughtContent>
</ChainOfThought>`;

const FROM_DATA = `import { ChainOfThoughtTrace, type ThoughtStep } from "@/components/arcane/chain-of-thought";

// Any model: map what it did to steps, and what each step did to its sub-steps. Store them with the reply.
const steps: ThoughtStep[] = [
  { label: "Searched the docs", results: ["Hippocampus", "Occipital"] },
  {
    label: "Compared the two pages",
    steps: [{ label: "Read Hippocampus" }, { label: "Read Occipital" }],
  },
  { label: "Writing the answer", status: "active" },
];

<ChainOfThoughtTrace label="Thought · 3 steps" steps={steps} thinking={isStreaming} />`;

const PREVIEW: ThoughtStep[] = [
  { label: "Recalled a similar request", results: ["go to the hippocampus page"] },
  {
    label: "Decided on 2 actions",
    steps: [
      { label: "Open Hippocampus" },
      { label: "Answer about Occipital", steps: [{ label: "Found the Occipital page" }, { label: "Picked the typo fact" }] },
    ],
  },
  { label: "Opening the page", status: "active" },
  { label: "Remember this turn", status: "pending" },
];

const PARTS: [string, string, string][] = [
  ["ChainOfThought", "open, defaultOpen, onOpenChange, className", "Holds the open state. Controlled or not."],
  ["ChainOfThoughtHeader", "thinking", "The row that opens the steps. While thinking, its node pulses and it reads Thinking."],
  ["ChainOfThoughtContent", "", "The steps. Animates open and closed."],
  ["ChainOfThoughtStep", "label, description, status, children", "One node. status is complete (default), active or pending."],
  ["ChainOfThoughtSubsteps", "", "Inside a step: its sub-steps, each with its own node, branching off the step's line. Nests."],
  ["ChainOfThoughtResults / Result", "", "Chips under a step: what a search, a recall or a tool came back with."],
  ["ChainOfThoughtTrace", "label, steps, thinking, defaultOpen", "The whole trace from a ThoughtStep[] array. A step's steps are its sub-steps."],
];

export default function ChainOfThoughtPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <header className="not-prose mb-12">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#C785F2]">ARCANE · Components</p>
        <h1 className="mt-3 max-w-3xl text-4xl font-extrabold leading-[1.08] tracking-[-0.03em] text-zinc-50 sm:text-5xl">
          Chain of Thought
        </h1>
        <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-zinc-300">
          A React component that shows the steps a model took, as a graph of nodes that opens from one header. A step
          can branch into sub-steps, each with its own node. It knows
          nothing about the model behind it: ARC 1, an LLM or a plain rule engine all hand it the same steps. The ARCANE
          chat uses it for every reply.
        </p>
      </header>

      <div className="text-zinc-300">
        <h2 id="preview" className={`${h2} mt-0`}>
          Preview
        </h2>
        <div className="not-prose max-w-md border border-zinc-800 bg-black p-5">
          <ChainOfThoughtTrace defaultOpen label="Thought · ARC 1 + Hippocampus + Occipital" steps={PREVIEW} />
        </div>

        <h2 id="install" className={h2}>
          Install
        </h2>
        <p className={p}>
          Copy the file below into your project, for example to{" "}
          <code className={code}>components/arcane/chain-of-thought.tsx</code>. It needs React 18 or later and
          Tailwind, nothing else. Set <code className={code}>--cot-accent</code> on a parent to change the node color, and{" "}
          <code className={code}>--cot-rail</code> to change the lines on a light background.
        </p>
        <CodeSnippet filename="chain-of-thought.tsx" language="tsx" code={SOURCE} maxHeight="24rem" />

        <h2 id="usage" className={h2}>
          Usage
        </h2>
        <p className={p}>Compose the parts when the steps are known as you write the page.</p>
        <CodeSnippet filename="Composed" language="tsx" code={COMPOSED} />
        <p className={p}>
          Or pass the steps as data. Keep the array with each reply, and while a reply streams, add steps and mark the
          current one <code className={code}>active</code>.
        </p>
        <CodeSnippet filename="From data" language="tsx" code={FROM_DATA} />

        <h2 id="parts" className={h2}>
          Parts
        </h2>
        <div className="not-prose overflow-x-auto">
          <table className="w-full min-w-[560px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className={th}>Part</th>
                <th className={th}>Props</th>
                <th className="py-2 font-medium">What it is</th>
              </tr>
            </thead>
            <tbody>
              {PARTS.map(([part, props, what]) => (
                <tr key={part} className="border-b border-zinc-900">
                  <td className={`${td} font-mono text-zinc-100`}>{part}</td>
                  <td className={`${td} font-mono text-xs`}>{props || "children"}</td>
                  <td className="py-2.5 align-top text-zinc-300">{what}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={`${p} mt-6 text-sm text-zinc-400`}>
          Show what the model did, not everything it considered: one step per action, and only the results that
          explain the next step.
        </p>
      </div>
    </div>
  );
}
