"use client";

/**
 * ARCANE harness · Chain of Thought: the steps a model took, as a collapsible line of nodes.
 * It knows nothing about the model behind it. Needs React 18+ and Tailwind; copy this file into any project.
 * Set --cot-accent on a parent to change the node color.
 */
import { createContext, useContext, useId, useState, type ReactNode } from "react";

export type ThoughtStatus = "complete" | "active" | "pending";

/** One step as data, for a trace you stored or streamed. */
export type ThoughtStep = { label: string; description?: string; status?: ThoughtStatus; results?: string[] };

const ThoughtContext = createContext<{ open: boolean; toggle: () => void; id: string } | null>(null);

function useThought() {
  const context = useContext(ThoughtContext);
  if (!context) throw new Error("ChainOfThought parts must be inside <ChainOfThought>");
  return context;
}

const accent = "var(--cot-accent, #C785F2)";

function Node({ status }: { status: ThoughtStatus }) {
  return (
    <span
      aria-hidden
      className={`flex size-[11px] shrink-0 items-center justify-center rounded-full border-2 ${status === "pending" ? "opacity-40" : ""}`}
      style={{ borderColor: accent, boxShadow: status === "active" ? `0 0 6px ${accent}` : undefined }}
    >
      {status !== "pending" && (
        <span className={`size-[3px] rounded-full ${status === "active" ? "animate-pulse" : ""}`} style={{ backgroundColor: accent }} />
      )}
    </span>
  );
}

export function ChainOfThought({
  open,
  defaultOpen = false,
  onOpenChange,
  className = "",
  children,
}: {
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  className?: string;
  children: ReactNode;
}) {
  const [own, setOwn] = useState(defaultOpen);
  const isOpen = open ?? own;
  const toggle = () => {
    setOwn(!isOpen);
    onOpenChange?.(!isOpen);
  };
  const id = useId();
  return (
    <ThoughtContext.Provider value={{ open: isOpen, toggle, id }}>
      <div className={`w-full font-mono ${className}`}>{children}</div>
    </ThoughtContext.Provider>
  );
}

/** The row that opens the steps. While `thinking`, its node pulses and the label reads "Thinking". */
export function ChainOfThoughtHeader({ thinking = false, children }: { thinking?: boolean; children?: ReactNode }) {
  const { open, toggle, id } = useThought();
  return (
    <button
      type="button"
      onClick={toggle}
      aria-expanded={open}
      aria-controls={id}
      className="group flex w-full items-center gap-2 text-[11px] tracking-wide text-zinc-500 transition-colors hover:text-zinc-300"
    >
      <Node status={thinking ? "active" : "complete"} />
      <span className={`min-w-0 flex-1 truncate text-left ${thinking ? "animate-pulse" : ""}`}>
        {children ?? (thinking ? "Thinking" : "Thought")}
      </span>
      <svg aria-hidden viewBox="0 0 24 24" className={`size-3.5 shrink-0 transition-transform ${open ? "rotate-180" : ""}`}>
        <path d="m6 9 6 6 6-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </button>
  );
}

export function ChainOfThoughtContent({ children }: { children: ReactNode }) {
  const { open, id } = useThought();
  return (
    <div id={id} inert={!open} className={`grid transition-[grid-template-rows] duration-200 ${open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}>
      <ol className="min-h-0 overflow-hidden pt-3">{children}</ol>
    </div>
  );
}

/** One node. A rail joins it to the header above and the next step below. `children` goes under the label. */
export function ChainOfThoughtStep({
  label,
  description,
  status = "complete",
  children,
}: {
  label: ReactNode;
  description?: ReactNode;
  status?: ThoughtStatus;
  children?: ReactNode;
}) {
  const rail = { backgroundColor: accent, opacity: 0.3 };
  return (
    <li className="group/step relative pb-3 pl-5 last:pb-0">
      <span aria-hidden className="absolute -top-3 left-[5px] hidden h-[15px] w-px group-first/step:block" style={rail} />
      <span aria-hidden className="absolute bottom-0 left-[5px] top-[17px] w-px group-last/step:hidden" style={rail} />
      <span className="absolute left-0 top-[3px]">
        <Node status={status} />
      </span>
      <p className={`text-[12px] leading-[17px] ${status === "pending" ? "text-zinc-600" : status === "active" ? "text-zinc-100" : "text-zinc-300"}`}>
        {label}
      </p>
      {description && <p className="mt-0.5 text-[11px] leading-relaxed text-zinc-500">{description}</p>}
      {children}
    </li>
  );
}

/** Chips under a step: what a search, a recall or a tool came back with. */
export function ChainOfThoughtResults({ children }: { children: ReactNode }) {
  return <div className="mt-1.5 flex flex-wrap gap-1.5">{children}</div>;
}

export function ChainOfThoughtResult({ children }: { children: ReactNode }) {
  return <span className="max-w-full truncate border border-zinc-800 bg-zinc-900 px-1.5 py-0.5 text-[11px] text-zinc-400">{children}</span>;
}

/** A whole trace from data. Closed by default. */
export function ChainOfThoughtTrace({
  label,
  steps,
  thinking = false,
  defaultOpen = false,
}: {
  label?: ReactNode;
  steps: ThoughtStep[];
  thinking?: boolean;
  defaultOpen?: boolean;
}) {
  return (
    <ChainOfThought defaultOpen={defaultOpen}>
      <ChainOfThoughtHeader thinking={thinking}>{label}</ChainOfThoughtHeader>
      <ChainOfThoughtContent>
        {steps.map(({ results, ...step }, index) => (
          <ChainOfThoughtStep key={index} {...step}>
            {results?.length ? (
              <ChainOfThoughtResults>
                {results.map((result) => (
                  <ChainOfThoughtResult key={result}>{result}</ChainOfThoughtResult>
                ))}
              </ChainOfThoughtResults>
            ) : null}
          </ChainOfThoughtStep>
        ))}
      </ChainOfThoughtContent>
    </ChainOfThought>
  );
}
