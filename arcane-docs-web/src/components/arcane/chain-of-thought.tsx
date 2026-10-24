"use client";

/**
 * ARCANE components · Chain of Thought: the steps a model took, as a collapsible graph of nodes.
 * A step can hold sub-steps, which branch off it with their own nodes, to any depth.
 * It knows nothing about the model behind it. Needs React 18+ and Tailwind; copy this file into any project.
 * Set --cot-accent on a parent to change the node color, and --cot-rail for the lines on a light background.
 */
import { Children, createContext, isValidElement, useContext, useId, useState, type ReactNode } from "react";

export type ThoughtStatus = "complete" | "active" | "pending";

/** One step as data, for a trace you stored or streamed. `steps` are its sub-steps. */
export type ThoughtStep = {
  label: string;
  description?: string;
  status?: ThoughtStatus;
  results?: string[];
  steps?: ThoughtStep[];
};

const ThoughtContext = createContext<{ open: boolean; toggle: () => void; id: string } | null>(null);
/** Where a step sits in its list, so it can draw its own connections. */
const PlaceContext = createContext({ root: true, first: true, last: true });

function useThought() {
  const context = useContext(ThoughtContext);
  if (!context) throw new Error("ChainOfThought parts must be inside <ChainOfThought>");
  return context;
}

const accent = "var(--cot-accent, #C785F2)";
// Opaque, so lines that meet or cross never look darker where they overlap.
const railColor = `var(--cot-rail, color-mix(in srgb, ${accent} 35%, #09090b))`;

// Geometry (px): an 11px node at x 0-11, its center line at x = 5, sub-steps indented 20px.
// From inside a sub-step, the parent's center line is at x = -15.
function Line({ className }: { className: string }) {
  return <span aria-hidden className={`absolute w-px ${className}`} style={{ backgroundColor: railColor }} />;
}

function Node({ status }: { status: ThoughtStatus }) {
  return (
    <span
      aria-hidden
      className={`relative flex size-[11px] shrink-0 items-center justify-center rounded-full border-2 ${status === "pending" ? "border-dashed" : ""}`}
      style={{ borderColor: status === "pending" ? railColor : accent }}
    >
      {status === "active" && (
        <span className="absolute inset-[-4px] animate-ping rounded-full opacity-40" style={{ backgroundColor: accent }} />
      )}
      {status !== "pending" && <span className="size-[3px] rounded-full" style={{ backgroundColor: accent }} />}
    </span>
  );
}

function StepList({ root, className = "", children }: { root: boolean; className?: string; children: ReactNode }) {
  const items = Children.toArray(children);
  return (
    <ol className={className}>
      {items.map((item, index) => (
        <PlaceContext.Provider key={isValidElement(item) ? item.key : index} value={{ root, first: index === 0, last: index === items.length - 1 }}>
          {item}
        </PlaceContext.Provider>
      ))}
    </ol>
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
      className={`group flex w-full items-center gap-2 text-[11px] tracking-wide transition-colors hover:text-zinc-200 ${open ? "text-zinc-300" : "text-zinc-500"}`}
    >
      <Node status={thinking ? "active" : "complete"} />
      <span className={`min-w-0 flex-1 truncate text-left ${thinking ? "animate-pulse" : ""}`}>
        {children ?? (thinking ? "Thinking" : "Thought")}
      </span>
      <svg aria-hidden viewBox="0 0 24 24" className={`size-3.5 shrink-0 transition-transform duration-200 ${open ? "rotate-180" : ""}`}>
        <path d="m6 9 6 6 6-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </button>
  );
}

export function ChainOfThoughtContent({ children }: { children: ReactNode }) {
  const { open, id } = useThought();
  return (
    <div id={id} inert={!open} className={`grid transition-[grid-template-rows,opacity] duration-200 ${open ? "grid-rows-[1fr]" : "grid-rows-[0fr] opacity-0"}`}>
      <div className="min-h-0 overflow-hidden">
        <StepList root className="pt-3">
          {children}
        </StepList>
      </div>
    </div>
  );
}

/** Sub-steps of a step: each branches off the step's line with its own node. Can nest. */
export function ChainOfThoughtSubsteps({ children }: { children: ReactNode }) {
  return (
    <StepList root={false} className="mt-2">
      {children}
    </StepList>
  );
}

/** One node. Lines join it to the header, its siblings and its sub-steps. Chips and sub-steps go in `children`. */
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
  const { root, first, last } = useContext(PlaceContext);
  const parts = Children.toArray(children);
  const substeps = parts.filter((part) => isValidElement(part) && part.type === ChainOfThoughtSubsteps);
  const rest = parts.filter((part) => !substeps.includes(part));
  // A top-level step's line runs on to the next one, through its own sub-steps. Otherwise it ends at its last sub-step.
  const continues = root && !last;
  return (
    <li className={`relative pl-5 transition-[opacity,translate] duration-300 starting:-translate-y-1 starting:opacity-0 ${last ? "" : root ? "pb-3" : "pb-2"}`}>
      {root && first && <Line className="-top-3 left-[5px] h-[15px]" />}
      {continues && <Line className="bottom-0 left-[5px] top-[14px]" />}
      {!root && (
        <>
          {/* the elbow from the parent's line into this node, and that line on past it to the next sub-step */}
          <span
            aria-hidden
            className="absolute -left-[15px] top-0 h-[9px] w-[15px] rounded-bl-[6px] border-b border-l"
            style={{ borderColor: railColor }}
          />
          {!last && <Line className="-left-[15px] bottom-0 top-0" />}
        </>
      )}
      <span className="absolute left-0 top-[3px]">
        <Node status={status} />
      </span>
      <div className="relative">
        {substeps.length > 0 && !continues && <Line className="-bottom-2 -left-[15px] top-[14px]" />}
        <p
          className={`text-[12px] leading-[17px] ${status === "pending" ? "text-zinc-600" : status === "active" ? "text-zinc-100" : root ? "text-zinc-300" : "text-zinc-400"}`}
        >
          {label}
        </p>
        {description && <p className="mt-0.5 text-[11px] leading-relaxed text-zinc-500">{description}</p>}
        {rest}
      </div>
      {substeps}
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

function StepFromData({ step: { results, steps, ...step } }: { step: ThoughtStep }) {
  return (
    <ChainOfThoughtStep {...step}>
      {results?.length ? (
        <ChainOfThoughtResults>
          {results.map((result) => (
            <ChainOfThoughtResult key={result}>{result}</ChainOfThoughtResult>
          ))}
        </ChainOfThoughtResults>
      ) : null}
      {steps?.length ? (
        <ChainOfThoughtSubsteps>
          {steps.map((substep, index) => (
            <StepFromData key={index} step={substep} />
          ))}
        </ChainOfThoughtSubsteps>
      ) : null}
    </ChainOfThoughtStep>
  );
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
        {steps.map((step, index) => (
          <StepFromData key={index} step={step} />
        ))}
      </ChainOfThoughtContent>
    </ChainOfThought>
  );
}
