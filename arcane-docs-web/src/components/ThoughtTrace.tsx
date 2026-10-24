"use client";

import { useState } from "react";
import { ChevronDown, Wrench } from "lucide-react";
import type { TraceStep } from "@/lib/docsChat";

/** One colour per nesting depth, like branch lanes in a git graph. */
const LANES = ["#C785F2", "#f5a524", "#4f9cff", "#34d399"];
const lane = (depth: number) => LANES[depth % LANES.length];
const rail = (color: string) => ({ backgroundColor: `${color}40` });

// Geometry (px): an 11px marker centred on a 1px rail at x = 5, text indented 20px under the header label.
function Dot({ color, pulse }: { color: string; pulse?: boolean }) {
  return (
    <span
      aria-hidden
      className="flex size-[11px] shrink-0 items-center justify-center rounded-full border-2"
      style={{ borderColor: color, boxShadow: pulse ? `0 0 6px ${color}99` : undefined }}
    >
      <span className={`size-[3px] rounded-full ${pulse ? "animate-pulse" : ""}`} style={{ backgroundColor: color }} />
    </span>
  );
}

function Chevron({ open }: { open: boolean }) {
  return <ChevronDown className={`mt-0.5 size-3 shrink-0 text-zinc-600 transition-transform group-hover:text-zinc-300 ${open ? "rotate-180" : ""}`} />;
}

function Step({ step, depth, hasNext, first }: { step: TraceStep; depth: number; hasNext: boolean; first?: boolean }) {
  // Top-level thoughts read like a narrative, so they start open; tool cards and sub-steps open on demand.
  const [open, setOpen] = useState(depth === 0 && !step.tool);
  const color = lane(depth);
  const parent = lane(depth - 1);
  const children = step.children ?? [];
  const showChildren = open && children.length > 0;
  const expandable = Boolean(step.detail) || children.length > 0;

  return (
    <li className="relative">
      {first && <span className="absolute -top-3 left-[5px] h-[15px] w-px" style={rail(color)} />}
      {depth === 0 && hasNext && <span className="absolute bottom-0 left-[5px] top-[17px] w-px" style={rail(color)} />}
      {depth > 0 && (
        <>
          {hasNext && <span className="absolute -left-[15px] bottom-0 top-0 w-px" style={rail(parent)} />}
          <span className="absolute -left-[15px] top-0 h-[9px] w-[15px] border-b border-l" style={{ borderColor: `${parent}59` }} />
        </>
      )}
      <div className={`relative pl-5 ${showChildren ? "pb-2.5" : hasNext ? "pb-3" : ""}`}>
        {showChildren && <span className="absolute bottom-0 left-[5px] top-[17px] w-px" style={rail(color)} />}
        {step.tool ? (
          <span
            aria-hidden
            className="absolute -left-[2px] top-[1px] flex size-[15px] items-center justify-center border"
            style={{ borderColor: `${color}80`, backgroundColor: `${color}1f`, color }}
          >
            <Wrench className="size-[9px]" />
          </span>
        ) : (
          <span className="absolute left-0 top-[3px]">
            <Dot color={color} />
          </span>
        )}
        {step.tool ? (
          <div className="border border-white/10 bg-white/[0.03]">
            <button
              type="button"
              onClick={() => setOpen((value) => !value)}
              disabled={!expandable}
              aria-expanded={expandable ? open : undefined}
              className="group flex w-full items-center gap-2 px-2 py-1.5 text-left hover:bg-white/[0.04] disabled:cursor-default disabled:hover:bg-transparent"
            >
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[11.5px] font-medium text-zinc-200">{step.title}</span>
                <code className="block truncate text-[10.5px]" style={{ color }}>
                  {step.tool}
                </code>
              </span>
              {expandable && <Chevron open={open} />}
            </button>
            {expandable && open && <p className="border-t border-white/10 px-2 py-2 text-[11.5px] italic leading-relaxed text-zinc-500">{step.detail}</p>}
          </div>
        ) : (
          <>
            <button
              type="button"
              onClick={() => setOpen((value) => !value)}
              disabled={!expandable}
              aria-expanded={expandable ? open : undefined}
              className="group flex w-full items-start gap-1 break-words text-left text-[12px] font-medium leading-[17px] text-zinc-300 hover:text-zinc-100 disabled:cursor-default disabled:hover:text-zinc-300"
            >
              <span className="flex-1">{step.title}</span>
              {expandable && <Chevron open={open} />}
            </button>
            {step.detail && open && <p className="mt-1 text-[11.5px] italic leading-relaxed text-zinc-500">{step.detail}</p>}
          </>
        )}
      </div>
      {showChildren && (
        <ul className={`ml-5 ${hasNext ? "pb-3" : ""}`} role="group" aria-label={step.title}>
          {children.map((child, index) => (
            <Step key={index} step={child} depth={depth + 1} hasNext={index < children.length - 1} />
          ))}
        </ul>
      )}
    </li>
  );
}

/** A reply's chain of thought: a header row that opens a graph of its steps. `thinking` shows the live header. */
export function ThoughtTrace({ label, steps = [], thinking = false }: { label?: string; steps?: TraceStep[]; thinking?: boolean }) {
  const [open, setOpen] = useState(false);
  const expandable = steps.length > 0;
  return (
    <div className="mb-2 font-mono">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        disabled={!expandable}
        aria-expanded={expandable ? open : undefined}
        className="group flex w-full items-center gap-[9px] text-[11px] tracking-wide text-zinc-500 transition-colors hover:text-zinc-300 disabled:cursor-default disabled:hover:text-zinc-500"
      >
        <span className={`transition-opacity group-hover:opacity-100 ${open || thinking ? "opacity-100" : "opacity-50"}`}>
          <Dot color={LANES[0]} pulse={thinking} />
        </span>
        <span className={`min-w-0 flex-1 truncate text-left ${thinking ? "animate-pulse" : ""}`}>
          {thinking ? "Thinking" : `Thought${label ? ` · ${label}` : ""}`}
        </span>
        {expandable && <ChevronDown className={`size-3.5 shrink-0 transition-transform ${open ? "rotate-180" : ""}`} />}
      </button>
      {open && expandable && (
        <ul className="mt-0.5 pt-3" aria-label="ARC 1's reasoning">
          {steps.map((step, index) => (
            <Step key={index} step={step} depth={0} hasNext={index < steps.length - 1} first={index === 0} />
          ))}
        </ul>
      )}
    </div>
  );
}
