import type { ReactNode } from "react";
import { Repeat } from "lucide-react";

/*
 * ARC 1 architecture, drawn in HTML so it reflows: two lanes side by side on
 * wide screens (request | tools), one column on phones, text never scales down.
 */

const REQUEST = "#B9DFE0";
const TOOLS = "#F294C0";
const BIND = "#C785F2";

function Arrow({ color = "#52525b", className = "" }: { color?: string; className?: string }) {
  return (
    <svg viewBox="0 0 12 24" className={`mx-auto block h-6 w-3 ${className}`} aria-hidden>
      <path d="M6 0v19" stroke={color} strokeWidth="1.5" />
      <path d="M1.5 15 6 21l4.5-6" fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

function Node({ title, children, accent, tag, grow = false }: {
  title: string; children?: ReactNode; accent?: string; tag?: string; grow?: boolean;
}) {
  return (
    <div className={`border border-zinc-800 bg-zinc-950 px-4 py-3 ${grow ? "flex-1" : ""}`} style={accent ? { borderLeft: `2px solid ${accent}` } : undefined}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h4 className="text-sm font-semibold text-zinc-100">{title}</h4>
        {tag && <span className="text-xs text-zinc-500">{tag}</span>}
      </div>
      {children && <div className="mt-1.5 text-[13px] leading-relaxed text-zinc-400">{children}</div>}
    </div>
  );
}

function Code({ children }: { children: ReactNode }) {
  return <code className="text-zinc-200">{children}</code>;
}

function Lane({ title, color, children }: { title: string; color: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col">
      <div className="mb-2 flex items-center gap-2 text-sm font-medium" style={{ color }}>
        <span className="h-2 w-2" style={{ background: color }} aria-hidden />
        {title}
      </div>
      {children}
    </div>
  );
}

const READOUTS = [
  { name: "fire", example: "convert_currency fires, 99%" },
  { name: "anchor", example: 'amount = "100", copied' },
  { name: "select", example: 'label = "billing"' },
  { name: "embedding", example: "128-d intent vector" },
];

export function Arc1ArchitectureDiagram() {
  return (
    <figure className="not-prose my-6">
      <div className="border border-zinc-800 bg-black p-4 sm:p-6">
        <div className="grid gap-8 md:grid-cols-2 md:gap-6">
          <Lane title="Your request, read once" color={REQUEST}>
            <Node title="Utterance" accent={REQUEST}>
              <span className="text-zinc-200">&ldquo;convert 100 USD to PHP&rdquo;</span>
              <br />
              Byte-level BPE tokens, right-padded and masked.
            </Node>
            <Arrow />
            <Node title="Perception blocks" tag="× 3, bidirectional" accent={REQUEST}>
              <ol className="mt-1 space-y-1">
                <li>
                  <Code>FieldAttention</Code>: every token sees every other real token
                </li>
                <li>
                  <Code>ResonantChannelMixer</Code>: low-rank mix, GSER gate
                </li>
                <li>
                  <Code>ConceptEngram</Code>: hashed n-gram memory, first block
                </li>
                <li>
                  <Code>FieldResonance</Code>: pull toward the sentence prototype, spike reset
                </li>
              </ol>
            </Node>
            <Arrow />
            <Node title="Utterance field U" tag="T × D" accent={REQUEST} grow>
              One vector per token. Its keys and values are computed once and shared by every probe.
            </Node>
          </Lane>

          <Lane title="Your tools, remembered" color={TOOLS}>
            <Node title="Schema texts" accent={TOOLS}>
              One short line per tool, argument, enum option, or classification label, such as{" "}
              <span className="text-zinc-200">&ldquo;amount (number). Amount to convert&rdquo;</span>.
            </Node>
            <Arrow />
            <Node title="Same perception blocks" tag="shared weights" accent={TOOLS}>
              Attention-pooled into one schema engram per text.
            </Node>
            <Arrow />
            <Node title="SchemaMemory" tag="cache" accent={TOOLS}>
              New texts are perceived in the same batch as the request, then kept, so a request is always one pass.
            </Node>
            <Arrow />
            <Node title="Probe" accent={TOOLS} grow>
              Engram + context engram (the tool, for an argument) + role.
            </Node>
          </Lane>
        </div>

        {/* Both lanes flow into binding. */}
        <div className="grid md:grid-cols-2 md:gap-6" aria-hidden>
          <Arrow color={REQUEST} className="hidden md:block" />
          <Arrow color={TOOLS} />
        </div>

        <div className="border px-4 py-4 sm:px-5" style={{ borderColor: BIND, background: "rgba(199,133,242,0.07)" }}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h4 className="text-sm font-semibold text-zinc-50">
              <Code>ResonantBinding</Code>
            </h4>
            <span className="inline-flex items-center gap-1.5 text-xs" style={{ color: BIND }}>
              <Repeat className="h-3.5 w-3.5" aria-hidden />
              repeats for K cycles (1 to 3)
            </span>
          </div>
          <p className="mt-1.5 text-[13px] leading-relaxed text-zinc-400">
            Every probe listens to the utterance field U in parallel. Each cycle uses the same weights:
          </p>
          <ol className="mt-3 grid gap-px bg-zinc-800 text-[13px] sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["Attend", "find the tokens the probe resonates with"],
              ["Hear", "read what those tokens say"],
              ["Gate", "a GSER spike gate decides how much to take in"],
              ["Settle", "integrate, then a small channel mix"],
            ].map(([t, d], i) => (
              <li key={t} className="bg-zinc-950 px-3 py-2.5">
                <span className="tabular-nums" style={{ color: BIND }}>
                  {i + 1}
                </span>{" "}
                <span className="font-medium text-zinc-100">{t}</span>
                <span className="block text-zinc-500">{d}</span>
              </li>
            ))}
          </ol>
        </div>

        <Arrow color={BIND} />

        <div className="grid grid-cols-2 gap-px bg-zinc-800 lg:grid-cols-4">
          {READOUTS.map((r) => (
            <div key={r.name} className="min-w-0 bg-zinc-950 px-3 py-3">
              <Code>{r.name}</Code>
              <div className="mt-1 text-xs leading-snug text-zinc-500">{r.example}</div>
            </div>
          ))}
        </div>
      </div>
      <figcaption className="mt-3 max-w-2xl text-sm leading-relaxed text-zinc-500">
        One request is one forward pass: the sentence and any new schema texts are perceived together, engrams
        become probes, and every probe binds at the same time. The embedding readout pools the utterance field
        directly.
      </figcaption>
    </figure>
  );
}
