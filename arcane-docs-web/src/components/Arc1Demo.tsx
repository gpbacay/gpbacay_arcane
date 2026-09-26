"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ListboxSelect } from "@/components/ui/listbox-select";
import {
  ArrowLeftRight,
  Check,
  CloudSun,
  Copy,
  FileText,
  Home,
  LayoutGrid,
  MessageSquare,
  Play,
  RotateCcw,
  Tags,
  X,
} from "lucide-react";

/* ------------------------------------------------------------------ types */

type Health = {
  ready: boolean;
  trained?: boolean;
  status?: string;
  parameters?: number;
  architecture?: string;
  active_cycles?: number;
  binding_cycles?: number;
  layers?: number;
  error?: string | null;
};

type Decision = {
  tool: string;
  param: string;
  kind: "anchor" | "select" | "fire";
  present: boolean;
  value: unknown;
  p: number;
  p_present?: number;
  span?: [number, number];
  surface?: string;
  distribution?: Record<string, number>;
};

type Call = { name: string; arguments: Record<string, unknown> };

type Stats = {
  tokens: number;
  probes: number;
  cycles: number;
  forward_passes?: number;
  schema_encoded?: number;
  schema_cached?: number;
};

type Result = {
  reasoning?: string;
  function_calls?: Call[];
  results?: unknown[];
  confidence?: number | null;
  source?: string;
  record?: Record<string, unknown>;
  label?: string;
  distribution?: Record<string, number>;
  decisions?: { tools: Record<string, number>; arguments: Decision[] } | Decision[];
  latency_ms?: number;
  stats?: Stats;
};

type ToolDef = {
  name: string;
  description: string;
  parameters: Array<{
    name: string;
    type: string;
    description: string;
    required?: boolean;
    enum?: string[];
  }>;
};

type Scene = {
  id: string;
  label: string;
  description: string;
  icon: typeof Home;
  mode: "run" | "extract" | "classify";
  tools: ToolDef[];
  prompts: string[];
  extractSchema?: Record<string, { type: string; description: string }>;
  labels?: string[];
  task?: string;
  state: Array<{ key: string; label: string; value: string }>;
};

/* ------------------------------------------------------------------ scenes */

const TOOL_WEATHER: ToolDef = {
  name: "get_weather",
  description: "Get the current weather for a city.",
  parameters: [{ name: "city", type: "string", description: "City name", required: true }],
};

const TOOL_LIGHTS: ToolDef = {
  name: "set_lights",
  description: "Set light brightness in a room (0-100).",
  parameters: [
    { name: "room", type: "string", description: "Room name", required: true },
    { name: "level", type: "integer", description: "Brightness 0-100", required: true },
  ],
};

const TOOL_FX: ToolDef = {
  name: "convert_currency",
  description: "Convert an amount from one currency to another.",
  parameters: [
    { name: "amount", type: "number", description: "Amount to convert", required: true },
    { name: "from_currency", type: "string", description: "Source currency code, e.g. USD", required: true },
    { name: "to_currency", type: "string", description: "Target currency code, e.g. PHP", required: true },
  ],
};

const TOOL_MSG: ToolDef = {
  name: "send_message",
  description: "Send a short message to a contact.",
  parameters: [
    { name: "to", type: "string", description: "Contact name or handle", required: true },
    { name: "message", type: "string", description: "Message body", required: true },
  ],
};

const SCENES: Scene[] = [
  {
    id: "home",
    label: "Smart home",
    description: "Lights and weather, two tools to choose between",
    icon: Home,
    mode: "run",
    tools: [TOOL_LIGHTS, TOOL_WEATHER],
    prompts: ["dim the living room to 30", "set the kitchen lights to 100", "what's the weather in Manila?"],
    state: [
      { key: "living room", label: "Living room", value: "Lights at 70%" },
      { key: "kitchen", label: "Kitchen", value: "Lights at 100%" },
      { key: "bedroom", label: "Bedroom", value: "Lights at 40%" },
    ],
  },
  {
    id: "all",
    label: "All tools",
    description: "Four tools at once, including requests none of them fit",
    icon: LayoutGrid,
    mode: "run",
    tools: [TOOL_WEATHER, TOOL_LIGHTS, TOOL_FX, TOOL_MSG],
    prompts: [
      "convert 250 EUR to JPY",
      "message Maya that dinner is ready",
      "is it raining in Tokyo?",
      "tell me a joke",
    ],
    state: [
      { key: "__calls", label: "Tools offered", value: "4 tools, 8 arguments" },
      { key: "__last", label: "Last action", value: "None yet" },
    ],
  },
  {
    id: "currency",
    label: "Currency",
    description: "An amount and two currency codes from one sentence",
    icon: ArrowLeftRight,
    mode: "run",
    tools: [TOOL_FX],
    prompts: ["convert 100 USD to PHP", "exchange 50 EUR to GBP", "how much is 2500 JPY in USD"],
    state: [
      { key: "__last", label: "Last conversion", value: "None yet" },
      { key: "__rates", label: "Rates", value: "Demo rates, not live" },
    ],
  },
  {
    id: "weather",
    label: "Weather",
    description: "One tool, one city argument",
    icon: CloudSun,
    mode: "run",
    tools: [TOOL_WEATHER],
    prompts: ["what's the weather in Lagos?", "how's it looking in Tokyo", "weather for Paris"],
    state: [
      { key: "lagos", label: "Lagos", value: "Not checked" },
      { key: "tokyo", label: "Tokyo", value: "Not checked" },
      { key: "paris", label: "Paris", value: "Not checked" },
    ],
  },
  {
    id: "messaging",
    label: "Messaging",
    description: "A contact and a free-text message body",
    icon: MessageSquare,
    mode: "run",
    tools: [TOOL_MSG],
    prompts: [
      "message Maya that dinner is ready",
      "tell Alex that the meeting moved to 3pm",
      "text Sam saying on my way",
    ],
    state: [
      { key: "maya", label: "Maya", value: "No messages" },
      { key: "alex", label: "Alex", value: "No messages" },
      { key: "sam", label: "Sam", value: "No messages" },
    ],
  },
  {
    id: "extract",
    label: "Extract",
    description: "Pull a typed record out of messy text",
    icon: FileText,
    mode: "extract",
    tools: [],
    prompts: [
      "Invoice for Ada Okonkwo. City: Lagos. Amount: 1200.",
      "Hi, I'm Rafael Cruz, I work at Globe Telecom and you can email me at rafael@example.com",
    ],
    extractSchema: {
      name: { type: "string", description: "Person name" },
      city: { type: "string", description: "City" },
      amount: { type: "number", description: "Amount of money" },
      company: { type: "string", description: "Company name" },
      email: { type: "string", description: "Email address" },
    },
    state: [],
  },
  {
    id: "classify",
    label: "Classify",
    description: "Pick one label for a message, from labels you choose",
    icon: Tags,
    mode: "classify",
    tools: [],
    // None of these sentences appear in ARC 1's training data.
    prompts: [
      "the headphones sound amazing",
      "my order arrived broken and support was rude",
      "the store opens at nine",
      "this charger is useless",
    ],
    labels: ["positive", "negative", "neutral"],
    task: "Classify the sentiment of the message.",
    state: [],
  },
];

/** "billing: charges, refunds" -> ["billing", "charges, refunds"]. */
function splitLabel(entry: string): [string, string] {
  const i = entry.indexOf(":");
  return i < 0 ? [entry.trim(), ""] : [entry.slice(0, i).trim(), entry.slice(i + 1).trim()];
}

/* Argument colors cycle through the ARCANE palette. */
const ARG_COLORS = ["#B9DFE0", "#F294C0", "#9DE4FA", "#C785F2", "#E8D38A"];

/* ------------------------------------------------------------------ helpers */

function pct(p: number) {
  return `${Math.round(p * 100)}%`;
}

function formatValue(v: unknown) {
  return typeof v === "string" ? `"${v}"` : JSON.stringify(v);
}

function argumentDecisions(result: Result): Decision[] {
  const d = result.decisions;
  if (!d) return [];
  return Array.isArray(d) ? d : d.arguments || [];
}

function toolProbs(result: Result): Record<string, number> {
  const d = result.decisions;
  if (!d || Array.isArray(d)) return {};
  return d.tools || {};
}

/** Split ``text`` into plain and anchored segments (non-overlapping, in order). */
function segmentText(text: string, anchors: Array<{ span: [number, number]; color: string; label: string }>) {
  const sorted = [...anchors]
    .filter((a) => a.span[1] > a.span[0] && a.span[1] <= text.length)
    .sort((a, b) => a.span[0] - b.span[0]);
  const out: Array<{ text: string; color?: string; label?: string }> = [];
  let pos = 0;
  for (const a of sorted) {
    if (a.span[0] < pos) continue; // overlapping anchor: keep the first
    if (a.span[0] > pos) out.push({ text: text.slice(pos, a.span[0]) });
    out.push({ text: text.slice(a.span[0], a.span[1]), color: a.color, label: a.label });
    pos = a.span[1];
  }
  if (pos < text.length) out.push({ text: text.slice(pos) });
  return out;
}

/* ------------------------------------------------------------------ pieces */

function StatusDot({ health, checking }: { health: Health | null; checking: boolean }) {
  let text = "Offline";
  let color = "bg-red-400";
  if (checking && !health) {
    text = "Connecting";
    color = "bg-zinc-500";
  } else if (health?.ready) {
    text = health.trained ? "Model ready" : "Untrained weights";
    color = health.trained ? "bg-emerald-400" : "bg-amber-400";
  } else if (health?.status === "loading") {
    text = "Loading model";
    color = "bg-amber-400";
  }
  return (
    <span className="inline-flex items-center gap-2 text-xs text-zinc-400" role="status">
      <span className={`h-2 w-2 rounded-full ${color}`} aria-hidden />
      {text}
    </span>
  );
}

function BindingView({ text, anchors }: {
  text: string;
  anchors: Array<{ span: [number, number]; color: string; label: string }>;
}) {
  // Words are laid out as wrapping flex items. An anchored span is a column
  // (word over its argument name) whose width is the wider of the two, so a
  // long label pushes its neighbours aside instead of overlapping them.
  // Pieces not separated by whitespace (e.g. "Manila" + "?") stay in one cluster.
  const clusters: Array<Array<{ text: string; color?: string; label?: string }>> = [];
  let startNew = true;
  for (const seg of segmentText(text, anchors)) {
    if (seg.color) {
      if (startNew || !clusters.length) clusters.push([]);
      clusters[clusters.length - 1].push(seg);
      startNew = false;
      continue;
    }
    for (const piece of seg.text.split(/(\s+)/)) {
      if (!piece) continue;
      if (/^\s+$/.test(piece)) {
        startNew = true;
        continue;
      }
      if (startNew || !clusters.length) clusters.push([]);
      clusters[clusters.length - 1].push({ text: piece });
      startNew = false;
    }
  }
  return (
    <p
      className="flex flex-wrap items-start gap-x-[0.3em] gap-y-2 text-lg leading-snug text-zinc-300 sm:text-xl"
      aria-label="Utterance with anchored arguments"
    >
      {clusters.map((cluster, i) => (
        <span key={i} className="inline-flex items-start">
          {cluster.map((s, j) =>
            s.color ? (
              <span key={j} className="inline-flex flex-col items-center">
                <mark
                  className="px-1 text-zinc-50"
                  style={{ boxShadow: `inset 0 -2px 0 ${s.color}`, background: `${s.color}24` }}
                >
                  {s.text}
                </mark>
                <span className="mt-1 whitespace-nowrap px-1 text-[11px] font-medium leading-none" style={{ color: s.color }}>
                  {s.label}
                </span>
              </span>
            ) : (
              <span key={j}>{s.text}</span>
            )
          )}
        </span>
      ))}
    </p>
  );
}

function FireBars({ probs, fired }: { probs: Record<string, number>; fired: Set<string> }) {
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return null;
  return (
    <div className="space-y-2.5">
      {entries.map(([name, p]) => {
        const on = fired.has(name);
        return (
          <div key={name} className="grid grid-cols-[minmax(0,9.5rem)_1fr_3rem] items-center gap-3 text-xs">
            <code className={`truncate ${on ? "text-zinc-100" : "text-zinc-500"}`}>{name}</code>
            <div className="relative h-1.5 bg-zinc-900" aria-hidden>
              <div
                className="absolute inset-y-0 left-0 transition-[width] duration-500 motion-reduce:transition-none"
                style={{ width: `${Math.max(2, p * 100)}%`, background: on ? "#C785F2" : "#3f3f46" }}
              />
              <div className="absolute inset-y-[-3px] left-1/2 w-px bg-zinc-600" title="Fire threshold" />
            </div>
            <span className={`text-right tabular-nums ${on ? "text-[#C785F2]" : "text-zinc-500"}`}>{pct(p)}</span>
          </div>
        );
      })}
    </div>
  );
}

function LabelEditor({ labels, onChange }: { labels: string[]; onChange: (labels: string[]) => void }) {
  const [draft, setDraft] = useState("");
  const add = () => {
    // A comma separates labels unless a colon starts a hint ("billing: charges, refunds").
    const parts = draft.includes(":") ? [draft.trim()] : draft.split(",").map((x) => x.trim()).filter(Boolean);
    const next = [...labels];
    for (const p of parts) {
      const name = splitLabel(p)[0].toLowerCase();
      if (name && !next.some((l) => splitLabel(l)[0].toLowerCase() === name)) next.push(p);
    }
    onChange(next.slice(0, 16));
    setDraft("");
  };
  return (
    <div className="mt-4">
      <label htmlFor="arc1-label-input" className="mb-2 block text-sm font-medium text-zinc-200">
        Labels
      </label>
      <div className="flex flex-wrap items-center gap-1.5 border border-zinc-800 bg-black p-2 focus-within:border-[#835BD9]">
        {labels.map((l) => (
          <span key={l} className="inline-flex max-w-full items-center gap-1 bg-zinc-900 py-1 pl-2.5 pr-1 text-xs text-zinc-200">
            <span className="truncate">
              {splitLabel(l)[0]}
              {splitLabel(l)[1] && <span className="text-zinc-500">: {splitLabel(l)[1]}</span>}
            </span>
            <button
              type="button"
              onClick={() => onChange(labels.filter((x) => x !== l))}
              aria-label={`Remove label ${splitLabel(l)[0]}`}
              className="p-0.5 text-zinc-500 hover:text-zinc-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-[#C785F2]"
            >
              <X className="h-3.5 w-3.5" aria-hidden />
            </button>
          </span>
        ))}
        <input
          id="arc1-label-input"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || (e.key === "," && !draft.includes(":"))) {
              e.preventDefault();
              add();
            } else if (e.key === "Backspace" && !draft && labels.length) {
              onChange(labels.slice(0, -1));
            }
          }}
          onBlur={() => draft.trim() && add()}
          placeholder={labels.length ? "Add a label" : "Type a label and press Enter"}
          className="min-w-[8rem] flex-1 bg-transparent px-1 py-1 text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
        />
      </div>
      <p className="mt-1.5 text-xs text-zinc-500">
        Press Enter to add a label. Add a hint after a colon, like <span className="text-zinc-300">billing: charges, refunds</span>.
        Labels work best when they relate to the words people actually use.
      </p>
    </div>
  );
}

function CopyButton({ text, label }: { text: string; label: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => {
          setDone(true);
          setTimeout(() => setDone(false), 1500);
        });
      }}
      className="inline-flex items-center gap-1.5 px-2 py-1 text-[11px] text-zinc-500 hover:text-zinc-200 focus-visible:outline focus-visible:outline-1 focus-visible:outline-[#C785F2]"
      aria-label={label}
    >
      {done ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      {done ? "Copied" : "Copy"}
    </button>
  );
}

/* ------------------------------------------------------------------ demo */

export function Arc1Demo() {
  const [health, setHealth] = useState<Health | null>(null);
  const [checking, setChecking] = useState(true);
  const [sceneId, setSceneId] = useState(SCENES[0].id);
  const [input, setInput] = useState(SCENES[0].prompts[0]);
  const [cycles, setCycles] = useState<number | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [ranText, setRanText] = useState("");
  const [state, setState] = useState<Record<string, string>>({});
  const [labels, setLabels] = useState<string[]>([]);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  const scene = useMemo(() => SCENES.find((s) => s.id === sceneId) || SCENES[0], [sceneId]);

  // Grow the input with its text so long requests are never cut off.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight + 2}px`;
  }, [input, sceneId]);
  const maxCycles = health?.binding_cycles ?? 3;
  const activeCycles = cycles ?? health?.active_cycles ?? maxCycles;

  const refreshHealth = useCallback(async () => {
    try {
      const res = await fetch("/api/arc1-health", { cache: "no-store" });
      setHealth((await res.json()) as Health);
    } catch {
      setHealth({ ready: false, status: "offline", error: "Could not reach the ARC 1 API." });
    } finally {
      setChecking(false);
    }
  }, []);

  // Poll quickly until the model is ready, then slowly.
  useEffect(() => {
    void refreshHealth();
    const id = setInterval(refreshHealth, health?.ready ? 20000 : 4000);
    return () => clearInterval(id);
  }, [refreshHealth, health?.ready]);

  const selectScene = useCallback((next: Scene) => {
    setSceneId(next.id);
    setInput(next.prompts[0] || "");
    setLabels(next.labels || []);
    setResult(null);
    setError(null);
    setState({});
  }, []);

  const applyEffects = useCallback((data: Result) => {
    const next: Record<string, string> = {};
    (data.function_calls || []).forEach((call, i) => {
      const r = (data.results?.[i] || {}) as Record<string, unknown>;
      if (call.name === "set_lights") {
        const room = String(call.arguments.room || "").toLowerCase();
        if (room) next[room] = `Lights at ${call.arguments.level}%`;
        next.__last = `Set ${room} lights to ${call.arguments.level}%`;
      }
      if (call.name === "convert_currency" && r.converted != null) {
        next.__last = `${r.amount} ${r.from} = ${r.converted} ${r.to}`;
      }
      if (call.name === "get_weather" && r.city) {
        next[String(r.city).toLowerCase()] = `${r.temp_c}°C, ${r.sky}`;
        next.__last = `Checked weather in ${r.city}`;
      }
      if (call.name === "send_message") {
        const to = String(call.arguments.to || "").toLowerCase();
        next[to] = `Queued: "${call.arguments.message}"`;
        next.__last = `Message queued for ${call.arguments.to}`;
      }
    });
    if (Object.keys(next).length) setState((prev) => ({ ...prev, ...next }));
  }, []);

  const run = useCallback(async () => {
    const text = input.trim();
    if (sending || !text) return;
    if (!health?.ready) {
      setError(health?.error || "ARC 1 is not running yet. Start it with npm run dev:with-arc1.");
      return;
    }
    if (scene.mode === "classify" && labels.length < 2) {
      setError("Add at least two labels to choose between.");
      return;
    }
    setSending(true);
    setError(null);
    try {
      const isRun = scene.mode === "run";
      const [url, body] =
        scene.mode === "run"
          ? ["/api/arc1-run", { prompt: text, tools: scene.tools, execute: true, cycles: activeCycles }]
          : scene.mode === "extract"
            ? ["/api/arc1-extract", { text, schema: scene.extractSchema, cycles: activeCycles }]
            : [
                "/api/arc1-classify",
                {
                  text,
                  labels: labels.map((l) => splitLabel(l)[0]),
                  descriptions: Object.fromEntries(labels.map(splitLabel).filter(([, d]) => d)),
                  task: scene.task,
                  cycles: activeCycles,
                },
              ];
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.error === "string" ? data.error : res.statusText);
      setResult(data as Result);
      setRanText(text);
      if (isRun) applyEffects(data as Result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "The request failed.");
    } finally {
      setSending(false);
    }
  }, [input, sending, health, scene, labels, activeCycles, applyEffects]);

  const decisions = useMemo(() => (result ? argumentDecisions(result) : []), [result]);
  const calls = result?.function_calls || [];
  const fired = new Set(calls.map((c) => c.name));
  const colorOf = useMemo(() => {
    const map = new Map<string, string>();
    decisions.forEach((d) => {
      const key = `${d.tool}.${d.param}`;
      if (!map.has(key)) map.set(key, ARG_COLORS[map.size % ARG_COLORS.length]);
    });
    return map;
  }, [decisions]);
  const shown = decisions.filter(
    (d) => d.present && (scene.mode === "extract" || fired.has(d.tool))
  );
  const anchors = shown
    .filter((d) => d.kind === "anchor" && d.span)
    .map((d) => ({ span: d.span as [number, number], color: colorOf.get(`${d.tool}.${d.param}`)!, label: d.param }));

  const offline = !checking && !health?.ready;
  const callJson = calls.length ? JSON.stringify(calls, null, 2) : "";

  return (
    <section
      aria-label="ARC 1 sandbox"
      className="not-prose border border-zinc-800 bg-zinc-950"
    >
      {/* Top bar: scenes + status */}
      <div className="flex flex-col gap-3 border-b border-zinc-800 px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-5">
        <div className="flex min-w-0 flex-col gap-1.5 sm:flex-row sm:items-center sm:gap-3">
          <ListboxSelect
            label="Example scene"
            value={sceneId}
            options={SCENES.map((s) => ({ value: s.id, label: s.label, description: s.description, icon: s.icon }))}
            onChange={(id) => {
              const next = SCENES.find((s) => s.id === id);
              if (next) selectScene(next);
            }}
            className="w-full sm:w-56"
          />
          <p className="min-w-0 text-xs leading-snug text-zinc-500">{scene.description}</p>
        </div>
        <div className="flex shrink-0 items-center gap-4">
          <StatusDot health={health} checking={checking} />
          {health?.parameters != null && (
            <span className="hidden text-xs tabular-nums text-zinc-500 sm:inline">
              {(health.parameters / 1e6).toFixed(2)}M parameters
            </span>
          )}
        </div>
      </div>

      {offline && (
        <div className="flex flex-col gap-2 border-b border-zinc-800 bg-[#835BD9]/10 px-4 py-3 text-sm text-zinc-300 sm:flex-row sm:items-center sm:justify-between sm:px-5">
          <span>
            The sandbox needs the local ARC 1 server. From <code className="text-zinc-100">arcane-docs-web</code>, run{" "}
            <code className="text-zinc-100">npm run dev:with-arc1</code>.
          </span>
          <CopyButton text="npm run dev:with-arc1" label="Copy start command" />
        </div>
      )}

      <div className="grid lg:grid-cols-[minmax(0,1fr)_16rem]">
        {/* Main column */}
        <div className="min-w-0 border-zinc-800 lg:border-r">
          {/* Input */}
          <div className="p-4 sm:p-5">
            <label htmlFor="arc1-input" className="mb-2 block text-sm font-medium text-zinc-200">
              {scene.mode === "extract" ? "Text to extract from" : scene.mode === "classify" ? "Text to classify" : "Ask for something"}
            </label>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start">
              <textarea
                id="arc1-input"
                ref={inputRef}
                value={input}
                rows={scene.mode === "run" ? 1 : 2}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void run();
                  }
                }}
                placeholder={
                  scene.mode === "extract"
                    ? "Paste a message, invoice, or signature"
                    : scene.mode === "classify"
                      ? "Paste a ticket, review, or message"
                      : "dim the kitchen to 20"
                }
                className="max-h-48 min-h-[2.75rem] min-w-0 flex-1 resize-none overflow-y-auto border border-zinc-800 bg-black px-3.5 py-2.5 text-sm text-zinc-100 outline-none placeholder:text-zinc-600 focus:border-[#835BD9]"
              />
              <button
                type="button"
                onClick={() => void run()}
                disabled={sending || !input.trim()}
                className="inline-flex h-[2.75rem] items-center justify-center gap-2 bg-white px-5 text-sm font-medium text-black transition-colors hover:bg-zinc-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#C785F2] disabled:opacity-40"
              >
                <Play className="h-3.5 w-3.5 fill-current" aria-hidden />
                {sending ? "Running" : scene.mode === "extract" ? "Extract" : scene.mode === "classify" ? "Classify" : "Run"}
              </button>
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {scene.prompts.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => {
                    setInput(p);
                    inputRef.current?.focus();
                  }}
                  className={`max-w-full truncate border px-2.5 py-1 text-left text-xs transition-colors focus-visible:outline focus-visible:outline-1 focus-visible:outline-[#C785F2] ${
                    input === p
                      ? "border-[#835BD9] bg-[#835BD9]/15 text-zinc-100"
                      : "border-zinc-800 text-zinc-500 hover:border-zinc-600 hover:text-zinc-200"
                  }`}
                >
                  {p}
                </button>
              ))}
            </div>
            {scene.mode === "classify" && <LabelEditor labels={labels} onChange={setLabels} />}
            {error && (
              <p role="alert" className="mt-3 border border-red-900/60 bg-red-950/30 px-3 py-2 text-sm text-red-200">
                {error}
              </p>
            )}
          </div>

          {/* Binding view */}
          <div className="border-t border-zinc-800 p-4 sm:p-5" aria-live="polite">
            {!result ? (
              <div className="py-6 text-sm text-zinc-500">
                {scene.mode === "classify"
                  ? "Classify a message to see the probability ARC 1 gives each of your labels."
                  : "Run a request to see which words ARC 1 anchored each argument to, and how strongly each tool fired."}
              </div>
            ) : (
              <div className="space-y-6">
                <div>
                  <h3 className="mb-2 text-sm font-medium text-zinc-400">What ARC 1 read</h3>
                  <BindingView text={ranText} anchors={anchors} />
                </div>

                {scene.mode === "classify" && result.distribution && (
                  <div>
                    <h3 className="mb-1 text-sm font-medium text-zinc-400">Label</h3>
                    <p className="mb-4 text-2xl font-semibold text-zinc-50">{result.label}</p>
                    <h3 className="mb-3 text-sm font-medium text-zinc-400">Label probability</h3>
                    <FireBars probs={result.distribution} fired={new Set(result.label ? [result.label] : [])} />
                  </div>
                )}

                {scene.mode === "run" && (
                  <div>
                    <h3 className="mb-3 text-sm font-medium text-zinc-400">Tool fire probability</h3>
                    <FireBars probs={toolProbs(result)} fired={fired} />
                  </div>
                )}

                {shown.length > 0 && (
                  <div>
                    <h3 className="mb-2 text-sm font-medium text-zinc-400">
                      {scene.mode === "extract" ? "Fields" : "Arguments"}
                    </h3>
                    <ul className="divide-y divide-zinc-900 border-y border-zinc-900">
                      {shown.map((d) => (
                        <li key={`${d.tool}.${d.param}`} className="flex items-baseline gap-3 py-2 text-sm">
                          <span
                            className="h-2 w-2 shrink-0 translate-y-[-1px]"
                            style={{ background: colorOf.get(`${d.tool}.${d.param}`) }}
                            aria-hidden
                          />
                          <code className="shrink-0 text-zinc-400">{d.param}</code>
                          <code className="min-w-0 flex-1 truncate text-zinc-100">{formatValue(d.value)}</code>
                          <span className="hidden shrink-0 text-xs text-zinc-500 sm:inline">
                            {d.kind === "anchor" ? "copied" : d.kind === "select" ? "selected" : "decided"}
                          </span>
                          <span className="w-10 shrink-0 text-right text-xs tabular-nums text-zinc-400">{pct(d.p)}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {scene.mode === "run" && calls.length === 0 && (
                  <p className="text-sm text-zinc-400">
                    No tool fired, so nothing was called. ARC 1 stays silent when a request doesn&apos;t match any
                    offered tool.
                  </p>
                )}

                {callJson && (
                  <div className="grid gap-4">
                    <div className="min-w-0">
                      <div className="mb-1 flex items-center justify-between">
                        <h3 className="text-sm font-medium text-zinc-400">Function calls</h3>
                        <CopyButton text={callJson} label="Copy function calls" />
                      </div>
                      <pre className="max-h-64 overflow-auto border border-zinc-800 bg-black p-3 text-xs leading-relaxed text-zinc-300">
                        {callJson}
                      </pre>
                    </div>
                    {result.results && result.results.length > 0 && (
                      <div className="min-w-0">
                        <h3 className="mb-1 py-1 text-sm font-medium text-zinc-400">Executed results</h3>
                        <pre className="max-h-64 overflow-auto border border-emerald-900/50 bg-emerald-950/20 p-3 text-xs leading-relaxed text-emerald-100/90">
                          {JSON.stringify(result.results, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}

                {result.source === "heuristic" && (
                  <p className="text-xs text-amber-300/80">
                    These weights are untrained, so a keyword fallback produced this call. Train ARC 1 for real decisions.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Side column: controls, measurements, state */}
        <aside className="border-t border-zinc-800 p-4 sm:p-5 lg:border-t-0">
          <fieldset>
            <legend className="mb-2 text-sm font-medium text-zinc-200">Binding cycles</legend>
            <div className="grid grid-flow-col gap-px bg-zinc-800" role="radiogroup">
              {Array.from({ length: maxCycles }, (_, i) => i + 1).map((c) => (
                <button
                  key={c}
                  type="button"
                  role="radio"
                  aria-checked={activeCycles === c}
                  onClick={() => setCycles(c)}
                  className={`py-1.5 text-sm tabular-nums transition-colors focus-visible:outline focus-visible:outline-1 focus-visible:outline-[#C785F2] ${
                    activeCycles === c ? "bg-[#835BD9] text-white" : "bg-zinc-950 text-zinc-400 hover:text-zinc-100"
                  }`}
                >
                  {c}
                </button>
              ))}
            </div>
            <p className="mt-2 text-xs leading-relaxed text-zinc-500">
              Fewer cycles answer faster. More cycles give each probe time to settle on the right words.
            </p>
          </fieldset>

          {result?.stats && (
            <dl className="mt-6 grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
              <div>
                <dt className="text-xs text-zinc-500">Latency</dt>
                <dd className="tabular-nums text-zinc-100">{result.latency_ms?.toFixed(1)} ms</dd>
              </div>
              <div>
                <dt className="text-xs text-zinc-500">Confidence</dt>
                <dd className="tabular-nums text-zinc-100">
                  {typeof result.confidence === "number" ? pct(result.confidence) : "n/a"}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-zinc-500">Tokens read</dt>
                <dd className="tabular-nums text-zinc-100">{result.stats.tokens}</dd>
              </div>
              <div>
                <dt className="text-xs text-zinc-500">Forward passes</dt>
                <dd className="tabular-nums text-zinc-100">{result.stats.forward_passes ?? 1}</dd>
              </div>
              <div>
                <dt className="text-xs text-zinc-500">Probes bound</dt>
                <dd className="tabular-nums text-zinc-100">{result.stats.probes}</dd>
              </div>
              <div className="col-span-2">
                <dt className="text-xs text-zinc-500">Schema engrams</dt>
                <dd className="text-zinc-100">
                  {result.stats.schema_cached ?? 0} cached, {result.stats.schema_encoded ?? 0} newly encoded
                </dd>
              </div>
            </dl>
          )}

          {scene.state.length > 0 && (
            <div className="mt-6">
              <h3 className="mb-2 text-sm font-medium text-zinc-200">Environment</h3>
              <ul className="space-y-2">
                {scene.state.map((row) => {
                  const live = state[row.key];
                  return (
                    <li key={row.key} className="border-l-2 pl-3 text-sm" style={{ borderColor: live ? "#C785F2" : "#27272a" }}>
                      <div className="text-zinc-300">{row.label}</div>
                      <div className={`text-xs ${live ? "text-zinc-100" : "text-zinc-500"}`}>{live || row.value}</div>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}

          {scene.mode === "run" && (
            <details className="group mt-6 text-sm">
              <summary className="cursor-pointer text-zinc-400 hover:text-zinc-200 focus-visible:outline focus-visible:outline-1 focus-visible:outline-[#C785F2]">
                Tool schemas sent
              </summary>
              <pre className="mt-2 max-h-72 overflow-auto border border-zinc-800 bg-black p-2 text-[11px] leading-relaxed text-zinc-400">
                {JSON.stringify(scene.tools, null, 2)}
              </pre>
            </details>
          )}

          <button
            type="button"
            onClick={() => selectScene(scene)}
            className="mt-6 inline-flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-200 focus-visible:outline focus-visible:outline-1 focus-visible:outline-[#C785F2]"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden />
            Reset scene
          </button>
        </aside>
      </div>
    </section>
  );
}
