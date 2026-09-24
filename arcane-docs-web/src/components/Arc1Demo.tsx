"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowLeftRight,
  CloudSun,
  FileText,
  Home,
  MessageSquare,
  Play,
  RotateCcw,
  Sparkles,
} from "lucide-react";

type Health = {
  ready: boolean;
  trained?: boolean;
  preset?: string;
  status?: string;
  parameters?: number;
  active_depth?: number;
  error?: string | null;
  model?: string;
};

type RunResult = {
  reasoning?: string;
  function_calls?: Array<{ name: string; arguments: Record<string, unknown> }>;
  results?: unknown[];
  confidence?: number;
  source?: string;
  raw?: string;
  record?: Record<string, unknown>;
};

type ToolDef = {
  name: string;
  description: string;
  parameters: Array<{
    name: string;
    type: string;
    description: string;
    required?: boolean;
  }>;
};

type ExampleScene = {
  id: string;
  label: string;
  blurb: string;
  icon: "home" | "weather" | "fx" | "extract" | "message";
  mode: "run" | "extract";
  tools: ToolDef[];
  prompts: string[];
  defaultPrompt: string;
  extractText?: string;
  extractSchema?: Record<string, { type: string; description: string }>;
  env: Array<{ label: string; value: string; hint?: string }>;
};

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
    { name: "from_currency", type: "string", description: "Source currency code", required: true },
    { name: "to_currency", type: "string", description: "Target currency code", required: true },
  ],
};

const TOOL_MSG: ToolDef = {
  name: "send_message",
  description: "Send a short message to a contact.",
  parameters: [
    { name: "to", type: "string", description: "Contact name", required: true },
    { name: "message", type: "string", description: "Message body", required: true },
  ],
};

const SCENES: ExampleScene[] = [
  {
    id: "smart-home",
    label: "Smart home",
    blurb: "Run the house with natural language.",
    icon: "home",
    mode: "run",
    tools: [TOOL_LIGHTS, TOOL_WEATHER],
    prompts: [
      "dim the living room to 30",
      "set kitchen lights to 100",
      "what's the weather in Manila?",
    ],
    defaultPrompt: "dim the living room to 30",
    env: [
      { label: "Living room", value: "lights 70%", hint: "ready" },
      { label: "Kitchen", value: "lights 100%", hint: "ready" },
      { label: "Bedroom", value: "lights 40%", hint: "idle" },
    ],
  },
  {
    id: "currency",
    label: "Currency",
    blurb: "Convert amounts with a single tool call.",
    icon: "fx",
    mode: "run",
    tools: [TOOL_FX],
    prompts: [
      "convert 100 USD to PHP",
      "exchange 50 EUR to GBP",
      "convert 2500 JPY to USD",
    ],
    defaultPrompt: "convert 100 USD to PHP",
    env: [
      { label: "USD", value: "1.00 base", hint: "demo rates" },
      { label: "PHP", value: "56.5 / USD", hint: "demo rates" },
      { label: "EUR", value: "0.92 / USD", hint: "demo rates" },
      { label: "GBP", value: "0.79 / USD", hint: "demo rates" },
    ],
  },
  {
    id: "weather",
    label: "Weather",
    blurb: "Ask for conditions in any city.",
    icon: "weather",
    mode: "run",
    tools: [TOOL_WEATHER],
    prompts: [
      "what's the weather in Lagos?",
      "how's it in Tokyo?",
      "weather for Paris",
    ],
    defaultPrompt: "what's the weather in Lagos?",
    env: [
      { label: "Lagos", value: "27°C clear", hint: "cached" },
      { label: "Tokyo", value: "18°C clear", hint: "cached" },
      { label: "Paris", value: "14°C clear", hint: "cached" },
    ],
  },
  {
    id: "messaging",
    label: "Messaging",
    blurb: "Route a short note to a contact.",
    icon: "message",
    mode: "run",
    tools: [TOOL_MSG],
    prompts: [
      "message Maya that dinner is ready",
      "tell Alex that the meeting slipped to 3pm",
      "ping Sam saying on my way",
    ],
    defaultPrompt: "message Maya that dinner is ready",
    env: [
      { label: "Maya", value: "online", hint: "contact" },
      { label: "Alex", value: "away", hint: "contact" },
      { label: "Sam", value: "online", hint: "contact" },
    ],
  },
  {
    id: "extract",
    label: "Extract",
    blurb: "Pull a typed record from messy text.",
    icon: "extract",
    mode: "extract",
    tools: [],
    prompts: [],
    defaultPrompt: "",
    extractText: "Invoice for Ada Okonkwo. City: Lagos. Amount: 1200.",
    extractSchema: {
      name: { type: "string", description: "Person name" },
      city: { type: "string", description: "City" },
      amount: { type: "string", description: "Amount" },
    },
    env: [
      { label: "Schema", value: "name · city · amount", hint: "fields" },
      { label: "Grounding", value: "spans from passage", hint: "on" },
    ],
  },
];

function statusLabel(health: Health | null, loading: boolean) {
  if (loading && !health) return { text: "Checking…", color: "bg-zinc-500" };
  if (!health) return { text: "Offline", color: "bg-red-500" };
  if (health.ready)
    return {
      text: health.trained ? "Model ready" : "Untrained model",
      color: health.trained ? "bg-emerald-400" : "bg-amber-400",
    };
  if (health.status === "loading") return { text: "Loading model…", color: "bg-amber-400" };
  return { text: "Offline", color: "bg-red-500" };
}

function SceneIcon({ kind }: { kind: ExampleScene["icon"] }) {
  const cls = "h-4 w-4";
  if (kind === "home") return <Home className={cls} />;
  if (kind === "weather") return <CloudSun className={cls} />;
  if (kind === "fx") return <ArrowLeftRight className={cls} />;
  if (kind === "message") return <MessageSquare className={cls} />;
  return <FileText className={cls} />;
}

function ResultView({ result }: { result: RunResult }) {
  const calls = result.function_calls || [];
  const conf = typeof result.confidence === "number" ? result.confidence : null;

  return (
    <div className="space-y-4">
      {conf != null && (
        <div>
          <div className="mb-1.5 flex items-center justify-between text-[11px] text-zinc-500">
            <span>Confidence</span>
            <span className="tabular-nums text-zinc-300">{(conf * 100).toFixed(0)}%</span>
          </div>
          <div className="h-1 overflow-hidden rounded-full bg-zinc-900">
            <div
              className="h-full rounded-full bg-[#C785F2] transition-all duration-500"
              style={{ width: `${Math.max(4, Math.min(100, conf * 100))}%` }}
            />
          </div>
        </div>
      )}

      {result.reasoning ? (
        <p className="text-sm leading-relaxed text-zinc-400">{result.reasoning}</p>
      ) : null}

      {calls.length > 0 ? (
        <div className="space-y-2">
          {calls.map((call, i) => (
            <div
              key={`${call.name}-${i}`}
              className="rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2.5"
            >
              <div className="mb-1 flex items-center gap-2">
                <Sparkles className="h-3.5 w-3.5 text-[#C785F2]" />
                <code className="text-sm font-medium text-zinc-100">{call.name}</code>
              </div>
              <pre className="overflow-x-auto text-[11px] leading-relaxed text-zinc-400">
                {JSON.stringify(call.arguments, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      ) : result.record && Object.keys(result.record).length > 0 ? (
        <div className="rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2.5">
          <div className="mb-1 text-[11px] text-zinc-500">Record</div>
          <pre className="overflow-x-auto text-[11px] leading-relaxed text-zinc-300">
            {JSON.stringify(result.record, null, 2)}
          </pre>
        </div>
      ) : (
        <p className="text-sm text-zinc-600">No tool call — request was refused or out of scope.</p>
      )}

      {result.results && result.results.length > 0 ? (
        <div>
          <div className="mb-1.5 text-[11px] text-zinc-500">Executed results</div>
          <div className="space-y-2">
            {result.results.map((item, i) => (
              <pre
                key={i}
                className="overflow-x-auto rounded-xl border border-emerald-900/40 bg-emerald-950/20 px-3 py-2 text-[11px] leading-relaxed text-emerald-200/90"
              >
                {JSON.stringify(item, null, 2)}
              </pre>
            ))}
          </div>
        </div>
      ) : null}

      {result.source ? (
        <p className="text-[10px] uppercase tracking-wider text-zinc-600">source · {result.source}</p>
      ) : null}
    </div>
  );
}

export function Arc1Demo() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [sceneId, setSceneId] = useState(SCENES[0].id);
  const [prompt, setPrompt] = useState(SCENES[0].defaultPrompt);
  const [extractText, setExtractText] = useState(SCENES[4].extractText || "");
  const [depth, setDepth] = useState(4);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [envOverride, setEnvOverride] = useState<Record<string, string>>({});

  const scene = useMemo(
    () => SCENES.find((s) => s.id === sceneId) || SCENES[0],
    [sceneId]
  );

  const refreshHealth = useCallback(async () => {
    try {
      const res = await fetch("/api/arc1-health", { cache: "no-store" });
      const data = (await res.json()) as Health;
      setHealth(data);
      if (typeof data.active_depth === "number") setDepth(data.active_depth);
    } catch {
      setHealth({
        ready: false,
        status: "offline",
        error: "Could not reach the ARC 1 API.",
      });
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    const id = setInterval(refreshHealth, 4000);
    return () => clearInterval(id);
  }, [refreshHealth]);

  const applyScene = useCallback((next: ExampleScene) => {
    setSceneId(next.id);
    setPrompt(next.defaultPrompt);
    if (next.extractText) setExtractText(next.extractText);
    setResult(null);
    setError(null);
    setEnvOverride({});
  }, []);

  const reset = useCallback(() => {
    applyScene(scene);
  }, [applyScene, scene]);

  const patchEnvFromResult = useCallback(
    (data: RunResult) => {
      const next: Record<string, string> = {};
      for (const call of data.function_calls || []) {
        if (call.name === "set_lights") {
          const room = String(call.arguments.room || "");
          const level = call.arguments.level;
          if (room) next[room] = `lights ${level}%`;
        }
        if (call.name === "convert_currency") {
          const r = (data.results?.[0] || {}) as Record<string, unknown>;
          if (r.converted != null) {
            next["last"] = `${r.amount} ${r.from} → ${r.converted} ${r.to}`;
          }
        }
        if (call.name === "get_weather") {
          const r = (data.results?.[0] || {}) as Record<string, unknown>;
          if (r.city) next[String(r.city)] = `${r.temp_c}°C ${r.sky}`;
        }
        if (call.name === "send_message") {
          next[String(call.arguments.to || "outbox")] = "queued";
        }
      }
      if (Object.keys(next).length) setEnvOverride((prev) => ({ ...prev, ...next }));
    },
    []
  );

  const run = useCallback(async () => {
    if (sending) return;
    if (!health?.ready) {
      setError(health?.error || "ARC 1 is not ready.");
      return;
    }
    setSending(true);
    setError(null);
    setResult(null);
    try {
      if (scene.mode === "run") {
        const res = await fetch("/api/arc1-run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt,
            tools: scene.tools,
            execute: true,
            depth,
          }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(typeof data.error === "string" ? data.error : res.statusText);
        const typed = data as RunResult;
        setResult(typed);
        patchEnvFromResult(typed);
      } else {
        const res = await fetch("/api/arc1-extract", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: extractText,
            schema: scene.extractSchema,
            depth,
          }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(typeof data.error === "string" ? data.error : res.statusText);
        setResult(data as RunResult);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setSending(false);
    }
  }, [sending, health, scene, prompt, depth, extractText, patchEnvFromResult]);

  const badge = statusLabel(health, healthLoading);

  const envRows = scene.env.map((row) => {
    const keyHit =
      envOverride[row.label] ||
      envOverride[row.label.toLowerCase()] ||
      undefined;
    return {
      ...row,
      value: keyHit || row.value,
      live: Boolean(keyHit),
    };
  });

  const currencyLive = scene.id === "currency" && envOverride.last;

  return (
    <div className="not-prose overflow-hidden rounded-2xl border border-zinc-800/90 bg-[#0c0c0e] shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
      {/* Sandbox header */}
      <div className="flex flex-col gap-3 border-b border-zinc-800/80 px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-5">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-semibold tracking-tight text-zinc-100">ARC 1 sandbox</h2>
            <span className="hidden text-zinc-600 sm:inline">·</span>
            <p className="hidden text-sm text-zinc-500 sm:block">{scene.blurb}</p>
          </div>
          <p className="mt-0.5 text-sm text-zinc-500 sm:hidden">{scene.blurb}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative">
            <select
              value={sceneId}
              onChange={(e) => {
                const next = SCENES.find((s) => s.id === e.target.value);
                if (next) applyScene(next);
              }}
              className="appearance-none rounded-full border border-zinc-700 bg-zinc-900 py-1.5 pl-3 pr-8 text-xs text-zinc-200 outline-none hover:border-zinc-500 focus:border-[#835BD9]"
              aria-label="Example scene"
            >
              {SCENES.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
            <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-[10px] text-zinc-500">
              ▾
            </span>
          </div>
          <button
            type="button"
            onClick={reset}
            className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs text-zinc-400 transition hover:bg-zinc-900 hover:text-zinc-200"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset
          </button>
          <div className="inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-950 px-3 py-1.5 text-xs text-zinc-400">
            <span className={`h-1.5 w-1.5 rounded-full ${badge.color}`} />
            {badge.text}
            {health?.parameters != null && (
              <span className="hidden text-zinc-600 sm:inline">
                · {(health.parameters / 1000).toFixed(0)}k params
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Example chips */}
      <div className="flex gap-2 overflow-x-auto border-b border-zinc-800/60 px-4 py-2.5 sm:px-5">
        {SCENES.map((s) => {
          const active = s.id === sceneId;
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => applyScene(s)}
              className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1.5 text-xs transition ${
                active
                  ? "bg-[#835BD9] text-white"
                  : "border border-zinc-800 bg-zinc-950 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200"
              }`}
            >
              <SceneIcon kind={s.icon} />
              {s.label}
            </button>
          );
        })}
      </div>

      <div className="grid lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.15fr)]">
        {/* Environment */}
        <div className="border-b border-zinc-800/80 p-4 sm:p-5 lg:border-b-0 lg:border-r">
          <div className="mb-3 flex items-center justify-between">
            <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
              Environment
            </span>
            <span className="text-[11px] text-zinc-600">{scene.label}</span>
          </div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-1">
            {envRows.map((row) => (
              <div
                key={row.label}
                className="rounded-xl border border-zinc-800 bg-zinc-950/70 px-3 py-3 transition"
              >
                <div className="mb-1 flex items-center gap-2">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      row.live || row.hint === "ready" || row.hint === "online"
                        ? "bg-emerald-400"
                        : "bg-zinc-600"
                    }`}
                  />
                  <span className="text-sm font-medium text-zinc-200">{row.label}</span>
                </div>
                <p className="text-xs text-zinc-400">{row.value}</p>
                {row.hint && (
                  <p className="mt-1 text-[10px] uppercase tracking-wider text-zinc-600">{row.hint}</p>
                )}
              </div>
            ))}
            {currencyLive && (
              <div className="rounded-xl border border-[#835BD9]/40 bg-[#835BD9]/10 px-3 py-3 sm:col-span-2 lg:col-span-1">
                <div className="mb-1 text-[11px] uppercase tracking-wider text-[#C785F2]">
                  Last conversion
                </div>
                <p className="text-sm text-zinc-100">{envOverride.last}</p>
              </div>
            )}
          </div>

          {scene.tools.length > 0 && (
            <div className="mt-4">
              <div className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                Tools in scope
              </div>
              <div className="flex flex-wrap gap-1.5">
                {scene.tools.map((t) => (
                  <code
                    key={t.name}
                    className="rounded-md border border-zinc-800 bg-black px-2 py-0.5 text-[10px] text-zinc-400"
                  >
                    {t.name}
                  </code>
                ))}
              </div>
            </div>
          )}

          <div className="mt-5">
            <label className="mb-2 flex items-center justify-between text-[11px] uppercase tracking-wider text-zinc-500">
              <span>Ladder depth</span>
              <span className="tabular-nums text-zinc-300">{depth}L</span>
            </label>
            <input
              type="range"
              min={2}
              max={4}
              step={2}
              value={depth}
              onChange={(e) => setDepth(Number(e.target.value))}
              className="w-full accent-[#835BD9]"
            />
            <p className="mt-1 text-[11px] text-zinc-600">Shallower = less compute on device.</p>
          </div>
        </div>

        {/* Query + result */}
        <div className="flex flex-col p-4 sm:p-5">
          <span className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
            {scene.mode === "extract" ? "Passage" : "Query"}
          </span>

          {scene.mode === "run" ? (
            <>
              <div className="flex flex-col gap-2 sm:flex-row">
                <input
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void run();
                    }
                  }}
                  placeholder="ask ARC 1 for something…"
                  className="min-w-0 flex-1 rounded-xl border border-zinc-800 bg-black px-3.5 py-2.5 text-sm text-zinc-100 outline-none placeholder:text-zinc-600 focus:border-[#835BD9]"
                />
                <button
                  type="button"
                  onClick={() => void run()}
                  disabled={sending || !health?.ready}
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-white px-4 py-2.5 text-sm font-medium text-black transition hover:bg-zinc-200 disabled:opacity-40"
                >
                  <Play className="h-3.5 w-3.5 fill-current" />
                  {sending ? "Running…" : "Run"}
                </button>
              </div>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {scene.prompts.map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => {
                      setPrompt(p);
                      setResult(null);
                    }}
                    className={`rounded-lg border px-2.5 py-1 text-left text-[11px] transition ${
                      prompt === p
                        ? "border-[#835BD9]/60 bg-[#835BD9]/15 text-zinc-100"
                        : "border-zinc-800 text-zinc-500 hover:border-zinc-600 hover:text-zinc-300"
                    }`}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </>
          ) : (
            <>
              <textarea
                value={extractText}
                onChange={(e) => setExtractText(e.target.value)}
                rows={4}
                className="w-full rounded-xl border border-zinc-800 bg-black px-3.5 py-2.5 text-sm text-zinc-100 outline-none focus:border-[#835BD9]"
              />
              <button
                type="button"
                onClick={() => void run()}
                disabled={sending || !health?.ready}
                className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-white px-4 py-2.5 text-sm font-medium text-black transition hover:bg-zinc-200 disabled:opacity-40 sm:w-auto"
              >
                <Play className="h-3.5 w-3.5 fill-current" />
                {sending ? "Extracting…" : "Extract"}
              </button>
            </>
          )}

          {error && (
            <p className="mt-3 rounded-lg border border-red-900/50 bg-red-950/30 px-3 py-2 text-sm text-red-300">
              {error}
            </p>
          )}

          <div className="mt-5 flex-1 border-t border-zinc-800/80 pt-4">
            <span className="mb-3 block text-[11px] font-medium uppercase tracking-wider text-zinc-500">
              Result
            </span>
            {result ? (
              <ResultView result={result} />
            ) : (
              <div className="flex min-h-[140px] items-center justify-center rounded-xl border border-dashed border-zinc-800 bg-zinc-950/40 px-4 text-center text-sm text-zinc-600">
                Pick an example chip, then Run — try currency, lights, or weather.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
