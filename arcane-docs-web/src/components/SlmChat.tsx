"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Role = "user" | "assistant";

type ChatMessage = {
  role: Role;
  content: string;
  model?: string;
  modelLabel?: string;
};

type ModelHealth = {
  id: string;
  label: string;
  ready: boolean;
  trained: boolean;
  status: string;
  error?: string | null;
  preset: string;
  parameters?: number | null;
  context?: number | null;
  architecture?: string | null;
  quality_note?: string;
};

type Health = {
  ready: boolean;
  trained?: boolean;
  preset?: string;
  status?: string;
  error?: string | null;
  default_model?: string;
  models?: ModelHealth[];
};

function statusLabel(model: ModelHealth | null, loading: boolean) {
  if (loading && !model) return { text: "Checking…", color: "bg-zinc-500" };
  if (!model) return { text: "Offline", color: "bg-red-500" };
  if (model.ready) {
    return {
      text: model.trained ? "Trained" : "Untrained",
      color: model.trained ? "bg-emerald-400" : "bg-amber-400",
    };
  }
  if (model.status === "loading") return { text: "Loading model…", color: "bg-amber-400" };
  return { text: "Offline", color: "bg-red-500" };
}

function parameterLabel(parameters?: number | null) {
  if (!parameters) return "";
  return `${(parameters / 1_000_000).toFixed(1)}M`;
}

export function SlmChat() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const refreshHealth = useCallback(async () => {
    try {
      const res = await fetch("/api/slm-health", { cache: "no-store" });
      const data = (await res.json()) as Health;
      setHealth(data);
      setSelectedModelId((current) => {
        if (current && data.models?.some((model) => model.id === current)) return current;
        return data.default_model || data.models?.[0]?.id || "";
      });
    } catch {
      setHealth({
        ready: false,
        status: "offline",
        error: "Could not reach the ARCANE SLM API.",
      });
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
    const id = setInterval(refreshHealth, 4000);
    return () => clearInterval(id);
  }, [refreshHealth]);

  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  const models: ModelHealth[] = health?.models?.length
    ? health.models
    : health
      ? [
          {
            id: health.preset || "default",
            label: health.preset || "ARCANE SLM",
            ready: health.ready,
            trained: Boolean(health.trained),
            status: health.status || (health.ready ? "ok" : "offline"),
            error: health.error,
            preset: health.preset || "default",
          },
        ]
      : [];
  const selectedModel = models.find((model) => model.id === selectedModelId) || models[0] || null;

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;
    if (!selectedModel?.ready) {
      setError(selectedModel?.error || "The selected model is not ready yet.");
      return;
    }
    setInput("");
    setError(null);
    const nextMessages: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(nextMessages);
    setSending(true);
    try {
      const res = await fetch("/api/slm-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: nextMessages.slice(0, -1),
          model: selectedModel.id,
          max_new_tokens: 48,
          temperature: 0.3,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(typeof data.error === "string" ? data.error : res.statusText);
      }
      setMessages([
        ...nextMessages,
        {
          role: "assistant",
          content: String(data.reply || "(empty)"),
          model: String(data.model || selectedModel.id),
          modelLabel: selectedModel.label,
        },
      ]);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Chat failed";
      setError(message);
      setMessages([
        ...nextMessages,
        { role: "assistant", content: `Could not generate a reply: ${message}` },
      ]);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }, [input, messages, selectedModel, sending]);

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  };

  const badge = statusLabel(selectedModel, healthLoading);

  return (
    <div className="not-prose flex flex-col border border-zinc-800 bg-zinc-950/80 min-h-[520px] h-[min(68vh,640px)]">
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b border-zinc-800">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">ARC 1 SLM</p>
          <p className="text-sm text-zinc-300">
            {selectedModel
              ? `${selectedModel.architecture || "causal"} decoder · ${parameterLabel(selectedModel.parameters)}`
              : "Causal decoder"}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="sr-only" htmlFor="arc1-model-select">
            Chat model
          </label>
          <select
            id="arc1-model-select"
            value={selectedModel?.id || ""}
            onChange={(event) => {
              setSelectedModelId(event.target.value);
              setError(null);
            }}
            disabled={sending || models.length === 0}
            className="min-w-52 bg-zinc-900 border border-zinc-700 px-2.5 py-2 text-xs text-zinc-200 focus:outline-none focus:border-[#C785F2]/70 disabled:opacity-50"
          >
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label}
                {model.parameters ? ` (${parameterLabel(model.parameters)})` : ""}
              </option>
            ))}
          </select>
          <div className="flex items-center gap-2 text-xs text-zinc-400 whitespace-nowrap">
            <span className={`inline-block size-2 rounded-full ${badge.color}`} />
            {badge.text}
          </div>
        </div>
      </div>

      {selectedModel?.quality_note && (
        <p className="px-4 py-2 border-b border-zinc-800 bg-amber-950/20 text-xs text-amber-200/90">
          {selectedModel.quality_note}
        </p>
      )}

      <div ref={scrollerRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {messages.length === 0 && (
          <div className="h-full min-h-[280px] flex flex-col items-center justify-center text-center px-6">
            <p className="text-zinc-200 font-medium">Chat with ARC 1, ARCANE&apos;s small language model</p>
            <p className="text-sm text-zinc-500 mt-2 max-w-md">
              {selectedModel?.ready
                ? selectedModel.trained
                  ? "Weights are loaded. This is a small next-token model: it continues your text rather than following instructions."
                  : "The decoder is live, but weights are random until you pretrain. Replies will look like noise."
                : "Start the local SLM API to talk to the real TensorFlow model."}
            </p>
          </div>
        )}
        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              role="article"
              aria-label={`${message.role === "user" ? "You" : "ARCANE"}: ${message.content}`}
              className={`max-w-[85%] px-3 py-2 text-sm leading-relaxed whitespace-pre-wrap break-words font-sans ${
                message.role === "user"
                  ? "bg-[#C785F2] text-black"
                  : "bg-zinc-900 text-zinc-200 border border-zinc-800"
              }`}
            >
              {message.content}
              {message.role === "assistant" && message.modelLabel && (
                <span className="block mt-1.5 text-[10px] uppercase tracking-wide text-zinc-500">
                  {message.modelLabel}
                </span>
              )}
            </div>
          </div>
        ))}
        {sending && (
          <div className="flex justify-start">
            <div className="bg-zinc-900 border border-zinc-800 px-3 py-2 text-sm text-zinc-500">
              {selectedModel ? `${selectedModel.label} is generating…` : "Thinking…"}
            </div>
          </div>
        )}
      </div>

      {error && (
        <p className="px-4 pb-2 text-sm text-red-400" role="alert">
          {error}
        </p>
      )}

      <form
        className="border-t border-zinc-800 p-3 flex gap-2 items-end"
        onSubmit={(event) => {
          event.preventDefault();
          void send();
        }}
      >
        <textarea
          ref={inputRef}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={onKeyDown}
          rows={2}
          placeholder={selectedModel?.ready ? "Message ARCANE…" : "Waiting for the SLM API…"}
          disabled={sending || !selectedModel?.ready}
          className="flex-1 resize-none bg-zinc-900 border border-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-[#C785F2]/60"
        />
        <button
          type="submit"
          disabled={sending || !input.trim() || !selectedModel?.ready}
          className="h-10 px-4 bg-[#C785F2] text-black text-sm font-semibold hover:bg-[#d49cf5] disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Send
        </button>
      </form>
    </div>
  );
}
