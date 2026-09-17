"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Role = "user" | "assistant";

type ChatMessage = {
  role: Role;
  content: string;
};

type Health = {
  ready: boolean;
  trained?: boolean;
  preset?: string;
  status?: string;
  error?: string | null;
};

function statusLabel(health: Health | null, loading: boolean) {
  if (loading && !health) return { text: "Checking…", color: "bg-zinc-500" };
  if (!health) return { text: "Offline", color: "bg-red-500" };
  if (health.ready) return { text: health.trained ? "Trained" : "Untrained", color: health.trained ? "bg-emerald-400" : "bg-amber-400" };
  if (health.status === "loading") return { text: "Loading model…", color: "bg-amber-400" };
  return { text: "Offline", color: "bg-red-500" };
}

export function SlmChat() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
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
    refreshHealth();
    const id = setInterval(refreshHealth, 4000);
    return () => clearInterval(id);
  }, [refreshHealth]);

  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;
    if (!health?.ready) {
      setError(health?.error || "The model is not ready yet.");
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
          max_new_tokens: 48,
          temperature: 0.7,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(typeof data.error === "string" ? data.error : res.statusText);
      }
      setMessages([
        ...nextMessages,
        { role: "assistant", content: String(data.reply || "(empty)") },
      ]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Chat failed";
      setError(msg);
      setMessages([
        ...nextMessages,
        { role: "assistant", content: `Could not generate a reply: ${msg}` },
      ]);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }, [health, input, messages, sending]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  };

  const badge = statusLabel(health, healthLoading);

  return (
    <div className="not-prose flex flex-col border border-zinc-800 bg-zinc-950/80 min-h-[520px] h-[min(68vh,640px)]">
      <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-zinc-800">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">ARCANE SLM</p>
          <p className="text-sm text-zinc-300">
            {health?.preset ? `${health.preset} decoder` : "Causal decoder"}
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-zinc-400">
          <span className={`inline-block size-2 rounded-full ${badge.color}`} />
          {badge.text}
        </div>
      </div>

      <div ref={scrollerRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {messages.length === 0 && (
          <div className="h-full min-h-[280px] flex flex-col items-center justify-center text-center px-6">
            <p className="text-zinc-200 font-medium">Chat with the ARCANE small language model</p>
            <p className="text-sm text-zinc-500 mt-2 max-w-md">
              {health?.ready
                ? health.trained
                  ? "Weights are loaded. The tiny checkpoint continues text (Shakespeare-style), not a chat-tuned assistant."
                  : "The decoder is live, but weights are random until you pretrain. Replies will look like noise."
                : "Start the local SLM API to talk to the real TensorFlow model."}
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={`${msg.role}-${i}`}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              role="article"
              aria-label={`${msg.role === "user" ? "You" : "ARCANE"}: ${msg.content}`}
              className={`max-w-[85%] px-3 py-2 text-sm leading-relaxed whitespace-pre-wrap break-words font-sans ${
                msg.role === "user"
                  ? "bg-[#C785F2] text-black"
                  : "bg-zinc-900 text-zinc-200 border border-zinc-800"
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}
        {sending && (
          <div className="flex justify-start">
            <div className="bg-zinc-900 border border-zinc-800 px-3 py-2 text-sm text-zinc-500">
              Thinking…
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
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          rows={2}
          placeholder={health?.ready ? "Message ARCANE…" : "Waiting for the SLM API…"}
          disabled={sending}
          className="flex-1 resize-none bg-zinc-900 border border-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-[#C785F2]/60"
        />
        <button
          type="submit"
          disabled={sending || !input.trim() || !health?.ready}
          className="h-10 px-4 bg-[#C785F2] text-black text-sm font-semibold hover:bg-[#d49cf5] disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Send
        </button>
      </form>
    </div>
  );
}
