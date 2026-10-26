"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { onArc1Status, rememberTurn, routeWithArc1, warmArc1 } from "@/lib/arc1Router";
import { answerDocs, navigationOnly, pageById, routeThought, withContext } from "@/lib/docsChat";
import { ChainOfThoughtTrace, type ThoughtStep } from "./arcane/chain-of-thought";

type Role = "user" | "assistant";

type ChatMessage = {
  role: Role;
  content: string;
  model?: string;
  modelLabel?: string;
  trace?: string;
  steps?: ThoughtStep[];
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

type Health = { models?: ModelHealth[] };

type Arc1Status = "idle" | "loading" | "ready" | "error";

const BROWSER: ModelHealth = {
  id: "arc1-browser",
  label: "ARC 1",
  ready: true,
  trained: true,
  status: "ok",
  preset: "browser",
  architecture: "browser",
  quality_note: "Same on-device model as the site chatbot. ARC 1, Hippocampus, and Occipital run in this browser.",
};

function parameterLabel(parameters?: number | null) {
  if (!parameters) return "";
  return `${(parameters / 1_000_000).toFixed(1)}M`;
}

function browserBadge(status: Arc1Status) {
  if (status === "ready") return { text: "On device", color: "bg-emerald-400" };
  if (status === "error") return { text: "Docs only", color: "bg-amber-400" };
  if (status === "loading") return { text: "Loading ARC 1…", color: "bg-amber-400" };
  return { text: "On device", color: "bg-emerald-400" };
}

function remoteBadge(model: ModelHealth) {
  if (model.ready) return { text: model.trained ? "Trained" : "Untrained", color: model.trained ? "bg-emerald-400" : "bg-amber-400" };
  return { text: "Offline", color: "bg-red-500" };
}

export function SlmChat() {
  const router = useRouter();
  const [remote, setRemote] = useState<ModelHealth[]>([]);
  const [selectedModelId, setSelectedModelId] = useState(BROWSER.id);
  const [arc1Status, setArc1Status] = useState<Arc1Status>("idle");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => onArc1Status(setArc1Status), []);

  useEffect(() => {
    warmArc1();
    let cancelled = false;
    fetch("/api/slm-health", { cache: "no-store" })
      .then((res) => res.json())
      .then((data: Health) => {
        if (cancelled) return;
        setRemote((data.models ?? []).filter((model) => model.ready && model.id !== BROWSER.id));
      })
      .catch(() => {
        if (!cancelled) setRemote([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  const models = [BROWSER, ...remote];
  const selectedModel = models.find((model) => model.id === selectedModelId) ?? BROWSER;
  const onDevice = selectedModel.architecture === "browser";

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;
    if (!selectedModel.ready) {
      setError(selectedModel.error || "The selected model is not ready yet.");
      return;
    }
    setInput("");
    setError(null);
    const history = messages.map((message) => ({ role: message.role, text: message.content }));
    const nextMessages: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(nextMessages);
    setSending(true);
    try {
      if (onDevice) {
        const asked = withContext(text, history);
        const route = await routeWithArc1(asked);
        const jump = navigationOnly(text) ?? (route?.intent === "none" ? route.pageId : null);
        const destination = pageById(jump);
        if (destination && jump) {
          const thought = routeThought(route, { label: `Opened ${destination.title}` });
          setMessages([
            ...nextMessages,
            {
              role: "assistant",
              content: `Opening ${destination.title}.`,
              model: BROWSER.id,
              modelLabel: BROWSER.label,
              trace: thought.label,
              steps: thought.steps,
            },
          ]);
          router.push(destination.href);
          return;
        }
        const docs = answerDocs(text, route);
        const byRules = answerDocs(text).pageId;
        if (byRules) rememberTurn(asked, byRules);
        const title = pageById(docs.pageId)?.title;
        const thought = routeThought(route, { label: title ? `Answered from the ${title} page` : "Answered from the docs" });
        setMessages([
          ...nextMessages,
          {
            role: "assistant",
            content: docs.text,
            model: BROWSER.id,
            modelLabel: BROWSER.label,
            trace: thought.label,
            steps: thought.steps,
          },
        ]);
        return;
      }

      const res = await fetch("/api/slm-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: nextMessages.slice(0, -1).map((message) => ({ role: message.role, content: message.content })),
          model: selectedModel.id,
          max_new_tokens: selectedModel.architecture === "hosted" ? 128 : 48,
          temperature: 0.3,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.error === "string" ? data.error : res.statusText);
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
        { role: "assistant", content: `Could not generate a reply: ${message}`, modelLabel: selectedModel.label },
      ]);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }, [input, messages, onDevice, router, selectedModel, sending]);

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  };

  const badge = onDevice ? browserBadge(arc1Status) : remoteBadge(selectedModel);

  return (
    <div className="not-prose flex flex-col border border-zinc-800 bg-zinc-950/80 min-h-[520px] h-[min(68vh,640px)]">
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b border-zinc-800">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">
            {onDevice ? "ARC 1" : selectedModel.architecture === "hosted" ? "Hosted SLM" : "ARC 1 SLM"}
          </p>
          <p className="text-sm text-zinc-300">
            {onDevice
              ? "On device · Hippocampus + Occipital"
              : selectedModel.architecture === "hosted"
                ? `SmolLM2-135M · open weights · ${parameterLabel(selectedModel.parameters)}`
                : `${selectedModel.architecture || "causal"} decoder · ${parameterLabel(selectedModel.parameters)}`}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="sr-only" htmlFor="arc1-model-select">
            Chat model
          </label>
          <select
            id="arc1-model-select"
            value={selectedModel.id}
            onChange={(event) => {
              setSelectedModelId(event.target.value);
              setError(null);
            }}
            disabled={sending}
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

      {selectedModel.quality_note && (
        <p
          className={`px-4 py-2 border-b text-xs ${
            onDevice
              ? "border-zinc-800 bg-zinc-900/60 text-zinc-400"
              : "border-zinc-800 bg-amber-950/20 text-amber-200/90"
          }`}
        >
          {selectedModel.quality_note}
        </p>
      )}

      <div ref={scrollerRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {messages.length === 0 && (
          <div className="h-full min-h-[280px] flex flex-col items-center justify-center text-center px-6">
            <p className="text-zinc-200 font-medium">
              {onDevice ? "Chat with ARC 1, in this browser" : `Chat with ${selectedModel.label}`}
            </p>
            <p className="text-sm text-zinc-500 mt-2 max-w-md">
              {onDevice
                ? "This is the same chatbot as the button on every docs page. It answers from the docs and stays available with no server."
                : selectedModel.trained
                  ? "Weights are loaded. This model continues text. It needs the local SLM server, which the on-device ARC 1 model does not."
                  : "The decoder is live, but its weights are random until you pretrain it."}
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
              {message.role === "assistant" && message.steps && message.steps.length > 0 && (
                <div className="mt-2 not-prose">
                  <ChainOfThoughtTrace label={message.trace ? `Thought · ${message.trace}` : "Thought"} steps={message.steps} />
                </div>
              )}
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
              {onDevice ? "ARC 1 is answering…" : `${selectedModel.label} is generating…`}
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
          placeholder="Message ARCANE…"
          disabled={sending || !selectedModel.ready}
          className="flex-1 resize-none bg-zinc-900 border border-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-[#C785F2]/60"
        />
        <button
          type="submit"
          disabled={sending || !input.trim() || !selectedModel.ready}
          className="h-10 px-4 bg-[#C785F2] text-black text-sm font-semibold hover:bg-[#d49cf5] disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Send
        </button>
      </form>
    </div>
  );
}
