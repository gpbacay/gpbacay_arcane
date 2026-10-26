"use client";

import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { ArrowUp, Check, Copy, MessageSquare, Mic, MicOff, MoreVertical, Pencil, Plus, Reply, Trash2, X } from "lucide-react";
import { rememberTurn, routeWithArc1, warmArc1 } from "@/lib/arc1Router";
import { answerDocs, arc1CanAnswer, generationFacts, groundedText, navigationOnly, pageById, routeThought, type ChatRoute } from "@/lib/docsChat";
import { appendTranscript, blankThread, removeThread, titleFrom, truncateBefore, type ThreadMsg } from "@/lib/chatThreads";
import { getSmolStatus, onSmolStatus, smolReady, smolStatusLabel, streamSmol, warmSmol, type SmolStatus } from "@/lib/smol";
import { ChainOfThoughtTrace, type ThoughtStep } from "./arcane/chain-of-thought";

type PageMsg = ThreadMsg & { modelLabel?: string };
type PageThread = { id: string; title: string; messages: PageMsg[] };
type PageStore = { chats: PageThread[]; currentId: string; modelId: string };

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

type SpeechRec = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: Event & { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onend: (() => void) | null;
  onerror: ((event: Event & { error?: string }) => void) | null;
};

const PAGE_KEY = "arcane_chat_page_v1";

const BROWSER: ModelHealth = {
  id: "arc1-browser",
  label: "ARC 1",
  ready: true,
  trained: true,
  status: "ok",
  preset: "browser",
  architecture: "browser",
  quality_note: "ARC 1 answers a short docs question. SmolLM2 writes the open, creative, and emotional ones in this browser.",
};

function speechCtor(): (new () => SpeechRec) | undefined {
  const host = window as typeof window & {
    SpeechRecognition?: new () => SpeechRec;
    webkitSpeechRecognition?: new () => SpeechRec;
  };
  return host.SpeechRecognition ?? host.webkitSpeechRecognition;
}

function loadStore(): PageStore {
  const fresh = blankThread(crypto.randomUUID());
  try {
    const stored = localStorage.getItem(PAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored) as PageStore;
      if (parsed.chats?.length && parsed.currentId) return { ...parsed, modelId: parsed.modelId || BROWSER.id };
    }
  } catch {
    /* a bad cache just starts empty */
  }
  return { chats: [fresh], currentId: fresh.id, modelId: BROWSER.id };
}

function parameterLabel(parameters?: number | null) {
  if (!parameters) return "";
  return `${(parameters / 1_000_000).toFixed(1)}M`;
}

async function writeWithSmol(
  text: string,
  route: ChatRoute | null,
  history: { role: "user" | "model"; text: string }[],
  onPartial: (text: string) => void,
): Promise<{ text: string; by: string }> {
  warmSmol();
  const state = getSmolStatus().state;
  const docs = () => answerDocs(text, route).text;
  if (!smolReady()) {
    if (state === "loading" || state === "idle") {
      return {
        text: "SmolLM2 is still loading in this browser. Ask again in a moment and it will answer in its own words.",
        by: "Passed to SmolLM2",
      };
    }
    return { text: docs(), by: "Answered from the docs" };
  }
  const facts = generationFacts(text, route);
  const fallback = docs();
  let latest = "";
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const reply = await streamSmol(
      text,
      facts,
      (partial) => {
        latest = partial;
        if (timer) return;
        timer = setTimeout(() => {
          timer = undefined;
          onPartial(latest);
        }, 120);
      },
      new AbortController().signal,
      history,
    );
    const written = groundedText(reply || latest, facts, fallback, text);
    return { text: written, by: written === fallback ? "Answered from the docs" : "Written by SmolLM2" };
  } catch {
    const written = groundedText(latest, facts, fallback, text);
    return { text: written, by: written === fallback ? "Answered from the docs" : "Written by SmolLM2" };
  } finally {
    clearTimeout(timer);
  }
}

export function SlmChat() {
  const router = useRouter();
  const [store, setStore] = useState<PageStore | null>(null);
  const [remote, setRemote] = useState<ModelHealth[]>([]);
  const [smol, setSmol] = useState<SmolStatus>(getSmolStatus);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agentsOpen, setAgentsOpen] = useState(false);
  const [replyTo, setReplyTo] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState("");
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [menuId, setMenuId] = useState<string | null>(null);
  const [listening, setListening] = useState(false);
  const [speechOn, setSpeechOn] = useState(false);
  const [speechError, setSpeechError] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const speechRef = useRef<SpeechRec | null>(null);

  useEffect(() => {
    setStore(loadStore());
    setSpeechOn(Boolean(speechCtor()));
    return () => speechRef.current?.abort();
  }, []);

  useEffect(() => {
    if (store) localStorage.setItem(PAGE_KEY, JSON.stringify(store));
  }, [store]);

  useEffect(() => onSmolStatus(setSmol), []);

  useEffect(() => {
    warmArc1();
    warmSmol();
    let cancelled = false;
    fetch("/api/slm-health", { cache: "no-store" })
      .then((res) => res.json())
      .then((data: Health) => {
        if (!cancelled) setRemote((data.models ?? []).filter((model) => model.ready && model.id !== BROWSER.id));
      })
      .catch(() => {
        if (!cancelled) setRemote([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const current = store?.chats.find((chat) => chat.id === store.currentId) ?? store?.chats[0];
  const models = [BROWSER, ...remote];
  const selectedModel = models.find((model) => model.id === store?.modelId) ?? BROWSER;
  const onDevice = selectedModel.architecture === "browser";

  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight });
  }, [current?.messages, sending]);

  const patchChat = (chatId: string, update: (chat: PageThread) => PageThread) => {
    setStore((prev) => (prev ? { ...prev, chats: prev.chats.map((chat) => (chat.id === chatId ? update(chat) : chat)) } : prev));
  };

  const put = (chatId: string, message: PageMsg) => {
    patchChat(chatId, (chat) => {
      const exists = chat.messages.some((item) => item.id === message.id);
      return {
        ...chat,
        messages: exists ? chat.messages.map((item) => (item.id === message.id ? message : item)) : [...chat.messages, message],
      };
    });
  };

  const answer = async (chatId: string, text: string, quote: string | undefined, history: PageMsg[], model: ModelHealth) => {
    const route = await routeWithArc1(text);
    const jump = navigationOnly(text) ?? (route?.intent === "none" ? route.pageId : null);
    const destination = pageById(jump);
    const thought = (outcome: ThoughtStep) => routeThought(route, outcome);
    const assistant = (body: string, outcome: ThoughtStep, extra: Partial<PageMsg> = {}): PageMsg => {
      const traced = thought(outcome);
      return { id: crypto.randomUUID(), role: "assistant", text: body, trace: traced.label, steps: traced.steps, ...extra };
    };
    if (destination && jump) {
      put(chatId, assistant(`Opening ${destination.title}.`, { label: `Opened ${destination.title}` }, { pageId: destination.id, modelLabel: "ARC 1" }));
      router.push(destination.href);
      return;
    }
    if (model.architecture !== "browser") {
      const res = await fetch("/api/slm-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: quote ? `Replying to: ${quote}\n\n${text}` : text,
          history: history.slice(-6).map((message) => ({ role: message.role, content: message.text })),
          model: model.id,
          max_new_tokens: model.architecture === "hosted" ? 128 : 48,
          temperature: 0.3,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.error === "string" ? data.error : res.statusText);
      put(chatId, assistant(String(data.reply || "(empty)"), { label: `${model.label} wrote the reply` }, { modelLabel: model.label }));
      return;
    }
    const direct = arc1CanAnswer(text);
    if (direct) {
      if (direct.pageId) rememberTurn(text, direct.pageId);
      const title = pageById(direct.pageId)?.title;
      put(
        chatId,
        assistant(direct.text, { label: title ? `Answered from the ${title} page` : "Answered by ARC 1" }, { pageId: direct.pageId, modelLabel: "ARC 1" }),
      );
      return;
    }
    const draftId = crypto.randomUUID();
    const written = await writeWithSmol(
      quote ? `Replying to: ${quote}\n\n${text}` : text,
      route,
      history.slice(-4).map((turn) => ({ role: turn.role === "assistant" ? "model" as const : "user" as const, text: turn.text })),
      (partial) => {
        const traced = thought({ label: "SmolLM2 is writing" });
        put(chatId, { id: draftId, role: "assistant", text: partial, trace: traced.label, steps: traced.steps, modelLabel: "SmolLM2" });
      },
    );
    const docs = answerDocs(text, route);
    const fromModel = written.by.startsWith("Written");
    const traced = thought({ label: written.by });
    put(chatId, {
      id: draftId,
      role: "assistant",
      text: written.text,
      trace: traced.label,
      steps: traced.steps,
      pageId: fromModel ? null : docs.pageId,
      modelLabel: fromModel ? "SmolLM2" : "ARC 1",
    });
  };

  const send = async (raw: string) => {
    const text = raw.trim();
    if (!text || sending || !current || !store) return;
    if (!selectedModel.ready) {
      setError(selectedModel.error || "The selected model is not ready yet.");
      return;
    }
    if (speechRef.current) {
      speechRef.current.onresult = null;
      speechRef.current.stop();
    }
    const chatId = current.id;
    const quote = replyTo ?? undefined;
    const history = current.messages;
    const model = selectedModel;
    setInput("");
    setReplyTo(null);
    setError(null);
    setSending(true);
    const user: PageMsg = { id: crypto.randomUUID(), role: "user", text, replyTo: quote };
    patchChat(chatId, (chat) => ({
      ...chat,
      title: chat.messages.length ? chat.title : titleFrom(text),
      messages: [...chat.messages, user],
    }));
    try {
      await answer(chatId, text, quote, history, model);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Chat failed";
      setError(message);
      put(chatId, { id: crypto.randomUUID(), role: "assistant", text: `Could not generate a reply: ${message}`, modelLabel: model.label });
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  };

  const saveEdit = async () => {
    const text = editingText.trim();
    if (!text || !current || !editingId || sending) return;
    const target = current.messages.find((message) => message.id === editingId);
    if (!target) return;
    setEditingId(null);
    if (target.role === "assistant") {
      patchChat(current.id, (chat) => ({
        ...chat,
        messages: chat.messages.map((message) => (message.id === editingId ? { ...message, text } : message)),
      }));
      return;
    }
    const kept = truncateBefore(current.messages, editingId);
    const user: PageMsg = { ...target, text };
    const chatId = current.id;
    patchChat(chatId, (chat) => ({ ...chat, title: kept.length ? chat.title : titleFrom(text), messages: [...kept, user] }));
    setSending(true);
    try {
      await answer(chatId, text, target.replyTo, kept, selectedModel);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Chat failed";
      setError(message);
    } finally {
      setSending(false);
    }
  };

  const confirmDelete = () => {
    if (!current || !pendingDeleteId) return;
    patchChat(current.id, (chat) => ({ ...chat, messages: truncateBefore(chat.messages, pendingDeleteId) }));
    setPendingDeleteId(null);
  };

  const newAgent = () => {
    if (!store) return;
    const chat = blankThread(crypto.randomUUID());
    setStore({ ...store, chats: [chat, ...store.chats], currentId: chat.id });
    setAgentsOpen(false);
    setReplyTo(null);
    setEditingId(null);
    setMenuId(null);
    setPendingDeleteId(null);
    inputRef.current?.focus();
  };

  const toggleSpeech = () => {
    if (listening) {
      speechRef.current?.stop();
      return;
    }
    const Ctor = speechCtor();
    if (!Ctor) {
      setSpeechError("Speech recognition is not supported by this browser.");
      return;
    }
    const recognition = new Ctor();
    const starting = input.trimEnd();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = navigator.language || "en-US";
    recognition.onresult = (event) => {
      const chunks: string[] = [];
      for (let index = 0; index < event.results.length; index += 1) chunks.push(event.results[index][0]?.transcript ?? "");
      setInput(appendTranscript(starting, chunks));
    };
    recognition.onerror = (event) => {
      setSpeechError(event.error === "not-allowed" ? "Microphone permission was denied." : "I could not hear that. Please try again.");
      setListening(false);
    };
    recognition.onend = () => {
      setListening(false);
      speechRef.current = null;
      inputRef.current?.focus();
    };
    speechRef.current = recognition;
    setSpeechError(null);
    setListening(true);
    try {
      recognition.start();
    } catch {
      setListening(false);
      setSpeechError("Unable to start the microphone. Please try again.");
    }
  };

  if (!store || !current) {
    return <div className="not-prose min-h-[520px] border border-zinc-800 bg-black" />;
  }

  const waiting = sending && current.messages.at(-1)?.role === "user";
  const badge = onDevice
    ? smolStatusLabel(smol)
    : selectedModel.architecture === "hosted"
      ? `SmolLM2-135M · ${parameterLabel(selectedModel.parameters)}`
      : `${selectedModel.label}${selectedModel.parameters ? ` · ${parameterLabel(selectedModel.parameters)}` : ""}`;

  return (
    <div className="not-prose flex h-[min(68vh,640px)] min-h-[520px] flex-col border border-zinc-800 bg-black">
      <header className="relative flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
        <img src="/arcane_logo_purple.svg" alt="" width={18} height={18} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-zinc-100">{current.title}</p>
          <p className="truncate text-[11px] text-zinc-500">{badge}</p>
        </div>
        <label className="sr-only" htmlFor="arc1-model-select">
          Chat model
        </label>
        <select
          id="arc1-model-select"
          aria-label="Chat model"
          value={selectedModel.id}
          disabled={sending}
          onChange={(event) => {
            setStore({ ...store, modelId: event.target.value });
            setError(null);
          }}
          className="max-w-40 bg-zinc-950 px-1 py-1 text-xs text-zinc-200 outline-none focus:border-[#C785F2] disabled:opacity-50"
        >
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.label}
              {model.parameters ? ` (${parameterLabel(model.parameters)})` : ""}
            </option>
          ))}
        </select>
        <button type="button" aria-label="New agent" onClick={newAgent} className="p-1 text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#C785F2]">
          <Plus className="h-4 w-4" />
        </button>
        <button
          type="button"
          aria-expanded={agentsOpen}
          aria-label="Agents"
          onClick={() => setAgentsOpen((value) => !value)}
          className="p-1 text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#C785F2]"
        >
          <MessageSquare className="h-4 w-4" />
        </button>
        {agentsOpen && (
          <div className="absolute right-2 top-full z-10 mt-1 w-60 border border-zinc-800 bg-black p-1 shadow-2xl">
            {store.chats.map((chat) => (
              <div key={chat.id} className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => {
                    setStore({ ...store, currentId: chat.id });
                    setAgentsOpen(false);
                    setEditingId(null);
                    setReplyTo(null);
                  }}
                  className="min-w-0 flex-1 truncate px-2 py-1.5 text-left text-sm text-zinc-200 hover:bg-zinc-900"
                >
                  {chat.id === current.id ? "✓ " : ""}
                  {chat.title}
                </button>
                <button
                  type="button"
                  aria-label={`Delete ${chat.title}`}
                  onClick={() => {
                    const next = removeThread(store.chats, chat.id, crypto.randomUUID());
                    setStore({ ...store, chats: next.threads, currentId: next.currentId });
                    setAgentsOpen(false);
                  }}
                  className="p-1 text-zinc-500 hover:text-red-400"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
            <button type="button" onClick={newAgent} className="flex w-full items-center gap-2 px-2 py-1.5 text-left text-sm text-[#C785F2] hover:bg-zinc-900">
              <Plus className="h-3.5 w-3.5" /> New agent
            </button>
          </div>
        )}
      </header>

      <div ref={scrollerRef} className="flex-1 space-y-4 overflow-y-auto px-3 pb-8 pt-3">
        {current.messages.length === 0 && (
          <p className="text-sm leading-relaxed text-zinc-400">
            Ask about the library, or say “open the quick start”. ARC 1 answers a short docs question. SmolLM2 writes the rest.
          </p>
        )}
        {current.messages.map((message) => {
          const page = pageById(message.pageId ?? null);
          if (editingId === message.id) {
            return (
              <div key={message.id} className="border border-zinc-700 bg-zinc-950 p-2">
                <textarea
                  value={editingText}
                  onChange={(event) => setEditingText(event.target.value)}
                  aria-label="Edit message"
                  className="min-h-16 w-full resize-none bg-transparent text-sm text-zinc-100 outline-none"
                />
                <div className="mt-1 flex justify-end gap-2">
                  <button type="button" onClick={() => setEditingId(null)} className="px-2 py-1 text-xs text-zinc-400 hover:text-zinc-100">
                    Cancel
                  </button>
                  <button type="button" onClick={() => void saveEdit()} disabled={!editingText.trim() || sending} className="px-2 py-1 text-xs text-[#C785F2] disabled:opacity-40">
                    Save
                  </button>
                </div>
              </div>
            );
          }
          const mine = message.role === "user";
          return (
            <div key={message.id} className={mine ? "flex justify-end" : ""}>
              <div
                className={`group relative pb-7 ${mine ? "w-fit max-w-[85%]" : "w-full"}`}
                onMouseLeave={() => setMenuId((id) => (id === message.id ? null : id))}
                onBlur={(event) => {
                  if (!event.currentTarget.contains(event.relatedTarget)) setMenuId((id) => (id === message.id ? null : id));
                }}
              >
                <div
                  role="article"
                  aria-label={`${mine ? "You" : "ARCANE"}: ${message.text}`}
                  className={mine ? "bg-zinc-900 px-3 py-2 text-sm text-zinc-100" : "text-sm leading-relaxed text-zinc-200"}
                >
                  {message.replyTo && <p className="mb-1 border-l border-[#C785F2] pl-2 text-[11px] text-zinc-500">{message.replyTo}</p>}
                  {!mine && message.trace && (
                    <div className="mb-2">
                      <ChainOfThoughtTrace label={`Thought · ${message.trace}`} steps={(message.steps ?? []).filter((step) => step.label)} />
                    </div>
                  )}
                  <p className="whitespace-pre-wrap">{message.text}</p>
                  {!mine && page && (
                    <button type="button" onClick={() => router.push(page.href)} className="mt-2 text-sm font-medium text-[#C785F2] underline hover:text-[#d49cf5]">
                      Open {page.title}
                    </button>
                  )}
                  {!mine && message.modelLabel && (
                    <span className="mt-1.5 block text-[10px] uppercase tracking-wide text-zinc-500">{message.modelLabel}</span>
                  )}
                </div>
                {pendingDeleteId === message.id ? (
                  <p className="mt-1 text-[11px] text-zinc-400">
                    Delete this and everything after it?{" "}
                    <button type="button" onClick={confirmDelete} className="text-red-400">
                      Delete
                    </button>{" "}
                    <button type="button" onClick={() => setPendingDeleteId(null)} className="text-zinc-300">
                      Cancel
                    </button>
                  </p>
                ) : (
                  <div className={`absolute bottom-0 z-10 flex gap-0.5 opacity-0 pointer-events-none group-hover:pointer-events-auto group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:opacity-100 [@media(hover:none)]:pointer-events-auto [@media(hover:none)]:opacity-100 ${mine ? "right-0" : "left-0"}`}>
                    <BubbleButton
                      label="Reply"
                      onClick={() => {
                        setMenuId(null);
                        setReplyTo(message.text.slice(0, 140));
                        inputRef.current?.focus();
                      }}
                    >
                      <Reply className="h-3.5 w-3.5" />
                    </BubbleButton>
                    <div className="relative">
                      <BubbleButton label="More" expanded={menuId === message.id} onClick={() => setMenuId((id) => (id === message.id ? null : message.id))}>
                        <MoreVertical className="h-3.5 w-3.5" />
                      </BubbleButton>
                      {menuId === message.id && (
                        <div role="menu" className={`absolute top-full z-20 mt-1 w-32 border border-zinc-800 bg-black py-1 shadow-2xl ${mine ? "right-0" : "left-0"}`}>
                          <MenuItem
                            label={copiedId === message.id ? "Copied" : "Copy"}
                            onClick={() => {
                              void navigator.clipboard.writeText(message.text);
                              setCopiedId(message.id);
                              window.setTimeout(() => setCopiedId((id) => (id === message.id ? null : id)), 1200);
                            }}
                          >
                            {copiedId === message.id ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                          </MenuItem>
                          <MenuItem
                            label="Edit"
                            disabled={sending}
                            onClick={() => {
                              setMenuId(null);
                              setEditingId(message.id);
                              setEditingText(message.text);
                            }}
                          >
                            <Pencil className="h-3.5 w-3.5" />
                          </MenuItem>
                          <MenuItem
                            label="Delete"
                            disabled={sending}
                            onClick={() => {
                              setMenuId(null);
                              setPendingDeleteId(message.id);
                            }}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </MenuItem>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
        {waiting && <ChainOfThoughtTrace thinking steps={[]} />}
      </div>

      {error && (
        <p className="px-3 pb-1 text-sm text-red-400" role="alert">
          {error}
        </p>
      )}

      <form
        className="border-t border-zinc-800 p-2"
        onSubmit={(event) => {
          event.preventDefault();
          void send(input);
        }}
      >
        {replyTo && (
          <p className="mb-1 flex items-start gap-2 px-1 text-[11px] text-zinc-400">
            <span className="min-w-0 flex-1 truncate border-l border-[#C785F2] pl-2">Replying to: {replyTo}</span>
            <button type="button" aria-label="Cancel reply" onClick={() => setReplyTo(null)} className="text-zinc-500 hover:text-zinc-200">
              <X className="h-3 w-3" />
            </button>
          </p>
        )}
        <div className="flex items-end gap-2">
          <textarea
            ref={inputRef}
            value={input}
            maxLength={2000}
            rows={1}
            placeholder="Ask about the docs"
            aria-label="Message ARCANE…"
            disabled={sending || !selectedModel.ready}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send(input);
              }
            }}
            className="max-h-24 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-zinc-100 outline-none placeholder:text-zinc-600 disabled:opacity-50"
          />
          <button
            type="button"
            onClick={toggleSpeech}
            disabled={!speechOn || sending}
            aria-pressed={listening}
            aria-label={listening ? "Stop voice input" : "Start voice input"}
            title={speechOn ? (listening ? "Stop listening" : "Speak your message") : "Voice input is unavailable in this browser"}
            className={`flex h-9 w-9 items-center justify-center border ${listening ? "border-red-400/40 bg-red-500/20 text-red-400" : "border-zinc-800 text-zinc-400 hover:text-zinc-100"} disabled:opacity-40`}
          >
            {listening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
          </button>
          <button type="submit" disabled={!input.trim() || sending || !selectedModel.ready} aria-label="Send message" className="flex h-9 w-9 items-center justify-center bg-[#C785F2] text-black disabled:opacity-40">
            <ArrowUp className="h-4 w-4" />
          </button>
        </div>
        {(listening || speechError) && <p className="px-1 pt-1 text-[11px] text-red-300">{speechError ?? "Listening… speak now"}</p>}
      </form>
    </div>
  );
}

function BubbleButton({
  label,
  onClick,
  disabled,
  expanded,
  children,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  expanded?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      aria-expanded={expanded}
      aria-haspopup={expanded === undefined ? undefined : "menu"}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className="p-1 text-zinc-500 hover:bg-zinc-900 hover:text-zinc-100 disabled:opacity-40"
    >
      {children}
    </button>
  );
}

function MenuItem({
  label,
  onClick,
  disabled,
  children,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      disabled={disabled}
      onClick={onClick}
      className="flex w-full items-center gap-2 px-2 py-1.5 text-left text-sm text-zinc-200 hover:bg-zinc-900 disabled:opacity-40"
    >
      {children}
      {label}
    </button>
  );
}
