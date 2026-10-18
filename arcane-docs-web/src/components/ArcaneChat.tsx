"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { ArrowUp, Check, Copy, MessageSquare, Mic, MicOff, MoreVertical, Pencil, Plus, Reply, Trash2, X } from "lucide-react";
import { onArc1Status, routeWithArc1, warmArc1 } from "@/lib/arc1Router";
import { appendTranscript, blankThread, removeThread, titleFrom, truncateBefore, type Thread, type ThreadMsg } from "@/lib/chatThreads";
import { answerDocs, navigationOnly, pageById, routeTrace, type ChatRoute } from "@/lib/docsChat";

const STORAGE_KEY = "arcane_docs_chat_v2";
const LEGACY_KEY = "arcane_docs_chat_v1";
const DOCS_MODEL = "docs";
const LANGUAGE_MODELS: ModelOption[] = [
  { id: "arc1-lm-v2", label: "ARC 1 LM v2", ready: false },
  { id: "arc1-lm-v1", label: "ARC 1 LM", ready: false },
  { id: "arc1-lm-100m", label: "ARC 1 LM 100M", ready: false },
];

type ModelOption = { id: string; label: string; ready: boolean };

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

function speechCtor(): (new () => SpeechRec) | undefined {
  const host = window as typeof window & {
    SpeechRecognition?: new () => SpeechRec;
    webkitSpeechRecognition?: new () => SpeechRec;
  };
  return host.SpeechRecognition ?? host.webkitSpeechRecognition;
}

type Store = { chats: Thread[]; currentId: string; modelId: string };

const STATUS_LABEL = {
  idle: "ARC 1",
  loading: "Loading ARC 1",
  ready: "ARC 1 on-device",
  error: "Docs only",
} as const;

function loadStore(): Store {
  const fresh = blankThread(crypto.randomUUID());
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored) as Store;
      if (parsed.chats?.length) return parsed;
    }
    const legacy = localStorage.getItem(LEGACY_KEY);
    if (legacy) {
      const messages = JSON.parse(legacy) as ThreadMsg[];
      if (Array.isArray(messages) && messages.length) {
        const chat = { ...fresh, title: titleFrom(messages.find((message) => message.role === "user")?.text ?? ""), messages };
        return { chats: [chat], currentId: chat.id, modelId: DOCS_MODEL };
      }
    }
  } catch {
    /* a bad cache just starts empty */
  }
  return { chats: [fresh], currentId: fresh.id, modelId: DOCS_MODEL };
}

async function languageReply(modelId: string, text: string, quote: string | undefined, history: ThreadMsg[]) {
  const res = await fetch("/api/slm-chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: quote ? `Replying to: ${quote}\n\n${text}` : text,
      history: history.slice(-6).map((message) => ({
        role: message.role === "assistant" ? "assistant" : "user",
        content: message.text,
      })),
      model: modelId,
      max_new_tokens: 48,
      temperature: 0.3,
    }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.error === "string" ? data.error : res.statusText);
  return String(data.reply || "(empty)");
}

export function ArcaneChat() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<keyof typeof STATUS_LABEL>("idle");
  const [store, setStore] = useState<Store | null>(null);
  const [agentsOpen, setAgentsOpen] = useState(false);
  const [replyTo, setReplyTo] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState("");
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [menuId, setMenuId] = useState<string | null>(null);
  const [models, setModels] = useState<ModelOption[]>([{ id: DOCS_MODEL, label: "ARC 1", ready: true }, ...LANGUAGE_MODELS]);
  const [listening, setListening] = useState(false);
  const [speechOn, setSpeechOn] = useState(false);
  const [speechError, setSpeechError] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const speechRef = useRef<SpeechRec | null>(null);

  useEffect(() => {
    setStore(loadStore());
    setSpeechOn(Boolean(speechCtor()));
    return () => speechRef.current?.abort();
  }, []);

  useEffect(() => {
    if (store) localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  }, [store]);

  useEffect(() => onArc1Status(setStatus), []);

  useEffect(() => {
    if (!open) return;
    warmArc1();
    const focus = window.setTimeout(() => inputRef.current?.focus(), 50);
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    fetch("/api/slm-health", { cache: "no-store" })
      .then((res) => res.json())
      .then((data: { models?: { id: string; label: string; ready: boolean }[] }) => {
        const remote = new Map((data.models ?? []).map((model) => [model.id, model]));
        const language = LANGUAGE_MODELS.map((model) => remote.get(model.id) ?? model);
        for (const model of remote.values()) {
          if (!language.some((item) => item.id === model.id)) language.push(model);
        }
        setModels([{ id: DOCS_MODEL, label: "ARC 1", ready: true }, ...language]);
      })
      .catch(() => setModels([{ id: DOCS_MODEL, label: "ARC 1", ready: true }, ...LANGUAGE_MODELS]));
    return () => {
      window.clearTimeout(focus);
      window.removeEventListener("keydown", onKey);
      speechRef.current?.abort();
    };
  }, [open]);

  const current = store?.chats.find((chat) => chat.id === store.currentId) ?? store?.chats[0];

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [current?.messages, open]);

  const patchChat = (chatId: string, update: (chat: Thread) => Thread) => {
    setStore((prev) =>
      prev ? { ...prev, chats: prev.chats.map((chat) => (chat.id === chatId ? update(chat) : chat)) } : prev,
    );
  };

  const answer = async (chatId: string, text: string, quote: string | undefined, history: ThreadMsg[]) => {
    const route: ChatRoute | null = await routeWithArc1(text);
    const jump = navigationOnly(text) ?? (route?.intent === "none" ? route.pageId : null);
    const page = pageById(jump);
    const modelId = store?.modelId ?? DOCS_MODEL;
    const model = models.find((item) => item.id === modelId);
    let message: ThreadMsg;
    if (page && jump) {
      message = { id: crypto.randomUUID(), role: "assistant", text: `Opening ${page.title}.`, trace: routeTrace(route), pageId: page.id };
      router.push(page.href);
      setOpen(false);
    } else if (modelId !== DOCS_MODEL && model?.ready) {
      try {
        const reply = await languageReply(modelId, text, quote, history);
        message = { id: crypto.randomUUID(), role: "assistant", text: reply, trace: `${routeTrace(route)} · ${model.label}` };
      } catch (error) {
        const fallback = answerDocs(text, route);
        const why = error instanceof Error ? error.message : "The language model failed";
        message = {
          id: crypto.randomUUID(),
          role: "assistant",
          text: fallback.text,
          pageId: fallback.pageId,
          trace: `${why}. ${routeTrace(route)}`,
        };
      }
    } else {
      const docs = answerDocs(text, route);
      const offline = modelId !== DOCS_MODEL ? `${model?.label ?? "That model"} is offline, so this reply comes from the docs. ` : "";
      message = { id: crypto.randomUUID(), role: "assistant", text: docs.text, pageId: docs.pageId, trace: `${offline}${routeTrace(route)}` };
    }
    patchChat(chatId, (chat) => ({ ...chat, messages: [...chat.messages, message] }));
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

  const send = async (raw: string) => {
    const text = raw.trim();
    if (!text || busy || !current || !store) return;
    if (speechRef.current) {
      speechRef.current.onresult = null;
      speechRef.current.stop();
    }
    const chatId = current.id;
    const quote = replyTo ?? undefined;
    setInput("");
    setReplyTo(null);
    setBusy(true);
    const user: ThreadMsg = { id: crypto.randomUUID(), role: "user", text, replyTo: quote };
    patchChat(chatId, (chat) => ({
      ...chat,
      title: chat.messages.length ? chat.title : titleFrom(text),
      messages: [...chat.messages, user],
    }));
    try {
      await answer(chatId, text, quote, current.messages);
    } finally {
      setBusy(false);
    }
  };

  const saveEdit = async () => {
    const text = editingText.trim();
    if (!text || !current || !editingId || busy) return;
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
    const user: ThreadMsg = { ...target, text };
    const chatId = current.id;
    patchChat(chatId, (chat) => ({ ...chat, title: kept.length ? chat.title : titleFrom(text), messages: [...kept, user] }));
    setBusy(true);
    try {
      await answer(chatId, text, target.replyTo, kept);
    } finally {
      setBusy(false);
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
  };

  if (!store || !current) return null;

  return (
    <>
      {!open && (
        <button
          type="button"
          onClick={() => setOpen(true)}
          onFocus={() => warmArc1()}
          className="fixed bottom-4 right-4 z-40 flex items-center gap-2 border border-zinc-700 bg-black px-3 py-2 text-sm text-zinc-100 shadow-2xl transition-colors hover:border-[#C785F2] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#C785F2]"
          aria-expanded={false}
          aria-haspopup="dialog"
        >
          <img src="/arcane_logo_purple.svg" alt="" width={18} height={18} />
          Ask ARCANE
        </button>
      )}

      {open && (
        <div
          role="dialog"
          aria-modal="false"
          aria-label="Ask ARCANE"
          className="fixed bottom-4 right-4 z-40 flex h-[min(36rem,calc(100vh-6rem))] w-[min(28rem,calc(100vw-2rem))] flex-col border border-zinc-800 bg-black shadow-2xl"
        >
          <header className="relative flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
            <img src="/arcane_logo_purple.svg" alt="" width={18} height={18} />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-semibold text-zinc-100">{current.title}</p>
              <p className="truncate text-[11px] text-zinc-500">{STATUS_LABEL[status]}</p>
            </div>
            <button
              type="button"
              aria-label="New agent"
              onClick={newAgent}
              className="p-1 text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#C785F2]"
            >
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
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close chat"
              className="p-1 text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#C785F2]"
            >
              <X className="h-4 w-4" />
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
                <button
                  type="button"
                  onClick={newAgent}
                  className="flex w-full items-center gap-2 px-2 py-1.5 text-left text-sm text-[#C785F2] hover:bg-zinc-900"
                >
                  <Plus className="h-3.5 w-3.5" /> New agent
                </button>
              </div>
            )}
          </header>

          <div ref={listRef} className="flex-1 space-y-4 overflow-y-auto px-3 pb-8 pt-3">
            {current.messages.length === 0 && (
              <p className="text-sm leading-relaxed text-zinc-400">
                Ask about the library, or say “open the quick start”. ARC 1 routes the message and answers from the docs unless you pick a language model.
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
                      <button
                        type="button"
                        onClick={() => void saveEdit()}
                        disabled={!editingText.trim() || busy}
                        className="px-2 py-1 text-xs text-[#C785F2] disabled:opacity-40"
                      >
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
                    className={`group relative w-fit pb-7 ${mine ? "max-w-[85%]" : "max-w-full"}`}
                    onMouseLeave={() => setMenuId((id) => (id === message.id ? null : id))}
                    onBlur={(event) => {
                      if (!event.currentTarget.contains(event.relatedTarget)) setMenuId((id) => (id === message.id ? null : id));
                    }}
                  >
                    <div className={mine ? "bg-zinc-900 px-3 py-2 text-sm text-zinc-100" : "text-sm leading-relaxed text-zinc-200"}>
                      {message.replyTo && <p className="mb-1 border-l border-[#C785F2] pl-2 text-[11px] text-zinc-500">{message.replyTo}</p>}
                      <p className="whitespace-pre-wrap">{message.text}</p>
                      {message.role === "assistant" && page && (
                        <button
                          type="button"
                          onClick={() => {
                            router.push(page.href);
                            setOpen(false);
                          }}
                          className="mt-2 text-sm font-medium text-[#C785F2] underline hover:text-[#d49cf5]"
                        >
                          Open {page.title}
                        </button>
                      )}
                      {message.trace && <p className="mt-1 text-[11px] text-zinc-500">{message.trace}</p>}
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
                          <BubbleButton
                            label="More"
                            expanded={menuId === message.id}
                            onClick={() => setMenuId((id) => (id === message.id ? null : message.id))}
                          >
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
                                disabled={busy}
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
                                disabled={busy}
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
          </div>

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
                aria-label="Message"
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void send(input);
                  }
                }}
                className="max-h-24 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
              />
              <button
                type="button"
                onClick={toggleSpeech}
                disabled={!speechOn || busy}
                aria-pressed={listening}
                aria-label={listening ? "Stop voice input" : "Start voice input"}
                title={speechOn ? (listening ? "Stop listening" : "Speak your message") : "Voice input is unavailable in this browser"}
                className={`flex h-9 w-9 items-center justify-center border ${listening ? "border-red-400/40 bg-red-500/20 text-red-400" : "border-zinc-800 text-zinc-400 hover:text-zinc-100"} disabled:opacity-40`}
              >
                {listening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
              </button>
              <button
                type="submit"
                disabled={!input.trim() || busy}
                aria-label="Send message"
                className="flex h-9 w-9 items-center justify-center bg-[#C785F2] text-black disabled:opacity-40"
              >
                <ArrowUp className="h-4 w-4" />
              </button>
            </div>
            {(listening || speechError) && (
              <p className="px-1 pt-1 text-[11px] text-red-300">{speechError ?? "Listening… speak now"}</p>
            )}
            <label className="mt-1 inline-flex w-fit items-center gap-2 px-1 text-[11px] text-zinc-500">
              Model
              <select
                aria-label="Model"
                value={models.some((model) => model.id === store.modelId) ? store.modelId : DOCS_MODEL}
                onChange={(event) => setStore({ ...store, modelId: event.target.value })}
                className="w-fit bg-zinc-950 px-1 py-1 text-xs text-zinc-200 outline-none focus:border-[#C785F2]"
              >
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label}
                    {model.ready ? "" : " (offline)"}
                  </option>
                ))}
              </select>
            </label>
          </form>
        </div>
      )}
    </>
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
