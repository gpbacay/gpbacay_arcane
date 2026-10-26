/**
 * SmolLM2 in the browser. The weights are SmolVLM-256M-Instruct Q4 (Apache-2.0); the vision encoder is not downloaded.
 * ARC 1 still answers a short docs hit. This model writes everything else. No server and no API key.
 */

export const SMOL_NAME = "SmolLM2";

export type SmolStatus = {
  state: "idle" | "loading" | "ready" | "unsupported" | "error";
  progress: number;
  phase?: "downloading" | "loading" | "warming";
};

export function smolStatusLabel(s: SmolStatus): string {
  switch (s.state) {
    case "loading":
      if (s.phase === "warming") return `Warming up ARC 1 + ${SMOL_NAME}`;
      return `${s.phase === "downloading" ? "Downloading" : "Loading"} ARC 1 + ${SMOL_NAME} ${Math.round(s.progress * 100)}%`;
    case "ready":
      return `ARC 1 + ${SMOL_NAME} · on device`;
    case "unsupported":
      return "Docs only · no WebGPU";
    case "error":
      return `Docs only · ${SMOL_NAME} failed to load`;
    default:
      return `ARC 1 + ${SMOL_NAME}`;
  }
}

let status: SmolStatus = { state: "idle", progress: 0 };
const listeners = new Set<(s: SmolStatus) => void>();
const setStatus = (s: SmolStatus) => {
  status = s;
  listeners.forEach((listener) => listener(s));
};

export const getSmolStatus = (): SmolStatus => status;

export function onSmolStatus(listener: (s: SmolStatus) => void): () => void {
  listeners.add(listener);
  listener(status);
  return () => void listeners.delete(listener);
}

type Gpu = { requestAdapter(): Promise<unknown> };

let worker: Worker | null = null;
let ready = false;
let generating = false;
let loading: Promise<void> | null = null;

type WorkerMsg =
  | { type: "status"; phase: "downloading" | "loading" | "warming"; progress: number }
  | { type: "ready" }
  | { type: "error"; message: string }
  | { type: "token"; id: number; text: string }
  | { type: "done"; id: number; text: string }
  | { type: "gen-error"; id: number; message: string };

type Pending = {
  id: number;
  onToken: (text: string) => void;
  resolve: (text: string) => void;
  reject: (err: Error) => void;
};

let pending: Pending | null = null;
let genSeq = 0;

function attach(w: Worker) {
  w.onmessage = (e: MessageEvent<WorkerMsg>) => {
    const msg = e.data;
    if (msg.type === "status") {
      setStatus({ state: "loading", phase: msg.phase, progress: msg.progress });
      return;
    }
    if (msg.type === "ready") {
      ready = true;
      setStatus({ state: "ready", progress: 1 });
      return;
    }
    if (msg.type === "error") {
      console.warn(`[${SMOL_NAME}] unavailable, answering from the docs:`, msg.message);
      setStatus({ state: "error", progress: 0 });
      return;
    }
    if (!pending || msg.id !== pending.id) return;
    if (msg.type === "token") pending.onToken(msg.text);
    else if (msg.type === "done") pending.resolve(msg.text);
    else pending.reject(new Error(msg.message));
  };
  w.onerror = (e) => {
    console.warn(`[${SMOL_NAME}] worker failed:`, e.message);
    setStatus({ state: "error", progress: 0 });
  };
}

/** Start the download. The browser caches it after the first visit. */
export function warmSmol(): void {
  if (loading || ready) return;
  setStatus({ state: "loading", phase: "downloading", progress: 0 });
  loading = (async () => {
    const adapter = await (navigator as Navigator & { gpu?: Gpu }).gpu?.requestAdapter();
    if (!adapter) {
      setStatus({ state: "unsupported", progress: 0 });
      return;
    }
    navigator.storage?.persist?.().catch(() => {});
    const w = new Worker(new URL("./smol.worker.ts", import.meta.url), { type: "module" });
    worker = w;
    attach(w);
    w.postMessage({ type: "load" });
  })().catch((e) => {
    console.warn(`[${SMOL_NAME}] unavailable, answering from the docs:`, e);
    setStatus({ state: "error", progress: 0 });
  });
}

export const smolReady = () => ready && worker !== null && !generating;

const FIRST_TOKEN_MS = 45_000;
const NEXT_TOKEN_MS = 10_000;

const SYSTEM = `You are the assistant on the ARCANE documentation site.
ARC 1 routes each question. You write the reply when it is open, creative, emotional, or not covered by one docs sentence.
Be warm and plain. One to three sentences, or a short list when they asked for one.
If they share a feeling, acknowledge it in one clause, then help.
Use the facts for anything about ARCANE. Do not invent function names, version numbers, or dates.
If the question is not about ARCANE or this site, one short friendly sentence is enough.`;

export type SmolTurn = { role: "user" | "model"; text: string };

function buildPrompt(question: string, facts: string, history: SmolTurn[], hint: string): string {
  const recent = history
    .filter((turn) => turn.text.trim())
    .slice(-4)
    .map((turn) => `${turn.role === "model" ? "Assistant" : "Visitor"}: ${turn.text.trim().slice(0, 400)}`)
    .join("\n");
  return `${SYSTEM}
Use the recent conversation for follow-ups. Always answer the newest question.${hint ? `\n\nRouter hint (not an answer): ${hint}` : ""}

Facts:
${facts}${recent ? `\n\nRecent chat:\n${recent}` : ""}

Question: ${question}`;
}

/** A finished line that already appeared: a 256M model loops this way. */
export function repeatsLine(text: string): boolean {
  const lines = text.split("\n").slice(0, -1).map((line) => line.trim()).filter((line) => line.length > 3);
  return new Set(lines).size < lines.length;
}

/** Stream a reply. `onText` gets the whole reply so far. */
export async function streamSmol(
  question: string,
  facts: string,
  onText: (text: string) => void,
  signal: AbortSignal,
  history: SmolTurn[] = [],
  hint = "",
): Promise<string> {
  const w = worker;
  if (!w || !ready) throw new Error(`${SMOL_NAME} is not loaded`);
  const id = ++genSeq;
  const prompt = buildPrompt(question, facts, history, hint);
  let text = "";
  let closed = false;
  let watchdog: ReturnType<typeof setTimeout> | undefined;
  let bail: (err: Error) => void = () => {};
  const failed = new Promise<never>((_, reject) => (bail = reject));
  failed.catch(() => {});
  const arm = (ms: number) => {
    clearTimeout(watchdog);
    watchdog = setTimeout(() => {
      w.postMessage({ type: "interrupt" });
      bail(new Error(`${SMOL_NAME} stopped responding`));
    }, ms);
  };
  signal.addEventListener("abort", () => w.postMessage({ type: "interrupt" }));
  const stopIfLooping = (final: boolean) => {
    if (!repeatsLine(final ? `${text}\n` : text)) return false;
    w.postMessage({ type: "interrupt" });
    bail(new Error(`${SMOL_NAME} started repeating itself`));
    return true;
  };

  generating = true;
  const run = new Promise<string>((resolve, reject) => {
    pending = {
      id,
      onToken: (t) => {
        if (closed || signal.aborted) {
          w.postMessage({ type: "interrupt" });
          return;
        }
        arm(NEXT_TOKEN_MS);
        text = t;
        if (stopIfLooping(false)) return;
        onText(text);
      },
      resolve,
      reject,
    };
    w.postMessage({ type: "generate", id, prompt, maxNewTokens: 160 });
  }).then((final) => {
    text = final || text;
    stopIfLooping(true);
    return text.trim();
  });
  run.catch(() => {}).finally(() => {
    if (pending?.id === id) pending = null;
    generating = false;
  });

  try {
    arm(FIRST_TOKEN_MS);
    await Promise.race([run, failed]);
    return text.trim();
  } catch (err) {
    if (text.trim()) return text.trim();
    throw err;
  } finally {
    closed = true;
    clearTimeout(watchdog);
  }
}
