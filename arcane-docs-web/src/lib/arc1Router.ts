/**
 * Browser ARC 1 router. Hippocampus holds the docs and mixes them with ARC 1 once the model has loaded.
 * Until then, retrieval is memory only.
 */
import { Hippocampus } from "gpbacay-arcane/hippocampus";
import { EXAMPLES, decide, type ChatRoute, type Classifier } from "./docsChat";

type Arc1 = Classifier;

let hippo: Hippocampus = new Hippocampus(null, { k: 8, weight: 0.7, memory: EXAMPLES });
let loading: Promise<void> | null = null;
let modelReady = false;
let status: "idle" | "loading" | "ready" | "error" = "idle";
const listeners = new Set<(next: typeof status) => void>();

function setStatus(next: typeof status) {
  status = next;
  listeners.forEach((listener) => listener(next));
}

export function onArc1Status(listener: (next: typeof status) => void): () => void {
  listeners.add(listener);
  listener(status);
  return () => listeners.delete(listener);
}

export function warmArc1(): void {
  if (loading || modelReady) return;
  setStatus("loading");
  loading = (async () => {
    const { load } = await import("gpbacay-arcane/web");
    const [model, config] = await Promise.all([
      fetch("/api/arc1-web?file=model").then((res) => {
        if (!res.ok) throw new Error(`model ${res.status}`);
        return res.arrayBuffer();
      }),
      fetch("/api/arc1-web?file=config").then((res) => {
        if (!res.ok) throw new Error(`config ${res.status}`);
        return res.json();
      }),
    ]);
    const agent: Arc1 = await load({ model: new Uint8Array(model), config });
    hippo = new Hippocampus(agent, { k: 8, weight: 0.7, memory: hippo.toJSON() });
    modelReady = true;
    setStatus("ready");
  })().catch((error) => {
    console.warn("[ARC 1] unavailable, answering from the docs:", error);
    loading = null;
    setStatus("error");
  });
}

export function rememberTurn(text: string, label: string): void {
  if (!text.trim()) return;
  hippo.remember(text, label);
}

export async function routeWithArc1(text: string): Promise<ChatRoute | null> {
  try {
    return await decide(hippo, text);
  } catch (error) {
    console.warn("[ARC 1] routing failed:", error);
    return null;
  }
}
