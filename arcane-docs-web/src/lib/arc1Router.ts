/**
 * Browser ARC 1 router. One on-device classify, mixed with Hippocampus examples in docsChat.ts.
 * Null until the model has loaded: the chat still answers from the docs.
 */
import { Hippocampus } from "gpbacay-arcane/hippocampus";
import { EXAMPLES, decide, type ChatRoute, type Classifier } from "./docsChat";

type Arc1 = Classifier;

let hippo: Classifier | null = null;
let loading: Promise<void> | null = null;
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
  if (loading || hippo) return;
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
    hippo = new Hippocampus(agent, { k: 8, weight: 0.7, memory: EXAMPLES });
    setStatus("ready");
  })().catch((error) => {
    console.warn("[ARC 1] unavailable, answering from the docs:", error);
    loading = null;
    setStatus("error");
  });
}

export async function routeWithArc1(text: string): Promise<ChatRoute | null> {
  if (!hippo) return null;
  try {
    return await decide(hippo, text);
  } catch (error) {
    console.warn("[ARC 1] routing failed:", error);
    return null;
  }
}
