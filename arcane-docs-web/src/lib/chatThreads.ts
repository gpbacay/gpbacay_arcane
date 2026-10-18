export type ThreadMsg = {
  id: string;
  role: "user" | "assistant";
  text: string;
  trace?: string;
  pageId?: string | null;
  /** Snippet of the bubble this message answers. */
  replyTo?: string;
};

export type Thread = {
  id: string;
  title: string;
  messages: ThreadMsg[];
};

export function blankThread(id: string): Thread {
  return { id, title: "New agent", messages: [] };
}

/** Spoken text appended to whatever was already in the box. Capped at 2000. */
export function appendTranscript(starting: string, chunks: string[]): string {
  const transcript = chunks.join("");
  const gap = starting && transcript ? " " : "";
  return `${starting}${gap}${transcript}`.slice(0, 2000);
}

export function titleFrom(text: string): string {
  const clean = text.trim().replace(/\s+/g, " ");
  if (!clean) return "New agent";
  return clean.length > 42 ? `${clean.slice(0, 42)}…` : clean;
}

/** Messages before `id`. Editing or deleting a bubble drops everything after it. */
export function truncateBefore<T extends { id: string }>(messages: T[], id: string): T[] {
  const index = messages.findIndex((message) => message.id === id);
  return index < 0 ? messages : messages.slice(0, index);
}

export function removeThread(threads: Thread[], id: string, nextId: string): { threads: Thread[]; currentId: string } {
  const rest = threads.filter((thread) => thread.id !== id);
  if (!rest.length) {
    const fresh = blankThread(nextId);
    return { threads: [fresh], currentId: fresh.id };
  }
  return { threads: rest, currentId: rest[0].id };
}
