// Arcane Capsule: a knowledge base packed into self-describing frames, searched with Hippocampus. No server, no
// vector database, no embedder. Rebuilt from QRyptoRAG (qr-video-rag): there the MP4 only held a copy of text a
// vector database already stored; here the frames are the whole store, and opening them rebuilds the search.
//
// pack({ id: text }) -> one payload (Uint8Array) per chunk: deflate(JSON ["arcane-capsule/1", id, chunk]), sized
// for one QR code. unpack(payloads) -> { memory, lost }: a Hippocampus whose examples are the chunks, labeled by
// document id, so memory.neighbors(query) retrieves passages and memory.react(query) says which document answers.
// The medium is up to you: capsule-video.mjs writes the payloads as QR frames of an MP4; the docs site draws them
// on canvases. This file uses only web APIs, so it runs in Node 18+ and the browser alike.
import { Hippocampus } from "./hippocampus.mjs";

export const TAG = "arcane-capsule/1";
export const MAX_PAYLOAD = 2331; // bytes a version-40 QR code holds at error correction M

const squeeze = async (bytes, stream) => new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(stream)).arrayBuffer());

/** Split `text` into chunks of at most `size` characters, cut at whitespace, each starting `overlap` characters
 *  before the previous one ended (moved forward to a word start). */
export function chunk(text, size = 500, overlap = 50) {
  const out = [];
  for (let i = 0; i < text.length; ) {
    let end = Math.min(i + size, text.length);
    if (end < text.length) {
      const cut = text.slice(i + (size >> 1), end + 1).search(/\s\S*$/);
      if (cut >= 0) end = i + (size >> 1) + cut;
    }
    const piece = text.slice(i, end).trim();
    if (piece) out.push(piece);
    if (end >= text.length) break;
    const next = text.slice(Math.max(end - overlap, i + 1), end).search(/(?<=\s)\S/);
    i = next >= 0 ? Math.max(end - overlap, i + 1) + next : end;
  }
  return out;
}

/** One payload per chunk of each document in `documents` ({ id: text }), in order. */
export async function pack(documents, { chunkSize = 500, overlap = 50 } = {}) {
  const payloads = [];
  for (const [id, text] of Object.entries(documents)) {
    for (const piece of chunk(text, chunkSize, overlap)) {
      const bytes = await squeeze(new TextEncoder().encode(JSON.stringify([TAG, id, piece])), new CompressionStream("deflate"));
      if (bytes.length > MAX_PAYLOAD) throw new Error(`a chunk of "${id}" compresses to ${bytes.length} bytes, over one QR code's ${MAX_PAYLOAD}; lower chunkSize`);
      payloads.push(bytes);
    }
  }
  return payloads;
}

/** Rebuild the knowledge base from payloads (null = a frame that could not be read). Returns { memory, lost }:
 *  memory is a Hippocampus over the chunks (label = document id; `options` go to it, e.g. { k: 5 }), lost counts
 *  payloads that were unreadable or not from a capsule. */
export async function unpack(payloads, options = {}) {
  const memory = new Hippocampus(null, options);
  let lost = 0;
  for (const bytes of payloads) {
    try {
      const [tag, id, text] = JSON.parse(new TextDecoder().decode(await squeeze(bytes, new DecompressionStream("deflate"))));
      if (tag !== TAG) throw new Error("not a capsule frame");
      memory.remember(text, id);
    } catch {
      lost++;
    }
  }
  return { memory, lost };
}
