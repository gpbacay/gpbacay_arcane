// Occipital: the visual pathway of a System 1 model's memory. Port of gpbacay_arcane/occipital.py.
//
// Occipital reads words by their letters, as the brain's visual word form area does, so Hippocampus recalls an
// example that shares no whole word with a request: a typo ("parcle"), another inflection ("refunded" / "refund"),
// a compound. new Hippocampus(model, { occipital: new Occipital() }) fuses its letter-n-gram scores with the word
// search; every decision model behind that Hippocampus (ARC 1 or any other) gets the better recall.
//
// It also reads memories written as images. pack(documents) -> one payload (Uint8Array) per chunk:
// deflate(JSON ["arcane-occipital/1", id, chunk, args?]), sized for one QR code. unpack(payloads) -> { memory, lost }:
// the Hippocampus they encode. occipital-video.mjs writes the payloads as QR frames of an MP4; the docs site draws
// them on canvases. This file uses only web APIs, so it runs in Node 18+ and the browser alike.
import { Hippocampus } from "./hippocampus.mjs";

export const TAG = "arcane-occipital/1";
export const MAX_PAYLOAD = 2331; // bytes a version-40 QR code holds at error correction M

// ------------------------------------------------------------ letter pathway
const K1 = 1.2, B = 0.75;
const WORD = /[\p{L}\p{N}]+/gu; // Python's [^\W_]+

/** Letter 3- and 4-grams of each word, padded with spaces so word starts and ends count: Map(gram -> count). */
function grams(text) {
  const out = new Map();
  for (const w of text.toLowerCase().match(WORD) ?? []) {
    const s = ` ${w} `;
    for (let n = 3; n <= 4; n++) for (let i = 0; i + n <= s.length; i++) out.set(s.slice(i, i + n), (out.get(s.slice(i, i + n)) || 0) + 1);
  }
  return out;
}

export class Occipital {
  /**
   * @param {{ weight?: number, sharpness?: number, minCoverage?: number }} [options]
   *   weight: share of the fused score from letters (the rest from Hippocampus's word search).
   *   sharpness: power on the fused score, so the closest examples dominate the vote (1 flattens it and dilutes
   *   the model's share). minCoverage: an example matched by letters alone counts only if it covers this share of
   *   the request's letter evidence (idf-weighted), so Hippocampus still abstains on requests unlike anything
   *   stored. Lower it for accuracy over abstention (see the docs' measurements). maxDf: a gram found in more
   *   than this share of the examples (and in more than one) is too common to say which example is meant: it
   *   still counts against coverage but scores nobody, which keeps lookups fast.
   * One Occipital per Hippocampus: it indexes that memory's examples.
   */
  constructor({ weight = 0.5, sharpness = 2, minCoverage = 0.3, maxDf = 0.1 } = {}) {
    Object.assign(this, { weight, sharpness, minCoverage, maxDf });
    this._postings = new Map(); // gram -> Map(id -> count)
    this._len = new Map(); // id -> grams in the example
    this._grams = new Map(); // id -> its distinct grams, so remove() touches only those
  }

  add(id, text) {
    this.remove(id);
    const g = grams(text);
    for (const [gram, c] of g) {
      if (!this._postings.has(gram)) this._postings.set(gram, new Map());
      this._postings.get(gram).set(id, c);
    }
    this._len.set(id, [...g.values()].reduce((a, b) => a + b, 0));
    this._grams.set(id, [...g.keys()]);
  }

  remove(id) {
    for (const gram of this._grams.get(id) ?? []) {
      const post = this._postings.get(gram);
      post.delete(id);
      if (!post.size) this._postings.delete(gram);
    }
    this._grams.delete(id);
    this._len.delete(id);
  }

  /** Fuse `lexical` (Map id -> word score, normalized to a top of 1) with letter scores for `text`, over the ids
   *  `allowed(id)` accepts: Map id -> fused score. */
  fuse(text, lexical, allowed = () => true) {
    const n = this._len.size;
    let avg = 0, total = 0;
    for (const len of this._len.values()) avg += len / n;
    const scores = new Map(), seen = new Map();
    for (const g of grams(text).keys()) {
      const post = this._postings.get(g), df = post?.size ?? 0;
      const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
      total += idf; // letters no example has count too: a request made of them is unlike the memory
      if (!post || df > Math.max(1, this.maxDf * n)) continue;
      for (const [id, tf] of post) {
        if (!allowed(id)) continue;
        scores.set(id, (scores.get(id) || 0) + (idf * tf * (K1 + 1)) / (tf + K1 * (1 - B + (B * this._len.get(id)) / avg)));
        seen.set(id, (seen.get(id) || 0) + idf);
      }
    }
    for (const id of scores.keys()) if (!lexical.has(id) && seen.get(id) < this.minCoverage * total) scores.delete(id);
    const top = Math.max(0, ...scores.values()) || 1, out = new Map();
    for (const [id, s] of lexical) out.set(id, (1 - this.weight) * s);
    for (const [id, s] of scores) out.set(id, (out.get(id) || 0) + (this.weight * s) / top);
    for (const [id, s] of out) out.set(id, s ** this.sharpness);
    return out;
  }
}

// --------------------------------------------------------- memories as images
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

/** One payload per chunk of each document in `documents`, in order: { id: text }, or [[id, text, args?], ...] when
 *  ids repeat. A Hippocampus memory packs as hippocampus.toJSON().map((e) => [e.label, e.text, e.arguments]); an
 *  entry with tool arguments is kept whole (one frame), since the arguments belong to its full text. */
export async function pack(documents, { chunkSize = 500, overlap = 50 } = {}) {
  const payloads = [];
  for (const [id, text, args] of Array.isArray(documents) ? documents : Object.entries(documents)) {
    for (const piece of args ? [text.trim()] : chunk(text, chunkSize, overlap)) {
      const bytes = await squeeze(new TextEncoder().encode(JSON.stringify([TAG, id, piece, ...(args ? [args] : [])])), new CompressionStream("deflate"));
      if (bytes.length > MAX_PAYLOAD) throw new Error(`a chunk of "${id}" compresses to ${bytes.length} bytes, over one QR code's ${MAX_PAYLOAD}; lower chunkSize`);
      payloads.push(bytes);
    }
  }
  return payloads;
}

/** Rebuild the knowledge base from payloads (null = a frame that could not be read). Returns { memory, lost }:
 *  memory is a Hippocampus over the chunks (label = document id), mixed with `model` if given (e.g. ARC 1, so the
 *  frames teach it); the other options go to Hippocampus, e.g. { k: 5, occipital: new Occipital() }. lost counts
 *  payloads that were unreadable or not Occipital frames. */
export async function unpack(payloads, { model = null, ...options } = {}) {
  const memory = new Hippocampus(model, options);
  let lost = 0;
  for (const bytes of payloads) {
    try {
      const [tag, id, text, args] = JSON.parse(new TextDecoder().decode(await squeeze(bytes, new DecompressionStream("deflate"))));
      if (tag !== TAG) throw new Error("not an Occipital frame");
      memory.remember(text, id, args ?? null);
    } catch {
      lost++;
    }
  }
  return { memory, lost };
}
