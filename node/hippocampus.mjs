// Hippocampus in JS: a fast-learning memory of decided examples for System 1 models. No server, no Python.
// Port of gpbacay_arcane/hippocampus.py, with the BM25 part of got.py's DocumentGraph as its search: a memory
// graph has no semantic links and one node per example, so its graph search reduces to normalized BM25.
//
// remember(text, label) stores a decided example; at request time the examples most like the input vote for
// their labels (weighted by search score), and the vote is mixed with the model's own probabilities. Tool
// examples can carry their arguments: a remembered request becomes a pattern ("rate {title} {stars} stars")
// that copies arguments from a new request's words. `model` is ARC 1 from gpbacay-arcane/web, anything with
// classify(text, labels, opts) -> { distribution } or run(prompt, tools, { toolPrior }) -> { function_calls },
// a function (text, labels, opts) -> { label: probability }, or null for memory-only decisions.
import { WORD_NUMBERS, coerceValue, normalizeTool, validateCalls } from "./tools.mjs";

// ------------------------------------------------------------------- search
const W = "[\\p{L}\\p{N}_]", NW = "[^\\p{L}\\p{N}_]"; // Python's Unicode \w and \W
const WB = `(?:(?<=${W})(?!${W})|(?<!${W})(?=${W}))`; // JS \b is ASCII-only, even with the u flag
const SPLIT = new RegExp(`${NW}+|_+`, "u"), WORDS = new RegExp(`${W}+`, "gu");
const STOPWORDS = new Set(
  ("a an and are as at be been but by can could did do does doing for from had has have how i if in into is it its " +
  "me my no not of on or our should so than that the their them then there these they this those to too was we " +
  "were what when where which while who whom why will with would you your about after all also any before between " +
  "both each few more most other over same some such only own under until up very just tell show give get use using").split(" "));
const SUFFIXES = ["ations", "ation", "ments", "ment", "ings", "ing", "ed"];
const K1 = 1.2, B = 0.75;
const AGREE = 3; // neighbors that must back a label before the memory alone counts as sure (see _decisive)

function stem(w) {
  if (w.length > 4 && w.endsWith("ies")) return w.slice(0, -3) + "y";
  for (const s of SUFFIXES) if (w.endsWith(s) && w.length - s.length >= 4) return w.slice(0, -s.length);
  if (w.length > 3 && w.endsWith("s") && !/(?:ss|us|is)$/.test(w)) w = w.slice(0, -1);
  return w.length > 4 && w.endsWith("e") ? w.slice(0, -1) : w;
}

const tokenize = (text) => text.replace(/\]\([^)]*\)/g, "]").toLowerCase().split(SPLIT)
  .filter((w) => w.length > 1 && !STOPWORDS.has(w)).map(stem);

// ----------------------------------------------------------------- patterns
// ponytail: English clause boundaries; a value containing "and" ("Tom and Jerry") is cut where a pattern has an open end
const JOINER = `(?:and|then|plus|also)${WB}`;
const CLAUSE_END = `(?=\\s*(?:$|[.,;!?\\n]|${JOINER}))`;
const CLAUSE_START = `(?:^|[.,;!?\\n]|${WB}${JOINER})${NW}*`;
const OPEN_VALUE = `(?=${W})(?:(?!${WB}${JOINER})[^.,;!?\\n])+?`; // a value at an open end holds no joiner
const NUMBER = "-?\\d+(?:[.,]\\d+)*|" + Object.keys(WORD_NUMBERS).sort((a, b) => b.length - a.length).join("|");
const NUMERIC = ["integer", "int", "number", "float"];
const esc = (s) => s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
const norm = (text) => text.toLowerCase().split(/\s+/).filter(Boolean).join(" "); // an example's id: case and spacing don't matter

/** { re, keys, fixed } for a remembered request, or null if it has no words. Argument values found in the text
 *  become slots typed by their parameter; values not in it (booleans, presets) are fixed by the pattern's words. */
function template(text, args, params) {
  const spans = [], fixed = {};
  for (const [key, value] of Object.entries(args)) {
    if (!params.has(key)) continue;
    const m = typeof value === "boolean" ? null : new RegExp(`(?<!${W})${esc(String(value))}(?!${W})`, "iu").exec(text);
    if (m && !spans.some(([s, e]) => m.index < e && s < m.index + m[0].length)) spans.push([m.index, m.index + m[0].length, key]);
    else fixed[key] = value;
  }
  spans.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const words = (s) => (s.match(WORDS) || []).map(esc);
  const items = [], keys = [];
  let pos = 0;
  for (const [s, e, key] of spans) {
    items.push(...words(text.slice(pos, s)), null);
    keys.push(key);
    pos = e;
  }
  items.push(...words(text.slice(pos)));
  if (!items.length) return null;
  let slot = 0;
  const last = items.length - 1;
  const parts = items.map((item, n) => {
    if (item !== null) return item;
    const p = params.get(keys[slot++]);
    if (p.enum?.length) return `(${p.enum.map(String).sort((a, b) => b.length - a.length).map(esc).join("|")})`;
    if (NUMERIC.includes(p.type.toLowerCase())) return `(${NUMBER})`;
    if (n === 0 || n === last) return `(${OPEN_VALUE}${n === last ? CLAUSE_END : ""})`;
    return "([^.,;!?\\n]+?)";
  });
  const head = items[0] === null ? CLAUSE_START : WB, tail = items[last] !== null ? WB : "";
  return { re: new RegExp(head + parts.join(`${NW}+`) + tail, "iu"), keys, fixed };
}

// --------------------------------------------------------------- hippocampus
export class Hippocampus {
  /**
   * @param {object|function|null} model see the header; null decides from memory alone.
   * @param {{ k?: number, weight?: number, memory?: object[]|object, maxSize?: number|null, confident?: number,
   *   threshold?: number, leak?: number, maxPending?: number, teacher?: function|null }} [options]
   *   teacher: async (text, labels) => label | { label: probability } | null, a stronger model that classify() asks
   *   when its own decision is under `confident`; its trusted answer is returned and remembered (see classify).
   *   k: neighbors that vote. weight: share of a classify distribution given to the vote when any neighbor is found.
   *   maxSize: cap on remembered examples; observe() evicts its own oldest writes past it (remembered ones never go).
   *   confident: observe() skips an example the memory and model already decide this confidently.
   *   threshold, leak, maxPending: observe() commits a candidate once its accumulated trust reaches threshold; leak is
   *   the share of every pending candidate's evidence lost per observe() call (0 = kept until cancelled or pushed
   *   out; about 1 / the calls between repeats of a text makes old evidence count for less); the stalest of
   *   maxPending candidates is dropped past it.
   *   memory: examples to start with: toJSON() output ([{ text, label, arguments }], the same shape as the ARC 1
   *   server's `memory` field) or a Python graph.to_json() (graph-of-thought v2).
   *   occipital: an Occipital (occipital.mjs) of its own, which also recalls examples by their letters, so typos and
   *   other word forms find their neighbors. null (default) searches words only.
   */
  constructor(model = null, { k = 8, weight = 0.5, memory = [], maxSize = null, confident = 0.8, threshold = 1, leak = 0, maxPending = 10000, teacher = null, occipital = null } = {}) {
    Object.assign(this, { model, k, weight, maxSize, confident, threshold, leak, maxPending, teacher, occipital });
    this._examples = new Map(); // norm(text) -> { text, label, arguments, source, tf, len }
    this._postings = new Map(); // term -> Map(norm(text) -> term count)
    this._labels = new Map(); // label -> number of examples
    this._pending = new Map(); // norm(text) -> Map(label -> accumulated trust)
    let examples = memory;
    if (!Array.isArray(memory)) {
      if (memory.format !== "graph-of-thought" || memory.version !== 2) throw new Error("memory must be an array of examples or a graph-of-thought v2 graph");
      const content = new Map(memory.nodes.map((n) => [n.nodeId, n.content]));
      examples = memory.documents.filter((d) => d.docId.startsWith("memory-"))
        .map((d) => ({ text: content.get(d.rootId), label: d.description || null, arguments: d.arguments, source: d.source }));
    }
    for (const ex of examples) this._store(ex.text, ex.label, ex.arguments, ex.source ?? "manual");
  }

  // ----------------------------------------------------------------- memory
  /** Store `text` decided as `label` (a class label or a tool name; null = no tool applies). `args` are that tool
   *  call's arguments; values that appear in `text` teach where each argument sits. Remembering the same text
   *  again replaces it, so corrections overwrite mistakes. */
  remember(text, label, args = null) {
    return this._store(text, label, args, "manual");
  }

  _store(text, label, args, source) {
    if (!String(text ?? "").trim()) throw new Error("remember() needs non-empty text");
    const id = norm(text);
    this._remove(id);
    // ponytail: one node per example; Python splits markdown headings and >2000-char texts into sections
    const tf = new Map();
    for (const t of tokenize(text.trim())) tf.set(t, (tf.get(t) || 0) + 1);
    this._examples.set(id, { text: text.trim(), label: label || null, arguments: args ? { ...args } : undefined, source, tf, len: [...tf.values()].reduce((a, b) => a + b, 0) });
    this._labels.set(label || null, (this._labels.get(label || null) || 0) + 1);
    this.occipital?.add(id, text.trim());
    for (const [t, c] of tf) {
      if (!this._postings.has(t)) this._postings.set(t, new Map());
      this._postings.get(t).set(id, c);
    }
    return id;
  }

  /** Learn from an outcome: `label` is what turned out right for `text` (a user correction, a tool that ran, a
   *  ground truth). Returns the new example's id, or null when nothing was stored.
   *
   *  Stores only what the memory (with its model) does not already decide confidently, so it grows where the model
   *  is weak. `trust` (0..1+) is the outcome's reliability; evidence for the same text and label accumulates, decays
   *  by `leak` per call, is cancelled by evidence for another label, and commits at `threshold`, so a noisy signal
   *  must repeat before it becomes memory. The default trust of 1 commits at once. Past `maxSize` the oldest
   *  observed example is forgotten. An example you remember()ed is never overwritten here (null is returned).
   *  `decisionLabels` is the decision space the "already decided" judgment is made
   *  over (default: every label in the memory); give it when the memory holds examples for other decisions.
   *  `opts` go to the model, as in classify(). */
  async observe(text, label, args = null, trust = 1, decisionLabels = null, opts = {}) {
    label = label || null;
    if (this._examples.get(norm(text))?.source === "manual") return null; // only remember()/forget() change a remembered example
    const space = decisionLabels ? [...new Set(decisionLabels.map(String))] : [...this._labels.keys()].filter(Boolean);
    if (label && !space.includes(label)) space.push(label);
    const seen = label === null || space.length < 2 // a model's choice among fewer than two labels says nothing
      ? this._vote(text, decisionLabels ? space : null).votes.get(label) ?? 0
      : (await this._decide(text, space, opts)).distribution[label] ?? 0;
    if (this.leak) {
      for (const [id, cands] of this._pending) {
        for (const [lab, v] of cands) v * (1 - this.leak) > 0.01 ? cands.set(lab, v * (1 - this.leak)) : cands.delete(lab);
        if (!cands.size) this._pending.delete(id);
      }
    }
    if (seen >= this.confident) return null;
    const id = norm(text);
    if (!this._pending.has(id) && this._pending.size >= this.maxPending) this._pending.delete(this._pending.keys().next().value); // the stalest
    if (!this._pending.has(id)) this._pending.set(id, new Map());
    const cands = this._pending.get(id);
    for (const [lab, v] of [...cands]) {
      if (lab === label) continue;
      if (v - trust <= 0.01) cands.delete(lab); // contradicting evidence
      else cands.set(lab, v - trust);
    }
    cands.set(label, (cands.get(label) ?? 0) + trust);
    if (cands.get(label) < this.threshold) return null;
    this._store(text, label, args, "auto"); // also clears the pending candidates for this text
    this._evict(id);
    return id;
  }

  _evict(keep) {
    // ponytail: O(memory) scan per eviction; keep a queue of auto ids if maxSize gets large
    while (this.maxSize && this._examples.size > this.maxSize) {
      const victim = [...this._examples].find(([id, ex]) => id !== keep && ex.source === "auto")?.[0];
      if (victim === undefined) return; // only remembered examples left: they are never evicted
      this._remove(victim);
    }
  }

  forget(text) {
    return this._remove(norm(text));
  }

  _remove(id) {
    this._pending.delete(id);
    const ex = this._examples.get(id);
    if (!ex) return false;
    for (const t of ex.tf.keys()) {
      const post = this._postings.get(t);
      post.delete(id);
      if (!post.size) this._postings.delete(t);
    }
    const n = this._labels.get(ex.label) - 1;
    if (n) this._labels.set(ex.label, n);
    else this._labels.delete(ex.label);
    this.occipital?.remove(id);
    return this._examples.delete(id);
  }

  /** Every remembered example as [{ text, label, arguments, source }] (arguments only for tools, source only for
   *  observed examples); JSON.stringify(hippocampus) saves the memory. */
  toJSON() {
    return [...this._examples.values()].map(({ text, label, arguments: a, source }) =>
      ({ text, label, ...(a && { arguments: a }), ...(source === "auto" && { source }) }));
  }

  // -------------------------------------------------------------- retrieval
  _search(text, labels) {
    const allowed = labels ? new Set(labels.map((l) => l ?? null)) : null;
    const n = this._examples.size;
    let scores = new Map(); // id -> score
    if (!n) return [];
    let avg = 0;
    for (const ex of this._examples.values()) avg += ex.len / n;
    for (const term of new Set(tokenize(text))) {
      const post = this._postings.get(term);
      if (!post) continue;
      const idf = Math.log(1 + (n - post.size + 0.5) / (post.size + 0.5));
      for (const [key, tf] of post) {
        const ex = this._examples.get(key);
        if (allowed && !allowed.has(ex.label)) continue;
        scores.set(key, (scores.get(key) || 0) + (idf * tf * (K1 + 1)) / (tf + K1 * (1 - B + (B * ex.len) / avg)));
      }
    }
    const top = Math.max(...scores.values());
    for (const [key, s] of scores) scores.set(key, s / top);
    if (this.occipital) scores = this.occipital.fuse(text, scores, (key) => !allowed || allowed.has(this._examples.get(key).label));
    return [...scores].map(([key, score]) => ({ ex: this._examples.get(key), score })).sort((a, b) => b.score - a.score).slice(0, this.k);
  }

  /** Remembered examples most like `text` (only those labeled one of `labels`, if given): [{ text, label, score }]. */
  neighbors(text, labels = null) {
    return this._search(text, labels).map(({ ex, score }) => ({ text: ex.text, label: ex.label, score }));
  }

  _vote(text, labels) {
    const hits = this.neighbors(text, labels), total = hits.reduce((s, h) => s + h.score, 0), votes = new Map();
    for (const h of hits) votes.set(h.label, (votes.get(h.label) || 0) + h.score / total);
    return { votes, neighbors: hits };
  }

  /** The memory's own decision, no model involved: `votes` (label -> share of the neighbors' score, summing to 1;
   *  the no-tool label null shows up as "null") and the `neighbors` that cast them. Empty when nothing similar is stored. */
  react(text, labels = null) {
    const { votes, neighbors } = this._vote(text, labels);
    return { votes: Object.fromEntries(votes), neighbors };
  }

  /** Arguments for `tool` copied from `prompt` through the most similar remembered request whose pattern matches it; null when none matches. */
  arguments(prompt, tool) {
    return this._match(prompt, normalizeTool(tool))?.[0] ?? null;
  }

  /** [arguments, [start, end] of the matched words in prompt] or null. */
  _match(prompt, tool) {
    const params = new Map(tool.parameters.map((p) => [p.name, p]));
    hits: for (const { ex } of this._search(prompt, [tool.name])) {
      const built = ex.arguments && template(ex.text, ex.arguments, params);
      const m = built && built.re.exec(prompt);
      if (!m) continue;
      const args = { ...built.fixed };
      for (const [i, key] of built.keys.entries()) {
        const [ok, value] = coerceValue(params.get(key).type, m[i + 1].trim());
        if (!ok) continue hits;
        args[key] = value;
      }
      return [args, [m.index, m.index + m[0].length]];
    }
    return null;
  }

  // -------------------------------------------------------------- decisions
  /** Pick one of `labels`: the model's distribution mixed with the memory's vote. Returns the model's own fields
   *  (if any) with label, confidence, distribution, source ("model", "memory" or "model+memory") and neighbors.
   *  With no model and no similar example, label is null: Hippocampus abstains rather than guess.
   *
   *  With a `teacher`, a decision under `confident` is put to it: a trusted answer becomes the result
   *  (source "teacher") and is remembered as an observed example, so the next similar request is decided without
   *  it. If the teacher throws, the memory's own decision stands, with `teacher_error`. A text you remember()ed is
   *  never put to the teacher. */
  async classify(text, labels, opts = {}) {
    labels = [...new Set(labels.map(String))];
    const out = await this._decide(text, labels, opts);
    const id = norm(text);
    if (!this.teacher || out.confidence >= this.confident || this._decisive(out) || this._examples.get(id)?.source === "manual") return out;
    let taught;
    try {
      taught = await this._askTeacher(text, labels);
    } catch (error) {
      return { ...out, teacher_error: `${error?.name ?? "Error"}: ${error?.message ?? error}` };
    }
    if (!taught) return out;
    const [label, distribution] = taught;
    this._store(text, label, null, "auto");
    this._evict(id);
    return { ...out, label, confidence: distribution[label], distribution, source: "teacher" };
  }

  /** The memory alone is sure: at least AGREE neighbors back the decided label and hold `confident` of the vote. The
   *  blended confidence can't say so, since it is capped by the model's own, weak, probability. */
  _decisive(out) {
    const total = out.neighbors.reduce((sum, h) => sum + h.score, 0);
    const backing = out.neighbors.filter((h) => h.label === out.label).map((h) => h.score);
    return backing.length >= AGREE && backing.reduce((a, b) => a + b, 0) >= this.confident * total;
  }

  /** [label, distribution] from the teacher when it is confident about one of `labels`, else null. */
  async _askTeacher(text, labels) {
    const answer = await this.teacher(text, labels);
    if (answer == null) return null;
    const given = typeof answer === "string" ? { [answer]: 1 } : answer;
    const total = labels.reduce((sum, l) => sum + (Number(given[l]) || 0), 0);
    if (!(total > 0)) return null;
    const distribution = Object.fromEntries(labels.map((l) => [l, (Number(given[l]) || 0) / total]));
    const label = labels.reduce((best, l) => (distribution[l] > distribution[best] ? l : best), labels[0]);
    return distribution[label] >= this.confident ? [label, distribution] : null;
  }

  async _decide(text, labels, opts) {
    let out = {}, modelDist = {};
    if (typeof this.model?.classify === "function") {
      out = { ...(await this.model.classify(text, labels, opts)) };
      modelDist = { ...out.distribution };
    } else if (typeof this.model === "function") modelDist = { ...(await this.model(text, labels, opts)) };
    const { votes, neighbors } = this._vote(text, labels), w = this.weight;
    let source = "model", score = (l) => modelDist[l] ?? 0;
    if (Object.keys(modelDist).length && votes.size) (source = "model+memory"), (score = (l) => (1 - w) * (modelDist[l] ?? 0) + w * (votes.get(l) ?? 0));
    else if (votes.size) (source = "memory"), (score = (l) => votes.get(l) ?? 0);
    const ranked = labels.map((l) => [l, score(l)]).sort((a, b) => b[1] - a[1]);
    const [label, confidence] = ranked.length && ranked[0][1] > 0 ? ranked[0] : [null, 0];
    return Object.assign(out, { label, confidence, distribution: Object.fromEntries(ranked), source, neighbors });
  }

  /** Tool calling: memory votes fire tools and remembered requests fill their arguments.
   *  The votes go to model.run as opts.toolPrior. Every example votes, including ones for tools not offered and
   *  null (no tool), so a request that resembles something else does not fire an offered tool. Then, for each
   *  offered tool, a remembered request whose pattern matches supplies the arguments it covers (over the model's)
   *  and adds the call if the model held it back; confidence is then null. Words a pattern matched belong to that
   *  call, so another model call whose argument copies them is dropped. memory_arguments names the tools whose
   *  arguments came from memory. Tools are not executed. For memory-only routing use react(). */
  async run(prompt, tools = [], opts = {}) {
    if (typeof this.model?.run !== "function") throw new TypeError("run() needs a model with run(prompt, tools, { toolPrior }); use react() for memory-only routing");
    const { votes, neighbors } = this._vote(prompt);
    const prior = Object.fromEntries([...votes].filter(([l]) => l));
    const out = { ...(await this.model.run(prompt, tools, { ...opts, toolPrior: Object.keys(prior).length ? prior : undefined })) };
    const offered = tools.map(normalizeTool);
    const calls = new Map(out.function_calls.map((c) => [c.name, { name: c.name, arguments: { ...c.arguments } }]));
    const filled = [], claimed = [];
    for (const tool of offered) {
      const found = this._match(prompt, tool);
      if (!found) continue;
      if (!calls.has(tool.name)) calls.set(tool.name, { name: tool.name, arguments: {} });
      Object.assign(calls.get(tool.name).arguments, found[0]);
      out.confidence = null; // the model's calibrated probability no longer describes this call
      filled.push(tool.name);
      claimed.push(found[1]);
    }
    const lower = prompt.toLowerCase();
    const overlapsClaimed = (v) => {
      const i = typeof v === "string" && v.trim() ? lower.indexOf(v.toLowerCase()) : -1;
      return i >= 0 && claimed.some(([s, e]) => i < e && s < i + v.length);
    };
    const kept = [...calls.values()].filter((c) => filled.includes(c.name) || !Object.values(c.arguments).some(overlapsClaimed));
    out.function_calls = validateCalls(kept, offered);
    out.results = [];
    if (filled.length) out.source = "model+memory";
    return Object.assign(out, { memory_arguments: filled, neighbors });
  }
}
