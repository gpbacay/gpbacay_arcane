// ARC 1 in the browser (or any JS runtime) on onnxruntime-web. No server, no Python.
// Port of gpbacay_arcane/tools.py Arc1Agent + arc1_codec.py + tokenization.py.
// Model files come from examples/export_arc1_onnx.py (web/arc1.onnx, web/arc1.json).
import * as ort from "onnxruntime-web";

const PAD = 0, BOS = 3, BYTE_OFFSET = 4, BASE_VOCAB = 260, NEG = -1e9;
const ROLE_TOOL = 0, ROLE_SPAN = 1, ROLE_BOOL = 2, ROLE_ENUM = 3, ROLE_OPTION = 4;
const EXTRACT_TOOL_NAME = "extract_record";
const EXTRACT_TOOL_DESCRIPTION = "Extract a typed record from the passage.";
const CLASSIFY_TASK = "Classify the text into one of the labels.";
const CLASSIFY_LABEL_DESCRIPTION = "The label that best fits the text";
const utf8 = new TextEncoder();
const unutf8 = new TextDecoder();

// ------------------------------------------------------------------ tokenizer
function makeTokenizer(merges) {
  const ranks = new Map(merges.map(([a, b], i) => [a * 65536 + b, i]));
  const width = (tok) => (tok < BASE_VOCAB ? 1 : width(merges[tok - BASE_VOCAB][0]) + width(merges[tok - BASE_VOCAB][1]));
  const encode = (text) => {
    let ids = Array.from(utf8.encode(text), (b) => b + BYTE_OFFSET);
    for (;;) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < bestRank) (bestRank = r), (best = i);
      }
      if (best < 0) return ids;
      ids.splice(best, 2, BASE_VOCAB + bestRank);
    }
  };
  return { encode, width };
}

// ---------------------------------------------------------------------- codec
const humanize = (name) => String(name).replaceAll("_", " ").replaceAll("-", " ").trim();
const toolText = (name, desc) => `${humanize(name)}. ${desc}`.trim();
const paramText = (name, type, desc, required) => `${humanize(name)} (${type || "string"}${required ? "" : ", optional"}). ${desc}`.trim();
const optionText = (value, desc) => (desc && desc.trim() ? `${humanize(value)}: ${desc.trim()}` : humanize(value));
const roleFor = (type, en) => (en && en.length ? ROLE_ENUM : ["boolean", "bool"].includes((type || "string").toLowerCase()) ? ROLE_BOOL : ROLE_SPAN);
const stripChars = (s, chars) => {
  let a = 0, b = s.length;
  while (a < b && chars.includes(s[a])) a++;
  while (b > a && chars.includes(s[b - 1])) b--;
  return s.slice(a, b);
};
const trimSurface = (s) => stripChars(stripChars(s.trim(), "\"'"), " .,!?;:");
const decodeBytes = (bytes) => unutf8.decode(bytes).replaceAll("�", ""); // Python errors="ignore"

function padBatch(seqs, maxLen) {
  const longest = Math.max(1, ...seqs.map((s) => s.length));
  const width = Math.min(Math.max(Math.ceil(longest / 8) * 8, 8), maxLen);
  const out = new Int32Array(seqs.length * width).fill(PAD);
  seqs.forEach((s, i) => out.set(s.slice(0, width), i * width));
  return new ort.Tensor("int32", out, [seqs.length, width]);
}

function bestSpan(start, end, lo, hi, t, maxWidth = 64) {
  const soft = (x) => {
    const z = x.slice(lo, hi).map((v) => v / t), m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  };
  const ps = soft(start), pe = soft(end), n = hi - lo;
  let bi = 0, bj = 0, bp = -1;
  for (let i = 0; i < n; i++)
    for (let j = i; j < n && j - i <= maxWidth; j++) if (ps[i] * pe[j] > bp) (bp = ps[i] * pe[j]), (bi = i), (bj = j);
  return [lo + bi, lo + bj, bp];
}

// ----------------------------------------------------------------- decoding
const WORD_NUMBERS = { zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  eleven: 11, twelve: 12, fifteen: 15, twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, hundred: 100, a: 1, an: 1 };
const roundHalfEven = (x) => (Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x));

function coerceValue(type, text) {
  type = (type || "string").toLowerCase();
  if (!["integer", "int", "number", "float"].includes(type)) return [Boolean(text), text];
  const m = text.match(/-?\d+(?:[.,]\d+)*/);
  const word = text.trim().toLowerCase();
  const num = m ? Number(m[0].replaceAll(",", "")) : word in WORD_NUMBERS ? WORD_NUMBERS[word] : NaN;
  if (Number.isNaN(num)) return [false, null];
  return [true, type === "integer" || type === "int" ? roundHalfEven(num) : num];
}

const sigmoid = (x, t) => 1 / (1 + Math.exp(-x / t));
const softmax = (xs, t) => {
  const z = xs.map((x) => x / t), m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};
const repr = (v) => (typeof v === "string" ? `'${v}'` : String(v));

function normalizeTool(t) {
  return {
    name: t.name, description: t.description || "",
    parameters: (t.parameters || []).map((p) => ({
      name: p.name, type: p.type || "string", description: p.description || "",
      required: p.required ?? true, enum: p.enum || null, enum_descriptions: p.enum_descriptions || null,
    })),
  };
}

function validateCalls(calls, tools) {
  const byName = new Map(tools.map((t) => [t.name, t]));
  return calls.flatMap((call) => {
    const spec = byName.get(call.name || "");
    if (!spec) return [];
    const allowed = new Map(spec.parameters.map((p) => [p.name, p]));
    const cleaned = {};
    for (const [k, v] of Object.entries(call.arguments || {})) {
      const p = allowed.get(k);
      if (p && !(p.enum && !p.enum.includes(String(v)))) cleaned[k] = v;
    }
    return spec.parameters.some((p) => p.required && !(p.name in cleaned)) ? [] : [{ name: spec.name, arguments: cleaned }];
  });
}

// ---------------------------------------------------------------------- load
const DEFAULT_MODEL = new URL("./web/arc1.onnx", import.meta.url);
const DEFAULT_CONFIG = new URL("./web/arc1.json", import.meta.url);
// Node's fetch() can't read file: URLs (the defaults outside a bundler), so read those from disk.
const isFile = (src) => String(src instanceof URL ? src.href : src).startsWith("file:");
const readFileUrl = async (src) => (await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ /* @vite-ignore */ "node:fs/promises")).readFile(new URL(src));

/**
 * Load ARC 1 in the browser.
 * @param {{ model?: string|URL|Uint8Array, config?: string|URL|object, toolThreshold?: number,
 *           presenceThreshold?: number, sessionOptions?: object, wasmPaths?: string }} [options]
 *   model / config: URL of arc1.onnx / arc1.json (or their contents). Default: the files shipped in this package.
 *   wasmPaths: where onnxruntime-web's .wasm files live (default in browsers: jsDelivr).
 */
export async function load(options = {}) {
  // Bundlers (Next.js, Vite) don't emit ort's .wasm runtime; fetch it from the CDN unless told otherwise.
  if (options.wasmPaths) ort.env.wasm.wasmPaths = options.wasmPaths;
  else if (typeof window !== "undefined" && !ort.env.wasm.wasmPaths)
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
  const cfgSrc = options.config ?? DEFAULT_CONFIG;
  const meta = typeof cfgSrc === "object" && !(cfgSrc instanceof URL) ? cfgSrc
    : isFile(cfgSrc) ? JSON.parse(await readFileUrl(cfgSrc)) : await (await fetch(cfgSrc)).json();
  let modelSrc = options.model ?? DEFAULT_MODEL;
  if (isFile(modelSrc)) modelSrc = await readFileUrl(modelSrc);
  const session = await ort.InferenceSession.create(modelSrc instanceof URL ? modelSrc.href : modelSrc, options.sessionOptions);
  const cfg = meta.config, d = cfg.d_model, cycles = meta.cycles;
  const temp = (k) => cfg.calibration?.[k] ?? 1;
  const tok = makeTokenizer(meta.tokenizer.merges);
  const toolThreshold = options.toolThreshold ?? 0.5, presenceThreshold = options.presenceThreshold ?? 0.5;
  const memory = new Map(); // schema text -> engram; ponytail: unbounded, add LRU if schemas churn

  const utterance = (text) => {
    const ids = tok.encode(text).slice(0, Math.max(cfg.seq_len - 1, 1));
    let pos = 0;
    const offsets = ids.map((t) => [pos, (pos += tok.width(t))]);
    return { ids: [BOS, ...ids], offsets, raw: utf8.encode(text).slice(0, pos), lo: 1, hi: ids.length + 1 };
  };
  const schemaIds = (text) => [BOS, ...tok.encode(text).slice(0, cfg.schema_len - 1)];
  const piece = (utt, s, e) => [utt.offsets[s - 1][0], utt.offsets[e - 1][1]];
  const tokensToText = (utt, s, e) => trimSurface(decodeBytes(utt.raw.slice(...piece(utt, s, e))));
  const tokensToCharSpan = (utt, s, e) => {
    const [a, b] = piece(utt, s, e), p = decodeBytes(utt.raw.slice(a, b)), core = trimSurface(p);
    const c0 = decodeBytes(utt.raw.slice(0, a)).length + Math.max(p.indexOf(core), 0);
    return [c0, c0 + core.length];
  };

  function plan(tools, withToolProbes = true) {
    const probes = [];
    for (const t of tools) {
      const tt = toolText(t.name, t.description);
      if (withToolProbes) probes.push({ role: ROLE_TOOL, a: tt, b: null, tool: t.name });
      for (const p of t.parameters) {
        const pt = paramText(p.name, p.type, p.description, p.required);
        probes.push({ role: roleFor(p.type, p.enum), a: pt, b: tt, tool: t.name, param: p });
        for (const opt of p.enum || [])
          probes.push({ role: ROLE_OPTION, a: optionText(opt, (p.enum_descriptions || {})[String(opt)]), b: pt, tool: t.name, param: p, option: String(opt) });
      }
    }
    return probes;
  }

  // One forward pass: perceive text + uncached schema texts, bind every probe.
  async function bind(text, probes) {
    const utt = utterance(text);
    const needed = [...new Set([...probes.map((p) => p.a), ...probes.filter((p) => p.b).map((p) => p.b)])];
    const cached = needed.filter((t) => memory.has(t)), missing = needed.filter((t) => !memory.has(t));
    const index = new Map([...cached, ...missing].map((t, i) => [t, i]));
    const bank = new Float32Array(cached.length * d);
    cached.forEach((t, i) => bank.set(memory.get(t), i * d));
    const i32 = (xs) => new ort.Tensor("int32", Int32Array.from(xs), [xs.length]);
    const r = await session.run({
      ids: padBatch([utt.ids, ...missing.map(schemaIds)], cfg.seq_len),
      bank: new ort.Tensor("float32", bank, [cached.length, d]),
      probe_a: i32(probes.map((p) => index.get(p.a))),
      probe_b: i32(probes.map((p) => (p.b ? index.get(p.b) : -1))),
      roles: i32(probes.map((p) => p.role)),
    });
    const rows = (t) => Array.from({ length: t.dims[0] }, (_, i) => t.data.subarray(i * t.dims[1], (i + 1) * t.dims[1]));
    missing.forEach((t, i) => memory.set(t, r.new_engrams.data.slice(i * d, (i + 1) * d)));
    const out = { fire: Array.from(r.fire.data), anchor_start: rows(r.anchor_start), anchor_end: rows(r.anchor_end),
      select_q: rows(r.select_q), select_k: rows(r.select_k), embedding: Array.from(r.embedding.data) };
    const stats = { tokens: utt.ids.length, probes: probes.length, cycles, forward_passes: 1,
      schema_encoded: missing.length, schema_cached: cached.length };
    return { utt, out, stats };
  }

  function anchor(utt, i, out, type, blocked) {
    let start = out.anchor_start[i], end = out.anchor_end[i];
    if (blocked) {
      start = start.map((v, k) => (blocked[k] ? NEG : v));
      end = end.map((v, k) => (blocked[k] ? NEG : v));
      if (blocked.slice(utt.lo, utt.hi).every(Boolean)) return { ok: false, value: null, p: 0 };
    }
    const [s, e, p] = bestSpan(start, end, utt.lo, utt.hi, temp("anchor"));
    const surface = tokensToText(utt, s, e);
    const [ok, value] = coerceValue(type, surface);
    return { ok, value, surface, tokens: [s, e], span: tokensToCharSpan(utt, s, e), p: ok ? p : 0 };
  }

  function readArguments(utt, probes, out, pFire) {
    const options = new Map();
    probes.forEach((p, i) => {
      if (p.role !== ROLE_OPTION) return;
      const k = `${p.tool}\0${p.param.name}`;
      options.set(k, [...(options.get(k) || []), i]);
    });
    const decisions = [], anchored = [];
    probes.forEach((pr, i) => {
      if (![ROLE_SPAN, ROLE_BOOL, ROLE_ENUM].includes(pr.role)) return;
      const param = pr.param, type = (param.type || "string").toLowerCase();
      const pPresent = param.required ? 1 : pFire[i];
      const dec = { tool: pr.tool, param: param.name, p_present: pPresent };
      if (pr.role === ROLE_BOOL) {
        const pt = pFire[i];
        Object.assign(dec, { kind: "fire", present: true, value: pt >= 0.5, p: Math.max(pt, 1 - pt), p_true: pt, p_present: 1 });
      } else if (pr.role === ROLE_ENUM) {
        const q = out.select_q[i];
        const logits = (options.get(`${pr.tool}\0${param.name}`) || []).map((j) => out.select_k[j].reduce((s, k, n) => s + k * q[n], 0) / Math.sqrt(d));
        const probs = softmax(logits, temp("select"));
        const best = probs.indexOf(Math.max(...probs));
        Object.assign(dec, { kind: "select", present: pPresent >= presenceThreshold, value: String(param.enum[best]), p: probs[best],
          distribution: Object.fromEntries(param.enum.map((o, k) => [String(o), probs[k]])) });
      } else if (utt.hi <= utt.lo) {
        Object.assign(dec, { kind: "anchor", present: false, value: null, p: 0 });
      } else {
        const a = anchor(utt, i, out, type);
        Object.assign(dec, { kind: "anchor", present: a.ok && pPresent >= presenceThreshold, value: a.value, surface: a.surface, span: a.span, p: a.p });
        anchored.push({ dec, i, type, required: param.required, tokens: a.tokens });
      }
      decisions.push(dec);
    });
    // Within one call a token belongs to at most one argument: strongest anchors claim first.
    const claimed = new Map(), width = out.anchor_start[0]?.length ?? 0;
    const live = anchored.filter((x) => x.dec.present && x.tokens).sort((x, y) => y.dec.p * y.dec.p_present - x.dec.p * x.dec.p_present);
    for (const { dec, i, type, required, tokens } of live) {
      if (!claimed.has(dec.tool)) claimed.set(dec.tool, new Array(width).fill(false));
      const taken = claimed.get(dec.tool);
      let [s, e] = tokens;
      if (taken.slice(s, e + 1).some(Boolean)) {
        if (!required) {
          Object.assign(dec, { present: false, note: "span already anchored by a stronger argument" });
          continue;
        }
        const a = anchor(utt, i, out, type, taken);
        Object.assign(dec, { present: a.ok, value: a.value, surface: a.surface, span: a.span, p: a.p, note: "re-anchored on unclaimed tokens" });
        if (!a.ok) continue;
        [s, e] = a.tokens;
      }
      taken.fill(true, s, e + 1);
    }
    return decisions;
  }

  const pick = (out, keep) => Object.fromEntries(Object.entries(out).map(([k, v]) => [k, k === "embedding" ? v : v.filter((_, i) => keep[i])]));

  return {
    /** Pick a tool and fill its arguments. tools: [{ name, description, parameters: [{ name, type, description, required, enum }] }] */
    async run(prompt, tools = []) {
      const t0 = performance.now();
      const active = tools.map(normalizeTool), probes = plan(active);
      let toolProbs = {}, decisions = [], stats = { tokens: 0, probes: 0, cycles };
      if (probes.length) {
        const b = await bind(prompt, probes);
        stats = b.stats;
        const pFire = b.out.fire.map((x) => sigmoid(x, temp("fire")));
        probes.forEach((p, i) => p.role === ROLE_TOOL && (toolProbs[p.tool] = pFire[i]));
        const keep = probes.map((p) => toolProbs[p.tool] >= toolThreshold);
        if (keep.some(Boolean)) decisions = readArguments(b.utt, probes.filter((_, i) => keep[i]), pick(b.out, keep), pFire.filter((_, i) => keep[i]));
      }
      const selected = active.filter((t) => (toolProbs[t.name] ?? 0) >= toolThreshold).sort((a, b) => toolProbs[b.name] - toolProbs[a.name]);
      let calls = [];
      const confs = [], notes = [];
      for (const t of selected) {
        const params = new Map(t.parameters.map((p) => [p.name, p]));
        const args = {}, missing = [], parts = [`${t.name} fires (p=${toolProbs[t.name].toFixed(2)})`];
        let conf = toolProbs[t.name];
        for (const dd of decisions.filter((x) => x.tool === t.name)) {
          const required = params.get(dd.param).required;
          if (dd.present) {
            args[dd.param] = dd.value;
            conf *= dd.p * (required ? 1 : dd.p_present);
            parts.push(`${dd.param}=${repr(dd.value)} (p=${dd.p.toFixed(2)})`);
          } else if (required) missing.push(dd.param);
          else conf *= 1 - dd.p_present;
        }
        if (missing.length) {
          notes.push(`${t.name} held back: could not anchor required ${missing.join(", ")}`);
          continue;
        }
        calls.push({ name: t.name, arguments: args });
        confs.push(conf);
        notes.push(parts.join("; "));
      }
      calls = validateCalls(calls, active);
      let confidence;
      if (calls.length) confidence = Math.min(...confs);
      else {
        confidence = 1 - Math.max(0, ...Object.values(toolProbs));
        notes.push("no offered tool fires");
      }
      const payload = { tools: toolProbs, arguments: decisions };
      return { reasoning: notes.join(". ") + ".", function_calls: calls, results: [], confidence, raw: JSON.stringify(payload),
        decisions: payload, source: "model", cycles, stats, latency_ms: +(performance.now() - t0).toFixed(2) };
    },

    /** Pull typed fields out of text. schema: { field: { type, description, enum? } } or { field: "description" } */
    async extract(text, schema) {
      const t0 = performance.now();
      const params = Object.entries(schema).map(([key, s]) => {
        s = s && typeof s === "object" ? s : { type: "string", description: String(s || key) };
        return { name: key, type: String(s.type ?? "string"), description: String(s.description ?? key), required: false, enum: s.enum || null };
      });
      const probes = plan([normalizeTool({ name: EXTRACT_TOOL_NAME, description: EXTRACT_TOOL_DESCRIPTION, parameters: params })], false);
      let decisions = [], stats = { tokens: 0, probes: 0, cycles };
      if (probes.length) {
        const b = await bind(text, probes);
        stats = b.stats;
        decisions = readArguments(b.utt, probes, b.out, b.out.fire.map((x) => sigmoid(x, temp("fire"))));
      }
      const record = {}, confs = [], notes = [];
      for (const dd of decisions) {
        if (dd.present) {
          record[dd.param] = dd.value;
          confs.push(dd.p * dd.p_present);
          notes.push(`${dd.param}=${repr(dd.value)} (p=${dd.p.toFixed(2)})`);
        } else {
          confs.push(1 - dd.p_present);
          notes.push(`${dd.param} not found (p_absent=${(1 - dd.p_present).toFixed(2)})`);
        }
      }
      return { record, function_calls: Object.keys(record).length ? [{ name: EXTRACT_TOOL_NAME, arguments: record }] : [],
        confidence: confs.length ? Math.min(...confs) : 0, reasoning: notes.join("; ") + ".", raw: JSON.stringify(decisions),
        decisions, source: "model", cycles, stats, latency_ms: +(performance.now() - t0).toFixed(2) };
    },

    /** Pick one label. opts: { task?, descriptions? } */
    async classify(text, labels, opts = {}) {
      const t0 = performance.now();
      labels = [...new Set(labels.map(String))].filter((x) => x.trim());
      if (!labels.length) throw new Error("classify() needs at least one label");
      const spec = normalizeTool({ name: "classify", description: opts.task || CLASSIFY_TASK, parameters: [
        { name: "label", type: "string", description: CLASSIFY_LABEL_DESCRIPTION, required: true, enum: labels, enum_descriptions: opts.descriptions } ] });
      const probes = plan([spec], false);
      const { utt, out, stats } = await bind(text, probes);
      const dec = readArguments(utt, probes, out, out.fire.map((x) => sigmoid(x, temp("fire"))))[0];
      const ranked = Object.entries(dec.distribution).sort((a, b) => b[1] - a[1]);
      return { label: dec.value, confidence: dec.p, distribution: Object.fromEntries(ranked),
        reasoning: ranked.slice(0, 3).map(([k, v]) => `${k} (p=${v.toFixed(2)})`).join(", ") + ".",
        source: "model", cycles, stats, latency_ms: +(performance.now() - t0).toFixed(2) };
    },

    /** L2-normalised embedding vector. */
    async embed(text) {
      const ids = utterance(text).ids, r = await session.run({
        ids: padBatch([ids], cfg.seq_len),
        bank: new ort.Tensor("float32", new Float32Array(d), [1, d]), // the graph needs >= 1 probe; its readout is ignored
        probe_a: new ort.Tensor("int32", Int32Array.of(0), [1]),
        probe_b: new ort.Tensor("int32", Int32Array.of(-1), [1]),
        roles: new ort.Tensor("int32", Int32Array.of(ROLE_TOOL), [1]),
      });
      return Array.from(r.embedding.data);
    },

    /** Free the ONNX session. */
    close: () => session.release(),
  };
}
