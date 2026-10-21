// Hippocampus (hippocampus.mjs) with stub models, then on the real ARC 1 web model. node test-hippocampus.mjs
// Mirrors tests/test_hippocampus.py.
import assert from "node:assert";
import { Hippocampus } from "./hippocampus.mjs";

const TICKETS = [["I was charged twice", "billing"], ["refund my last invoice", "billing"],
  ["my parcel never arrived", "shipping"], ["track my delivery", "shipping"]];
const stub = { // ARC 1 stand-in: classify prefers the first label; run records the tool prior
  classify: async (text, labels) => ({ label: labels[0], confidence: 0.7, latency_ms: 1,
    distribution: Object.fromEntries(labels.map((l, i) => [l, i === 0 ? 0.7 : 0.3 / (labels.length - 1)])) }),
  run: async (prompt, tools, opts) => ({ function_calls: [], tool_prior: opts.toolPrior }),
};
const hippocampus = (model = null) => {
  const h = new Hippocampus(model);
  for (const [text, label] of TICKETS) h.remember(text, label);
  return h;
};

// memory overrules any model and falls back to it
const firstLabel = (text, labels) => Object.fromEntries(labels.map((l, i) => [l, i === 0 ? 0.7 : 0.3]));
for (const model of [stub, firstLabel]) {
  const out = await hippocampus(model).classify("why was my card charged twice for the invoice", ["shipping", "billing"]);
  assert.equal(out.label, "billing");
  assert.equal(out.source, "model+memory");
  assert.deepStrictEqual(new Set(out.neighbors.map((h) => h.label)), new Set(["billing"]));
  assert.ok(Math.abs(Object.values(out.distribution).reduce((a, b) => a + b) - 1) < 1e-9);
  assert.equal((await hippocampus(model).classify("where is my parcel", ["sales", "account"])).source, "model");
}
assert.equal((await hippocampus(stub).classify("track it", ["shipping", "billing"])).latency_ms, 1); // model fields kept

// memory alone decides or abstains
let h = hippocampus();
assert.equal((await h.classify("track my parcel", ["billing", "shipping"])).label, "shipping");
const none = await h.classify("hello there", ["billing", "shipping"]);
assert.equal(none.label, null);
assert.equal(none.source, "model");
assert.deepStrictEqual(h.react("charged twice").votes, { billing: 1 });
await assert.rejects(h.run("set an alarm", []), TypeError);

// memory is dynamic
h.remember("I was charged twice", "fraud"); // same text again: the label is corrected, not duplicated
assert.equal(h.toJSON().length, 4);
assert.equal((await h.classify("charged twice", ["billing", "fraud"])).label, "fraud");
assert.ok(h.forget("I was charged twice") && h.toJSON().length === 3);
assert.deepStrictEqual(h.react("charged twice").votes, {});
assert.ok(!h.forget("I was charged twice"));

// the tool prior counts votes for other tools and for no tool
h = new Hippocampus(stub);
h.remember("set an alarm for 7am", "set_alarm");
h.remember("wake me up at six with an alarm", "set_alarm");
h.remember("thanks, alarm sounds good", null);
const prior = (await h.run("set an alarm for 6am")).tool_prior;
assert.deepStrictEqual(Object.keys(prior), ["set_alarm"]);
assert.ok(prior.set_alarm > 0.5 && prior.set_alarm < 1); // the null example took a share
assert.equal((await h.run("deploy the build")).tool_prior, undefined); // nothing similar: the model alone
const restored = new Hippocampus(stub, { memory: JSON.parse(JSON.stringify(h)) });
assert.deepStrictEqual(restored.toJSON(), h.toJSON());
assert.deepStrictEqual(new Set(Object.keys(restored.react("set an alarm").votes)), new Set(["set_alarm", "null"]));

// remembered requests fill arguments for any model
const badSpans = { // fires rate_movie with a wrong title and never calls toggle_bluetooth
  run: async (prompt) => ({ confidence: 0.9,
    function_calls: prompt.includes("rate") ? [{ name: "rate_movie", arguments: { title: "away deserves", stars: 5 } }] : [] }),
};
const RATE = { name: "rate_movie", description: "Rate a movie", parameters: [{ name: "title" }, { name: "stars", type: "integer" }] };
const BLUETOOTH = { name: "toggle_bluetooth", description: "Turn bluetooth on or off", parameters: [{ name: "enabled", type: "boolean" }] };
h = new Hippocampus(badSpans);
h.remember("rate Dune 4 stars", "rate_movie", { title: "Dune", stars: 4 });
h.remember("switch bluetooth off", "toggle_bluetooth", { enabled: false });
let out = await h.run("please rate spirited away 5 stars", [RATE, BLUETOOTH]);
assert.deepStrictEqual(out.function_calls, [{ name: "rate_movie", arguments: { title: "spirited away", stars: 5 } }]);
assert.deepStrictEqual(out.memory_arguments, ["rate_movie"]);
assert.equal(out.confidence, null);
out = await h.run("ok. Also switch bluetooth off", [RATE, BLUETOOTH]);
assert.deepStrictEqual(out.function_calls, [{ name: "toggle_bluetooth", arguments: { enabled: false } }]); // held back, added
assert.deepStrictEqual((await h.run("switch bluetooth on", [BLUETOOTH])).function_calls, []); // "off" is part of the pattern
const reloaded = new Hippocampus(badSpans, { memory: JSON.parse(JSON.stringify(h)) });
assert.deepStrictEqual(reloaded.arguments("rate Coco two stars", RATE), { title: "Coco", stars: 2 });
assert.equal(reloaded.arguments("rate it", RATE), null);
assert.deepStrictEqual(reloaded.arguments("RATE Amélie 3 stars", RATE), { title: "Amélie", stars: 3 }); // Unicode words

// a model call copying words a pattern explains is dropped
const wrongTool = { run: async (prompt) => ({ function_calls: [{ name: "play_podcast", arguments: { name: prompt } }] }) };
const PODCAST = { name: "play_podcast", description: "Play a podcast", parameters: [{ name: "name" }] };
h = new Hippocampus(wrongTool);
h.remember("Inception deserves 3 stars", "rate_movie", { title: "Inception", stars: 3 });
out = await h.run("Parasite deserves 5 stars", [RATE, PODCAST]);
assert.deepStrictEqual(out.function_calls, [{ name: "rate_movie", arguments: { title: "Parasite", stars: 5 } }]);
assert.equal((await h.run("play Serial", [RATE, PODCAST])).function_calls[0].name, "play_podcast");
// observe stores only what is surprising
h = new Hippocampus();
assert.ok(await h.observe("I was charged twice", "billing")); // nothing known: stored
assert.equal(await h.observe("  I WAS charged   twice ", "billing"), null); // same text, already decided: skipped
assert.ok(await h.observe("track my delivery", "shipping"));
assert.equal(h.toJSON().length, 2);
assert.ok(await h.observe("I was charged twice", "fraud")); // the outcome disagrees: corrected, not duplicated
assert.equal(h.toJSON().length, 2);
assert.equal((await h.classify("charged twice", ["billing", "fraud"])).label, "fraud");
assert.ok(await new Hippocampus(stub).observe("my parcel never arrived", "shipping")); // model unsure (0.7) and no memory

// observe judges over the given decision space and spares remembered examples
const alphaBeta = () => {
  const m = new Hippocampus();
  m.remember("alpha beta", "x");
  m.remember("alpha beta gamma delta epsilon", "o");
  m.remember("alpha beta gamma zeta", "o");
  return m;
};
assert.ok(await alphaBeta().observe("alpha beta gamma", "x")); // among x and o, memory leans o: surprising
assert.equal(await alphaBeta().observe("alpha beta gamma", "x", null, 1, ["x", "z"]), null); // o is another decision
h = alphaBeta();
assert.equal(await h.observe("alpha beta", "z"), null); // remembered: untouched
assert.deepStrictEqual(h.react("alpha beta", ["x"]).votes, { x: 1 });
h.forget("alpha beta");
assert.ok(await h.observe("alpha beta", "z")); // forgotten first, then observe may write

// observe needs repeated trusted evidence to commit
h = new Hippocampus(null, { threshold: 1, leak: 0.5 });
const evidence = (label) => h.observe("refund my invoice", label, null, 0.6);
assert.equal(await evidence("billing"), null); // 0.6
assert.equal(await evidence("shipping"), null); // billing decays to 0.3, then is cancelled
assert.equal(await evidence("billing"), null); // shipping cancelled, billing starts over
assert.equal(await evidence("billing"), null); // 0.3 + 0.6 = 0.9 < 1
assert.equal(h.toJSON().length, 0);
assert.ok(await evidence("billing")); // 0.45 + 0.6 >= 1: committed
assert.deepStrictEqual(h.toJSON().map((e) => e.label), ["billing"]);

// pending evidence does not decay by default and is bounded
for (const [options, commits] of [[{ maxPending: 2 }, false], [{}, true]]) {
  h = new Hippocampus(null, options);
  await h.observe("refund my invoice", "billing", null, 0.6);
  for (let i = 0; i < 50; i++) await h.observe(`unrelated filler number ${i} zzz${i}`, "other", null, 0.1);
  assert.equal((await h.observe("refund my invoice", "billing", null, 0.6)) !== null, commits); // maxPending=2 pushed it out
}

// eviction spares remembered examples and drops the oldest observed
h = new Hippocampus(null, { maxSize: 3 });
h.remember("pinned example one", "a");
await h.observe("alpha beta gamma", "a");
await h.observe("delta epsilon zeta", "b");
await h.observe("eta theta iota", "c"); // over the cap: the oldest observed example goes
assert.equal(h.toJSON().length, 3);
assert.deepStrictEqual(h.react("alpha beta gamma").votes, {});
assert.deepStrictEqual(h.react("delta epsilon zeta").votes, { b: 1 });
assert.deepStrictEqual(h.react("pinned example one").votes, { a: 1 });
const capped = new Hippocampus(null, { maxSize: 1 });
capped.remember("one", "a");
capped.remember("two", "b");
assert.equal(capped.toJSON().length, 2); // remember is never capped

// observed examples keep their source through save and load
h = new Hippocampus();
await h.observe("I was charged twice", "billing");
h.remember("track my delivery", "shipping");
const saved = JSON.parse(JSON.stringify(h));
assert.deepStrictEqual(saved.map((e) => e.source), ["auto", undefined]);
const loaded = new Hippocampus(null, { memory: saved, maxSize: 2 });
await loaded.observe("alpha beta gamma", "c"); // over the cap: the loaded observed example is the one to go
assert.deepStrictEqual(loaded.toJSON().map((e) => e.text), ["track my delivery", "alpha beta gamma"]);
console.log("hippocampus ok");

// on ARC 1 itself: a tool arc1-tiny was never trained on fires and gets its arguments from two examples
const { load } = await import("./web.mjs");
const arc1 = await load();
try {
  h = new Hippocampus(arc1);
  h.remember("rate Dune 4 stars", "rate_movie", { title: "Dune", stars: 4 });
  h.remember("give Alien five stars", "rate_movie", { title: "Alien", stars: 5 });
  out = await h.run("rate Coco 3 stars", [RATE]);
  assert.deepStrictEqual(out.function_calls, [{ name: "rate_movie", arguments: { title: "Coco", stars: 3 } }]);
  assert.ok(out.decisions.tools.rate_movie > 0.5); // the memory's prior fired the tool through noisy-OR
  console.log("hippocampus + arc1 ok", JSON.stringify(out.function_calls));
} finally {
  await arc1.close();
}
