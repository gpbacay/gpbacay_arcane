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
