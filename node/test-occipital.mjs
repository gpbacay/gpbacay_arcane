// Occipital (occipital.mjs, occipital-video.mjs). node test-occipital.mjs
import assert from "node:assert";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { chunk, pack, unpack, MAX_PAYLOAD, Occipital } from "./occipital.mjs";
import { writeVideo, readVideo } from "./occipital-video.mjs";
import { Hippocampus } from "./hippocampus.mjs";
import { load } from "./web.mjs";

const RATE = { name: "rate_movie", description: "Rate a movie", parameters: [{ name: "title" }, { name: "stars", type: "integer" }] };

const DOCS = {
  billing: "Refunds are issued to the original card within five business days. A duplicate charge is reversed automatically once the bank confirms it. ".repeat(6),
  shipping: "Parcels ship from Cebu every weekday. Tracking numbers arrive by email, and a lost parcel is replaced after ten days without a scan. ".repeat(6),
};

// chunks cut at whitespace, stay under size, overlap, and cover the text
const text = DOCS.billing.trim();
const pieces = chunk(text, 120, 30);
assert.ok(pieces.length > 1);
for (const p of pieces) {
  assert.ok(p.length <= 120 && text.includes(p));
  assert.ok(/^\S/.test(p) && /\S$/.test(p));
  assert.ok(text.includes(" " + p) || text.startsWith(p), `cut mid-word: ${p.slice(0, 20)}`);
}
for (const word of text.split(/\s+/)) assert.ok(pieces.some((p) => p.split(" ").includes(word)));
assert.ok(pieces[0].slice(-30).includes(pieces[1].split(" ").slice(0, 2).join(" ")), "no overlap");
assert.deepStrictEqual(chunk("   "), []);
assert.deepStrictEqual(chunk("short"), ["short"]);

// the letter pathway: typos find the neighbors words miss, abstention stays, the index follows the memory
const TICKETS = [["I was charged twice", "billing"], ["refund my last invoice", "billing"],
  ["my parcel never arrived", "shipping"], ["track my delivery", "shipping"]];
const tickets = (occipital = null) => {
  const h = new Hippocampus(null, { occipital });
  for (const [t, l] of TICKETS) h.remember(t, l);
  return h;
};
const TYPO = "my parcle nevr arived", LABELS = ["billing", "shipping"];
assert.equal((await tickets().classify(TYPO, LABELS)).label, null);
assert.equal((await tickets(new Occipital()).classify(TYPO, LABELS)).neighbors[0].text, "my parcel never arrived");
assert.equal((await tickets(new Occipital()).classify("hello partner", LABELS)).label, null);
assert.ok(tickets(new Occipital({ minCoverage: 0 })).react("hello partner").neighbors.length);
assert.equal((await tickets(new Occipital()).classify("I was charged twice for the invoice", LABELS)).neighbors[0].text, "I was charged twice");
const followed = tickets(new Occipital());
followed.forget("my parcel never arrived");
assert.ok(followed.neighbors(TYPO).every((n) => n.text !== "my parcel never arrived"));
followed.remember("my parcel never arrived", "billing");
assert.equal((await followed.classify(TYPO, LABELS)).label, "billing");
assert.deepStrictEqual([...tickets(new Occipital()).occipital.fuse(TYPO, new Map())].map(([, s]) => +s.toFixed(6)), [0.25]); // = Python

// payloads round-trip into a search; foreign or broken frames are counted, not fatal
const payloads = await pack(DOCS, { chunkSize: 200 });
assert.ok(payloads.every((p) => p.length <= MAX_PAYLOAD));
const { memory, lost } = await unpack([...payloads, null, new Uint8Array([1, 2, 3])], { k: 3 });
assert.equal(lost, 2);
assert.equal(memory.neighbors("my parcel was lost, no tracking scan").length, 3);
assert.equal(memory.neighbors("my parcel was lost, no tracking scan")[0].label, "shipping");
assert.deepStrictEqual(Object.keys(memory.react("charged twice, refund to card").votes), ["billing"]);
const noise = Array.from({ length: 2000 }, (_, i) => ((i * 7919) % 100003).toString(36)).join(" "); // barely compresses
await assert.rejects(pack({ big: noise }, { chunkSize: 9000 }), /lower chunkSize/);

// the video round-trips and still opens after lossy re-encoding (needs ffmpeg)
const dir = mkdtempSync(path.join(tmpdir(), "occipital-"));
try {
  const file = path.join(dir, "kb.mp4");
  const written = await writeVideo(file, DOCS, { chunkSize: 200 });
  assert.equal(written.frames, payloads.length);
  const opened = await readVideo(file);
  assert.equal(opened.frames, written.frames);
  assert.equal(opened.lost, 0);
  assert.equal(opened.memory.neighbors("duplicate charge reversed")[0].label, "billing");
  console.log(`video: ${written.frames} frames, ${written.bytes} bytes for ${written.textBytes} bytes of text`);

  // an ARC 1 memory with tool arguments, read back from video, teaches ARC 1 a tool it was never trained on
  const taught = new Hippocampus();
  taught.remember("rate Dune 4 stars", "rate_movie", { title: "Dune", stars: 4 });
  taught.remember("give Alien five stars", "rate_movie", { title: "Alien", stars: 5 });
  taught.remember("thanks, that's all", null);
  await writeVideo(file, taught.toJSON().map((e) => [e.label, e.text, e.arguments]));
  const arc1 = await load();
  try {
    const { memory: skill, lost: skillLost } = await readVideo(file, { model: arc1 });
    assert.equal(skillLost, 0);
    assert.deepStrictEqual(skill.toJSON(), taught.toJSON());
    const out = await skill.run("rate Coco 3 stars", [RATE]);
    assert.deepStrictEqual(out.function_calls, [{ name: "rate_movie", arguments: { title: "Coco", stars: 3 } }]);
    console.log("occipital + arc1 ok", JSON.stringify(out.function_calls));
  } finally {
    await arc1.close();
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
}
console.log("occipital ok");
