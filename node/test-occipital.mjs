// Occipital (occipital.mjs, occipital-video.mjs). node test-occipital.mjs
import assert from "node:assert";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { chunk, pack, unpack, MAX_PAYLOAD } from "./occipital.mjs";
import { writeVideo, readVideo } from "./occipital-video.mjs";

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
} finally {
  rmSync(dir, { recursive: true, force: true });
}
console.log("occipital ok");
