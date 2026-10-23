// Occipital -> Hippocampus -> ARC 1, on intents ARC 1 never trained on. node bench-occipital-arc1.mjs [shots...]
// The protocol of examples/benchmark_hippocampus.py (clinc): CLINC150 test utterances for the intents held out of
// ARC 1's training, 5 labels each, the memory holding `shots` train utterances per intent. Each request is also run
// with typos (half of the 4+ letter words get one dropped, swapped or replaced letter). Out-of-scope: CLINC150's
// oos_test with 5 random labels, where memory alone should abstain. Occipital's settings were chosen on the val
// split, not this one. Last, the memory goes through an MP4 and back, and must decide exactly as before.
// Needs data/external/clinc150_data_full.json (see the Python benchmark for the download).
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { load } from "./web.mjs";
import { Hippocampus } from "./hippocampus.mjs";
import { Occipital } from "./occipital.mjs";
import { writeVideo, readVideo } from "./occipital-video.mjs";

const HELD_OUT = new Set(["time", "timezone", "restaurant_reservation", "restaurant_reviews", "restaurant_suggestion"]);
const TASK = "Label the intent of the request.", N = 600;
const data = JSON.parse(readFileSync(new URL("../data/external/clinc150_data_full.json", import.meta.url), "utf8"));
const intents = [...new Set(data.train.map(([, l]) => l))].filter((l) => !HELD_OUT.has(l)).sort().filter((_, i) => i % 5 === 0);
const human = (l) => l.replaceAll("_", " ");
let seed;
const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
const sample = (xs, k) => [...xs].sort(() => rand() - 0.5).slice(0, k); // ponytail: biased shuffle, fine for a fixed-seed draw
const typo = (t) => t.replace(/[a-z]{4,}/g, (w) => {
  if (rand() > 0.5) return w;
  const i = 1 + Math.floor(rand() * (w.length - 2)), r = rand();
  if (r < 1 / 3) return w.slice(0, i) + w.slice(i + 1);
  if (r < 2 / 3) return w.slice(0, i) + w[i + 1] + w[i] + w.slice(i + 2);
  return w.slice(0, i) + "abcdefghijklmnopqrstuvwxyz"[Math.floor(rand() * 26)] + w.slice(i + 1);
});
const pct = (h, n) => `${((100 * h) / n).toFixed(1)}%`;

const arc1 = await load();
const dir = mkdtempSync(path.join(tmpdir(), "occipital-arc1-"));
try {
  for (const shots of (process.argv.slice(2).length ? process.argv.slice(2) : ["1", "5", "10"]).map(Number)) {
    seed = 1;
    const examples = intents.flatMap((i) => sample(data.train.filter(([, l]) => l === i), shots).map(([t]) => [human(i), t]));
    seed = 11;
    const clean = sample(data.test.filter(([, l]) => intents.includes(l)), N).map(([t, l]) => [t, human(l), sample([l, ...sample(intents.filter((i) => i !== l), 4)], 5).map(human)]);
    seed = 99;
    const typos = clean.map(([t, l, ls]) => [typo(t), l, ls]);
    seed = 7;
    const oos = data.oos_test.map(([t]) => [t, sample(intents, 5).map(human)]);

    const build = (model, occipital) => {
      const h = new Hippocampus(model, { occipital });
      for (const [label, text] of examples) h.remember(text, label);
      return h;
    };
    const rows = {}, accuracy = async (h, set) => {
      let hit = 0;
      for (const [t, l, ls] of set) hit += (await (h ? h.classify(t, ls, { task: TASK }) : arc1.classify(t, ls, { task: TASK }))).label === l;
      return pct(hit, set.length);
    };
    rows["ARC 1 alone"] = { clean: await accuracy(null, clean), typos: await accuracy(null, typos), "out-of-scope abstained": "–" };
    for (const [name, occipital] of [["Hippocampus", null], ["Hippocampus + Occipital", () => new Occipital()],
      ["Hippocampus + Occipital(minCoverage 0.1)", () => new Occipital({ minCoverage: 0.1 })]]) {
      const memory = build(null, occipital?.());
      let abstained = 0;
      for (const [t, ls] of oos) abstained += (await memory.classify(t, ls)).label === null;
      rows[`${name}, memory alone`] = { clean: await accuracy(memory, clean), typos: await accuracy(memory, typos), "out-of-scope abstained": pct(abstained, oos.length) };
      const both = build(arc1, occipital?.());
      rows[`ARC 1 + ${name}`] = { clean: await accuracy(both, clean), typos: await accuracy(both, typos), "out-of-scope abstained": "–" };
    }

    // the memory through a video: identical decisions
    const file = path.join(dir, `clinc-${shots}.mp4`);
    const written = await writeVideo(file, examples);
    const direct = build(arc1, new Occipital());
    const { memory: fromVideo, lost } = await readVideo(file, { model: arc1, occipital: new Occipital() });
    let same = 0;
    for (const [t, , ls] of typos) {
      const [a, b] = [await direct.classify(t, ls, { task: TASK }), await fromVideo.classify(t, ls, { task: TASK })];
      same += a.label === b.label && Math.abs(a.confidence - b.confidence) < 1e-9;
    }
    console.log(`\n${shots} example(s) per intent, ${intents.length} intents, n = ${N} (+ ${oos.length} out-of-scope)`);
    console.table(rows);
    console.log(`video: ${written.frames} frames, ${written.bytes} bytes, ${lost} lost; identical decisions to the in-memory Hippocampus: ${same}/${typos.length}`);
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
  await arc1.close();
}
