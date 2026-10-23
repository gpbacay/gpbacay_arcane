# gpbacay-arcane (Node.js)

The ARCANE 5.0 Node runtime provides the grounded-decision and neural-memory parts of the System 1 stack: ARC 1 tool calling, field extraction, classification, embeddings, and Hippocampus memory.

The model runs locally in a Python child process, so you also need the Python package:

```sh
pip install gpbacay-arcane
npm install gpbacay-arcane
```

```js
const { load } = require("gpbacay-arcane");

const agent = await load(); // bundled arc1-tiny; load({ python, model }) to override

await agent.run("is it raining in Tokyo?", [
  { name: "get_weather", description: "Get the current weather for a city.",
    parameters: [{ name: "city", description: "City name" }] },
]);
// { function_calls: [{ name: "get_weather", arguments: { city: "Tokyo" } }], confidence, ... }

await agent.extract("My name is Maria Santos and I live in Cebu", { name: "Person name", city: "City" });
await agent.classify("the headphones sound amazing", ["positive", "negative", "neutral"]);
await agent.embed("hello");

agent.close();
```

- `load()` resolves once the model is loaded. The first load takes a while because TensorFlow starts up. Later calls take milliseconds.
- The Python interpreter is `options.python`, then `$ARC1_PYTHON`, then `python` on Windows or `python3` elsewhere.
- Tools are not executed here. Use `function_calls` in your own code.

## Node.js without Python

`gpbacay-arcane/web` also runs in plain Node.js (ESM) with no Python install:

```sh
npm install gpbacay-arcane onnxruntime-web
```

```js
import { load } from "gpbacay-arcane/web";

const agent = await load(); // reads the bundled arc1.onnx / arc1.json from disk
const r = await agent.classify("I was charged twice", ["billing", "shipping"]);
await agent.close();
```

## In the browser (no server, no Python)

`gpbacay-arcane/web` runs the same model client-side with [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) (WebAssembly). The model is 5.5 MB, and the browser fetches it once and caches it. Results match the Python package.

```sh
npm install gpbacay-arcane onnxruntime-web
```

Next.js (App Router), in a client component:

```jsx
"use client";
import { useEffect, useRef, useState } from "react";
import { load } from "gpbacay-arcane/web";

export default function Arc1Demo() {
  const agent = useRef(null);
  const [out, setOut] = useState("loading ARC 1...");
  useEffect(() => {
    load().then((a) => { agent.current = a; setOut("ready"); });
    return () => agent.current?.close();
  }, []);
  const ask = async () => {
    const r = await agent.current.run("is it raining in Tokyo?", [
      { name: "get_weather", description: "Get the current weather for a city.",
        parameters: [{ name: "city", description: "City name" }] },
    ]);
    setOut(JSON.stringify(r.function_calls));
  };
  return <><button onClick={ask} disabled={out.startsWith("loading")}>Ask</button><pre>{out}</pre></>;
}
```

- The API matches the Node API: `run`, `extract`, `classify`, `embed` and `close`, with the same result objects. The one exception is `cycles`: it is fixed at export time and can't be passed per call.
- By default `load()` uses `arc1.onnx` / `arc1.json` from this package, and the bundler emits them as assets. To serve them yourself (for example from `public/arc1/`), use `load({ model: "/arc1/arc1.onnx", config: "/arc1/arc1.json" })`.
- In browsers, onnxruntime's `.wasm` runtime is fetched from jsDelivr. To self-host it, pass `load({ wasmPaths: "/ort/" })`.
- To re-export after training: `python examples/export_arc1_onnx.py --model your.rcn --out node/web` (needs `pip install tf2onnx onnxruntime`). The script also checks that the ONNX outputs match TensorFlow.

## Hippocampus: teach ARC 1 with examples (browser and Node, no Python)

`Hippocampus` stores decided examples and mixes their vote into ARC 1's decisions. New labels and new tools work from a few examples, with no fine-tuning. It is a port of the Python `Hippocampus`: same API and same decisions, but `classify`, `run` and `observe` are async.

```js
import { load, Hippocampus } from "gpbacay-arcane/web";

const hippocampus = new Hippocampus(await load()); // or any { classify } / { run } model; new Hippocampus() = memory only

hippocampus.remember("I was charged twice this month", "billing");
hippocampus.remember("rate Dune 4 stars", "rate_movie", { title: "Dune", stars: 4 }); // tool + its arguments
hippocampus.remember("thanks, that's all", null); // null = no tool applies

await hippocampus.classify("why is my card charged again?", ["billing", "shipping"]); // { label, source, neighbors, ... }
await hippocampus.run("rate spirited away 5 stars", tools);
// function_calls -> [{ name: "rate_movie", arguments: { title: "spirited away", stars: 5 } }], memory_arguments -> ["rate_movie"]
hippocampus.react("charged again?"); // the memory's vote alone, no model call
hippocampus.forget("thanks, that's all");

// Let it decide what to keep: give it an outcome (a correction, a tool that ran) and it stores only what is surprising
await hippocampus.observe("my card was declined", "billing");              // stored, or null if already decided confidently
await hippocampus.observe("refund my invoice", "billing", null, 0.6);     // trust 0.6: stored once evidence reaches threshold (1)
// observe(text, label, args, trust, decisionLabels): the 5th argument is the decision space it judges over (default: every label)
// Or give it a stronger model to learn from: classify() asks it when unsure, returns its answer and remembers it
// const hippocampus = new Hippocampus(await load(), { teacher: async (text, labels) => askMyLlm(text, labels) });
// the teacher returns a label, { label: probability } or null; an answer under `confident` (0.8) is not trusted
// new Hippocampus(model, { maxSize: 5000 }) forgets the oldest observed example past the cap; remembered ones stay,
// and observe never overwrites an example you remember()ed

const saved = JSON.stringify(hippocampus); // [{ text, label, arguments, source }] (source: "auto" for observed examples)
new Hippocampus(agent, { memory: JSON.parse(saved) }); // also accepts Python's hippocampus.graph.to_json()
```

- For memory-only use, `import { Hippocampus } from "gpbacay-arcane/hippocampus"`. It doesn't need onnxruntime-web.
- `run` passes the memory's vote to the model as `opts.toolPrior`. ARC 1 web combines it with its own firing probability by noisy-OR. Tools are not executed.

## Occipital: the visual pathway of Hippocampus

`Occipital` gives a `Hippocampus` recall by letters as well as words, the way the brain's visual word form area reads, so typos and other word forms still find their neighbors. It changes which examples vote, not the model, so ARC 1 or any other model behind the memory decides better. It's a port of the Python `Occipital` and returns the same scores.

```js
import { load, Hippocampus } from "gpbacay-arcane/web";
import { Occipital } from "gpbacay-arcane/occipital";

const hippocampus = new Hippocampus(await load(), { occipital: new Occipital() }); // one Occipital per Hippocampus
hippocampus.remember("my parcel never arrived", "shipping");
await hippocampus.classify("my parcle nevr arived", ["billing", "shipping"]); // "shipping", through its letters
```

- `node bench-occipital-arc1.mjs 1 5 10` uses 29 CLINC150 intents ARC 1 never trained on. With clean requests, ARC 1 + Hippocampus went from 83.2 / 89.7 / 94.3% to 84.0 / 91.0 / 95.8% at 1, 5 and 10 examples per intent. With typos it went from 66.2 / 74.8 / 81.8% to 66.7 / 76.3 / 83.0%. Memory-only abstention on out-of-scope requests stayed within 0.5 points.
- Options: `{ weight: 0.5, sharpness: 2, minCoverage: 0.3, maxDf: 0.1 }`. `minCoverage: 0.1` adds about 5 more points on typos, but a memory-only Hippocampus then abstains much less.
- It doesn't help tool calling, and with one example per tool it costs about 1% of correct refusals, so leave it off for a tool memory.

### Memories as video

A memory or a set of documents can be written as QR-code frames of an MP4 and read back into a `Hippocampus`, with no vector database, embedder or server.

```sh
npm install gpbacay-arcane qrcode jsqr ffmpeg-static   # or set $FFMPEG_PATH / have ffmpeg on PATH
```

```js
import { writeVideo, readVideo } from "gpbacay-arcane/occipital-video";

await writeVideo("kb.mp4", { "refunds.md": refundsText, "shipping.md": shippingText });   // documents, or
await writeVideo("skill.mp4", hippocampus.toJSON().map((e) => [e.label, e.text, e.arguments])); // a memory

const { memory, lost } = await readVideo("skill.mp4", { model: await load(), occipital: new Occipital() });
memory.neighbors("how long do refunds take?");   // [{ text, label, score }]; lost = frames that could not be read
```

- Each passage (500 characters, 50 overlap, cut at whitespace) or example is stored as `deflate(["arcane-occipital/1", id, text, args?])` in one self-describing QR frame, so a damaged frame loses only its own entry.
- `gpbacay-arcane/occipital` also holds `chunk`, `pack` and `unpack` with no dependencies, for the browser or any other medium.
- A memory read from video decides exactly like the original: 600 of 600 identical decisions in `bench-occipital-arc1.mjs`.
- `node bench-occipital.mjs ../docs` measures the 13 docs in this repo (119 KB of text, 43 KB gzipped). The result is 269 frames in 681 KB, **5.7× the text size**, written in 2 s and read in 5.5 s. Every frame still decodes after re-encoding at H.264 CRF 35 or VP9. It's not compression.
