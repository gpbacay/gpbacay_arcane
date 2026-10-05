# gpbacay-arcane (Node.js)

ARC 1 tool calling, field extraction, classification, and embeddings from Node.js.

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
