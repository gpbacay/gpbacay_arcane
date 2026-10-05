// Parity test: web.mjs (onnxruntime-web) vs the Python model (index.js). node test-web.mjs
import assert from "node:assert";
import { createRequire } from "node:module";
import { load } from "./web.mjs";

const py = await createRequire(import.meta.url)("./index.js").load();
const web = await load(); // default: the bundled web/arc1.onnx + arc1.json, read from disk in Node

const weather = { name: "get_weather", description: "Get the current weather for a city.",
  parameters: [{ name: "city", description: "City name" }, { name: "unit", required: false, enum: ["celsius", "fahrenheit"] }] };
const lights = { name: "set_lights", description: "Set the brightness of the lights in a room.",
  parameters: [{ name: "room", description: "Room name" }, { name: "level", type: "integer", description: "Brightness percent" },
    { name: "on", type: "boolean", description: "Turn on" }] };
const cases = [
  ["run", "is it raining in Tokyo?", [weather]],
  ["run", "dim the bedroom lights to 30 percent", [weather, lights]],
  ["run", "tell me a joke", [weather, lights]],
  ["run", "how hot is it in Paris in fahrenheit", [weather, lights]], // second call hits the schema cache
  ["extract", "My name is Maria Santos and I live in Cebu, I am 31", { name: "Person name", city: "City", age: { type: "integer", description: "Age" } }],
  ["classify", "the headphones sound amazing", ["positive", "negative", "neutral"]],
  ["classify", "I was charged twice", ["billing", "shipping", "tech support"], { task: "Route the ticket to a team" }],
];

const close = (a, b, k) => assert.ok(Math.abs(a - b) < 1e-3, `${k}: ${a} vs ${b}`);
try {
  for (const [method, ...args] of cases) {
    const [a, b] = [await py[method](...args), await web[method](...args)];
    assert.deepStrictEqual(b.function_calls ?? b.label, a.function_calls ?? a.label, `${method} ${args[0]}`);
    if (a.record) assert.deepStrictEqual(b.record, a.record);
    close(a.confidence, b.confidence, `${method} ${args[0]} confidence`);
    console.log(`ok  ${method.padEnd(8)} ${JSON.stringify(b.function_calls ?? b.label)}  conf=${b.confidence.toFixed(3)}  ${b.latency_ms}ms`);
  }
  const [ea, eb] = [await py.embed("hello there"), await web.embed("hello there")];
  close(ea.reduce((s, x, i) => s + x * eb[i], 0), 1, "embed cosine");
  await assert.rejects(web.classify("x", []), /at least one label/);
  console.log("parity ok");
} finally {
  py.close();
  await web.close();
}
