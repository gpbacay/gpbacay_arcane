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
