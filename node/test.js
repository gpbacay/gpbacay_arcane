// Smoke test: node test.js (needs gpbacay-arcane installed for the chosen Python).
const assert = require("node:assert");
const { load } = require("./index.js");

(async () => {
  const agent = await load();
  try {
    const weather = { name: "get_weather", description: "Get the current weather for a city.", parameters: [{ name: "city", description: "City name" }] };
    const run = await agent.run("is it raining in Tokyo?", [weather]);
    assert.deepStrictEqual(run.function_calls, [{ name: "get_weather", arguments: { city: "Tokyo" } }]);

    const ex = await agent.extract("My name is Maria Santos and I live in Cebu", { name: "Person name", city: "City" });
    assert.strictEqual(ex.record.city, "Cebu");

    const cl = await agent.classify("the headphones sound amazing", ["positive", "negative", "neutral"]);
    assert.ok(["positive", "negative", "neutral"].includes(cl.label));

    const vec = await agent.embed("hello");
    assert.ok(vec.length > 0 && typeof vec[0] === "number");

    await assert.rejects(agent.classify("x", []), /at least one label/);
    console.log("ok");
  } finally {
    agent.close();
  }
})();
