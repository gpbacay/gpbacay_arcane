// node --experimental-strip-types src/lib/docsChat.test.mts
import assert from "node:assert/strict";
import { Hippocampus } from "gpbacay-arcane/hippocampus";
import { appendTranscript, blankThread, removeThread, titleFrom, truncateBefore } from "./chatThreads.ts";
import { EXAMPLES, answerDocs, decide, navigationOnly } from "./docsChat.ts";

const hippo = new Hippocampus(null, { k: 8, weight: 0.7, memory: EXAMPLES });

const heldOut: [string, { intent?: "docs" | "none"; pageId?: string | null }][] = [
  ["scroll me over to the install docs", { intent: "none", pageId: "installation" }],
  ["open the hippocampus page", { intent: "none", pageId: "hippocampus" }],
  ["jump to the quick start", { intent: "none", pageId: "quick-start" }],
  ["what does ARC 1 actually return", { intent: "docs" }],
  ["how do I install the package", { intent: "docs", pageId: null }],
  ["write me a haiku about rain", { intent: "none", pageId: null }],
];

for (const [text, expect] of heldOut) {
  const route = await decide(hippo, text);
  if (expect.intent) assert.equal(route.intent, expect.intent, `${text} intent ${route.intent} calls ${route.calls}`);
  if ("pageId" in expect) assert.equal(route.pageId, expect.pageId, `${text} page ${route.pageId}`);
}

assert.equal(navigationOnly("open the installation page"), "installation");
assert.equal(navigationOnly("show me the fruit fly demo"), "fruitfly");
assert.equal(navigationOnly("what is hippocampus?"), null);
assert.match(answerDocs("how do I install arcane").text, /pip install gpbacay-arcane/);
assert.match(answerDocs("who are you").text, /ARC 1/);
assert.match(answerDocs("what is the weather in paris", { intent: "none", pageId: null, confidence: 1, latencyMs: 1, calls: ["none"] }).text, /only answer/);

const older = { id: "a", role: "user" as const, text: "first" };
const newer = { id: "b", role: "assistant" as const, text: "second" };
assert.deepEqual(truncateBefore([older, newer], "b").map((message) => message.id), ["a"]);
assert.equal(titleFrom("  how   do I install arcane and also tune hippocampus memory  "), "how do I install arcane and also tune hipp…");
assert.equal(appendTranscript("how do I", ["install arcane"]), "how do I install arcane");
assert.equal(appendTranscript("", ["open the quick start"]), "open the quick start");
assert.equal(appendTranscript("a".repeat(1998), ["xyz"]).length, 2000);
const only = blankThread("only");
assert.equal(removeThread([only], "only", "fresh").currentId, "fresh");
const kept = removeThread([only, blankThread("other")], "only", "unused");
assert.deepEqual(kept.threads.map((thread) => thread.id), ["other"]);

console.log("docs chat ok");
