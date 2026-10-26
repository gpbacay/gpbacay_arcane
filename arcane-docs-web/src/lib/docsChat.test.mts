// node --experimental-strip-types src/lib/docsChat.test.mts
import assert from "node:assert/strict";
import { Hippocampus } from "gpbacay-arcane/hippocampus";
import { Occipital } from "gpbacay-arcane/occipital";
import { appendTranscript, blankThread, removeThread, titleFrom, truncateBefore } from "./chatThreads.ts";
import { EXAMPLES, answerDocs, arc1CanAnswer, decide, generationFacts, groundedText, missesPageFact, navigationOnly, routeThought, ungroundedDetail, withContext } from "./docsChat.ts";

const hippo = new Hippocampus(null, { k: 8, weight: 0.7, memory: EXAMPLES, occipital: new Occipital() }); // as arc1Router.ts

const heldOut: [string, { intent?: "docs" | "none"; pageId?: string | null }][] = [
  ["scroll me over to the install docs", { intent: "none", pageId: "installation" }],
  ["open the hippocampus page", { intent: "none", pageId: "hippocampus" }],
  ["jump to the quick start", { intent: "none", pageId: "quick-start" }],
  ["opn the hipocampus page", { intent: "none", pageId: "hippocampus" }], // typos: found through Occipital
  ["go to instalation", { intent: "none", pageId: "installation" }],
  ["show me the instal page", { intent: "none", pageId: "installation" }],
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
assert.match(arc1CanAnswer("how do I install arcane")?.text ?? "", /pip install gpbacay-arcane/);
assert.equal(arc1CanAnswer("write me a haiku about rain"), null);
assert.equal(arc1CanAnswer("I feel lost, can you explain this in a kinder and more encouraging way"), null);
assert.match(answerDocs("is ARC 1 a system 1 decision model").text, /System 1 decision model/);
assert.match(answerDocs("why should I use hippocampus").text, /fast memory of decided examples/);
assert.match(answerDocs("who are you").text, /ARC 1/);
assert.match(answerDocs("what is the weather in paris", { intent: "none", pageId: null, recallId: null, source: "memory", confidence: 1, latencyMs: 1, calls: ["none"] }).text, /only answer/);
assert.match(answerDocs("who made arcane").text, /Gianne P\. Bacay/);
assert.doesNotMatch(answerDocs("who made arcane").text, /September/);
assert.match(answerDocs("when was arcane made").text, /3 September 2025/);
assert.match(answerDocs("when was arcane made").text, /13 October 2026/);
assert.doesNotMatch(answerDocs("when was arcane made").text, /Gianne/);
assert.match(answerDocs("when?", { intent: "docs", pageId: null, recallId: "about", source: "memory", confidence: 1, latencyMs: 1, calls: [] }).text, /13 October 2026/);
const about = await decide(hippo, "who is the creator of this project");
assert.equal(about.recallId, "about", `recalled ${about.recallId}`);
assert.equal(about.source, "memory");
assert.match(answerDocs("the person behind it", about).text, /Gianne P\. Bacay/);
const thought = routeThought(await decide(hippo, "opn the hipocampus page"), { label: "Opened Hippocampus" });
assert.match(thought.label, /Hippocampus \+ Occipital/);
assert.deepEqual(
  thought.steps.map((step) => step.label),
  ["Recalled a similar request", "Decided to open Hippocampus", "Opened Hippocampus"],
);
assert.match(thought.steps[0].results?.[0] ?? "", /hippocampus/i, "the recall backs the decision");
const split = routeThought(await decide(hippo, "open hippocampus and then show the install docs"), { label: "Opened" });
const parts = split.steps.find((step) => step.steps);
assert.equal(parts?.label, "Split the message into 2 parts");
assert.deepEqual(parts?.steps?.map((step) => step.results?.[0]), ["open hippocampus", "show the install docs"]);
assert.equal(withContext("when?", [{ role: "user", text: "who made arcane" }]), "who made arcane. when?");
assert.equal(withContext("how do I install the package from a source checkout", []), "how do I install the package from a source checkout");

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

assert.match(arc1CanAnswer("what is your primary directive?")?.text ?? "", /primary directive/);
assert.match(arc1CanAnswer("what is your primary directive?")?.text ?? "", /not in them/);
const directiveFacts = generationFacts("what is your primary directive?", null);
assert.match(directiveFacts, /Primary directive/);
assert.doesNotMatch(directiveFacts, /fruit fly/i);
const memoryFacts = generationFacts("what does the memory do", null);
assert.match(memoryFacts, /Hippocampus/);
const authorFacts = generationFacts("who made this", null);
assert.match(authorFacts, /Gianne P\. Bacay/);
assert.doesNotMatch(authorFacts, /September/);
assert.equal(ungroundedDetail("Hippocampus stores decided examples.", memoryFacts), null);
assert.equal(ungroundedDetail("It was released in October 1999.", memoryFacts), "October");
assert.equal(groundedText("It was released in October 1999.", memoryFacts, "Not in the docs."), "Not in the docs.");
assert.match(groundedText("It is a fast memory of decided examples.", memoryFacts, "Not in the docs."), /memory/);
const invented = "Hippocampus has been used extensively in my brain for decades.";
assert.equal(missesPageFact(invented, memoryFacts), true);
assert.equal(groundedText(invented, memoryFacts, "Not in the docs."), "Not in the docs.");
assert.equal(missesPageFact("rain on the glass", generationFacts("write a haiku about rain", null), "write a haiku about rain"), false);

console.log("docs chat ok");
