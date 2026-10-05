import Link from "next/link";
import { Gauge, Plug, RefreshCw, SearchCheck } from "lucide-react";
import { BarChart } from "@/components/BarChart";
import { CodeSnippet } from "@/components/CodeSnippet";
import { Mermaid } from "@/components/markdown";

const h2Base = "scroll-mt-28 text-2xl font-bold tracking-tight text-zinc-100 mb-4 border-b border-zinc-800 pb-2";
const h2 = `${h2Base} mt-16`;
const code = "bg-zinc-900 px-1.5 py-0.5 text-zinc-100";
const p = "max-w-2xl leading-relaxed";
const link = "font-medium text-[#C785F2] underline hover:text-[#d49cf5]";
const th = "py-2 pr-4 font-medium";
const td = "py-2.5 pr-4 tabular-nums text-zinc-300";

const README_URL = "https://github.com/gpbacay/gpbacay_arcane#hippocampus--a-fast-learning-memory-for-any-decision-model";

// Points gained over the base model alone, from the tables below (arc1-tiny, held-out intents and tools).
const GAINS: [string, string, string, string][] = [
  ["+24.0", "points", "Intent accuracy, 1 example per label", "56.5% → 80.5%"],
  ["+36.8", "points", "Intent accuracy, 10 examples per label", "56.5% → 93.3%"],
  ["+43.7", "points", "Fully correct tool calls, 5 examples per tool", "42.3% → 86.0%"],
  ["+43.6", "points", "Right tool chosen, 10 examples per tool", "55.7% → 99.3%"],
];

const CAPABILITIES = [
  {
    icon: Plug,
    title: "Any model",
    body: "A classifier, an LLM asked for probabilities, a tool-calling model, or no model at all.",
  },
  {
    icon: RefreshCw,
    title: "Learns without training",
    body: "Remember or forget an example and the next request uses it. No fine-tuning.",
  },
  {
    icon: Gauge,
    title: "Scales",
    body: "15,000 examples, 150 labels: 0.13 ms to add one, 3.4 ms per decision.",
  },
  {
    icon: SearchCheck,
    title: "Shows its evidence",
    body: "Every decision lists the remembered examples behind it.",
  },
];

const BRAIN: [string, string, string][] = [
  ["Learns one experience at once", "remember(text, label)", "Takes effect on the next request, no training."],
  ["A cue brings back similar episodes", "Retrieval of the 8 most similar examples", "BM25 keyword scoring over the stored examples."],
  ["Recalled episodes bias the cortex", "Vote mixed with the model's probabilities", "Neither side decides alone; combined they beat both."],
  ["Correcting or losing a memory", "Re-remember the same text, or forget()", "A wrong example is overwritten, not baked into weights."],
  ["Knowing what you remembered", "neighbors returned with each decision", "Every decision lists the examples behind it."],
];

const MODELS: [string, string][] = [
  ["None", "Memory only. Decides from the vote; returns label=None when nothing similar is stored, instead of guessing."],
  ["(text, labels) -> {label: probability}", "Any function: a scikit-learn classifier, an embedding model, an LLM asked to score the labels."],
  ["object with classify(text, labels, **kw)", "Returns a dict with \"distribution\". Arc1Agent already does; extra keyword arguments such as task= pass through."],
  ["object with run(prompt, tools=, tool_prior=, execute=)", "For tool calling; returns function_calls. tool_prior maps tool name to the probability it applies, and Arc1Agent.run combines it with its own firing probability. Arguments from a matching remembered request then replace the model's, whatever the model."],
];

const CLINC: [string, string, string, string][] = [
  ["none", "56.5%", "–", "–"],
  ["1", "–", "64.7%", "80.5%"],
  ["2", "–", "74.5%", "85.3%"],
  ["5", "–", "87.3%", "90.5%"],
  ["10", "–", "92.0%", "93.3%"],
];

const TOOLS: [string, string, string, string, string][] = [
  ["none (model alone)", "55.7%", "42.3%", "–", "18.6%"],
  ["1", "77.0%", "50.3%", "64.0%", "53.3%"],
  ["2", "83.7%", "52.0%", "70.7%", "64.1%"],
  ["5", "93.7%", "54.7%", "86.0%", "83.5%"],
  ["10", "99.3%", "54.7%", "99.3%", "100%"],
];

export default function HippocampusPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <header className="not-prose mb-12">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#C785F2]">ARCANE · Harness</p>
        <h1 className="mt-3 max-w-3xl text-4xl font-extrabold leading-[1.08] tracking-[-0.03em] text-zinc-50 sm:text-5xl">
          Hippocampus
        </h1>
        <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-zinc-300">
          A fast-learning memory, modeled on the brain&apos;s hippocampus, that makes a model better at your labels
          and tools without retraining it.
          RAG lets a language model use your knowledge by retrieving text when a request arrives. Many models,
          such as classifiers, routers and tool-calling models, can&apos;t read retrieved text, so Hippocampus retrieves
          past <em>decisions</em> instead. The examples most like the request vote for their labels or tools, the
          vote is combined with the model&apos;s own probabilities, and remembered tool calls show where each
          argument sits in a request.
        </p>
        <p className="mt-4 max-w-2xl text-[17px] leading-relaxed text-zinc-300">
          Add an example and the next request uses it; the weights never change. In our benchmarks, a single
          example per label lifted accuracy on unseen intents by 24 points, and 5 examples per tool lifted fully
          correct tool calls by 44 points. See{" "}
          <a href="#improvement" className={link}>
            how much it improves
          </a>
          .
        </p>
      </header>

      <div className="text-zinc-300">
        <h2 id="what-it-does" className={h2Base}>
          What it does
        </h2>
        <ul className="not-prose mt-6 grid gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-2 lg:grid-cols-4">
          {CAPABILITIES.map(({ icon: Icon, title, body }) => (
            <li key={title} className="bg-zinc-950 p-4 sm:p-5">
              <Icon className="h-5 w-5 text-[#C785F2]" aria-hidden />
              <p className="mt-3 font-semibold text-zinc-100">{title}</p>
              <p className="mt-1.5 text-sm leading-relaxed text-zinc-400">{body}</p>
            </li>
          ))}
        </ul>
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          Examples are stored in the same document graph as the{" "}
          <Link href="/docs/graph-of-thought" className={link}>
            Grounded Graph of Thought
          </Link>
          . Search needs only numpy, with no API keys or vector database.
        </p>

        <h2 id="brain" className={h2}>
          Inspired by the brain
        </h2>
        <p className={p}>
          Complementary learning systems theory says the brain keeps two learners. The neocortex learns slowly and
          holds general knowledge in its connections. The hippocampus stores single episodes at once without
          rewiring the cortex, and when a similar cue returns it recalls them to steer the cortex&apos;s decision.
          Hippocampus plays that second role next to any model, which plays the first.
        </p>
        <div className="not-prose mt-6 overflow-x-auto">
          <table className="w-full min-w-[560px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className={th}>In the brain</th>
                <th className={th}>Here</th>
                <th className="py-2 font-medium">What it means</th>
              </tr>
            </thead>
            <tbody>
              {BRAIN.map(([brain, here, body]) => (
                <tr key={brain} className="border-b border-zinc-900 align-top">
                  <td className="py-2.5 pr-4 text-zinc-100">{brain}</td>
                  <td className="py-2.5 pr-4 font-mono text-xs text-zinc-300">{here}</td>
                  <td className="py-2.5 text-zinc-400">{body}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          The analogy is loose. The brain consolidates: it replays hippocampal memories during sleep to train the
          cortex slowly, whereas Hippocampus never moves examples into the model&apos;s weights, so the memory keeps
          growing. Its retrieval matches keywords, not meaning and context. And it stores decided examples and counts
          votes, closer to exemplar theories of categorization than to full episodic memory.
        </p>

        <h2 id="improvement" className={h2}>
          How much it improves
        </h2>
        <p className={p}>
          Hippocampus adds accuracy in proportion to how many examples you give it, and the first few help most. Gains
          below are percentage points over the same model with no memory, on text and tools the model was never
          trained on.
        </p>
        <dl className="not-prose mt-6 grid gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-2 lg:grid-cols-4">
          {GAINS.map(([gain, unit, label, range]) => (
            <div key={label} className="bg-zinc-950 p-4 sm:p-5">
              <dd className="text-3xl font-semibold tabular-nums text-zinc-50">
                {gain} <span className="text-sm font-normal text-zinc-500">{unit}</span>
              </dd>
              <dt className="mt-2 text-sm text-zinc-300">{label}</dt>
              <dd className="mt-1 text-xs tabular-nums text-zinc-500">{range}</dd>
            </div>
          ))}
        </dl>
        <ul className="mt-6 max-w-2xl list-disc space-y-2 pl-5 text-sm leading-relaxed text-zinc-300">
          <li>
            <strong className="text-zinc-100">Classification:</strong> one example per label takes accuracy from
            56.5% to 80.5%; ten reach 93.3%. The model and the memory are each weaker alone (56.5% and 64.7% at one
            example), so the gain comes from combining them.
          </li>
          <li>
            <strong className="text-zinc-100">Tool selection:</strong> the right tool is chosen 55.7% of the time
            with no examples, 77.0% with one per tool and 93.7% with five.
          </li>
          <li>
            <strong className="text-zinc-100">Tool arguments:</strong> storing the arguments raises arguments
            right from 18.6% to 83.5% with five examples, and fully correct calls from 42.3% to 86.0%.
          </li>
          <li>
            <strong className="text-zinc-100">Memory alone:</strong> with no model, 15,000 examples across 150
            intents decide 80.6% correctly.
          </li>
        </ul>
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          These numbers were measured with one base model, the bundled arc1-tiny. How much another model gains
          depends on how much it already gets right: a model that is weak on your labels gains the most, and one
          that already scores near 100% has little left to gain. The{" "}
          <a href="#results" className={link}>
            results
          </a>{" "}
          section has the full tables and how to reproduce them.
        </p>

        <h2 id="quick-start" className={h2}>
          Quick start
        </h2>
        <CodeSnippet filename="Terminal" language="bash" lineNumbers={false} code="pip install gpbacay-arcane" />
        <CodeSnippet
          filename="hippocampus.py"
          language="python"
          code={`from gpbacay_arcane import Hippocampus, load_arc1

hippocampus = Hippocampus(load_arc1())        # or any model, or Hippocampus() for memory only

# Labels: a text and the label it should get
hippocampus.remember("I was charged twice this month", "billing")
hippocampus.remember("my parcel never arrived", "shipping")

out = hippocampus.classify("why is my card charged again?", ["billing", "shipping"])
out["label"], out["source"]   # "billing", "model+memory"; out["confidence"] is the mixed probability
out["neighbors"]              # the remembered examples that voted

# Tools: a request, the tool, and the call's arguments
hippocampus.remember("rate Dune 4 stars", "rate_movie", {"title": "Dune", "stars": 4})
hippocampus.remember("switch bluetooth off", "toggle_bluetooth", {"enabled": False})
hippocampus.remember("thanks, that's all", None)   # None = no tool applies

out = hippocampus.run("rate spirited away 5 stars", tools=my_tools)
out["function_calls"]     # [{"name": "rate_movie", "arguments": {"title": "spirited away", "stars": 5}}]
out["memory_arguments"]   # ["rate_movie"]: these arguments came from a remembered request

hippocampus.react("charged again?")                              # the memory's vote alone, no model call
hippocampus.remember("I was charged twice this month", "fraud")  # same text: the label is corrected
hippocampus.forget("thanks, that's all")`}
        />

        <h2 id="how-it-works" className={h2}>
          How it works
        </h2>
        <Mermaid
          minWidth={560}
          chart={`flowchart LR
  EX["remember(text, label)"] --> G["Document graph<br/>one node per example"]
  Q["Request"] --> S["BM25 search<br/>top 8 examples"]
  G --> S
  S --> V["Vote<br/>score-weighted share per label"]
  Q --> M["Model<br/>its own probabilities"]
  V --> C{"Combine"}
  M --> C
  S --> A["Patterns<br/>rate {title} {stars} stars"]
  C --> OUT["Decision<br/>+ neighbors as evidence"]
  A --> OUT`}
        />
        <ol className="mt-4 max-w-2xl list-decimal space-y-2 pl-5 text-sm leading-relaxed text-zinc-300">
          <li>
            <strong className="text-zinc-100">Remember.</strong> Each example becomes one graph node. Its label is
            stored beside it, not in the searchable text. The id is a hash of the text, so remembering the same text
            again replaces the label.
          </li>
          <li>
            <strong className="text-zinc-100">Retrieve.</strong> BM25 keyword scoring finds the 8 examples most like
            the request. For <code className={code}>classify</code>, only examples with one of the offered labels
            count.
          </li>
          <li>
            <strong className="text-zinc-100">Vote.</strong> Each label gets its share of the neighbors&apos; total
            score, so the votes sum to 1. This is the k-nearest-neighbor step of{" "}
            <a href="https://arxiv.org/abs/1911.00172" target="_blank" rel="noreferrer" className={link}>
              kNN-LM
            </a>
            , applied to decisions instead of next words.
          </li>
          <li>
            <strong className="text-zinc-100">Combine.</strong> For <code className={code}>classify</code>, the vote
            and the model&apos;s distribution are averaged (<code className={code}>weight=0.5</code>). For{" "}
            <code className={code}>run</code>, the vote is passed to the model as{" "}
            <code className={code}>tool_prior</code>; ARC 1, for example, fires a tool when either source is
            confident (noisy-OR).
          </li>
          <li>
            <strong className="text-zinc-100">Fill arguments.</strong> A remembered call becomes a pattern: its
            argument values turn into typed slots, so &quot;rate Dune 4 stars&quot; becomes{" "}
            <code className={code}>{"rate {title} {stars} stars"}</code>. The most similar pattern that matches the
            request supplies the arguments, copied from the request&apos;s own words, so a value is never invented.
            Values that aren&apos;t in the text, like <code className={code}>enabled: false</code> for &quot;switch
            bluetooth off&quot;, come with the pattern&apos;s words. A matching pattern also adds a call the model
            held back, and drops a model call that copied the same words, since a word belongs to one argument.
            The call&apos;s <code className={code}>confidence</code> is then <code className={code}>None</code>: the
            model&apos;s calibrated probability no longer describes it.
          </li>
          <li>
            <strong className="text-zinc-100">Fall back.</strong> A label with no examples, or a request with no
            similar example, is decided by the model alone. With no model either, Hippocampus abstains.
          </li>
        </ol>
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          For tool calling, every example votes, including examples for tools that aren&apos;t offered and examples
          labeled <code className={code}>None</code>. A request that looks like something else therefore takes votes
          away from the offered tools instead of firing one.
        </p>

        <h2 id="any-model" className={h2}>
          Use it with any model
        </h2>
        <p className={p}>
          The memory never calls a model, so Hippocampus works with anything that turns a text and a list of labels into
          probabilities.
        </p>
        <div className="not-prose mt-6 overflow-x-auto">
          <table className="w-full min-w-[520px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className={th}>model=</th>
                <th className="py-2 font-medium">What it needs</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map(([name, body]) => (
                <tr key={name} className="border-b border-zinc-900 align-top">
                  <td className="py-2.5 pr-4 font-mono text-xs text-zinc-100">{name}</td>
                  <td className="py-2.5 text-zinc-400">{body}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <CodeSnippet
          filename="any_model.py"
          language="python"
          code={`from gpbacay_arcane import Hippocampus

# A scikit-learn pipeline (vectorizer + classifier) trained on your labels
def sklearn_model(text, labels):
    probs = dict(zip(pipeline.classes_, pipeline.predict_proba([text])[0]))
    return {label: probs.get(label, 0.0) for label in labels}

hippocampus = Hippocampus(sklearn_model)
hippocampus.remember("my card was declined at checkout", "billing")   # fix a blind spot without retraining

# Memory only: a router that learns from every confirmed decision
router = Hippocampus()
for text, label in confirmed_tickets:
    router.remember(text, label)
router.classify(new_ticket, ["billing", "shipping", "account"])  # label=None when nothing is similar`}
        />

        <h2 id="results" className={h2}>
          Results
        </h2>
        <p className={p}>
          All results use the bundled{" "}
          <Link href="/docs/arc-1" className={link}>
            ARC 1
          </Link>{" "}
          (arc1-tiny) as the base model, with the same weights throughout and no fine-tuning. The memory holds a few
          examples per label; test texts are never in it. Run{" "}
          <code className={code}>python examples/benchmark_hippocampus.py --shots 1 2 5 10</code> to reproduce.
        </p>
        <BarChart
          title="Intents the model never trained on, real text"
          subtitle="CLINC150 test utterances, 5 labels (chance 20%), one remembered example per label, n = 600"
          data={[
            { label: "Model alone", value: 0.565 },
            { label: "Memory alone", value: 0.647 },
            { label: "Hippocampus (model + memory)", value: 0.805 },
          ]}
          max={1}
          ticks={[0, 0.25, 0.5, 0.75, 1]}
          unit="percent"
          highlight="Hippocampus (model + memory)"
        />
        <p className={`${p} text-sm text-zinc-400`}>
          Neither source is enough on its own with one example. Combined, they beat both, so the model is still doing
          real work rather than being overruled by a lookup.
        </p>

        <h3 className="mt-10 text-lg font-semibold text-zinc-100">Real text, more examples</h3>
        <div className="not-prose mt-4 overflow-x-auto">
          <table className="w-full min-w-[480px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className={th}>Examples per label</th>
                <th className={th}>Model alone</th>
                <th className={th}>Memory alone</th>
                <th className="py-2 font-medium">Hippocampus</th>
              </tr>
            </thead>
            <tbody>
              {CLINC.map(([shots, arc1, memory, both]) => (
                <tr key={shots} className="border-b border-zinc-900">
                  <td className={td}>{shots}</td>
                  <td className={td}>{arc1}</td>
                  <td className={td}>{memory}</td>
                  <td className="py-2.5 font-semibold tabular-nums text-zinc-100">{both}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h3 className="mt-10 text-lg font-semibold text-zinc-100">Tools the model never trained on</h3>
        <p className={`${p} mt-2 text-sm text-zinc-400`}>
          The held-out tool split, n = 300. Fully correct means the right tools with every argument right. Correct
          refusals stayed at 100% in every row.
        </p>
        <div className="not-prose mt-4 overflow-x-auto">
          <table className="w-full min-w-[480px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className={th}>Examples per tool</th>
                <th className={th}>Right tool</th>
                <th className={th}>Fully correct, labels only</th>
                <th className={th}>Fully correct, with arguments</th>
                <th className="py-2 font-medium">Arguments right</th>
              </tr>
            </thead>
            <tbody>
              {TOOLS.map(([shots, sel, labelsOnly, exact, args]) => (
                <tr key={shots} className="border-b border-zinc-900">
                  <td className={td}>{shots}</td>
                  <td className={td}>{sel}</td>
                  <td className={td}>{labelsOnly}</td>
                  <td className="py-2.5 pr-4 font-semibold tabular-nums text-zinc-100">{exact}</td>
                  <td className="py-2.5 tabular-nums text-zinc-300">{args}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          Votes alone fix which tool fires, but calls stay near 55% because the base model copies the wrong words for
          arguments of tools it never trained on. Storing the arguments fixes that: with 5 examples, 86.0% of calls
          are fully right, the same as the{" "}
          <Link href="/docs/arc-1" className={link}>
            86.0%
          </Link>{" "}
          it scores on the tools it was trained on (measured on a different test set).
        </p>
        <p className="not-prose mt-4 max-w-2xl border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
          <strong>Read the 10-example row with care.</strong> These held-out tools are tested with the same sentence
          patterns the examples were written from, only with new values, so by 10 examples the memory has seen every
          phrasing. Real requests vary more. A phrasing no example covers falls back to the base model&apos;s own
          argument copying, so store varied examples.
        </p>

        <h3 className="mt-10 text-lg font-semibold text-zinc-100">Scale</h3>
        <p className={`${p} mt-2`}>
          Memory only, all 150 CLINC150 intents with 100 examples each (15,000 examples), 1,000 test utterances, on a
          busy laptop CPU:
        </p>
        <dl className="not-prose mt-4 grid max-w-2xl gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-4">
          {[
            ["80.6%", "150-way accuracy"],
            ["0.13 ms", "to add an example"],
            ["3.4 ms", "median decision"],
            ["11 ms", "90th percentile"],
          ].map(([value, label]) => (
            <div key={label} className="bg-zinc-950 p-4">
              <dt className="text-xs text-zinc-500">{label}</dt>
              <dd className="mt-1 text-xl font-semibold tabular-nums text-zinc-100">{value}</dd>
            </div>
          ))}
        </dl>
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          Adding an example never scans the memory, and a search only touches examples that share words with the
          request. Hippocampus&apos;s default graph leaves out the semantic links the document graph normally adds: on
          CLINC150 they lowered accuracy (0.811 to 0.790 at 15,000 examples) and made each insert take 15 ms instead
          of 0.09 ms.
        </p>

        <h2 id="saving" className={h2}>
          Saving the memory
        </h2>
        <p className={p}>
          Examples are documents whose ids start with <code className={code}>memory-</code>, with the label in the
          document&apos;s description and any tool arguments beside it. Saving the graph saves the memory, and a
          graph that also holds your documents keeps the two apart: documents never vote.
        </p>
        <CodeSnippet
          filename="save.py"
          language="python"
          code={`import json
from gpbacay_arcane import DocumentGraph, Hippocampus

json.dump(hippocampus.graph.to_json(), open("memory.json", "w"))

hippocampus = Hippocampus(model, DocumentGraph.from_json(json.load(open("memory.json"))))
hippocampus.labels()   # {"memory-3f2a...": "billing", ...}`}
        />

        <h2 id="limitations" className={h2}>
          Limitations
        </h2>
        <ul className="not-prose mt-2 max-w-2xl divide-y divide-zinc-900 border-y border-zinc-800">
          {[
            ["Measured on one model", "Every improvement figure on this page comes from arc1-tiny. Other models were not benchmarked, so treat the gains as an example of the effect, not a guarantee."],
            ["Keyword retrieval", "A request that shares no words with any example gets no vote and falls back to the model. Using ARC 1's embeddings as DocumentGraph(embed=...) did not meaningfully help on CLINC150."],
            ["Argument patterns", "A pattern only matches phrasings like a remembered one; others fall back to the model's arguments. A value at the start or end of a pattern stops at punctuation or at and, then, plus or also, so \"Tom and Jerry\" in that position is cut. English joining words only."],
            ["Uncalibrated mix", "Even if the model's probabilities are calibrated, the combined confidence has not been re-fitted, so treat it as a ranking."],
            ["Tested up to 15,000 examples", "Lookups slow down when common words match many examples. Expect that somewhere past about 100,000; it has not been measured."],
            ["Shared graphs", "If the graph also holds documents, each lookup filters across all stored examples. The default graph holds only examples."],
            ["Tool calling needs tool_prior", "Any model can classify, but run() needs a model that accepts tool_prior. Arc1Agent does; other tool-calling models need that one argument."],
          ].map(([title, body]) => (
            <li key={title} className="py-3.5">
              <p className="font-medium text-zinc-100">{title}</p>
              <p className="mt-1 text-sm leading-relaxed text-zinc-400">{body}</p>
            </li>
          ))}
        </ul>
        <p className={`${p} mt-6 text-sm text-zinc-400`}>
          Source: <code className={code}>gpbacay_arcane/hippocampus.py</code>. See also the{" "}
          <a href={README_URL} target="_blank" rel="noreferrer" className={link}>
            repository README
          </a>
          .
        </p>
      </div>
    </div>
  );
}
