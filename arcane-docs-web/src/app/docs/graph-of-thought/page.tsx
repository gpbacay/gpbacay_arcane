import Link from "next/link";
import { GitMerge, Network, RefreshCw, Route } from "lucide-react";
import { CodeSnippet } from "@/components/CodeSnippet";
import { Mermaid } from "@/components/markdown";

const h2Base = "scroll-mt-28 text-2xl font-bold tracking-tight text-zinc-100 mb-4 border-b border-zinc-800 pb-2";
const h2 = `${h2Base} mt-16`;
const code = "bg-zinc-900 px-1.5 py-0.5 text-zinc-100";
const p = "max-w-2xl leading-relaxed";
const link = "font-medium text-[#C785F2] underline hover:text-[#d49cf5]";

const GOT_REPO = "https://github.com/gpbacay/graph-of-thought";
const README_URL = "https://github.com/gpbacay/gpbacay_arcane#grounded-graph-of-thought--document-knowledge-graph-and-verified-reasoning";

const CAPABILITIES = [
  {
    icon: Network,
    title: "No vector database",
    body: "Keyword scoring plus a walk along document links. Needs only numpy, with no API keys.",
  },
  {
    icon: Route,
    title: "Explains every hit",
    body: "Each result says how it was reached, such as a reference from Troubleshooting.",
  },
  {
    icon: RefreshCw,
    title: "Live documents",
    body: "Add, replace or remove a document at any time. Only that document's nodes change.",
  },
  {
    icon: GitMerge,
    title: "Verified reasoning",
    body: "Every sentence is checked against the documents. Unsupported ones trigger a search, then a rewrite.",
  },
];

const EDGES: [string, string, string][] = [
  ["parent-child", "Heading hierarchy. Walking up to a parent counts half as much as walking down.", "0.8"],
  ["reference", "A section's text names another section's title, as in \"see Configuration\".", "0.75"],
  ["next", "Reading order between neighbouring sections and chunks.", "0.4"],
  ["semantic", "Shared distinctive terms, or embedding similarity when you pass an embedder. Up to 5 per node.", "0.3 to 0.9"],
];

const OPS: [string, string][] = [
  ["ops.retrieve(query=None)", "Searches the graph for the question or a sub-query and attaches the hits as evidence."],
  ["ops.generate(k)", "Branches each thought into k candidate answers in one LLM call."],
  ["ops.keep_best(n)", "Keeps the n most grounded thoughts, weighted by how relevant their sections are to the question, and drops weak or off-topic ones while a better one exists."],
  ["ops.aggregate()", "Merges several thoughts into one, the step a tree of thoughts can't do. The merge is kept only if it is at least as grounded as its best parent."],
  ["ops.refine(max_rounds)", "Searches the graph with each unsupported sentence first, then asks the LLM to rewrite only what is still unsupported. A rewrite that lowers grounding is discarded."],
];

const TOOLS: [string, string][] = [
  ["got_search", "Find sections about a topic, with snippets and how each was reached"],
  ["got_read_node", "Full text of a section, with its parent, children and links"],
  ["got_outline", "Table of contents of a document, with node ids"],
  ["got_list_documents", "Documents in the graph"],
  ["got_add_document, got_remove_document", "Change the graph. Only with writable=True"],
  ["got_reason", "Cited answer through Graph-of-Thoughts. Only when you pass a reasoner"],
];

export default function GraphOfThoughtPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <header className="not-prose mb-12">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#C785F2]">ARCANE · Harness</p>
        <h1 className="mt-3 max-w-3xl text-4xl font-extrabold leading-[1.08] tracking-[-0.03em] text-zinc-50 sm:text-5xl">
          Grounded Graph of Thought
        </h1>
        <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-zinc-300">
          A document knowledge graph, retrieval that explains itself and Graph-of-Thoughts reasoning that checks
          every sentence against your documents. Give it your documents and it finds the sections that answer a
          question, including the ones a keyword search would miss because they are only linked to the match.
        </p>
        <p className="mt-4 max-w-2xl text-[17px] leading-relaxed text-zinc-300">
          Retrieval runs offline with no API keys, embeddings or vector database. Add an LLM and the same graph
          verifies its reasoning: unsupported sentences are searched for, rewritten or reported, and only sections
          that back a sentence are cited.
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
          The document graph is a Python port of{" "}
          <a href={GOT_REPO} target="_blank" rel="noreferrer" className={link}>
            graph-of-thought
          </a>
          . The verified reasoning and ARCANE tool specs are added on top.
        </p>

        <h2 id="quick-start" className={h2}>
          Quick start
        </h2>
        <CodeSnippet filename="Terminal" language="bash" lineNumbers={false} code="pip install gpbacay-arcane" />
        <CodeSnippet
          filename="retrieve.py"
          language="python"
          code={`from gpbacay_arcane import DocumentGraph

guide = """# User Guide

## Installation
Run pip install gpbacay-arcane.

## Configuration
Edit config.json to customize settings.

## Deployment
Push the build to the server.

## Troubleshooting
If config fails, see the Configuration section.
"""

graph = DocumentGraph()
graph.add_document(guide, "User Guide")   # the same title again replaces it

for hit in graph.search("config fails"):
    print(hit["title"], hit["edgeType"], round(hit["score"], 2))
# Troubleshooting None 1.0             <- direct match
# Configuration reference 0.78         <- "see the Configuration section"
# Deployment next 0.34
# Installation parent-child 0.23

context = graph.retrieve("deploy")       # "### Deployment [user-guide#3]\\nPush the build..."`}
        />
        <p className={`${p} mt-4`}>
          Pass <code className={code}>context</code> to any language model as grounding. Each hit also has{" "}
          <code className={code}>path</code>, <code className={code}>hops</code> and{" "}
          <code className={code}>lexicalScore</code>, and <code className={code}>graph.outline(&quot;user-guide&quot;)</code>{" "}
          prints the table of contents with node ids.
        </p>

        <h2 id="how-retrieval-works" className={h2}>
          How retrieval works
        </h2>
        <Mermaid
          minWidth={520}
          chart={`flowchart LR
  DOC["Markdown documents"] --> PARSE["Headings become nodes<br/>long sections become chunks"]
  PARSE --> G["Document graph<br/>parent-child, reference, next, semantic"]
  Q["Query"] --> BM["BM25 keyword scoring<br/>(+ embedding similarity)"]
  G --> BM
  BM --> SEEDS["Top seeds"]
  SEEDS --> WALK["Bounded multi-source walk<br/>score weakens each hop"]
  WALK --> HITS["Ranked hits<br/>with path and edge type"]`}
        />
        <ol className="mt-4 max-w-2xl list-decimal space-y-2 pl-5 text-sm leading-relaxed text-zinc-300">
          <li>
            <strong className="text-zinc-100">Index.</strong> Each document gets a root node and one node per heading.
            Sections longer than 2,000 characters are split into chunk nodes. <code className={code}>#</code> lines
            inside code blocks are ignored.
          </li>
          <li>
            <strong className="text-zinc-100">Seed.</strong> BM25 scores every section against the query. Words are
            reduced to their stems, so <code className={code}>deploy</code> matches &quot;Deployment&quot;.
          </li>
          <li>
            <strong className="text-zinc-100">Walk.</strong> Starting from the top 5 seeds, a Dijkstra search follows
            edges for up to 2 hops. Each hop multiplies the score by the edge weight and 0.85, and paths that fall below
            0.1 stop.
          </li>
          <li>
            <strong className="text-zinc-100">Rank.</strong> A section that matches the query and is also reached
            through the graph scores higher than either alone. A query only touches the part of the graph it activates.
          </li>
        </ol>
        <div className="not-prose mt-6 overflow-x-auto">
          <table className="w-full min-w-[520px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className="py-2 pr-4 font-medium">Edge</th>
                <th className="py-2 pr-4 font-medium">Meaning</th>
                <th className="py-2 font-medium">Weight</th>
              </tr>
            </thead>
            <tbody>
              {EDGES.map(([type, meaning, weight]) => (
                <tr key={type} className="border-b border-zinc-900 align-top">
                  <td className="py-2.5 pr-4 font-mono text-xs text-zinc-100">{type}</td>
                  <td className="py-2.5 pr-4 text-zinc-400">{meaning}</td>
                  <td className="py-2.5 tabular-nums text-zinc-300">{weight}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h2 id="reasoning" className={h2}>
          Grounded Graph-of-Thoughts reasoning
        </h2>
        <p className={p}>
          <code className={code}>GroundedGraphOfThought</code> follows{" "}
          <a href="https://arxiv.org/abs/2308.09687" target="_blank" rel="noreferrer" className={link}>
            Besta et al., 2023
          </a>
          , except that the LLM never grades its own answers. Each thought&apos;s sentences are checked against the
          document graph when the thought is created, by how much of each sentence&apos;s rare-word weight its evidence
          covers. No LLM call is involved. That one check ranks candidates, decides whether a merge or rewrite is
          kept, turns each unsupported sentence into a graph search, and picks the citations.
        </p>
        <Mermaid
          minWidth={560}
          chart={`flowchart LR
  R["retrieve"] --> GEN["generate 3"]
  GEN --> KB["keep best 2<br/>by grounding"]
  KB --> AGG["aggregate<br/>kept only if not less grounded"]
  AGG --> CHK{"unsupported<br/>sentences?"}
  CHK -- "no" --> OUT["answer + citations"]
  CHK -- "yes" --> SEARCH["search graph<br/>with each sentence"]
  SEARCH --> CHK2{"still<br/>unsupported?"}
  CHK2 -- "no" --> OUT
  CHK2 -- "yes" --> REW["LLM rewrites<br/>those sentences"]
  REW --> CHK`}
        />
        <CodeSnippet
          filename="reason.py"
          language="python"
          code={`from gpbacay_arcane import GroundedGraphOfThought
from gpbacay_arcane.got import ops

def my_llm(prompt: str) -> str:
    ...  # call any model (hosted, Ollama, local) and return its text

got = GroundedGraphOfThought(graph, llm=my_llm)  # max_llm_calls=16, min_support=0.5
result = got.reason("My config fails. What should I check?")

result["answer"]       # final answer
result["claims"]       # each sentence with its support (0-1) and the sections backing it
result["unsupported"]  # sentences the documents do not back
result["citations"]    # only sections that back a supported sentence
result["thoughts"]     # the whole thought graph: ids, parents, evidence, grounding
result["llm_calls"], result["budget_exhausted"]

graph.support("Set DATABASE_URL before starting.", ["user-guide#3"])  # (1.0, ["user-guide#3"], mass)

# Optional entailment check: (claim, evidence) -> probability, e.g. an NLI model
got = GroundedGraphOfThought(graph, llm=my_llm, verify=my_nli)

# Custom plan
got.reason(question, [ops.retrieve(), ops.generate(5), ops.keep_best(3), ops.aggregate(), ops.refine(3)])`}
        />
        <ul className="not-prose mt-6 max-w-2xl divide-y divide-zinc-900 border-y border-zinc-800">
          {OPS.map(([name, body]) => (
            <li key={name} className="py-3">
              <p className="font-mono text-xs text-zinc-100">{name}</p>
              <p className="mt-1 text-sm leading-relaxed text-zinc-400">{body}</p>
            </li>
          ))}
        </ul>
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          The default plan makes 1 to 4 LLM calls: one to generate, one to merge if two candidates are grounded,
          and up to two rewrites if sentences stay unsupported after searching. Every call is capped by{" "}
          <code className={code}>max_llm_calls</code>, and <code className={code}>budget_exhausted</code> tells you
          when a step was skipped.
        </p>

        <h2 id="embeddings" className={h2}>
          Embeddings
        </h2>
        <p className={p}>
          Without embeddings, semantic edges come from shared distinctive terms. Pass any{" "}
          <code className={code}>text -&gt; vector</code> function and sections are linked by meaning instead, and the
          closest sections by meaning also become search seeds. Links and seeds need a cosine similarity of at least
          0.3 by default; change it with <code className={code}>min_similarity</code>.
        </p>
        <CodeSnippet
          filename="embed.py"
          language="python"
          code={`from gpbacay_arcane import DocumentGraph

graph = DocumentGraph(embed=my_sentence_embedder)   # any text -> vector function
graph.add_document(guide, "User Guide")`}
        />
        <p className="not-prose mt-4 border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
          <strong>Don&apos;t use arc1-tiny&apos;s embedder here yet.</strong> Its similarities don&apos;t follow meaning
          (&quot;install the package with pip&quot; vs &quot;how to set up the library&quot; scores -0.02), so it adds
          misleading links. Use a sentence-embedding model, or leave <code className="text-amber-100">embed</code> unset.
          See{" "}
          <Link href="/docs/arc-1" className="underline">
            ARC 1
          </Link>{" "}
          for what it is trained on.
        </p>

        <h2 id="agent-tools" className={h2}>
          Agent tools
        </h2>
        <p className={p}>
          <code className={code}>got_tools(graph)</code> returns ARCANE <code className={code}>ToolSpec</code>s, so an
          agent can search and read the graph itself. Give <code className={code}>spec.schema_dict()</code> to any
          tool-calling model and run its calls with <code className={code}>execute_tools</code>.
        </p>
        <div className="not-prose mt-6 overflow-x-auto">
          <table className="w-full min-w-[520px] border-collapse text-left text-sm">
            <tbody>
              {TOOLS.map(([name, body]) => (
                <tr key={name} className="border-b border-zinc-900 align-top">
                  <td className="py-2.5 pr-4 font-mono text-xs text-zinc-100">{name}</td>
                  <td className="py-2.5 text-zinc-400">{body}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <CodeSnippet
          filename="tools.py"
          language="python"
          code={`from gpbacay_arcane import got_tools
from gpbacay_arcane.tools import execute_tools

tools = got_tools(graph)                    # read-only; writable=True adds add/remove
schemas = [t.schema_dict() for t in tools]  # {"name", "description", "parameters"}

# After your model returns tool calls:
execute_tools([{"name": "got_search", "arguments": {"query": "deploy"}}], tools)`}
        />
        <p className="not-prose mt-4 border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
          <strong>Use an LLM to drive the tools for now.</strong> The bundled arc1-tiny was not trained on these tools
          and rarely picks the right one. Fine-tune ARC 1 on graph-tool examples before handing them to{" "}
          <code className="text-amber-100">Arc1Agent</code>.
        </p>

        <h2 id="saving-graphs" className={h2}>
          Saving and sharing graphs
        </h2>
        <p className={p}>
          <code className={code}>graph.to_json()</code> returns a plain dictionary and{" "}
          <code className={code}>DocumentGraph.from_json(data)</code> rebuilds the graph. The format is the same as the
          Node.js{" "}
          <a href={GOT_REPO} target="_blank" rel="noreferrer" className={link}>
            graph-of-thought
          </a>{" "}
          v2 package, so a graph built in Python can be served by its MCP server, and the reverse.
        </p>
        <CodeSnippet
          filename="save.py"
          language="python"
          code={`import json

json.dump(graph.to_json(), open("graph.json", "w"))
graph = DocumentGraph.from_json(json.load(open("graph.json")))`}
        />

        <h2 id="limitations" className={h2}>
          Limitations
        </h2>
        <ul className="not-prose mt-2 max-w-2xl divide-y divide-zinc-900 border-y border-zinc-800">
          {[
            ["Markdown headings only", "Plain text without headings is kept under the document root and split into chunks."],
            ["Simple stemming", "Common suffixes are removed, but it is not a full stemmer, so some word forms still miss."],
            ["Size", "Semantic linking compares each new section with every other section, which suits up to tens of thousands of sections."],
            ["Term links age", "Term-based links use word rarity at the time a document is added. Re-add a document to refresh them."],
            ["Lexical verification", "The check counts shared words, so an honest paraphrase scores lower and a negated claim (\"never edit config.json\") still matches. Pass verify= (an NLI model, for example) to catch contradictions."],
            ["Not benchmarked yet", "The design is tested with scripted models only. Its effect on answer quality against a real LLM has not been measured."],
            ["Sequential LLM calls", "Reasoning steps call the model one at a time, so a plan's latency is the sum of its calls."],
            ["Tool routing", "Use an LLM to pick tools; arc1-tiny is not trained for them yet."],
          ].map(([title, body]) => (
            <li key={title} className="py-3.5">
              <p className="font-medium text-zinc-100">{title}</p>
              <p className="mt-1 text-sm leading-relaxed text-zinc-400">{body}</p>
            </li>
          ))}
        </ul>
        <p className={`${p} mt-6 text-sm text-zinc-400`}>
          Source: <code className={code}>gpbacay_arcane/got.py</code>. See also the{" "}
          <a href={README_URL} target="_blank" rel="noreferrer" className={link}>
            repository README
          </a>
          .
        </p>
      </div>
    </div>
  );
}
