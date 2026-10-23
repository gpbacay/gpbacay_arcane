import Link from "next/link";
import { Eye, FileVideo, Plug, ShieldCheck } from "lucide-react";
import { BarChart } from "@/components/BarChart";
import { OccipitalDemo } from "@/components/OccipitalDemo";
import { CodeSnippet } from "@/components/CodeSnippet";
import { Mermaid } from "@/components/markdown";

const h2Base = "scroll-mt-28 text-2xl font-bold tracking-tight text-zinc-100 mb-4 border-b border-zinc-800 pb-2";
const h2 = `${h2Base} mt-16`;
const code = "bg-zinc-900 px-1.5 py-0.5 text-zinc-100";
const p = "max-w-2xl leading-relaxed";
const link = "font-medium text-[#C785F2] underline hover:text-[#d49cf5]";
const th = "py-2 pr-4 font-medium";
const td = "py-2.5 pr-4 tabular-nums text-zinc-300";

const CAPABILITIES = [
  {
    icon: Eye,
    title: "Reads by letters",
    body: "Typos and other word forms still find the remembered examples a word search misses.",
  },
  {
    icon: Plug,
    title: "Any model",
    body: "It changes which examples vote, not the model, so ARC 1, a classifier or an LLM router all gain.",
  },
  {
    icon: ShieldCheck,
    title: "Keeps abstention",
    body: "A request unlike anything stored still gets no neighbors, so a memory-only Hippocampus still abstains.",
  },
  {
    icon: FileVideo,
    title: "Memories as video",
    body: "A whole memory can be written as QR-code frames of an MP4 and read back into any Hippocampus.",
  },
];

const BRAIN: [string, string, string][] = [
  ["The visual word form area recognizes words from their letters", "Letter 3- and 4-grams of every word", "\"parcle\" still shares \" pa\", \"par\", \"arc\", \"parc\"… with \"parcel\"."],
  ["The ventral stream carries what was read to the medial temporal lobe", "Letter scores fused with Hippocampus's word scores", "The closest examples, by either route, vote."],
  ["Familiar shapes are read at a glance, novel ones are not", "min_coverage", "A request made of unfamiliar letters gets no neighbors: abstain."],
  ["The visual cortex holds what is seen as images", "QR-code frames of an MP4", "A memory stored as pictures and read back by sight."],
];

// python examples/benchmark_hippocampus.py --shots 1 5 10 [--occipital]: CLINC150 held-out intents, 5 labels, n = 600.
const GAINS: [string, string, string, string, string][] = [
  ["1", "56.5%", "80.5%", "81.3%", "+0.8"],
  ["5", "56.5%", "90.5%", "92.2%", "+1.7"],
  ["10", "56.5%", "93.5%", "94.5%", "+1.0"],
];

// node bench-occipital-arc1.mjs 1 5 10 (node/): the same intents, its own sample, with typos and out-of-scope requests.
const TYPOS: [string, string, string, string, string, string][] = [
  ["1", "48.0%", "66.2%", "66.7%", "71.8%", "88.7% → 88.7%"],
  ["5", "48.0%", "74.8%", "76.3%", "82.0%", "69.2% → 68.8%"],
  ["10", "48.0%", "81.8%", "83.0%", "87.8%", "60.1% → 59.6%"],
];

// node bench-occipital.mjs ../docs (node/): the 13 Markdown files in docs/.
const SIZES = [
  { label: "Text", value: 118766 },
  { label: "Text, gzipped", value: 42994 },
  { label: "Occipital video", value: 680660, detail: "269 frames" },
];

export default function OccipitalPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <header className="not-prose mb-12">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#C785F2]">ARCANE · Visual pathway</p>
        <h1 className="mt-3 max-w-3xl text-4xl font-extrabold leading-[1.08] tracking-[-0.03em] text-zinc-50 sm:text-5xl">
          Occipital
        </h1>
        <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-zinc-300">
          The visual pathway of{" "}
          <Link href="/docs/hippocampus" className={link}>
            Hippocampus
          </Link>
          . Hippocampus recalls remembered decisions by whole words. Occipital also reads them by their letters, the way
          the brain&apos;s visual word form area does, so a typo or another word form still finds its neighbors.
        </p>
        <p className="mt-4 max-w-2xl text-[17px] leading-relaxed text-zinc-300">
          Better recall means a better vote, and the vote steers whatever System 1 model sits behind the memory. With{" "}
          <Link href="/docs/arc-1" className={link}>
            ARC 1
          </Link>
          , it added 0.8 to 1.7 points on intents ARC 1 never trained on, and about 5 to 7 points on requests with typos
          when tuned for noisy input. See{" "}
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

        <h2 id="brain" className={h2}>
          Inspired by the brain
        </h2>
        <p className={p}>
          Reading starts in the occipital lobe. The visual word form area recognizes a word from its letters, which is
          why a misspelled word is still read, and the ventral stream carries what was read toward the medial temporal
          lobe, where the hippocampus binds it to memory. Occipital plays that role for Hippocampus.
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
          The analogy is loose. The visual word form area is learned from years of reading; Occipital&apos;s letter
          features are fixed, and nothing in it is trained.
        </p>

        <h2 id="improvement" className={h2}>
          How much it improves
        </h2>
        <p className={p}>
          ARC 1 (arc1-tiny) with a Hippocampus of CLINC150 examples for 29 intents it never trained on, 5 labels per
          request, n = 600. The same examples, the same model; the only change is the Occipital.
        </p>
        <div className="not-prose mt-6 overflow-x-auto">
          <table className="w-full min-w-[520px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className={th}>Examples per intent</th>
                <th className={th}>ARC 1 alone</th>
                <th className={th}>ARC 1 + Hippocampus</th>
                <th className={th}>+ Occipital</th>
                <th className="py-2 font-medium">Gain</th>
              </tr>
            </thead>
            <tbody>
              {GAINS.map(([shots, alone, hippo, occ, gain]) => (
                <tr key={shots} className="border-b border-zinc-900">
                  <td className={td}>{shots}</td>
                  <td className={td}>{alone}</td>
                  <td className={td}>{hippo}</td>
                  <td className="py-2.5 pr-4 font-semibold tabular-nums text-zinc-100">{occ}</td>
                  <td className="py-2.5 tabular-nums text-zinc-300">{gain}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          Run <code className={code}>python examples/benchmark_hippocampus.py --shots 1 5 10 --occipital</code> to
          reproduce. Memory alone (no model), all 150 CLINC150 intents with 15,000 examples: 80.1% with words, 83.2%
          with Occipital.
        </p>

        <h3 className="mt-10 text-lg font-semibold text-zinc-100">Requests with typos</h3>
        <p className={`${p} mt-2 text-sm text-zinc-400`}>
          The same test with half of every request&apos;s longer words given one dropped, swapped or wrong letter, and
          memory-only abstention on CLINC150&apos;s 1,000 out-of-scope requests. Run{" "}
          <code className={code}>node bench-occipital-arc1.mjs 1 5 10</code> in <code className={code}>node/</code>.
        </p>
        <div className="not-prose mt-4 overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className={th}>Examples per intent</th>
                <th className={th}>ARC 1 alone</th>
                <th className={th}>+ Hippocampus</th>
                <th className={th}>+ Occipital</th>
                <th className={th}>+ Occipital, min_coverage 0.1</th>
                <th className="py-2 font-medium">Out-of-scope abstained (memory alone)</th>
              </tr>
            </thead>
            <tbody>
              {TYPOS.map(([shots, alone, hippo, occ, low, oos]) => (
                <tr key={shots} className="border-b border-zinc-900">
                  <td className={td}>{shots}</td>
                  <td className={td}>{alone}</td>
                  <td className={td}>{hippo}</td>
                  <td className="py-2.5 pr-4 font-semibold tabular-nums text-zinc-100">{occ}</td>
                  <td className={td}>{low}</td>
                  <td className="py-2.5 tabular-nums text-zinc-400">{oos}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          With the default settings, abstention stays within half a point of words alone. Lowering{" "}
          <code className={code}>min_coverage</code> to 0.1 lifts ARC 1 on typos by another 4.8 to 5.7 points, but a
          memory-only Hippocampus then abstains far less often (55.2% instead of 69.2% at 5 examples), so use it when a
          model always decides anyway.
        </p>

        <h2 id="try-it" className={h2}>
          Try it
        </h2>
        <p className={p}>
          The demo stores your text as frames, reads every frame back from its pixels, and searches what it read twice:
          by words only, and by words and letters. The default query has two typos and no other word in common with the text.
        </p>
        <OccipitalDemo />

        <h2 id="quick-start" className={h2}>
          Quick start
        </h2>
        <CodeSnippet
          filename="occipital.py"
          language="python"
          code={`from gpbacay_arcane import Hippocampus, Occipital, load_arc1

hippocampus = Hippocampus(load_arc1(), occipital=Occipital())   # any model, or None for memory only

hippocampus.remember("my parcel never arrived", "shipping")
hippocampus.remember("I was charged twice", "billing")

hippocampus.classify("my parcle nevr arived", ["billing", "shipping"])
# "shipping": found through its letters; words alone share only "my", a stopword`}
        />
        <CodeSnippet
          filename="occipital.mjs"
          language="javascript"
          code={`import { load, Hippocampus } from "gpbacay-arcane/web";
import { Occipital } from "gpbacay-arcane/occipital";

const hippocampus = new Hippocampus(await load(), { occipital: new Occipital() });`}
        />
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          Give every Hippocampus its own Occipital: it indexes that memory&apos;s examples as they are remembered,
          observed and forgotten, and a loaded graph brings its examples with it. Python and JavaScript return the same
          scores.
        </p>

        <h2 id="how-it-works" className={h2}>
          How it works
        </h2>
        <Mermaid
          chart={`flowchart LR
  Q["Request"] --> W["Word search<br/>BM25 over stems"]
  Q --> L["Letter search<br/>BM25 over 3- and 4-grams"]
  L --> G{"Covers 30%<br/>of the letters?"}
  W --> F["Fuse<br/>half words, half letters, squared"]
  G -- yes --> F
  F --> V["Top 8 vote"]
  V --> M["Mixed with any model"]`}
        />
        <ol className="mt-4 max-w-2xl list-decimal space-y-2 pl-5 text-sm leading-relaxed text-zinc-300">
          <li>
            <strong className="text-zinc-100">Index.</strong> Every remembered example is split into the letter 3- and
            4-grams of its words, padded so word starts and ends count (&quot; par&quot;, &quot;cel &quot;).
          </li>
          <li>
            <strong className="text-zinc-100">Score.</strong> A request is scored against those grams with BM25. A gram
            found in more than 10% of the examples (<code className={code}>max_df</code>) is too common to say which
            example is meant, so it scores nobody, which also keeps lookups fast.
          </li>
          <li>
            <strong className="text-zinc-100">Gate.</strong> An example found only by letters counts if it covers at
            least 30% of the request&apos;s letter evidence (<code className={code}>min_coverage</code>, idf-weighted,
            counting letters no example has). This is what keeps abstention.
          </li>
          <li>
            <strong className="text-zinc-100">Fuse.</strong> Word and letter scores, each scaled to a top of 1, are
            averaged and squared (<code className={code}>sharpness=2</code>), so the closest examples dominate the vote.
            Without the square, letters spread the vote thin and the model&apos;s share was diluted: ARC 1 lost 2 points
            at one example per intent.
          </li>
        </ol>
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          <code className={code}>weight</code>, <code className={code}>sharpness</code> and{" "}
          <code className={code}>min_coverage</code> were chosen on CLINC150&apos;s validation split. The default{" "}
          <code className={code}>max_df</code> was chosen for speed and for keeping no-tool refusals at 100%, with the
          test numbers visible.
        </p>

        <h2 id="video" className={h2}>
          Memories as video
        </h2>
        <p className={p}>
          A memory can also be written as QR-code frames of an MP4, one example or passage per frame, and read back into
          any Hippocampus (Node.js). A memory read from video made the same decisions with the same confidence as the
          original on 600 of 600 requests, so a taught skill ships as one file.
        </p>
        <CodeSnippet filename="Terminal" language="bash" lineNumbers={false} code="npm install gpbacay-arcane qrcode jsqr ffmpeg-static" />
        <CodeSnippet
          filename="skill.mjs"
          language="javascript"
          code={`import { load, Hippocampus } from "gpbacay-arcane/web";
import { Occipital } from "gpbacay-arcane/occipital";
import { writeVideo, readVideo } from "gpbacay-arcane/occipital-video";

const teach = new Hippocampus();
teach.remember("rate Dune 4 stars", "rate_movie", { title: "Dune", stars: 4 });
await writeVideo("skill.mp4", teach.toJSON().map((e) => [e.label, e.text, e.arguments]));

const { memory, lost } = await readVideo("skill.mp4", { model: await load(), occipital: new Occipital() });

// Documents work too: each passage is labeled by its document
await writeVideo("kb.mp4", { "refunds.md": refundsText, "shipping.md": shippingText });`}
        />
        <BarChart
          title="Bytes to store 13 docs (119 KB of text)"
          subtitle="node bench-occipital.mjs ../docs: 500-character passages, 4 px per module, H.264 CRF 23"
          data={SIZES}
          max={8e5}
          ticks={[0, 2e5, 4e5, 6e5, 8e5]}
          unit="mb"
          highlight="Occipital video"
        />
        <p className="not-prose mt-2 max-w-2xl border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
          <strong>This is not compression.</strong> The video is 5.7× its text and 16× the gzipped text. It is for
          when one self-contained file is worth that size. Every frame still decoded after re-encoding to H.264 at CRF
          35 or to VP9, and reading runs at about 50 frames per second.
        </p>

        <h2 id="limitations" className={h2}>
          Limitations
        </h2>
        <ul className="not-prose mt-2 max-w-2xl divide-y divide-zinc-900 border-y border-zinc-800">
          {[
            ["Modest by default", "With the default settings ARC 1 gains 0.5 to 1.7 points, on clean and typo'd requests alike. The larger typo gains need min_coverage 0.1, which costs memory-only abstention."],
            ["Not for tool memories", "Tool choice and exact calls did not change, and with one example per tool ARC 1 fired a tool on 1.2% of requests that need none (100% to 98.8% correct refusals). Leave Occipital off for a tool-calling Hippocampus."],
            ["Slower", "At 15,000 examples a decision takes 16 ms instead of 4.2 ms (median, Python). Lower max_df for speed: at 0.05 it is 9 ms and 81.8% instead of 83.2%."],
            ["Spelling, not meaning", "Letters catch typos and shared word parts, not synonyms: \"cancel my order\" and \"stop my purchase\" still share nothing."],
            ["One benchmark", "Every figure comes from CLINC150 with arc1-tiny, and the typos are synthetic. Real user errors and other models were not measured."],
            ["Video is Node only", "Writing and reading MP4s needs Node.js and FFmpeg; the letter pathway itself runs in Python, Node and the browser."],
          ].map(([title, body]) => (
            <li key={title} className="py-3.5">
              <p className="font-medium text-zinc-100">{title}</p>
              <p className="mt-1 text-sm leading-relaxed text-zinc-400">{body}</p>
            </li>
          ))}
        </ul>
        <p className={`${p} mt-6 text-sm text-zinc-400`}>
          Source: <code className={code}>gpbacay_arcane/occipital.py</code>, <code className={code}>node/occipital.mjs</code>{" "}
          and <code className={code}>node/occipital-video.mjs</code>.
        </p>
      </div>
    </div>
  );
}
