import Link from "next/link";
import { FileVideo, Film, SearchCheck, ServerOff } from "lucide-react";
import { BarChart } from "@/components/BarChart";
import { CapsuleDemo } from "@/components/CapsuleDemo";
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
    icon: FileVideo,
    title: "One file",
    body: "Documents and their search index live in one MP4 that you can copy, stream or archive.",
  },
  {
    icon: ServerOff,
    title: "No vector database",
    body: "Opening the file rebuilds a Hippocampus search over the chunks. You don't need an embedder, API key or server.",
  },
  {
    icon: Film,
    title: "Survives re-encoding",
    body: "Every frame still decoded after re-encoding to H.264 at CRF 35 or to VP9 at CRF 40.",
  },
  {
    icon: SearchCheck,
    title: "Shows its evidence",
    body: "Results are the stored passages themselves, each labeled with the document it came from.",
  },
];

// node bench-capsule.mjs ../docs (node/), and QRyptoRAG v1's encoder settings on the same 13 files.
const SIZES = [
  { label: "Text", value: 118766 },
  { label: "Text, gzipped", value: 42994 },
  { label: "Arcane Capsule", value: 673207, detail: "269 frames" },
  { label: "QRyptoRAG v1", value: 6323159, detail: "270 frames" },
];

// Same corpus, one setting changed from the defaults (scale 4, CRF 23, 500-character chunks). Every row decoded 100%.
const SWEEP: [string, string, string][] = [
  ["Defaults", "5.7×", "Every frame decodes, with headroom for re-encoding."],
  ["CRF 35", "4.8×", "A lower-quality encode. It decodes here, with less headroom."],
  ["1,000-character chunks", "4.3×", "Half the frames, but retrieval returns coarser passages."],
  ["Scale 8 (8 px per module)", "4.5×", "Opens 2.3× slower, because jsQR reads 4× the pixels."],
  ["Scale 2", "14.6×", "Smaller frames compress worse."],
  ["Scale 3", "40.0×", "Module edges fall off H.264's block grid, which most likely explains the size."],
];

const CHANGES: [string, string, string][] = [
  ["What's in a frame", "A chunk's plain text", "deflate([version, document id, chunk]) in QR byte mode"],
  ["Search index", "Embeddings in a separate vector database, which also stored the text", "None stored. Opening the frames rebuilds a Hippocampus (BM25)"],
  ["Default embedder", "A sin/cos hash of the string, with no notion of meaning", "Not needed. Keyword search over the decoded chunks"],
  ["Chunking", "Fixed character windows that cut words in half", "Cut at whitespace, with overlap that starts on a word"],
  ["Frames to video", "PNG files in a temp folder under the working directory", "Raw pixels piped straight into FFmpeg"],
  ["Reading", "One FFmpeg call per retrieved frame, plus PNG decoding", "One FFmpeg pass that streams raw frames to jsQR"],
  ["Size claim", "\"80–95% smaller\"", "Measured at 5.7× the text, on this page"],
];

export default function CapsulePage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <header className="not-prose mb-12">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#C785F2]">ARCANE · Portable memory</p>
        <h1 className="mt-3 max-w-3xl text-4xl font-extrabold leading-[1.08] tracking-[-0.03em] text-zinc-50 sm:text-5xl">
          Arcane Capsule
        </h1>
        <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-zinc-300">
          A knowledge base in one video file. Capsule packs documents into QR-code frames of an MP4. Opening it decodes the
          frames and rebuilds a{" "}
          <Link href="/docs/hippocampus" className={link}>
            Hippocampus
          </Link>{" "}
          search over them, so retrieval works offline from the file alone.
        </p>
        <p className="mt-4 max-w-2xl text-[17px] leading-relaxed text-zinc-300">
          Capsule is the successor to QRyptoRAG. It is portable, but it doesn&apos;t compress: a capsule is about 5.7× the
          size of its text. See{" "}
          <a href="#size" className={link}>
            measured size
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

        <h2 id="try-it" className={h2}>
          Try it
        </h2>
        <p className={p}>
          The demo packs your text into frames, draws each frame as a QR code, reads every code back from its pixels and
          searches what it read. The Node package goes one step further and writes the frames to an MP4.
        </p>
        <CapsuleDemo />

        <h2 id="quick-start" className={h2}>
          Quick start
        </h2>
        <CodeSnippet
          filename="Terminal"
          language="bash"
          lineNumbers={false}
          code="npm install gpbacay-arcane qrcode jsqr ffmpeg-static"
        />
        <CodeSnippet
          filename="capsule.mjs"
          language="javascript"
          code={`import { writeCapsule, openCapsule } from "gpbacay-arcane/capsule-video";

await writeCapsule("kb.mp4", { "refunds.md": refundsText, "shipping.md": shippingText });
// { frames, bytes, textBytes }

const { memory, lost } = await openCapsule("kb.mp4");  // lost = frames that could not be read
memory.neighbors("how long do refunds take?");         // [{ text, label: "refunds.md", score }]
memory.react("how long do refunds take?").votes;       // { "refunds.md": 1 }: which document answers

// Mix the memory with ARC 1 to route requests to the right document
import { load, Hippocampus } from "gpbacay-arcane/web";
const router = new Hippocampus(await load(), { memory: memory.toJSON() });`}
        />
        <p className={`${p} mt-4 text-sm text-zinc-400`}>
          FFmpeg comes from <code className={code}>$FFMPEG_PATH</code>, then <code className={code}>ffmpeg-static</code>{" "}
          if it&apos;s installed, then <code className={code}>ffmpeg</code> on your PATH. For another medium, such as
          printed codes, a canvas or a different video library, use{" "}
          <code className={code}>gpbacay-arcane/capsule</code>. It has no dependencies and runs in the browser:{" "}
          <code className={code}>pack(documents)</code> returns one payload per QR code, and{" "}
          <code className={code}>unpack(payloads)</code> rebuilds the search.
        </p>

        <h2 id="how-it-works" className={h2}>
          How it works
        </h2>
        <Mermaid
          chart={`flowchart LR
  D["Documents<br/>{ id: text }"] --> C["Chunk<br/>500 chars, cut at whitespace"]
  C --> Z["Deflate<br/>[version, id, chunk]"]
  Z --> Q["QR code<br/>byte mode, level M"]
  Q --> V["H.264 MP4<br/>one frame per chunk"]
  V --> R["Open: FFmpeg → jsQR<br/>every frame"]
  R --> H["Hippocampus<br/>label = document id"]
  H --> S["neighbors / react"]`}
        />
        <ol className="mt-4 max-w-2xl list-decimal space-y-2 pl-5 text-sm leading-relaxed text-zinc-300">
          <li>
            <strong className="text-zinc-100">Chunk.</strong> Text is cut into chunks of up to 500 characters at
            whitespace. Each chunk overlaps the previous one by about 50 characters and starts on a word.
          </li>
          <li>
            <strong className="text-zinc-100">Pack.</strong> Each chunk is stored with its document id and a format
            version, then deflated. A chunk that won&apos;t fit in one QR code is an error, not a silent truncation.
          </li>
          <li>
            <strong className="text-zinc-100">Encode.</strong> Each payload becomes a QR code at 4 pixels per module.
            All frames are centered on one square size and piped as raw pixels into FFmpeg (libx264, stillimage
            tuning).
          </li>
          <li>
            <strong className="text-zinc-100">Open.</strong> One FFmpeg pass streams raw frames to jsQR. Each frame
            describes itself, so a frame that fails to decode loses only its own chunk, and{" "}
            <code className={code}>lost</code> counts those frames.
          </li>
          <li>
            <strong className="text-zinc-100">Search.</strong> The chunks go into a{" "}
            <Link href="/docs/hippocampus" className={link}>
              Hippocampus
            </Link>{" "}
            labeled by document id. <code className={code}>neighbors</code> returns passages ranked by BM25, and{" "}
            <code className={code}>react</code> votes on which document answers.
          </li>
        </ol>

        <h2 id="size" className={h2}>
          Measured size
        </h2>
        <p className={p}>
          The corpus is the 13 Markdown files in this repository&apos;s <code className={code}>docs/</code> folder. To
          reproduce, run <code className={code}>node bench-capsule.mjs ../docs</code> in{" "}
          <code className={code}>node/</code>.
        </p>
        <BarChart
          title="Bytes to store 13 docs (119 KB of text)"
          subtitle="QRyptoRAG v1: its default encoder settings on the same files"
          data={SIZES}
          max={7e6}
          ticks={[0, 2e6, 4e6, 6e6]}
          unit="mb"
          highlight="Arcane Capsule"
        />
        <dl className="not-prose mt-6 grid max-w-2xl gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-4">
          {[
            ["5.7×", "video vs. text"],
            ["9.4×", "smaller than QRyptoRAG v1"],
            ["2 s", "to write 269 frames"],
            ["5.5 s", "to open (about 50 frames/s)"],
          ].map(([value, label]) => (
            <div key={label} className="bg-zinc-950 p-4">
              <dt className="text-xs text-zinc-500">{label}</dt>
              <dd className="mt-1 text-xl font-semibold tabular-nums text-zinc-100">{value}</dd>
            </div>
          ))}
        </dl>
        <p className="not-prose mt-6 max-w-2xl border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
          <strong>This is not compression.</strong> QRyptoRAG&apos;s README claimed storage 80–95% smaller than the text.
          Measured, its settings made a video 53× larger. A capsule is 5.7× larger than its text and 16× larger than the
          gzipped text. Use a capsule when one self-contained file that survives video pipelines is worth that size.
          To make storage smaller, gzip the text.
        </p>

        <h3 className="mt-10 text-lg font-semibold text-zinc-100">Settings</h3>
        <div className="not-prose mt-4 overflow-x-auto">
          <table className="w-full min-w-[520px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className={th}>Setting</th>
                <th className={th}>Video / text</th>
                <th className="py-2 font-medium">Note</th>
              </tr>
            </thead>
            <tbody>
              {SWEEP.map(([setting, ratio, note]) => (
                <tr key={setting} className="border-b border-zinc-900 align-top">
                  <td className="py-2.5 pr-4 text-zinc-100">{setting}</td>
                  <td className={td}>{ratio}</td>
                  <td className="py-2.5 text-zinc-400">{note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h2 id="from-qryptorag" className={h2}>
          What changed from QRyptoRAG
        </h2>
        <div className="not-prose mt-6 overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className={th}></th>
                <th className={th}>QRyptoRAG v1 (qr-video-rag)</th>
                <th className="py-2 font-medium">Arcane Capsule</th>
              </tr>
            </thead>
            <tbody>
              {CHANGES.map(([what, before, after]) => (
                <tr key={what} className="border-b border-zinc-900 align-top">
                  <td className="py-2.5 pr-4 text-zinc-100">{what}</td>
                  <td className="py-2.5 pr-4 text-zinc-400">{before}</td>
                  <td className="py-2.5 text-zinc-300">{after}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h2 id="limitations" className={h2}>
          Limitations
        </h2>
        <ul className="not-prose mt-2 max-w-2xl divide-y divide-zinc-900 border-y border-zinc-800">
          {[
            ["Larger than its text", "A capsule is about 5.7× its text on the default settings. It is a portable format, not a storage saving."],
            ["Opening decodes every frame", "About 50 frames per second, so 10 MB of text (about 23,000 frames) takes several minutes to open. Cache JSON.stringify(memory) next to a large capsule and load that instead."],
            ["Keyword retrieval", "Search is BM25 over the decoded chunks, so a query that shares no words with a passage won't find it. You can't add an embedder yet."],
            ["Node only for video", "Writing and reading the MP4 needs Node and FFmpeg. The browser can pack and unpack payloads, but it doesn't write video, and there is no Python version yet."],
            ["Tested on one corpus", "Sizes and decode rates come from this repository's docs. Other text, other codecs or downscaled re-uploads were not measured."],
          ].map(([title, body]) => (
            <li key={title} className="py-3.5">
              <p className="font-medium text-zinc-100">{title}</p>
              <p className="mt-1 text-sm leading-relaxed text-zinc-400">{body}</p>
            </li>
          ))}
        </ul>
        <p className={`${p} mt-6 text-sm text-zinc-400`}>
          Source: <code className={code}>node/capsule.mjs</code> and <code className={code}>node/capsule-video.mjs</code>.
        </p>
      </div>
    </div>
  );
}
