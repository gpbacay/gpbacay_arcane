"use client";

import { useState } from "react";
import QRCode from "qrcode";
import jsQR from "jsqr";
import { Occipital, pack, unpack } from "gpbacay-arcane/occipital";
import { Hippocampus } from "gpbacay-arcane/hippocampus";

const SAMPLE = `# Refunds
Refunds go back to the original card within five business days. A duplicate charge is reversed automatically once the bank confirms it, and you get an email when it is done.

# Shipping
Parcels ship from Cebu every weekday. Tracking numbers arrive by email. A parcel with no tracking scan for ten days is treated as lost and replaced at no cost.

# Accounts
You can change your email address under Settings. Two-factor sign-in uses an authenticator app; if you lose the phone, recovery codes restore access.`;

type Built = { frames: string[]; textBytes: number; payloadBytes: number; documents: number; lost: number; words: Hippocampus; letters: Hippocampus };

/** "# Title" sections become documents named by their title; text before the first heading is "notes". */
function toDocuments(text: string) {
  const docs: Record<string, string> = {};
  text.split(/^#\s+/m).forEach((part, i) => {
    const title = i === 0 ? "notes" : part.split("\n", 1)[0].trim();
    const body = (i === 0 ? part : part.slice(part.indexOf("\n") + 1)).trim();
    if (body) docs[title] = docs[title] ? `${docs[title]}\n${body}` : body;
  });
  return docs;
}

export function OccipitalDemo() {
  const [text, setText] = useState(SAMPLE);
  const [query, setQuery] = useState("my parcle has no traking");
  const [built, setBuilt] = useState<Built | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const build = async () => {
    setBusy(true);
    setError(null);
    try {
      const docs = toDocuments(text);
      const payloads = await pack(docs, { chunkSize: 160, overlap: 30 }); // small, so a short sample spans several frames
      const frames: string[] = [], read: (Uint8Array | null)[] = [];
      for (const payload of payloads) {
        // Draw the frame, then read it back from its pixels: what readVideo does with each video frame.
        const canvas = document.createElement("canvas");
        await QRCode.toCanvas(canvas, [{ data: payload, mode: "byte" }], { errorCorrectionLevel: "M", margin: 2, scale: 4 });
        const { data, width, height } = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height);
        const qr = jsQR(data, width, height, { inversionAttempts: "dontInvert" });
        read.push(qr ? Uint8Array.from(qr.binaryData) : null);
        frames.push(canvas.toDataURL());
      }
      const { memory: letters, lost } = await unpack(read, { k: 3, occipital: new Occipital() });
      const words = new Hippocampus(null, { k: 3, memory: letters.toJSON() });
      setBuilt({
        frames,
        textBytes: new TextEncoder().encode(Object.values(docs).join("")).length,
        payloadBytes: payloads.reduce((n, p) => n + p.length, 0),
        documents: Object.keys(docs).length,
        lost,
        words,
        letters,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBuilt(null);
    } finally {
      setBusy(false);
    }
  };

  const sides = built && query.trim()
    ? ([["Words only (Hippocampus)", built.words], ["Words + letters (Hippocampus + Occipital)", built.letters]] as const).map(([title, memory]) => ({
        title,
        hits: memory.neighbors(query),
        votes: Object.entries(memory.react(query).votes).sort((a, b) => b[1] - a[1]),
      }))
    : [];

  return (
    <div className="not-prose mt-6 border border-zinc-800 bg-zinc-950">
      <div className="border-b border-zinc-800 p-4 sm:p-5">
        <label htmlFor="occipital-text" className="text-sm font-medium text-zinc-200">
          Documents <span className="font-normal text-zinc-500">(each &quot;# Title&quot; section is one document)</span>
        </label>
        <textarea
          id="occipital-text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={9}
          className="mt-2 w-full resize-y border border-zinc-800 bg-zinc-900 p-3 font-mono text-xs leading-relaxed text-zinc-200 focus:border-[#C785F2] focus:outline-none"
        />
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={build}
            disabled={busy || !text.trim()}
            className="border border-[#C785F2] bg-[#C785F2]/10 px-4 py-2 text-sm font-medium text-[#e2c2f9] hover:bg-[#C785F2]/20 disabled:opacity-50"
          >
            {busy ? "Packing…" : built ? "Pack again" : "Pack into frames"}
          </button>
          <span className="text-xs text-zinc-500">Runs in your browser. Nothing is uploaded.</span>
        </div>
        {error && <p className="mt-3 text-sm text-red-400">{error}</p>}
      </div>

      {built && (
        <>
          <dl className="grid grid-cols-2 gap-px border-b border-zinc-800 bg-zinc-800 sm:grid-cols-5">
            {[
              [String(built.documents), "documents"],
              [String(built.frames.length), "QR frames"],
              [`${built.textBytes} B`, "text"],
              [`${built.payloadBytes} B`, "payloads"],
              [`${built.frames.length - built.lost}/${built.frames.length}`, "decoded from pixels"],
            ].map(([value, label]) => (
              <div key={label} className="bg-zinc-950 p-3 last:col-span-2 sm:last:col-span-1">
                <dt className="text-xs text-zinc-500">{label}</dt>
                <dd className="mt-1 text-lg font-semibold tabular-nums text-zinc-100">{value}</dd>
              </div>
            ))}
          </dl>

          <ol className="grid grid-cols-3 gap-2 border-b border-zinc-800 p-4 sm:grid-cols-6 sm:p-5">
            {built.frames.map((src, i) => (
              <li key={i} className="text-center">
                {/* eslint-disable-next-line @next/next/no-img-element -- a data URL drawn in the browser */}
                <img src={src} alt={`QR frame ${i + 1} of ${built.frames.length}`} className="w-full bg-white [image-rendering:pixelated]" />
                <span className="mt-1 block text-[11px] tabular-nums text-zinc-500">frame {i + 1}</span>
              </li>
            ))}
          </ol>

          <div className="p-4 sm:p-5">
            <label htmlFor="occipital-query" className="text-sm font-medium text-zinc-200">
              Search what was read back
            </label>
            <input
              id="occipital-query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="mt-2 w-full border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:border-[#C785F2] focus:outline-none"
            />
            <p className="mt-2 text-xs text-zinc-500">Try a typo or another word form; the left side needs whole words in common.</p>
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              {sides.map(({ title, hits, votes }) => (
                <section key={title} aria-label={title}>
                  <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-400">{title}</h4>
                  <p className="mt-2 text-xs text-zinc-400">
                    Answering document:{" "}
                    {votes.length ? (
                      votes.map(([doc, share]) => (
                        <span key={doc} className="mr-3 font-mono text-zinc-200">
                          {doc} {Math.round(share * 100)}%
                        </span>
                      ))
                    ) : (
                      <span className="text-zinc-500">none (abstains)</span>
                    )}
                  </p>
                  <ul className="mt-2 space-y-2">
                    {hits.map((hit) => (
                      <li key={hit.text} className="border border-zinc-800 p-3">
                        <p className="text-xs text-zinc-500">
                          <span className="font-mono text-[#C785F2]">{hit.label}</span> · score {hit.score.toFixed(2)}
                        </p>
                        <p className="mt-1 text-sm leading-relaxed text-zinc-300">{hit.text}</p>
                      </li>
                    ))}
                  </ul>
                </section>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
