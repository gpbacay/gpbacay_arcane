import Link from "next/link";
import { SlmChat } from "@/components/SlmChat";

export default function ChatPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-8">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Chat with ARCANE
        </h1>
        <p className="text-xl text-zinc-400">
          ARC 1 answers a short docs question. SmolLM2, running in this browser, writes the open, creative, and
          emotional ones. No server and no API key.
        </p>
      </div>

      <SlmChat />

      <div className="space-y-6 text-zinc-300 mt-10">
        <h2
          id="how-this-works"
          className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2"
        >
          How this works
        </h2>
        <p className="leading-relaxed">
          The default model is the{" "}
          <Link href="/docs/arc-1" className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium">
            ARC 1
          </Link>{" "}
          checkpoint shipped with the site. A short question it recognizes is answered from the docs. Anything it
          does not cover is written by SmolLM2 (the text decoder of SmolVLM-256M Q4) on WebGPU. The first visit
          downloads that model into the browser cache. A local SLM server can still add extra models to the menu.
        </p>
      </div>
    </div>
  );
}
