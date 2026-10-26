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
          The same on-device chatbot as the button on every docs page. ARC 1, Hippocampus, and Occipital run in
          the browser, so this page answers with no server.
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
          checkpoint shipped with the site, loaded through onnxruntime-web. Hippocampus recalls similar questions and
          Occipital tolerates typos. The reply comes from the docs. A local SLM server, when it is running, adds its
          ready models to the same menu. Those extras are not required for the page to answer.
        </p>
      </div>
    </div>
  );
}
