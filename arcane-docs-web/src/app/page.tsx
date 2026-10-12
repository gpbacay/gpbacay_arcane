"use client";

import { useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  BrainCircuit,
  Check,
  Copy,
  DatabaseZap,
  Network,
  Route,
} from "lucide-react";
import BlockDrift from "@/components/BlockDrift";
import { SiteFooter } from "@/components/footer";
import { SiteHeader } from "@/components/header";
import { SidebarProvider } from "@/components/ui/sidebar";

const capabilities = [
  {
    icon: BrainCircuit,
    title: "Adaptive inference",
    body: "Align internal state through bounded feedback and resonance while processing the current input.",
  },
  {
    icon: Route,
    title: "Grounded decisions",
    body: "Return typed tools, labels, and values tied directly to the input instead of generating prose.",
  },
  {
    icon: DatabaseZap,
    title: "Neural memory",
    body: "Recall relevant decisions and examples to guide the next response without retraining the model.",
  },
  {
    icon: Network,
    title: "Biological mechanisms",
    body: "Build with feedback, spiking gates, local plasticity, and homeostatic activity control.",
  },
];

function PipInstall() {
  const [copied, setCopied] = useState(false);
  const command = "pip install gpbacay-arcane";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy command", err);
    }
  };

  return (
    <div className="mx-auto flex w-fit max-w-full items-center justify-center gap-3 border border-zinc-700 bg-black/65 px-4 py-3 text-center backdrop-blur-md transition-colors hover:border-[#C785F2]/60 sm:px-5">
      <code className="whitespace-nowrap font-mono text-[13px] text-zinc-100 min-[380px]:text-sm sm:text-base">
        <span className="text-[#C785F2]">pip</span> install gpbacay-arcane
      </code>
      <button
        onClick={handleCopy}
        className="p-2 text-zinc-500 transition-colors hover:bg-zinc-900 hover:text-[#C785F2] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#C785F2]"
        aria-label={copied ? "Installation command copied" : "Copy installation command"}
      >
        {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
      </button>
    </div>
  );
}

export default function Home() {
  return (
    <SidebarProvider>
      <div className="min-h-screen w-full overflow-x-hidden bg-black font-sans text-white selection:bg-[#C785F2]/30">
        <SiteHeader />

        <main>
          <section className="relative flex min-h-[calc(100vh-64px)] items-center overflow-hidden border-b border-zinc-900">
            <div className="absolute inset-0 z-0 bg-[#07020c]" aria-hidden="true">
              <BlockDrift
                near="#0E0318"
                far="#944FDE"
                edge="#050108"
                grid={17}
                blockSize={12}
                gap={20}
                layers={15}
                density={10}
                cluster={10}
                clearCentre={4}
                speed={16}
                className="h-full w-full opacity-95"
              />
            </div>
            <div className="absolute inset-0 z-[1] bg-[radial-gradient(circle_at_center,rgba(0,0,0,0.2)_0%,rgba(0,0,0,0.5)_48%,rgba(0,0,0,0.78)_100%)]" aria-hidden="true" />
            <div className="absolute inset-0 z-[1] bg-gradient-to-b from-black/10 via-transparent to-black" aria-hidden="true" />

            <div className="relative z-10 mx-auto flex w-full max-w-5xl flex-col items-center px-5 py-20 text-center sm:px-8 lg:px-10">
              <div className="flex w-full flex-col items-center animate-in fade-in slide-in-from-bottom-4 duration-1000">
                <p className="mb-5 text-xs font-semibold uppercase tracking-[0.22em] text-[#C785F2]">
                  Augmented Reconstruction of Consciousness through Artificial Neural Evolution
                </p>
                <h1 className="max-w-5xl text-balance text-4xl font-extrabold leading-[1.02] tracking-[-0.045em] text-white sm:text-6xl lg:text-7xl">
                  The first library for adaptive System 1 AI models.
                </h1>
                <p className="mt-6 max-w-2xl text-base leading-relaxed text-zinc-300 sm:text-lg">
                  Build fast, grounded models that adapt during inference, remember experience, and act without
                  token-by-token generation.
                </p>

                <div className="mt-8 w-full">
                  <PipInstall />
                </div>

                <div className="mt-7 flex w-full flex-col justify-center gap-3 sm:w-auto sm:flex-row">
                  <Link
                    href="/docs/quick-start"
                    className="inline-flex h-12 items-center justify-center gap-2 bg-white px-6 text-sm font-bold text-black transition-colors hover:bg-zinc-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#C785F2]"
                  >
                    Start building <ArrowRight className="h-4 w-4" aria-hidden="true" />
                  </Link>
                  <Link
                    href="/docs/system-1"
                    className="inline-flex h-12 items-center justify-center border border-zinc-700 bg-black/40 px-6 text-sm font-bold text-zinc-100 backdrop-blur transition-colors hover:border-zinc-500 hover:bg-zinc-900/70"
                  >
                    Explore the system
                  </Link>
                </div>
              </div>
            </div>
          </section>

          <section className="mx-auto max-w-7xl px-6 py-20 lg:px-10 lg:py-28">
            <div className="grid gap-10 lg:grid-cols-[0.72fr_1.28fr]">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#C785F2]">The System 1 stack</p>
                <h2 className="mt-4 text-balance text-3xl font-bold tracking-tight text-zinc-100 sm:text-4xl">
                  Four capabilities. One adaptive architecture.
                </h2>
                <p className="mt-5 max-w-xl leading-relaxed text-zinc-400">
                  ARCANE combines the parts a System 1 model needs to perceive, adapt, recall experience, and produce
                  a direct decision, with each mechanism available as a standalone component.
                </p>
              </div>
              <div className="grid gap-px border border-zinc-800 bg-zinc-800 sm:grid-cols-2">
                {capabilities.map(({ icon: Icon, title, body }) => (
                  <article key={title} className="bg-zinc-950 p-6">
                    <Icon className="h-6 w-6 text-[#C785F2]" aria-hidden="true" />
                    <h3 className="mt-5 font-semibold text-zinc-100">{title}</h3>
                    <p className="mt-2 text-sm leading-relaxed text-zinc-400">{body}</p>
                  </article>
                ))}
              </div>
            </div>
          </section>

          <section className="border-y border-zinc-900 bg-zinc-950/55">
            <div className="mx-auto grid max-w-7xl gap-8 px-6 py-20 lg:grid-cols-3 lg:px-10">
              {[
                ["01", "Act", "ARC 1 returns typed tool calls, extracted fields, and classifications with calibrated confidence.", "/docs/arc-1", "Explore ARC 1"],
                ["02", "Adapt", "Resonance, plasticity, and homeostasis give models controlled ways to refine state beyond a fixed forward pass.", "/docs/neural-resonance", "Understand adaptation"],
                ["03", "Remember", "Hippocampus and Grounded Graph of Thought add example memory, retrieval, and evidence-aware reasoning.", "/docs/hippocampus", "Explore memory"],
              ].map(([number, title, body, href, label]) => (
                <article key={title} className="group border-t border-zinc-800 pt-6">
                  <p className="font-mono text-xs text-[#C785F2]">{number}</p>
                  <h2 className="mt-4 text-2xl font-bold text-zinc-100">{title}</h2>
                  <p className="mt-3 min-h-20 text-sm leading-relaxed text-zinc-400">{body}</p>
                  <Link href={href} className="mt-6 inline-flex items-center gap-2 text-sm font-semibold text-zinc-200 transition-colors group-hover:text-[#C785F2]">
                    {label} <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" aria-hidden="true" />
                  </Link>
                </article>
              ))}
            </div>
          </section>
        </main>

        <SiteFooter />
      </div>
    </SidebarProvider>
  );
}
