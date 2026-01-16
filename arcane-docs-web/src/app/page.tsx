"use client";

import { useState } from "react";
import Link from "next/link";
import { Check, Copy } from "lucide-react";
import LetterGlitch from "@/components/LetterGlitch";
import { SiteFooter } from "@/components/footer";
import { SiteHeader } from "@/components/header";
import { SidebarProvider } from "@/components/ui/sidebar";

function PipInstall() {
  const [copied, setCopied] = useState(false);
  const command = "pip install gpbacay-arcane";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy!", err);
    }
  };

  return (
    <div className="flex items-center gap-4 bg-zinc-900/50 backdrop-blur-md border border-zinc-800 rounded-none px-6 py-4 transition-all hover:border-[#C785F2]/50 hover:bg-zinc-900/80 group relative min-w-[300px] sm:min-w-[400px]">
      <code className="text-base sm:text-lg font-mono tracking-tight transition-colors flex-1 text-left">
        <span className="text-[#C785F2]">pip</span>{" "}
        <span className="text-[#B9DFE0] group-hover:text-white transition-colors">install gpbacay-arcane</span>
      </code>
      <button
        onClick={handleCopy}
        className="p-2 rounded-none hover:bg-zinc-800 transition-colors text-zinc-500 hover:text-[#C785F2]"
        title={copied ? "Copied!" : "Copy to clipboard"}
      >
        {copied ? (
          <Check className="h-4 w-4 sm:h-5 sm:w-5 animate-in zoom-in duration-300" />
        ) : (
          <Copy className="h-4 w-4 sm:h-5 sm:w-5 animate-in zoom-in duration-300" />
        )}
      </button>
    </div>
  );
}

export default function Home() {
  return (
    <SidebarProvider>
      <div className="flex flex-col bg-black text-white selection:bg-[#C785F2]/30 overflow-x-hidden relative font-sans w-full">
        <SiteHeader />
        <div className="h-[calc(100vh-64px)] flex flex-col items-center justify-center relative">
          {/* Letter Glitch Background */}
          <div className="absolute inset-0 z-0">
            <LetterGlitch
              glitchSpeed={50}
              centerVignette={true}
              outerVignette={false}
              smooth={true}
            />
          </div>

          <main className="z-10 flex flex-col items-center text-center px-4 animate-in fade-in zoom-in duration-1000 max-w-4xl">
            <h1 className="text-4xl font-extrabold tracking-tight sm:text-6xl bg-clip-text text-transparent bg-gradient-to-b from-white via-white to-white/40 mb-10 drop-shadow-2xl max-w-3xl leading-[1.1]">
              The world&apos;s first neuromimetic semantic foundation model library
            </h1>
            
            <div className="flex flex-col items-center gap-6 mb-12 animate-in fade-in slide-in-from-bottom-4 duration-1000 delay-300">
              <PipInstall />
            </div>

            <div className="flex flex-col sm:flex-row gap-4 sm:gap-6 w-full sm:w-auto">
              <Link
                href="/docs"
                className="group relative inline-flex h-11 sm:h-13 items-center justify-center overflow-hidden rounded-none bg-white px-8 sm:px-12 font-bold text-black transition-all hover:bg-zinc-200 hover:scale-105 active:scale-95 shadow-[0_0_30px_rgba(255,255,255,0.2)] text-sm sm:text-base"
              >
                <span className="mr-2 sm:mr-3">Explore Docs</span>
                <svg className="h-4 w-4 sm:h-5 sm:w-5 transition-transform group-hover:translate-x-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </Link>
            </div>
          </main>
        </div>
        <SiteFooter />
      </div>
    </SidebarProvider>
  );
}