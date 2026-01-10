"use client";

import Link from "next/link";
import Image from "next/image";
import { GitHubStarButton } from "@/components/github-star-button";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

export function SiteHeader() {
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const router = useRouter();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setSearchOpen(true);
      }
      if (e.key === "Escape") {
        setSearchOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      // Basic mock search - just go to docs for now but could be expanded
      console.log("Searching for:", searchQuery);
      setSearchOpen(false);
      router.push(`/docs?q=${encodeURIComponent(searchQuery)}`);
    }
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-zinc-800 bg-black/95 backdrop-blur supports-[backdrop-filter]:bg-black/60">
      <div className="container flex h-16 items-center pl-8 pr-8 mx-auto">
        <div className="mr-4 hidden md:flex items-center">
          <Link href="/" className="mr-8 flex items-center space-x-3 group">
            <Image
              src="/arcane_logo_purple.svg"
              alt="Arcane Logo"
              width={32}
              height={32}
              className="rounded-full transition-transform group-hover:rotate-12"
            />
            <span className="hidden font-bold sm:inline-block text-zinc-100 text-lg tracking-tight">
              A.R.C.A.N.E.
            </span>
          </Link>
          <nav className="flex items-center space-x-8 text-sm font-medium">
            <Link
              href="/docs"
              className="text-zinc-200 transition-colors hover:text-[#C785F2]"
            >
              Documentation
            </Link>
            <Link
              href="/docs/layers"
              className="text-zinc-400 transition-colors hover:text-zinc-100"
            >
              Layers
            </Link>
          </nav>
        </div>

        <div className="flex flex-1 items-center justify-between space-x-4 md:justify-end">
          <div className="w-full flex-1 md:w-auto md:flex-none">
            <button
              onClick={() => setSearchOpen(true)}
              className="relative inline-flex items-center w-full md:w-64 h-10 text-sm text-zinc-500 border border-zinc-800 rounded-full px-4 whitespace-nowrap bg-zinc-950 transition-all hover:border-[#835BD9] hover:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-[#835BD9]/50"
            >
              <svg className="w-4 h-4 mr-3 text-zinc-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <span>Search documentation...</span>
              <kbd className="ml-auto pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border border-zinc-800 bg-zinc-900 px-1.5 font-mono text-[10px] font-medium text-zinc-500 opacity-100">
                <span className="text-xs">Ctrl</span>K
              </kbd>
            </button>
          </div>
          <nav className="flex items-center gap-4">
            <GitHubStarButton />
          </nav>
        </div>
      </div>

      {searchOpen && (
        <div
          className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-md animate-in fade-in duration-200"
          onClick={() => setSearchOpen(false)}
        >
          <div
            className="fixed left-1/2 top-[20%] -translate-x-1/2 w-full max-w-2xl px-4 animate-in slide-in-from-top-4 duration-300"
            onClick={(e) => e.stopPropagation()}
          >
            <form onSubmit={handleSearch} className="relative">
              <input
                type="text"
                placeholder="Search documentation..."
                className="w-full bg-zinc-900 border-2 border-[#835BD9] rounded-full px-12 py-4 text-zinc-100 text-lg placeholder-zinc-500 focus:outline-none shadow-[0_0_30px_rgba(131,91,217,0.3)] font-sans"
                autoFocus
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <svg className="absolute left-4 top-1/2 -translate-y-1/2 w-6 h-6 text-zinc-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <div className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-zinc-500 bg-zinc-950/50 px-2 py-1 rounded">
                ESC to close
              </div>
            </form>
            <div className="mt-4 bg-zinc-900 border border-zinc-800 rounded-xl p-6 shadow-2xl">
              <p className="text-sm font-medium text-zinc-400 mb-4 uppercase tracking-wider">Quick Links</p>
              <div className="grid grid-cols-2 gap-3 text-zinc-300">
                <button onClick={() => { router.push('/docs/installation'); setSearchOpen(false); }} className="p-3 bg-zinc-950 border border-zinc-800 rounded-lg hover:border-[#C785F2] hover:text-white transition-all text-left">
                  🚀 Installation Guide
                </button>
                <button onClick={() => { router.push('/docs/quick-start'); setSearchOpen(false); }} className="p-3 bg-zinc-950 border border-zinc-800 rounded-lg hover:border-[#C785F2] hover:text-white transition-all text-left">
                  ⚡ Quick Start
                </button>
                <button onClick={() => { router.push('/docs/neural-resonance'); setSearchOpen(false); }} className="p-3 bg-zinc-950 border border-zinc-800 rounded-lg hover:border-[#C785F2] hover:text-white transition-all text-left">
                  🧠 Neural Resonance
                </button>
                <button onClick={() => { router.push('/docs/layers'); setSearchOpen(false); }} className="p-3 bg-zinc-950 border border-zinc-800 rounded-lg hover:border-[#C785F2] hover:text-white transition-all text-left">
                  🧬 Biological Layers
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
