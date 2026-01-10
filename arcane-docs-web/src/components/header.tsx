"use client";

import Link from "next/link";
import Image from "next/image";
import { GitHubStarButton } from "@/components/github-star-button";
import { useState, useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";

export function SiteHeader() {
  const [searchOpen, setSearchOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const router = useRouter();
  const pathname = usePathname();
  const isDocsPage = pathname?.startsWith("/docs");

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
      <div className="container flex h-16 items-center px-4 sm:px-6 mx-auto">
        {/* Mobile menu button */}
        {!isDocsPage && (
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden mr-4 p-2 rounded-md text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 transition-colors"
            aria-label="Toggle mobile menu"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              {mobileMenuOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        )}

        {/* Logo and Desktop Navigation */}
        <div className="mr-4 flex items-center">
          <Link href="/" className="flex items-center space-x-3 group">
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

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center ml-8 space-x-8 text-sm font-medium">
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

        {/* Search and GitHub */}
        <div className="flex flex-1 items-center justify-end gap-2 sm:gap-4">
          <div className="w-full max-w-[140px] min-[450px]:max-w-sm md:w-auto md:flex-none">
            <button
              onClick={() => setSearchOpen(true)}
              className="relative inline-flex items-center w-full md:w-64 h-9 sm:h-10 text-xs sm:text-sm text-zinc-500 border border-zinc-800 rounded-full px-3 sm:px-4 whitespace-nowrap bg-zinc-950 transition-all hover:border-[#835BD9] hover:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-[#835BD9]/50"
            >
              <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4 mr-2 sm:mr-3 text-zinc-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <span className="hidden min-[450px]:inline truncate">Search documentation...</span>
              <span className="min-[450px]:hidden">Search...</span>
              <kbd className="hidden sm:inline-flex ml-auto pointer-events-none h-5 select-none items-center gap-1 rounded border border-zinc-800 bg-zinc-900 px-1.5 font-mono text-[10px] font-medium text-zinc-500 opacity-100">
                <span className="text-xs">Ctrl</span>K
              </kbd>
            </button>
          </div>
          <nav className="flex items-center shrink-0">
            <GitHubStarButton />
          </nav>
        </div>
      </div>

      {/* Mobile Navigation Drawer */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="md:hidden border-t border-zinc-800 bg-black/95 backdrop-blur overflow-hidden"
          >
            <nav className="px-6 py-8 space-y-6">
              <Link
                href="/docs"
                className="flex items-center text-lg font-bold text-zinc-200 hover:text-[#C785F2] transition-colors"
                onClick={() => setMobileMenuOpen(false)}
              >
                Documentation
              </Link>
              <Link
                href="/docs/layers"
                className="flex items-center text-lg font-bold text-zinc-400 hover:text-zinc-100 transition-colors"
                onClick={() => setMobileMenuOpen(false)}
              >
                Layers
              </Link>
              <div className="border-t border-zinc-800 pt-6 mt-6">
                <Link
                  href="/"
                  className="flex items-center text-lg font-bold text-zinc-400 hover:text-zinc-100 transition-colors"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  Home
                </Link>
              </div>
            </nav>
          </motion.div>
        )}
      </AnimatePresence>

      {searchOpen && (
        <div
          className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-md animate-in fade-in duration-200"
          onClick={() => setSearchOpen(false)}
        >
          <div
            className="fixed left-1/2 top-[15%] sm:top-[20%] -translate-x-1/2 w-full max-w-2xl px-4 animate-in slide-in-from-top-4 duration-300"
            onClick={(e) => e.stopPropagation()}
          >
            <form onSubmit={handleSearch} className="relative">
              <input
                type="text"
                placeholder="Search documentation..."
                className="w-full bg-zinc-900 border-2 border-[#835BD9] rounded-full px-10 sm:px-12 py-3 sm:py-4 text-zinc-100 text-base sm:text-lg placeholder-zinc-500 focus:outline-none shadow-[0_0_30px_rgba(131,91,217,0.3)] font-sans"
                autoFocus
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <svg className="absolute left-3 sm:left-4 top-1/2 -translate-y-1/2 w-5 h-5 sm:w-6 sm:h-6 text-zinc-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <div className="absolute right-3 sm:right-4 top-1/2 -translate-y-1/2 text-xs text-zinc-500 bg-zinc-950/50 px-2 py-1 rounded hidden sm:block">
                ESC to close
              </div>
            </form>
            <div className="mt-4 bg-zinc-900 border border-zinc-800 rounded-xl p-4 sm:p-6 shadow-2xl">
              <p className="text-sm font-medium text-zinc-400 mb-4 uppercase tracking-wider">Quick Links</p>
              <div className="grid grid-cols-2 gap-3 text-zinc-300">
                <button onClick={() => { router.push('/docs/installation'); setSearchOpen(false); }} className="p-3 bg-zinc-950 border border-zinc-800 rounded-lg hover:border-[#C785F2] hover:text-white transition-all text-left text-sm">
                  🚀 Installation Guide
                </button>
                <button onClick={() => { router.push('/docs/quick-start'); setSearchOpen(false); }} className="p-3 bg-zinc-950 border border-zinc-800 rounded-lg hover:border-[#C785F2] hover:text-white transition-all text-left text-sm">
                  ⚡ Quick Start
                </button>
                <button onClick={() => { router.push('/docs/neural-resonance'); setSearchOpen(false); }} className="p-3 bg-zinc-950 border border-zinc-800 rounded-lg hover:border-[#C785F2] hover:text-white transition-all text-left text-sm">
                  🧠 Neural Resonance
                </button>
                <button onClick={() => { router.push('/docs/layers'); setSearchOpen(false); }} className="p-3 bg-zinc-950 border border-zinc-800 rounded-lg hover:border-[#C785F2] hover:text-white transition-all text-left text-sm">
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
