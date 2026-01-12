"use client";

import Link from "next/link";
import Image from "next/image";
import { GitHubStarButton } from "@/components/github-star-button";
import { useState, useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import {
  CommandDialog,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
} from "@/components/ui/command";
import { docsConfig } from "@/config/docs";
import { Search, FileText, Zap, Brain, Layers, Cpu, BarChart3, Terminal } from "lucide-react";

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
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const handleSearchSelect = (href: string) => {
    setSearchOpen(false);
    router.push(href);
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
              href="/docs/blog"
              className="text-zinc-400 transition-colors hover:text-zinc-100"
            >
              Blog
            </Link>
          </nav>
        </div>

        {/* Search and GitHub */}
        <div className="flex flex-1 items-center justify-end gap-2 sm:gap-4">
          <div className="w-full max-w-[140px] min-[450px]:max-w-sm md:w-auto md:flex-none">
            <button
              onClick={() => setSearchOpen(true)}
              className="relative inline-flex items-center w-full md:w-64 h-9 sm:h-10 text-xs sm:text-sm text-zinc-500 border border-zinc-800 rounded-full px-3 sm:px-4 whitespace-nowrap bg-zinc-950 transition-all hover:border-[#835BD9]/50 hover:bg-zinc-900 focus:outline-none focus:ring-1 focus:ring-[#835BD9]/50"
            >
              <Search className="w-3.5 h-3.5 sm:w-4 sm:h-4 mr-2 sm:mr-3 text-zinc-400" />
              <span className="hidden min-[450px]:inline truncate">Search documentation...</span>
              <span className="min-[450px]:hidden">Search...</span>
              <kbd className="hidden sm:inline-flex ml-auto pointer-events-none h-5 select-none items-center gap-1 rounded border border-zinc-800 bg-zinc-900 px-1.5 font-mono text-[10px] font-medium text-zinc-500 opacity-100">
                <span className="text-xs text-zinc-600">Ctrl</span>K
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
                href="/docs/blog"
                className="flex items-center text-lg font-bold text-zinc-400 hover:text-zinc-100 transition-colors"
                onClick={() => setMobileMenuOpen(false)}
              >
                Blog
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

      <CommandDialog open={searchOpen} onOpenChange={setSearchOpen}>
        <CommandInput 
          placeholder="Search documentation..." 
          value={searchQuery}
          onValueChange={setSearchQuery}
        />
        <CommandList className="max-h-[70vh]">
          <CommandEmpty>No results found.</CommandEmpty>
          {docsConfig.sidebarNav.map((group) => (
            <CommandGroup key={group.title} heading={group.title}>
              {group.items.map((item) => {
                const Icon = 
                  item.title.includes("Introduction") ? FileText :
                  item.title.includes("Installation") || item.title.includes("Quick Start") ? Zap :
                  item.title.includes("Neural Resonance") ? Brain :
                  item.title.includes("Layers") ? Layers :
                  item.title.includes("Model") || item.title.includes("GSER") ? Cpu :
                  item.title.includes("CLI") ? Terminal :
                  item.title.includes("Benchmarks") ? BarChart3 :
                  FileText;
                
                return (
                  <CommandItem
                    key={item.href}
                    value={`${group.title} ${item.title}`}
                    onSelect={() => handleSearchSelect(item.href)}
                    className="flex items-center gap-3 py-3"
                  >
                    <Icon className="h-4 w-4 text-zinc-400" />
                    <span>{item.title}</span>
                  </CommandItem>
                );
              })}
            </CommandGroup>
          ))}
        </CommandList>
      </CommandDialog>
    </header>
  );
}
