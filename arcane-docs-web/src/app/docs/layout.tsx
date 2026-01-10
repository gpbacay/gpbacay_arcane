"use client";

import Link from "next/link";
import { SiteHeader } from "@/components/header";
import { DocsSidebar } from "@/components/sidebar";
import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { TableOfContents } from "@/components/TableOfContents";
import { AnimatePresence, motion } from "framer-motion";

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [headings, setHeadings] = useState<{ id: string; text: string; level: number }[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [tocOpen, setTocOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    // Extract headings from the page content for the TOC
    const extractHeadings = () => {
      const headingElements = Array.from(document.querySelectorAll("h2, h3"));
      const extracted = headingElements.map((el) => {
        // Ensure element has an ID for linking
        if (!el.id) {
          el.id = el.textContent?.toLowerCase().replace(/\s+/g, "-") || "";
        }
        return {
          id: el.id,
          text: el.textContent || "",
          level: parseInt(el.tagName[1]),
        };
      });
      setHeadings(extracted);
    };

    // Need to wait for content to be rendered
    const timer = setTimeout(extractHeadings, 800);
    return () => clearTimeout(timer);
  }, [pathname]);

  return (
    <div className="flex min-h-screen flex-col bg-black selection:bg-[#C785F2]/20 font-sans">
      <SiteHeader />

      {/* Mobile Controls */}
      <div className="md:hidden flex items-center justify-between px-6 py-3 border-b border-zinc-800 bg-black/50 backdrop-blur-sm sticky top-16 z-40">
        <button
          onClick={() => setSidebarOpen(true)}
          className="flex items-center gap-2 text-sm font-medium text-zinc-400 hover:text-zinc-100 transition-colors py-1 px-2 rounded-md hover:bg-zinc-900"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
          Menu
        </button>

        {headings.length > 0 && (
          <button
            onClick={() => setTocOpen(!tocOpen)}
            className="flex items-center gap-2 text-sm font-medium text-zinc-400 hover:text-zinc-100 transition-colors py-1 px-2 rounded-md hover:bg-zinc-900"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
            </svg>
            On Page
          </button>
        )}
      </div>

      {/* Mobile Sidebar Overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="md:hidden fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm"
            onClick={() => setSidebarOpen(false)}
          >
            <motion.div
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="fixed left-0 top-0 h-full w-80 bg-black border-r border-zinc-800 flex flex-col"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between p-6 border-b border-zinc-800">
                <h2 className="text-lg font-bold text-white tracking-tight uppercase">Navigation</h2>
                <button
                  onClick={() => setSidebarOpen(false)}
                  className="p-2 rounded-full text-zinc-400 hover:text-white hover:bg-zinc-900 transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="overflow-y-auto flex-1">
                <DocsSidebar />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Mobile TOC Overlay */}
      <AnimatePresence>
        {tocOpen && headings.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="md:hidden fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm"
            onClick={() => setTocOpen(false)}
          >
            <motion.div
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="fixed right-0 top-0 h-full w-80 bg-black border-l border-zinc-800 flex flex-col"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between p-6 border-b border-zinc-800">
                <h2 className="text-lg font-bold text-white tracking-tight uppercase">On This Page</h2>
                <button
                  onClick={() => setTocOpen(false)}
                  className="p-2 rounded-full text-zinc-400 hover:text-white hover:bg-zinc-900 transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="p-6 overflow-y-auto flex-1">
                <TableOfContents headings={headings} />
                <div className="pt-10 mt-10 border-t border-zinc-900">
                  <Link
                    href="https://github.com/gpbacay/gpbacay_arcane/issues/new"
                    target="_blank"
                    className="text-xs text-zinc-600 hover:text-zinc-400 transition-colors flex items-center gap-2 group"
                    rel="noreferrer"
                    onClick={() => setTocOpen(false)}
                  >
                    Report an issue
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  </Link>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Desktop Layout */}
      <div className="container flex-1 items-start md:grid md:grid-cols-[240px_minmax(0,1fr)_260px] md:gap-14 mx-auto px-4 md:px-8">
        {/* Desktop Sidebar - Left */}
        <div className="hidden md:block sticky top-16 h-[calc(100vh-4rem)] overflow-y-auto scrollbar-hide border-r border-zinc-900">
          <DocsSidebar />
        </div>

        {/* Main Content - Center */}
        <main className="relative py-6 md:py-12">
          <div className="mx-auto w-full min-w-0 max-w-3xl break-words px-4 md:px-0">
            {children}
          </div>
        </main>

        {/* Desktop Table of Contents - Right */}
        <aside className="hidden md:block sticky top-16 h-[calc(100vh-4rem)]">
          <div className="sticky top-16 p-6">
            <TableOfContents headings={headings} />

            <div className="pt-10 mt-10 border-t border-zinc-900 px-1">
              <Link
                href="https://github.com/gpbacay/gpbacay_arcane/issues/new"
                target="_blank"
                className="text-xs text-zinc-600 hover:text-zinc-400 transition-colors flex items-center gap-2 group"
                rel="noreferrer"
              >
                Report an issue
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </Link>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

