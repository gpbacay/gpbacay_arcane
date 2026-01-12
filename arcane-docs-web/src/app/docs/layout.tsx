"use client";

import Link from "next/link";
import { SiteHeader } from "@/components/header";
import { DocsSidebar } from "@/components/sidebar";
import { DocsPager } from "@/components/pager";
import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { TableOfContents } from "@/components/TableOfContents";
import { AnimatePresence, motion } from "framer-motion";
import { SidebarProvider, Sidebar, SidebarContent, SidebarTrigger, SidebarRail, useSidebar } from "@/components/ui/sidebar";
import { SiteFooter } from "@/components/footer";

function MobileMenuButton({ tocOpen, setTocOpen, headings }: { tocOpen: boolean, setTocOpen: (open: boolean) => void, headings: { id: string; text: string; level: number }[] }) {
  const { setOpenMobile } = useSidebar();
  
  return (
    <div className="md:hidden flex items-center justify-between px-6 py-3 border-b border-zinc-800 bg-black/50 backdrop-blur-sm sticky top-16 z-40">
      <button
        onClick={() => setOpenMobile(true)}
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
  );
}

function DocsLayoutContent({ children, headings, tocOpen, setTocOpen }: { children: React.ReactNode, headings: { id: string; text: string; level: number }[], tocOpen: boolean, setTocOpen: (open: boolean) => void }) {
  const { state, isMobile } = useSidebar();

  return (
    <div className="flex min-h-screen flex-col bg-black selection:bg-[#C785F2]/20 font-sans w-full">
      <SiteHeader />

      <MobileMenuButton tocOpen={tocOpen} setTocOpen={setTocOpen} headings={headings} />

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
      <div className="flex-1 flex w-full relative">
        <Sidebar collapsible="offcanvas" className="top-16 border-r border-zinc-900 bg-black h-[calc(100vh-4rem)]">
          <SidebarContent className="bg-black scrollbar-hide">
            <DocsSidebar />
          </SidebarContent>
          <SidebarRail />
        </Sidebar>

          <div className="flex-1 flex flex-col min-w-0">
            <div className="flex-1 flex items-start lg:grid lg:grid-cols-[1fr_260px] lg:gap-10 xl:gap-14 mx-auto px-4 md:px-8 w-full">
              {/* Main Content - Center */}
              <main className="relative py-6 md:py-12 lg:px-4 xl:px-0 w-full">
                {state === "collapsed" && !isMobile && (
                  <div className="absolute left-[-12px] top-[26px] hidden md:block">
                    <SidebarTrigger className="text-zinc-500 hover:text-zinc-200 hover:bg-zinc-900/50 transition-colors" />
                  </div>
                )}
                <div className="mx-auto w-full min-w-0 max-w-3xl break-words px-4 md:px-0">
                  {children}
                  <DocsPager />
                </div>
              </main>

              {/* Desktop Table of Contents - Right */}
              <aside className="hidden lg:block sticky top-16 h-[calc(100vh-4rem)] overflow-y-auto scrollbar-hide scroll-smooth">
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
            <SiteFooter />
          </div>
      </div>
    </div>
  );
}

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [headings, setHeadings] = useState<{ id: string; text: string; level: number }[]>([]);
  const [tocOpen, setTocOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    // Extract headings from the page content for the TOC
    const extractHeadings = () => {
      const mainContent = document.querySelector("main");
      if (!mainContent) return;

      const headingElements = Array.from(mainContent.querySelectorAll("h2, h3"));
      const extracted = headingElements
        .map((el) => {
          // Ensure element has an ID for linking
          if (!el.id) {
            el.id = el.textContent?.toLowerCase().replace(/\s+/g, "-") || "";
          }
          return {
            id: el.id,
            text: el.textContent || "",
            level: parseInt(el.tagName[1]),
          };
        })
        .filter((h) => h.text.toLowerCase() !== "command palette");
      setHeadings(extracted);
    };

    // Need to wait for content to be rendered
    const timer = setTimeout(extractHeadings, 800);
    return () => clearTimeout(timer);
  }, [pathname]);

  return (
    <SidebarProvider>
      <DocsLayoutContent headings={headings} tocOpen={tocOpen} setTocOpen={setTocOpen}>
        {children}
      </DocsLayoutContent>
    </SidebarProvider>
  );
}

