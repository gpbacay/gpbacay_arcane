"use client";

import { SiteHeader } from "@/components/header";
import { DocsSidebar } from "@/components/sidebar";
import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { TableOfContents } from "@/components/TableOfContents";

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [headings, setHeadings] = useState<{ id: string; text: string; level: number }[]>([]);
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
      <div className="container flex-1 items-start md:grid md:grid-cols-[240px_minmax(0,1fr)_260px] md:gap-14 mx-auto px-4 md:px-8">
        <DocsSidebar />
        <main className="relative py-12">
          <div className="mx-auto w-full min-w-0 max-w-3xl break-words">
            {children}
          </div>
        </main>
        {/* On This Page Sidebar */}
        <aside className="fixed top-28 z-30 hidden h-[calc(100vh-8rem)] w-full shrink-0 md:sticky md:block pr-4">
          <div className="sticky top-28">
            <TableOfContents headings={headings} />

            <div className="pt-10 mt-10 border-t border-zinc-900 px-1">
              <Link
                href="https://github.com/gpbacay/gpbacay_arcane/issues/new"
                target="_blank"
                className="text-xs text-zinc-600 hover:text-zinc-400 transition-colors flex items-center gap-2 group"
                rel="noreferrer"
              >
              </Link>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

// Minimal missing Link import fix in the component above
import Link from "next/link";
