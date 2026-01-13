"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { docsConfig } from "@/config/docs";

export function DocsPager() {
    const pathname = usePathname();

    const allLinks = docsConfig.sidebarNav.flatMap((section) => section.items);
    const currentIndex = allLinks.findIndex((link) => link.href === pathname);

    if (currentIndex === -1) {
        return null;
    }

    const prev = currentIndex > 0 ? allLinks[currentIndex - 1] : null;
    const next = currentIndex < allLinks.length - 1 ? allLinks[currentIndex + 1] : null;

    return (
        <div className="flex flex-row items-center justify-between pt-12 mt-12 border-t border-zinc-900 px-1">
            {prev ? (
                <Link
                    href={prev.href}
                    className="group flex flex-row items-center gap-3 rounded-none border border-zinc-800 px-5 py-4 transition-all hover:border-[#C785F2]/50 hover:bg-zinc-900/30 max-w-[48%]"
                >
                    <svg className="w-4 h-4 text-zinc-500 group-hover:text-[#C785F2] transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M15 19l-7-7 7-7" />
                    </svg>
                    <span className="text-sm font-bold text-zinc-300 group-hover:text-zinc-100 transition-colors line-clamp-1">
                        {prev.title}
                    </span>
                </Link>
            ) : (
                <div />
            )}
            {next ? (
                <Link
                    href={next.href}
                    className="group flex flex-row items-center gap-3 rounded-none border border-zinc-800 px-5 py-4 text-right transition-all hover:border-[#C785F2]/50 hover:bg-zinc-900/30 max-w-[48%]"
                >
                    <span className="text-sm font-bold text-zinc-300 group-hover:text-zinc-100 transition-colors line-clamp-1">
                        {next.title}
                    </span>
                    <svg className="w-4 h-4 text-zinc-500 group-hover:text-[#C785F2] transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5l7 7-7 7" />
                    </svg>
                </Link>
            ) : (
                <div />
            )}
        </div>
    );
}
