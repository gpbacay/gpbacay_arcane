"use client";

import { useEffect, useState, useMemo } from "react";
import { cn } from "@/utils/cn";

interface TOCProps {
    headings: { id: string; text: string; level: number }[];
}

export function TableOfContents({ headings }: TOCProps) {
    const [activeId, setActiveId] = useState<string>("");
    const itemHeight = 36;
    const xBase = 4;
    const xIndent = 20;

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                const visibleEntries = entries
                    .filter(e => e.isIntersecting)
                    .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);

                if (visibleEntries.length > 0) {
                    setActiveId(visibleEntries[0].target.id);
                }
            },
            { rootMargin: "-20% 0% -60% 0%", threshold: 0 }
        );

        headings.forEach((heading) => {
            const element = document.getElementById(heading.id);
            if (element) observer.observe(element);
        });

        return () => observer.disconnect();
    }, [headings]);

    const activeIndex = headings.findIndex(h => h.id === activeId);

    // Pre-calculate segments so we can render them individually and highlight them perfectly
    const segments = useMemo(() => {
        const result: { path: string; id: string; x: number; y: number }[] = [];
        let currentX = xBase;

        headings.forEach((h, i) => {
            const targetX = h.level === 3 ? xIndent : xBase;
            const startY = i * itemHeight;
            const endY = (i + 1) * itemHeight;
            const midY = startY + itemHeight / 2;

            let segmentsPath = "";

            // If we need to transition to a new X, do it at the very top of the item
            if (currentX !== targetX) {
                const dx = targetX - currentX;
                const dy = 12; // Height of the diagonal kink

                // Diagonal transition + remaining vertical
                segmentsPath = `M ${currentX} ${startY} L ${targetX} ${startY + dy} V ${endY}`;
                currentX = targetX;
            } else {
                // Just vertical
                segmentsPath = `M ${currentX} ${startY} V ${endY}`;
            }

            result.push({
                path: segmentsPath,
                id: h.id,
                x: targetX,
                y: startY
            });
        });

        return result;
    }, [headings, xBase, xIndent, itemHeight]);

    if (headings.length === 0) return null;

    return (
        <div className="relative font-sans select-none px-1">
            <div className="flex items-center gap-3 text-zinc-300 mb-4 md:mb-6 group cursor-default">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-zinc-500 transition-colors group-hover:text-zinc-300">
                    <path d="M3 12h18M3 6h18M3 18h18" />
                </svg>
                <span className="text-xs font-bold uppercase tracking-[0.1em] opacity-70">On this page</span>
            </div>

            <div className="relative flex">
                {/* SVG Navigation Guide */}
                <div className="absolute left-0 top-0 w-[40px] pointer-events-none">
                    <svg
                        width="40"
                        height={headings.length * itemHeight}
                        viewBox={`0 0 40 ${headings.length * itemHeight}`}
                        fill="none"
                    >
                        {/* Background segments */}
                        {segments.map((seg) => (
                            <path
                                key={`bg-${seg.id}`}
                                d={seg.path}
                                stroke="rgba(39, 39, 42, 0.6)" // zinc-800/60
                                strokeWidth="1.5"
                                fill="none"
                            />
                        ))}

                        {/* Active (Highlight) segment */}
                        {activeIndex !== -1 && (
                            <path
                                d={segments[activeIndex].path}
                                stroke="white"
                                strokeWidth="2.5"
                                fill="none"
                                strokeLinecap="round"
                                className="transition-all duration-300 ease-in-out"
                                style={{
                                    filter: "drop-shadow(0 0 8px rgba(255, 255, 255, 0.4))"
                                }}
                            />
                        )}
                    </svg>
                </div>

                <ul className="flex-1 space-y-0 relative z-10">
                    {headings.map((heading, i) => (
                        <li
                            key={heading.id}
                            className={cn(
                                "h-[36px] flex items-center transition-all duration-300",
                                heading.level === 3 ? "pl-10" : "pl-6"
                            )}
                        >
                            <a
                                href={`#${heading.id}`}
                                onClick={(e) => {
                                    e.preventDefault();
                                    const el = document.getElementById(heading.id);
                                    if (el) {
                                        const offset = 100;
                                        const bodyRect = document.body.getBoundingClientRect().top;
                                        const elementRect = el.getBoundingClientRect().top;
                                        const elementPosition = elementRect - bodyRect;
                                        const offsetPosition = elementPosition - offset;
                                        window.scrollTo({ top: offsetPosition, behavior: 'smooth' });
                                    }
                                }}
                                className={cn(
                                    "text-[13px] transition-all duration-300 block truncate w-full",
                                    activeId === heading.id
                                        ? "text-zinc-50 font-bold scale-[1.02] translate-x-1"
                                        : "text-zinc-500 hover:text-zinc-200"
                                )}
                            >
                                {heading.text}
                            </a>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
}
