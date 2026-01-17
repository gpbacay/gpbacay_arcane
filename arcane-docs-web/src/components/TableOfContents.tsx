"use client";

import { useEffect, useState, useMemo, useRef } from "react";
import { cn } from "@/lib/utils";

interface TOCProps {
    headings: { id: string; text: string; level: number }[];
}

export function TableOfContents({ headings }: TOCProps) {
    const [activeId, setActiveId] = useState<string>("");
    const [itemMetrics, setItemMetrics] = useState<{ id: string; y: number; height: number; x: number }[]>([]);
    const listRef = useRef<HTMLUListElement>(null);
    
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
            { rootMargin: "-80px 0% -80% 0%", threshold: 0 }
        );

        headings.forEach((heading) => {
            const element = document.getElementById(heading.id);
            if (element) observer.observe(element);
        });

        return () => observer.disconnect();
    }, [headings]);

    // Measure positions of each item to draw the path correctly even if text wraps
    useEffect(() => {
        const measure = () => {
            if (!listRef.current) return;
            const items = Array.from(listRef.current.children) as HTMLElement[];
            const metrics = items.map((item, i) => ({
                id: headings[i].id,
                y: item.offsetTop,
                height: item.offsetHeight,
                x: headings[i].level === 3 ? xIndent : xBase
            }));
            setItemMetrics(metrics);
        };

        measure();
        window.addEventListener('resize', measure);
        // Also remeasure after a short delay to account for font loading/layout shifts
        const timer = setTimeout(measure, 500);
        
        return () => {
            window.removeEventListener('resize', measure);
            clearTimeout(timer);
        };
    }, [headings]);

    const segments = useMemo(() => {
        if (itemMetrics.length === 0) return [];
        
        const result: { path: string; id: string; x: number; y: number }[] = [];
        let currentX = xBase;

        itemMetrics.forEach((m, i) => {
            const targetX = m.x;
            const startY = m.y;
            const endY = m.y + m.height;
            
            let segmentsPath = "";

            if (currentX !== targetX) {
                const dy = Math.min(12, m.height / 2);
                segmentsPath = `M ${currentX} ${startY} L ${targetX} ${startY + dy} V ${endY}`;
                currentX = targetX;
            } else {
                segmentsPath = `M ${currentX} ${startY} V ${endY}`;
            }

            result.push({
                path: segmentsPath,
                id: m.id,
                x: targetX,
                y: startY
            });
        });

        return result;
    }, [itemMetrics]);

    const activeIndex = headings.findIndex(h => h.id === activeId);
    
    // Automatically scroll the TOC list to keep the active item in view
    useEffect(() => {
        if (activeId && listRef.current) {
            const activeItem = listRef.current.querySelector(`[data-id="${activeId}"]`);
            if (activeItem) {
                activeItem.scrollIntoView({
                    behavior: "smooth",
                    block: "nearest",
                    inline: "start"
                });
            }
        }
    }, [activeId]);

    const totalHeight = itemMetrics.length > 0 ? itemMetrics[itemMetrics.length - 1].y + itemMetrics[itemMetrics.length - 1].height : 0;

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
                        height={totalHeight}
                        viewBox={`0 0 40 ${totalHeight}`}
                        fill="none"
                        preserveAspectRatio="none"
                    >
                        {/* Background segments */}
                        {segments.map((seg) => (
                            <path
                                key={`bg-${seg.id}`}
                                d={seg.path}
                                stroke="rgba(113, 113, 122, 0.8)" // zinc-500/80 - more visible
                                strokeWidth="1.5"
                                fill="none"
                                strokeLinecap="round"
                            />
                        ))}

                        {/* Active (Highlight) segment */}
                        {activeIndex !== -1 && segments[activeIndex] && (
                            <path
                                d={segments[activeIndex].path}
                                stroke="white"
                                strokeWidth="2"
                                fill="none"
                                strokeLinecap="square"
                                className="transition-all duration-300 ease-in-out"
                                style={{
                                    filter: "drop-shadow(0 0 8px rgba(255, 255, 255, 0.6))"
                                }}
                            />
                        )}
                    </svg>
                </div>

                <ul ref={listRef} className="flex-1 space-y-0 relative z-10">
                    {headings.map((heading) => (
                        <li
                            key={heading.id}
                            data-id={heading.id}
                            className={cn(
                                "min-h-[32px] flex items-center transition-all duration-300 py-1",
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
                                        
                                        // Update URL hash without jumping
                                        window.history.pushState(null, "", `#${heading.id}`);
                                    }
                                }}
                                className={cn(
                                    "text-[13px] leading-snug transition-all duration-300 block w-full",
                                    activeId === heading.id
                                        ? "text-zinc-50 font-semibold translate-x-0.5"
                                        : "text-zinc-500 hover:text-zinc-300"
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
