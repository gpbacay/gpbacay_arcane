"use client";

import { useEffect, useState, useMemo, useRef } from "react";
import { cn } from "@/lib/utils";

interface TOCProps {
    headings: { id: string; text: string; level: number }[];
    onNavigate?: () => void;
}

// Sticky site header (64px) plus the mobile "Menu / On Page" bar, with breathing room.
const SCROLL_OFFSET = 120;

export function TableOfContents({ headings, onNavigate }: TOCProps) {
    const [activeId, setActiveId] = useState<string>("");
    const [itemMetrics, setItemMetrics] = useState<{ id: string; y: number; height: number; x: number }[]>([]);
    const listRef = useRef<HTMLUListElement>(null);
    
    const xBase = 4;
    const xIndent = 20;

    // Set by a click; keeps the clicked entry active while the page scrolls past other headings.
    const lockRef = useRef<{ id: string; timer: number } | null>(null);

    // Active heading = the last one whose top has passed the header, or the last one at the page bottom.
    useEffect(() => {
        let frame = 0;
        const update = () => {
            frame = 0;
            if (lockRef.current) return;
            const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
            let current = headings[0]?.id ?? "";
            for (const h of headings) {
                const el = document.getElementById(h.id);
                if (el && el.getBoundingClientRect().top <= SCROLL_OFFSET + 8) current = h.id;
            }
            if (atBottom && headings.length) current = headings[headings.length - 1].id;
            setActiveId(current);
        };
        const onScroll = () => {
            if (!frame) frame = requestAnimationFrame(update);
        };
        update();
        window.addEventListener("scroll", onScroll, { passive: true });
        window.addEventListener("resize", onScroll);
        return () => {
            window.removeEventListener("scroll", onScroll);
            window.removeEventListener("resize", onScroll);
            if (frame) cancelAnimationFrame(frame);
        };
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
    
    // Keep the active entry visible by scrolling only the TOC's own container. scrollIntoView would also
    // touch the window and cancel the page's smooth scroll mid-way, landing on the wrong heading.
    useEffect(() => {
        const list = listRef.current;
        const item = list?.querySelector<HTMLElement>(`[data-id="${CSS.escape(activeId)}"]`);
        let box = list?.parentElement ?? null;
        while (box && box.scrollHeight <= box.clientHeight) box = box.parentElement;
        if (!item || !box || box === document.documentElement || box === document.body) return;
        const b = box.getBoundingClientRect();
        const r = item.getBoundingClientRect();
        if (r.top < b.top || r.bottom > b.bottom) box.scrollBy({ top: r.top - b.top - b.height / 2, behavior: "smooth" });
    }, [activeId]);

    const goTo = (id: string) => {
        const el = document.getElementById(id);
        if (!el) return;
        setActiveId(id);
        if (lockRef.current) clearTimeout(lockRef.current.timer);
        lockRef.current = { id, timer: window.setTimeout(() => (lockRef.current = null), 900) };
        const top = el.getBoundingClientRect().top + window.scrollY - SCROLL_OFFSET;
        window.scrollTo({ top, behavior: "smooth" });
        window.history.replaceState(null, "", `#${id}`);
        onNavigate?.();
    };

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
                                    goTo(heading.id);
                                }}
                                aria-current={activeId === heading.id ? "location" : undefined}
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
