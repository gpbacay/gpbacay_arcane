"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface TOCProps {
  headings: { id: string; text: string; level: number }[];
  onNavigate?: () => void;
}

const SCROLL_OFFSET = 120;

export function TableOfContents({ headings, onNavigate }: TOCProps) {
  const [activeId, setActiveId] = useState<string>("");
  const [itemMetrics, setItemMetrics] = useState<{ id: string; y: number; height: number; x: number }[]>([]);
  const listRef = useRef<HTMLUListElement>(null);

  const xBase = 4;
  const xIndent = 20;
  const lockRef = useRef<{ id: string; timer: number } | null>(null);

  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      if (lockRef.current) return;
      const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
      let current = headings[0]?.id ?? "";
      for (const heading of headings) {
        const element = document.getElementById(heading.id);
        if (element && element.getBoundingClientRect().top <= SCROLL_OFFSET + 8) current = heading.id;
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

  useEffect(() => {
    const measure = () => {
      if (!listRef.current) return;
      const items = Array.from(listRef.current.children) as HTMLElement[];
      const metrics = items.map((item, index) => ({
        id: headings[index].id,
        y: item.offsetTop,
        height: item.offsetHeight,
        x: headings[index].level === 3 ? xIndent : xBase,
      }));
      setItemMetrics(metrics);
    };

    measure();
    window.addEventListener("resize", measure);
    const timer = setTimeout(measure, 500);
    return () => {
      window.removeEventListener("resize", measure);
      clearTimeout(timer);
    };
  }, [headings]);

  const segments = useMemo(() => {
    if (itemMetrics.length === 0) return [];
    const result: { path: string; id: string }[] = [];
    let currentX = xBase;

    itemMetrics.forEach((metric) => {
      const targetX = metric.x;
      const startY = metric.y;
      const endY = metric.y + metric.height;
      let path = "";

      if (currentX !== targetX) {
        const curveHeight = Math.min(12, metric.height / 2);
        path = `M ${currentX} ${startY} L ${targetX} ${startY + curveHeight} V ${endY}`;
        currentX = targetX;
      } else {
        path = `M ${currentX} ${startY} V ${endY}`;
      }

      result.push({ path, id: metric.id });
    });

    return result;
  }, [itemMetrics]);

  const activeIndex = headings.findIndex((heading) => heading.id === activeId);
  const activeGroupIds = useMemo(() => {
    if (activeIndex < 0) return new Set<string>();
    let parentIndex = activeIndex;
    while (parentIndex > 0 && headings[parentIndex].level !== 2) parentIndex -= 1;
    let endIndex = parentIndex + 1;
    while (endIndex < headings.length && headings[endIndex].level !== 2) endIndex += 1;
    return new Set(headings.slice(parentIndex, endIndex).map((heading) => heading.id));
  }, [activeIndex, headings]);

  useEffect(() => {
    const list = listRef.current;
    const item = list?.querySelector<HTMLElement>(`[data-id="${CSS.escape(activeId)}"]`);
    let box = list?.parentElement ?? null;
    while (box && box.scrollHeight <= box.clientHeight) box = box.parentElement;
    if (!item || !box || box === document.documentElement || box === document.body) return;
    const containerBounds = box.getBoundingClientRect();
    const itemBounds = item.getBoundingClientRect();
    if (itemBounds.top < containerBounds.top || itemBounds.bottom > containerBounds.bottom) {
      box.scrollBy({ top: itemBounds.top - containerBounds.top - containerBounds.height / 2, behavior: "smooth" });
    }
  }, [activeId]);

  const goTo = (id: string) => {
    const element = document.getElementById(id);
    if (!element) return;
    setActiveId(id);
    if (lockRef.current) clearTimeout(lockRef.current.timer);
    lockRef.current = { id, timer: window.setTimeout(() => (lockRef.current = null), 1400) };
    element.scrollIntoView({ behavior: "smooth", block: "start" });
    window.history.replaceState(null, "", `#${id}`);
    onNavigate?.();
  };

  const totalHeight = itemMetrics.length > 0
    ? itemMetrics[itemMetrics.length - 1].y + itemMetrics[itemMetrics.length - 1].height
    : 0;

  if (headings.length === 0) return null;

  return (
    <div className="relative select-none px-1 font-sans">
      <div className="group mb-4 flex cursor-default items-center gap-3 text-zinc-300 md:mb-6">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-zinc-500 transition-colors group-hover:text-zinc-300">
          <path d="M3 12h18M3 6h18M3 18h18" />
        </svg>
        <span className="text-xs font-bold uppercase tracking-[0.1em] opacity-70">On this page</span>
      </div>

      <div className="relative flex">
        <div className="pointer-events-none absolute left-0 top-0 w-[40px]">
          <svg width="40" height={totalHeight} viewBox={`0 0 40 ${totalHeight}`} fill="none" preserveAspectRatio="none">
            {segments.map((segment) => (
              <path
                key={`bg-${segment.id}`}
                d={segment.path}
                stroke="rgba(113, 113, 122, 0.8)"
                strokeWidth="1.5"
                fill="none"
                strokeLinecap="round"
              />
            ))}

            {segments
              .filter((segment) => activeGroupIds.has(segment.id))
              .map((segment) => (
                <path
                  key={`group-${segment.id}`}
                  d={segment.path}
                  stroke="rgba(199, 133, 242, 0.65)"
                  strokeWidth="1.75"
                  fill="none"
                  strokeLinecap="square"
                  className="transition-all duration-300 ease-in-out"
                />
              ))}

            {activeIndex !== -1 && segments[activeIndex] && (
              <path
                d={segments[activeIndex].path}
                stroke="white"
                strokeWidth="2"
                fill="none"
                strokeLinecap="square"
                className="transition-all duration-300 ease-in-out"
                style={{ filter: "drop-shadow(0 0 8px rgba(255, 255, 255, 0.6))" }}
              />
            )}
          </svg>
        </div>

        <ul ref={listRef} className="relative z-10 flex-1 space-y-0">
          {headings.map((heading) => {
            const exactActive = activeId === heading.id;
            const groupActive = activeGroupIds.has(heading.id);
            return (
              <li
                key={heading.id}
                data-id={heading.id}
                className={cn(
                  "flex min-h-[32px] items-center py-1 transition-all duration-300",
                  heading.level === 3 ? "pl-10" : "pl-6"
                )}
              >
                <a
                  href={`#${heading.id}`}
                  onClick={(event) => {
                    event.preventDefault();
                    goTo(heading.id);
                  }}
                  aria-current={exactActive ? "location" : undefined}
                  className={cn(
                    "block w-full text-[13px] leading-snug transition-all duration-300",
                    exactActive && "translate-x-0.5 font-semibold text-zinc-50",
                    !exactActive && groupActive && "text-zinc-300 hover:text-zinc-100",
                    !groupActive && "text-zinc-500 hover:text-zinc-300"
                  )}
                >
                  {heading.text}
                </a>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
