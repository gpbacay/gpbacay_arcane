"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface TOCProps {
  headings: { id: string; text: string; level: number }[];
  onNavigate?: () => void;
}

// Height of the sticky header: a section scrolled under it is out of view.
const SCROLL_OFFSET = 120;
const X_BASE = 4;
const X_INDENT = 20;

type Metric = { y: number; height: number; x: number };

export function TableOfContents({ headings, onNavigate }: TOCProps) {
  // Every heading whose section is on screen, like fumadocs: one section, or several when they are short.
  const [activeIds, setActiveIds] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const listRef = useRef<HTMLUListElement>(null);

  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const tops = headings.map((heading) => document.getElementById(heading.id)?.getBoundingClientRect().top ?? Infinity);
      // A section runs from its heading to the next one; the last runs to the end of the page.
      const visible = headings.filter(
        (_, index) => tops[index] < window.innerHeight && (tops[index + 1] ?? Infinity) > SCROLL_OFFSET,
      );
      const next = visible.length ? visible.map((heading) => heading.id) : headings.slice(0, 1).map((heading) => heading.id);
      setActiveIds((prev) => (prev.join() === next.join() ? prev : next));
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
    const list = listRef.current;
    if (!list) return;
    const measure = () =>
      setMetrics(
        (Array.from(list.children) as HTMLElement[]).map((item, index) => ({
          y: item.offsetTop,
          height: item.offsetHeight,
          x: headings[index]?.level === 3 ? X_INDENT : X_BASE,
        })),
      );
    measure();
    // Fonts loading or the sidebar resizing rewraps items, which moves the line.
    const observer = new ResizeObserver(measure);
    observer.observe(list);
    return () => observer.disconnect();
  }, [headings]);

  // The list follows the page: whenever the active sections change, it scrolls to keep them centered.
  useEffect(() => {
    const list = listRef.current;
    if (!list || !activeIds.length) return;
    let box = list.parentElement;
    while (box && box.scrollHeight <= box.clientHeight) box = box.parentElement;
    if (!box || box === document.documentElement || box === document.body) return;
    const items = activeIds
      .map((id) => list.querySelector<HTMLElement>(`[data-id="${CSS.escape(id)}"]`))
      .filter((item): item is HTMLElement => Boolean(item));
    if (!items.length) return;
    const middle = (items[0].getBoundingClientRect().top + items[items.length - 1].getBoundingClientRect().bottom) / 2;
    const boxBounds = box.getBoundingClientRect();
    box.scrollTo({ top: box.scrollTop + middle - boxBounds.top - box.clientHeight / 2, behavior: "smooth" });
  }, [activeIds]);

  const goTo = (id: string) => {
    const element = document.getElementById(id);
    if (!element) return;
    element.scrollIntoView({ behavior: "smooth", block: "start" });
    window.history.replaceState(null, "", `#${id}`);
    onNavigate?.();
  };

  if (headings.length === 0) return null;

  // One line through every item, stepping in and out with the heading level.
  let path = "";
  let x = X_BASE;
  metrics.forEach((metric, index) => {
    if (index === 0) path += `M ${metric.x} ${metric.y}`;
    else if (metric.x !== x) path += ` L ${metric.x} ${metric.y + Math.min(12, metric.height / 2)}`;
    path += ` V ${metric.y + metric.height}`;
    x = metric.x;
  });
  const height = metrics.length ? metrics[metrics.length - 1].y + metrics[metrics.length - 1].height : 0;
  const line = (stroke: string) =>
    `<svg xmlns="http://www.w3.org/2000/svg" width="40" height="${height}"><path d="${path}" stroke="${stroke}" stroke-width="2" fill="none"/></svg>`;
  const mask = `url("data:image/svg+xml,${encodeURIComponent(line("black"))}")`;

  // The thumb covers the active items; it is the line itself, revealed through a mask, so it follows the bends.
  const first = headings.findIndex((heading) => activeIds.includes(heading.id));
  const last = headings.findLastIndex((heading) => activeIds.includes(heading.id));
  const thumb =
    first >= 0 && metrics[last] ? { top: metrics[first].y, height: metrics[last].y + metrics[last].height - metrics[first].y } : null;

  return (
    <div className="relative select-none px-1 font-sans">
      <div className="group mb-4 flex cursor-default items-center gap-3 text-zinc-300 md:mb-6">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-zinc-500 transition-colors group-hover:text-zinc-300">
          <path d="M3 12h18M3 6h18M3 18h18" />
        </svg>
        <span className="text-xs font-bold uppercase tracking-[0.1em] opacity-70">On this page</span>
      </div>

      <div className="relative flex">
        {height > 0 && (
          <div
            aria-hidden
            className="pointer-events-none absolute left-0 top-0 w-[40px]"
            style={{ height, maskImage: mask, WebkitMaskImage: mask, maskRepeat: "no-repeat", WebkitMaskRepeat: "no-repeat" }}
          >
            <div className="absolute inset-0 bg-zinc-700" />
            {thumb && (
              <div
                className="absolute inset-x-0 bg-white transition-[top,height] duration-300 ease-out motion-reduce:transition-none"
                style={thumb}
              />
            )}
          </div>
        )}

        <ul ref={listRef} className="relative z-10 flex-1">
          {headings.map((heading) => {
            const active = activeIds.includes(heading.id);
            return (
              <li
                key={heading.id}
                data-id={heading.id}
                className={cn("flex min-h-[32px] items-center py-1", heading.level === 3 ? "pl-10" : "pl-6")}
              >
                <a
                  href={`#${heading.id}`}
                  onClick={(event) => {
                    event.preventDefault();
                    goTo(heading.id);
                  }}
                  aria-current={active ? "location" : undefined}
                  className={cn(
                    "block w-full text-[13px] leading-snug transition-colors duration-300",
                    active ? "text-zinc-50" : "text-zinc-500 hover:text-zinc-300",
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
