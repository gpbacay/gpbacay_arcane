"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { docsConfig } from "@/config/docs";
import { useEffect, useState, useRef, useMemo } from "react";
import { SidebarTrigger } from "@/components/ui/sidebar";

function SidebarGroup({ item, pathname }: { item: any, pathname: string }) {
  const [itemMetrics, setItemMetrics] = useState<{ y: number; height: number; active: boolean }[]>([]);
  const groupRef = useRef<HTMLDivElement>(null);
  const [activeIdx, setActiveIdx] = useState(-1);

  useEffect(() => {
    const idx = item.items.findIndex((subItem: any) => subItem.href === pathname);
    setActiveIdx(idx);
  }, [pathname, item.items]);

  useEffect(() => {
    const measure = () => {
      if (!groupRef.current) return;
      const links = Array.from(groupRef.current.querySelectorAll('a')) as HTMLElement[];
      const metrics = links.map((link, i) => ({
        y: link.offsetTop,
        height: link.offsetHeight,
        active: i === activeIdx
      }));
      setItemMetrics(metrics);
    };

    measure();
    window.addEventListener('resize', measure);
    const timer = setTimeout(measure, 100);
    return () => {
      window.removeEventListener('resize', measure);
      clearTimeout(timer);
    };
  }, [item.items, activeIdx]);

  const xBase = 2;
  const xActive = 8;

  const totalHeight = itemMetrics.length > 0 
    ? itemMetrics[itemMetrics.length - 1].y + itemMetrics[itemMetrics.length - 1].height 
    : 0;

  return (
    <div className="space-y-3 relative">
      <div className="flex items-center justify-between pr-2">
        <h4 className="text-xs font-bold uppercase tracking-[0.2em] text-zinc-500 px-4 transition-colors">
          {item.title}
        </h4>
        {item.title === "Getting Started" && (
          <SidebarTrigger className="h-6 w-6 text-zinc-500 hover:text-zinc-200 hover:bg-zinc-900 transition-colors" />
        )}
      </div>
      <div className="relative flex px-2" ref={groupRef}>
        {/* SVG Indicator Guide */}
        <div className="absolute left-2 top-0 w-4 h-full pointer-events-none">
          <svg 
            width="16" 
            height={totalHeight} 
            viewBox={`0 0 16 ${totalHeight}`} 
            fill="none"
            preserveAspectRatio="none"
          >
            {/* Background segments */}
            {itemMetrics.map((m, i) => (
              <path
                key={`bg-${i}`}
                d={`M ${xBase} ${m.y} V ${m.y + m.height}`}
                stroke="rgba(39, 39, 42, 0.4)"
                strokeWidth="1"
              />
            ))}

            {/* Active Highlight with Sharp Curve */}
            {activeIdx !== -1 && itemMetrics[activeIdx] && (
              <path
                d={`M ${xBase} ${itemMetrics[activeIdx].y} 
                   L ${xActive} ${itemMetrics[activeIdx].y + 6} 
                   V ${itemMetrics[activeIdx].y + itemMetrics[activeIdx].height - 6} 
                   L ${xBase} ${itemMetrics[activeIdx].y + itemMetrics[activeIdx].height}`}
                stroke="#C785F2"
                strokeWidth="2"
                fill="none"
                strokeLinecap="square"
                strokeLinejoin="miter"
                className="transition-all duration-300 ease-in-out"
                style={{
                  filter: "drop-shadow(0 0 8px rgba(199, 133, 242, 0.6))"
                }}
              />
            )}
          </svg>
        </div>

        <div className="grid grid-flow-row auto-rows-max text-sm space-y-1 flex-1 pl-6">
          {item.items.map((subItem: any, subIndex: number) => (
            <Link
              key={subIndex}
              href={subItem.href}
              className={cn(
                "group flex w-full items-center rounded-none px-3 py-2 transition-all duration-200",
                pathname === subItem.href
                  ? "bg-[#C785F2]/5 text-[#C785F2] font-semibold"
                  : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900"
              )}
            >
              {subItem.title}
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}

export function DocsSidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-full">
      <div className="py-8 px-4 md:px-6">
        <div className="space-y-12">
          {/* Mobile-only Global Navigation */}
          <div className="md:hidden space-y-3 mb-8 pb-8 border-b border-zinc-800">
            <h4 className="text-xs font-bold uppercase tracking-[0.2em] text-zinc-500 px-4">
              Navigation
            </h4>
            <div className="grid grid-flow-row auto-rows-max text-sm space-y-1">
              <Link href="/" className="flex w-full items-center rounded-none px-4 py-2 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 transition-all">
                Home
              </Link>
              <Link href="/docs" className="flex w-full items-center rounded-none px-4 py-2 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 transition-all">
                Documentation
              </Link>
              <Link href="/docs/layers" className="flex w-full items-center rounded-none px-4 py-2 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 transition-all">
                Layers
              </Link>
            </div>
          </div>

          {docsConfig.sidebarNav.map((item, index) => (
            <SidebarGroup key={index} item={item} pathname={pathname} />
          ))}
        </div>
      </div>
    </aside>
  );
}
