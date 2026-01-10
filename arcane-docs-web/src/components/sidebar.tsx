"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/utils/cn";
import { docsConfig } from "@/config/docs";

export function DocsSidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed top-16 z-30 -ml-2 hidden h-[calc(100vh-4rem)] w-full shrink-0 md:sticky md:block">
      <div className="h-full overflow-y-auto py-8 pl-8 pr-6 scrollbar-hide">
        <div className="space-y-8">
          {docsConfig.sidebarNav.map((item, index) => (
            <div key={index} className="space-y-3">
              <h4 className="text-xs font-bold uppercase tracking-[0.2em] text-zinc-500 px-2 transition-colors">
                {item.title}
              </h4>
              <div className="grid grid-flow-row auto-rows-max text-sm space-y-1">
                {item.items.map((subItem, subIndex) => (
                  <Link
                    key={subIndex}
                    href={subItem.href}
                    className={cn(
                      "group flex w-full items-center rounded-lg px-3 py-2.5 transition-all duration-200",
                      pathname === subItem.href
                        ? "bg-indigo-600/10 text-[#C785F2] font-semibold border-l-2 border-[#C785F2]"
                        : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 border-l-2 border-transparent"
                    )}
                  >
                    {subItem.title}
                    {pathname === subItem.href && (
                       <span className="ml-auto flex h-1.5 w-1.5 rounded-full bg-[#C785F2] shadow-[0_0_8px_#C785F2]" />
                    )}
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
