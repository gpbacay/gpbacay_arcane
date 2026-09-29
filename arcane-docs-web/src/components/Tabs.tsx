"use client";

import { useState, type ReactNode } from "react";

type Tab = { id: string; label: string; content: ReactNode };

// Every panel stays mounted (just hidden) so code blocks render once with the page.
export function Tabs({ name, tabs }: { name: string; tabs: Tab[] }) {
  const [active, setActive] = useState(tabs[0].id);
  return (
    <div className="mt-6">
      <div role="tablist" className="not-prose flex flex-wrap gap-px border border-zinc-800 bg-zinc-800 sm:inline-flex">
        {tabs.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            role="tab"
            id={`${name}-tab-${id}`}
            aria-selected={active === id}
            aria-controls={`${name}-panel-${id}`}
            onClick={() => setActive(id)}
            className={`px-4 py-2 text-sm font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#C785F2] ${
              active === id ? "bg-white text-black" : "bg-zinc-950 text-zinc-300 hover:bg-zinc-900"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {tabs.map(({ id, content }) => (
        <div key={id} role="tabpanel" id={`${name}-panel-${id}`} aria-labelledby={`${name}-tab-${id}`} className="mt-4" hidden={active !== id}>
          {content}
        </div>
      ))}
    </div>
  );
}
