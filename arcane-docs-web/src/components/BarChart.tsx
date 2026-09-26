"use client";

import { useState } from "react";

export type BarDatum = { label: string; value: number; detail?: string };

type Props = {
  title: string;
  subtitle?: string;
  data: BarDatum[];
  max: number;
  ticks: number[];
  unit: "percent" | "mb";
  highlight?: string;
};

const FORMAT = {
  percent: (v: number) => `${Math.round(v * 100)}%`,
  mb: (v: number) => `${(v / 1e6).toFixed(v < 1e6 && v > 0 ? 2 : 1)} MB`,
};

// Axis ticks stay short so they never collide at phone width; the unit rides on the last tick.
const TICK = {
  percent: FORMAT.percent,
  mb: (v: number, last: boolean) => `${v / 1e6}${last ? " MB" : ""}`,
};

const BAR = "#9B6BE6"; // validated against the dark surface (lightness band + 3:1 contrast)
const BAR_MUTED = "#52525b";

/** Horizontal single-series bar chart: label | bar | value, with hover/focus tooltips. */
export function BarChart({ title, subtitle, data, max, ticks, unit, highlight }: Props) {
  const [active, setActive] = useState<number | null>(null);
  const format = FORMAT[unit];
  const pct = (v: number) => `${Math.min(100, (v / max) * 100)}%`;

  return (
    <figure className="not-prose my-6 border border-zinc-800 bg-zinc-950 p-4 sm:p-5">
      <figcaption className="mb-4">
        <div className="text-sm font-semibold text-zinc-100">{title}</div>
        {subtitle && <div className="mt-0.5 text-xs text-zinc-500">{subtitle}</div>}
      </figcaption>
      <div role="list" className="grid grid-cols-[minmax(0,7.5rem)_1fr_3.75rem] items-center gap-x-3 sm:grid-cols-[minmax(0,12rem)_1fr_4.5rem]">
        {data.map((d, i) => {
          const on = highlight ? d.label === highlight : true;
          return (
            <div
              key={d.label}
              role="listitem"
              tabIndex={0}
              aria-label={`${d.label}: ${format(d.value)}${d.detail ? `, ${d.detail}` : ""}`}
              onPointerEnter={() => setActive(i)}
              onPointerLeave={() => setActive(null)}
              onFocus={() => setActive(i)}
              onBlur={() => setActive(null)}
              className="col-span-3 grid grid-cols-subgrid items-center py-1.5 outline-none focus-visible:bg-zinc-900"
            >
              <span className="truncate text-xs text-zinc-300 sm:text-sm">{d.label}</span>
              <span className="relative block h-3.5">
                {ticks.map((t) => (
                  <span key={t} className="absolute inset-y-[-6px] w-px bg-zinc-800" style={{ left: pct(t) }} aria-hidden />
                ))}
                <span
                  className="absolute inset-y-0 left-0 rounded-r-[4px]"
                  style={{ width: pct(d.value), background: on ? BAR : BAR_MUTED }}
                  aria-hidden
                />
                {active === i && d.detail && (
                  <span className="pointer-events-none absolute bottom-full left-0 z-10 mb-2 whitespace-nowrap border border-zinc-700 bg-black px-2 py-1 text-xs text-zinc-200 shadow-lg">
                    {d.detail}
                  </span>
                )}
              </span>
              <span className="text-right text-xs tabular-nums text-zinc-200 sm:text-sm">{format(d.value)}</span>
            </div>
          );
        })}
        {/* axis */}
        <span aria-hidden />
        <span className="relative mt-1 block h-4" aria-hidden>
          {ticks.map((t, i) => (
            <span key={t} className="absolute -translate-x-1/2 whitespace-nowrap text-[10px] tabular-nums text-zinc-500" style={{ left: pct(t) }}>
              {TICK[unit](t, i === ticks.length - 1)}
            </span>
          ))}
        </span>
        <span aria-hidden />
      </div>
    </figure>
  );
}
