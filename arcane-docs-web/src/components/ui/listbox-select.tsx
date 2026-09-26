"use client";

import { useCallback, useEffect, useId, useRef, useState, type ComponentType } from "react";
import { Check, ChevronDown } from "lucide-react";

export type ListboxOption = {
  value: string;
  label: string;
  description?: string;
  icon?: ComponentType<{ className?: string }>;
};

type Props = {
  label: string;
  value: string;
  options: ListboxOption[];
  onChange: (value: string) => void;
  className?: string;
};

/**
 * Accessible single-select listbox: a trigger button that opens a list of
 * options with icons and descriptions. Keyboard: Arrow keys, Home/End, Enter
 * or Space to choose, Escape or Tab to close. Typing a letter jumps to it.
 */
export function ListboxSelect({ label, value, options, onChange, className = "" }: Props) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(() => Math.max(0, options.findIndex((o) => o.value === value)));
  const rootRef = useRef<HTMLDivElement | null>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const listRef = useRef<HTMLUListElement | null>(null);
  const selected = options.find((o) => o.value === value) || options[0];
  const SelectedIcon = selected?.icon;

  const openList = useCallback(() => {
    setActive(Math.max(0, options.findIndex((o) => o.value === value)));
    setOpen(true);
  }, [options, value]);

  const close = useCallback((focusButton = true) => {
    setOpen(false);
    if (focusButton) buttonRef.current?.focus();
  }, []);

  const choose = useCallback(
    (index: number) => {
      const option = options[index];
      if (option) onChange(option.value);
      close();
    },
    [options, onChange, close]
  );

  useEffect(() => {
    if (!open) return;
    listRef.current?.focus();
    const onPointer = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onPointer);
    return () => document.removeEventListener("pointerdown", onPointer);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    listRef.current?.querySelector<HTMLElement>(`[data-index="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [active, open]);

  const onListKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => Math.min(options.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(0, i - 1));
    } else if (e.key === "Home") {
      e.preventDefault();
      setActive(0);
    } else if (e.key === "End") {
      e.preventDefault();
      setActive(options.length - 1);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      choose(active);
    } else if (e.key === "Escape") {
      e.preventDefault();
      close();
    } else if (e.key === "Tab") {
      close(false);
    } else if (e.key.length === 1) {
      const start = (active + 1) % options.length;
      const order = [...options.slice(start), ...options.slice(0, start)];
      const hit = order.find((o) => o.label.toLowerCase().startsWith(e.key.toLowerCase()));
      if (hit) setActive(options.indexOf(hit));
    }
  };

  return (
    <div ref={rootRef} className={`relative ${className}`}>
      <span id={`${id}-label`} className="sr-only">
        {label}
      </span>
      <button
        ref={buttonRef}
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-labelledby={`${id}-label ${id}-value`}
        onClick={() => (open ? close() : openList())}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            openList();
          }
        }}
        className="flex w-full items-center gap-2.5 border border-zinc-700 bg-black px-3 py-2 text-left text-sm text-zinc-100 transition-colors hover:border-zinc-500 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-[#C785F2]"
      >
        {SelectedIcon && <SelectedIcon className="h-4 w-4 shrink-0 text-[#C785F2]" />}
        <span id={`${id}-value`} className="min-w-0 flex-1 truncate">
          {selected?.label}
        </span>
        <ChevronDown
          className={`h-4 w-4 shrink-0 text-zinc-500 transition-transform motion-reduce:transition-none ${open ? "rotate-180" : ""}`}
          aria-hidden
        />
      </button>

      {open && (
        <ul
          ref={listRef}
          role="listbox"
          tabIndex={-1}
          aria-labelledby={`${id}-label`}
          aria-activedescendant={`${id}-opt-${active}`}
          onKeyDown={onListKey}
          className="absolute left-0 right-0 z-50 mt-1 max-h-80 overflow-auto border border-zinc-700 bg-zinc-950 py-1 shadow-[0_12px_32px_rgba(0,0,0,0.6)] outline-none sm:right-auto sm:min-w-[20rem]"
        >
          {options.map((o, i) => {
            const Icon = o.icon;
            const isSelected = o.value === value;
            return (
              <li
                key={o.value}
                id={`${id}-opt-${i}`}
                data-index={i}
                role="option"
                aria-selected={isSelected}
                onPointerMove={() => setActive(i)}
                onClick={() => choose(i)}
                className={`flex cursor-pointer items-start gap-3 px-3 py-2.5 ${i === active ? "bg-[#835BD9]/20" : ""}`}
              >
                {Icon && <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${isSelected ? "text-[#C785F2]" : "text-zinc-500"}`} />}
                <span className="min-w-0 flex-1">
                  <span className="block text-sm text-zinc-100">{o.label}</span>
                  {o.description && <span className="mt-0.5 block text-xs leading-snug text-zinc-500">{o.description}</span>}
                </span>
                {isSelected && <Check className="mt-0.5 h-4 w-4 shrink-0 text-[#C785F2]" aria-hidden />}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
