"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

export function GitHubStarButton() {
  const [stars, setStars] = useState<number | null>(null);

  useEffect(() => {
    fetch("https://api.github.com/repos/gpbacay/gpbacay_arcane")
      .then((res) => res.json())
      .then((data) => {
        if (typeof data.stargazers_count === "number") {
          setStars(data.stargazers_count);
        }
      })
      .catch((err) => console.error("Failed to fetch stars", err));
  }, []);

  return (
    <Link
      href="https://github.com/gpbacay/gpbacay_arcane"
      target="_blank"
      rel="noreferrer"
      className="group flex items-center overflow-hidden rounded-full bg-indigo-600 text-xs font-bold text-white transition-all hover:bg-indigo-500 shadow-lg shadow-indigo-500/20"
    >
      <span className="flex h-9 items-center px-4 bg-[#835BD9]">
        Star On GitHub
      </span>
      <span className="flex h-9 items-center gap-1 bg-black px-4 transition-colors group-hover:bg-zinc-900 border-l border-white/10">
        <svg viewBox="0 0 24 24" fill="currentColor" className="h-4 w-4 text-white">
            <path fillRule="evenodd" d="M10.788 3.21c.448-1.077 1.976-1.077 2.424 0l2.082 5.007 5.404.433c1.164.093 1.636 1.545.749 2.305l-4.117 3.527 1.257 5.273c.271 1.136-.964 2.033-1.96 1.425L12 18.354 7.373 21.18c-.996.608-2.231-.29-1.96-1.425l1.257-5.273-4.117-3.527c-.887-.76-.415-2.212.749-2.305l5.404-.433 2.082-5.006z" clipRule="evenodd" />
        </svg>
        {stars !== null ? stars.toLocaleString() : "..."}
      </span>
    </Link>
  );
}
