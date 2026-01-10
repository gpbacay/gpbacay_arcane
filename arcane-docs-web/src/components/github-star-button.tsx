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
      className="group flex items-center overflow-hidden rounded-full bg-indigo-600 text-[10px] sm:text-xs font-bold text-white transition-all hover:bg-indigo-500 shadow-lg shadow-indigo-500/20"
    >
      <span className="hidden sm:flex h-8 sm:h-9 items-center px-3 sm:px-4 bg-[#835BD9]">
        Star On GitHub
      </span>
      <span className="flex sm:hidden h-8 items-center px-2 bg-[#835BD9]">
        <svg viewBox="0 0 24 24" fill="currentColor" className="h-4 w-4 text-white">
          <path d="M12 .5C5.37.5 0 5.87 0 12.5c0 5.3 3.438 9.8 8.205 11.385.6.11.82-.26.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.292 24 17.792 24 12.5 24 5.87 18.63.5 12 .5z"/>
        </svg>
      </span>
      <span className="flex h-8 sm:h-9 items-center gap-1 bg-black px-2 sm:px-4 transition-colors group-hover:bg-zinc-900 border-l border-white/10">
        <svg viewBox="0 0 24 24" fill="currentColor" className="h-3 w-3 sm:h-4 sm:w-4 text-white">
            <path fillRule="evenodd" d="M10.788 3.21c.448-1.077 1.976-1.077 2.424 0l2.082 5.007 5.404.433c1.164.093 1.636 1.545.749 2.305l-4.117 3.527 1.257 5.273c.271 1.136-.964 2.033-1.96 1.425L12 18.354 7.373 21.18c-.996.608-2.231-.29-1.96-1.425l1.257-5.273-4.117-3.527c-.887-.76-.415-2.212.749-2.305l5.404-.433 2.082-5.006z" clipRule="evenodd" />
        </svg>
        {stars !== null ? stars.toLocaleString() : "..."}
      </span>
    </Link>
  );
}
