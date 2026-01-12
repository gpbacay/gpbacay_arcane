import Image from "next/image";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { cn } from "@/lib/utils";
import posts from "@/config/blog.json";

export default function BlogPage() {
  return (
    <div className="text-zinc-400">
      {/* Header */}
      <div className="mb-16">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Latest News and Insights
        </h1>
        <p className="text-xl text-zinc-400">
          Stay updated with the latest trends and insights in the world of neuromimetic AI.
        </p>
      </div>

      {/* Blog List */}
      <div className="space-y-10">
        {posts.map((post) => (
          <div 
            key={post.id}
            className="group relative flex flex-col xl:flex-row gap-8 p-6 rounded-2xl border border-zinc-800/50 bg-zinc-900/20 hover:bg-zinc-900/40 transition-all duration-300 hover:border-zinc-700/50"
          >
            {/* Left: Image */}
            <div className="relative w-full xl:w-[300px] aspect-video xl:aspect-square overflow-hidden rounded-xl border border-zinc-800">
              <Image
                src={post.image}
                alt={post.title}
                fill
                className="object-cover transition-transform duration-500 group-hover:scale-105"
              />
            </div>

            {/* Right: Content */}
            <div className="flex-1 flex flex-col justify-between text-left">
              <div>
                <h2 id={post.slug} className="text-xl md:text-2xl font-bold text-white mb-4 leading-tight group-hover:text-purple-400 transition-colors text-left border-none mt-0">
                  {post.title}
                </h2>
                
                {/* Author Meta */}
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-8 h-8 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center overflow-hidden shrink-0">
                    <Image 
                      src={post.authorImage || "/arcane_logo_purple.svg"} 
                      alt={post.author} 
                      width={32} 
                      height={32}
                      className={cn(
                        "object-cover",
                        !post.authorImage && "opacity-70 p-1"
                      )}
                    />
                  </div>
                  <div className="flex flex-col text-left">
                    <span className="text-sm font-bold text-zinc-200 leading-none mb-1">{post.author}</span>
                    <span className="text-[11px] text-zinc-500 uppercase tracking-widest font-medium">{post.date}</span>
                  </div>
                </div>

                <p className="text-zinc-500 text-sm md:text-base leading-relaxed line-clamp-3 text-left m-0">
                  {post.excerpt}
                </p>
              </div>

              {/* Read More */}
              <div className="mt-8 flex justify-end">
                <Link 
                  href={`/docs/blog/${post.slug}`}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-900 border border-zinc-800 text-xs font-bold text-zinc-100 hover:bg-white hover:text-black transition-all group/btn no-underline"
                >
                  <ArrowUpRight className="w-3.5 h-3.5 transition-transform group-hover/btn:translate-x-0.5 group-hover/btn:-translate-y-0.5" />
                  Read More
                </Link>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
