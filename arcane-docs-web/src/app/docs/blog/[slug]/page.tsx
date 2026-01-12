import { Markdown } from "@/components/markdown";
import fs from "fs";
import path from "path";
import Image from "next/image";
import Link from "next/link";
import { ExternalLink, Calendar, User, ChevronLeft } from "lucide-react";
import { notFound } from "next/navigation";
import posts from "@/config/blog.json";

export default async function BlogPostPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;

  const post = posts.find(p => p.slug === slug);

  if (!post) {
    return notFound();
  }

  let content = "";

  if (post.contentFile) {
    const blogPath = path.join(process.cwd(), "..", "docs", post.contentFile);
    try {
      content = fs.readFileSync(blogPath, "utf8");
      // Remove main title
      content = content.replace(/^# .*\n/, "");
    } catch (error) {
      content = "Content coming soon...";
    }
  } else {
    content = `## Content coming soon...\n\nThank you for your interest in "${post.title}". This article is currently being prepared for the web documentation. Please check back later or follow the author for updates.`;
  }

  return (
    <div className="text-zinc-300">
      <Link 
        href="/docs/blog" 
        className="inline-flex items-center gap-2 text-sm font-medium text-zinc-500 hover:text-zinc-100 transition-colors mb-8 group no-underline"
      >
        <ChevronLeft className="w-4 h-4 transition-transform group-hover:-translate-x-1" />
        Back to all insights
      </Link>

      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-6 text-zinc-100 leading-tight">
          {post.title}
        </h1>
        
        <div className="flex flex-wrap items-center gap-6 text-sm text-zinc-400 font-medium">
          <div className="flex items-center gap-2">
            {post.authorImage ? (
              <div className="w-6 h-6 rounded-full overflow-hidden border border-zinc-800">
                <Image 
                  src={post.authorImage} 
                  alt={post.author} 
                  width={24} 
                  height={24}
                  className="object-cover"
                />
              </div>
            ) : (
              <User className="w-4 h-4 text-purple-400" />
            )}
            <span>{post.author}</span>
          </div>
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-purple-400" />
            <span>{post.date}</span>
          </div>
          {post.mediumUrl && (
            <Link 
              href={post.mediumUrl}
              target="_blank"
              className="flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors group no-underline"
            >
              <span className="border-b border-purple-400/30 group-hover:border-purple-300">Read on Medium</span>
              <ExternalLink className="w-4 h-4" />
            </Link>
          )}
        </div>
      </div>

      {/* Hero Image */}
      <div className="relative w-full aspect-video overflow-hidden rounded-2xl border border-zinc-800 mb-12 shadow-2xl">
        <Image
          src={post.image}
          alt={post.title}
          fill
          className="object-cover"
          priority
        />
      </div>

      <div className="prose prose-zinc dark:prose-invert max-w-none prose-h3:text-zinc-100 prose-h3:mt-12 prose-p:text-zinc-400 prose-p:leading-8 prose-li:text-zinc-400">
        <Markdown content={content} />
      </div>
    </div>
  );
}
