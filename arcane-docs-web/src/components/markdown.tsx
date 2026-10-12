"use client";

import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import mermaid from "mermaid";
import { useEffect, useState, useId } from "react";
import type { BundledLanguage } from "@/components/ui/code-block";
import { CodeSnippet } from "@/components/CodeSnippet";

if (typeof window !== "undefined") {
  mermaid.initialize({
    startOnLoad: false,
    theme: 'dark',
    securityLevel: 'loose',
    fontFamily: 'monospace',
    themeVariables: {
      darkMode: true,
      background: '#09090b',
      primaryColor: '#18181b',
      primaryTextColor: '#fff',
      primaryBorderColor: '#3f3f46',
      lineColor: '#71717a',
      secondaryColor: '#27272a',
      tertiaryColor: '#18181b',
    }
  });
}

export const Mermaid = ({ chart }: { chart: string; minWidth?: number }) => {
  const id = "mermaid-" + useId().replace(/:/g, "");
  const [svg, setSvg] = useState("");
  const verticalChart = chart
    .replace(/^(\s*)(flowchart|graph)\s+(LR|RL)\b/m, "$1$2 TD")
    .replace(/^(\s*)direction\s+(LR|RL)\b/gm, "$1direction TB");

  useEffect(() => {
    if (typeof window !== "undefined") {
      try {
        mermaid.render(id, verticalChart).then(({ svg }) => {
          setSvg(svg);
        });
      } catch (error) {
        console.error("Mermaid error:", error);
      }
    }
  }, [verticalChart, id]);

  if (!svg) return <div className="text-zinc-500 text-xs animate-pulse">Loading diagram...</div>;

  return (
    <div
      className="mermaid not-prose my-8 flex min-h-[240px] justify-center overflow-hidden border border-zinc-900/50 bg-zinc-950/30 p-4 sm:p-6 [&_svg]:mx-auto [&_svg]:h-auto [&_svg]:max-w-full"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
};

interface MarkdownProps {
  content: string;
  className?: string;
}

export function Markdown({ content, className }: MarkdownProps) {
  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkGfm]}
        rehypePlugins={[rehypeKatex]}
        components={{
          h1: ({ children }) => <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100">{children}</h1>,
          h2: ({ children }) => <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">{children}</h2>,
          h3: ({ children }) => <h3 className="text-xl font-bold text-zinc-100 mt-8 mb-4">{children}</h3>,
          p: ({ children }) => <p className="text-zinc-300 leading-7 mb-4">{children}</p>,
          ul: ({ children }) => <ul className="list-disc pl-6 space-y-2 marker:text-zinc-500 mb-4">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-6 space-y-2 marker:text-zinc-500 mb-4">{children}</ol>,
          li: ({ children }) => <li className="text-zinc-300">{children}</li>,
          code: ({ children, className }) => {
            const isInline = !className || !className.includes('language-');
            if (className === 'language-mermaid') {
              // Determine if children is string or array
              const chartContent = Array.isArray(children) ? children.join('') : String(children);
              return <Mermaid chart={chartContent} />;
            }
            return isInline ? (
              <code className="bg-zinc-800 px-1.5 py-0.5 rounded-none text-zinc-200 text-sm font-mono">{children}</code>
            ) : (
              <code className={className}>{children}</code>
            );
          },
          pre: ({ children, node }) => {
            // Check if the code block is mermaid to avoid wrapping it in standard pre styling
            const childNode = node?.children?.[0];
            let isMermaid = false;

            // Safe check for the AST node structure
            if (childNode && typeof childNode === 'object' && 'tagName' in childNode && childNode.tagName === 'code') {
              const properties = 'properties' in childNode ? childNode.properties : {};
              const className = properties && typeof properties === 'object' && 'className' in properties ? properties.className : null;

              if (Array.isArray(className) && className.includes('language-mermaid')) {
                isMermaid = true;
              }
            }

            if (isMermaid) {
              return <div className="relative">{children}</div>;
            }

            let language = "text";
            let code = "";
            if (childNode && typeof childNode === "object" && "tagName" in childNode && childNode.tagName === "code") {
              const properties = "properties" in childNode ? childNode.properties : {};
              const classes = properties && typeof properties === "object" && "className" in properties
                ? properties.className
                : [];
              const languageClass = Array.isArray(classes)
                ? classes.find((value) => typeof value === "string" && value.startsWith("language-"))
                : undefined;
              if (typeof languageClass === "string") language = languageClass.replace("language-", "");
              if ("children" in childNode && Array.isArray(childNode.children)) {
                code = childNode.children
                  .map((child) => typeof child === "object" && child && "value" in child ? String(child.value) : "")
                  .join("")
                  .replace(/\n$/, "");
              }
            }

            return (
              <CodeSnippet
                filename={language === "text" ? "Code" : language}
                language={language as BundledLanguage}
                code={code}
              />
            );
          },
          table: ({ children }) => (
            <div className="my-6 w-full overflow-y-auto">
              <table className="w-full border-collapse border border-zinc-800 text-sm">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-zinc-900 text-zinc-100">{children}</thead>,
          th: ({ children }) => <th className="border border-zinc-800 px-4 py-2 font-bold">{children}</th>,
          td: ({ children }) => <td className="border border-zinc-800 px-4 py-2 text-zinc-300">{children}</td>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
