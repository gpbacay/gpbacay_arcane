import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";

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
          li: ({ children }) => <li className="text-zinc-300">{children}</li>,
          code: ({ children, className }) => {
            const isInline = !className || !className.includes('language-');
            return isInline ? (
              <code className="bg-zinc-800 px-1.5 py-0.5 rounded-none text-zinc-200 text-sm font-mono">{children}</code>
            ) : (
              <code className={className}>{children}</code>
            );
          },
          pre: ({ children }) => (
            <pre className="bg-zinc-950 p-4 rounded-none border border-zinc-800 overflow-x-auto text-sm text-zinc-300 my-6">
              {children}
            </pre>
          ),
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
