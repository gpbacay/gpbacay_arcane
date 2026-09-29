"use client";

import type { BundledLanguage } from "@/components/kibo-ui/code-block";
import {
  CodeBlock,
  CodeBlockBody,
  CodeBlockContent,
  CodeBlockCopyButton,
  CodeBlockFilename,
  CodeBlockFiles,
  CodeBlockHeader,
  CodeBlockItem,
} from "@/components/kibo-ui/code-block";

type CodeSnippetProps = {
  filename: string;
  language: BundledLanguage;
  code: string;
  lineNumbers?: boolean;
};

// The kibo-ui code block themes itself with the shadcn `dark` variant. This site is dark by
// design but doesn't set `.dark` on <html>, so the wrapper scopes it to the block.
export function CodeSnippet({ filename, language, code, lineNumbers = true }: CodeSnippetProps) {
  const data = [{ language, filename, code }];
  return (
    <div className="dark not-prose mt-4">
      <CodeBlock data={data} defaultValue={language} className="rounded-none border-zinc-800">
        <CodeBlockHeader className="justify-between border-zinc-800 bg-zinc-900/60">
          <CodeBlockFiles>
            {(item) => (
              <CodeBlockFilename key={item.language} value={item.language} className="bg-transparent text-zinc-400">
                {item.filename}
              </CodeBlockFilename>
            )}
          </CodeBlockFiles>
          <CodeBlockCopyButton aria-label="Copy code" />
        </CodeBlockHeader>
        <CodeBlockBody>
          {(item) => (
            <CodeBlockItem key={item.language} value={item.language} lineNumbers={lineNumbers} className="bg-zinc-950">
              <CodeBlockContent language={item.language as BundledLanguage}>{item.code}</CodeBlockContent>
            </CodeBlockItem>
          )}
        </CodeBlockBody>
      </CodeBlock>
    </div>
  );
}
