
import { CodeSnippet } from "@/components/CodeSnippet";

export default function InstallationPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          Installation
        </h1>
        <p className="text-xl text-zinc-400">
          Install the ARCANE System 1 toolkit and its core runtime.
        </p>
      </div>

      <div className="space-y-8 text-zinc-300 leading-7">
        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Prerequisites
        </h2>
        <ul className="list-disc pl-6 space-y-2 marker:text-zinc-500">
          <li>Python 3.11+</li>
          <li>TensorFlow 2.12+</li>
        </ul>

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Install from PyPI (Recommended)
        </h2>
        <CodeSnippet filename="Terminal" language="bash" lineNumbers={false} code="pip install gpbacay-arcane" />

        <h2 className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Install from Source
        </h2>
        <p>
          If you want to contribute or use the latest development version:
        </p>
        <CodeSnippet filename="Terminal" language="bash" lineNumbers={false} code={`git clone https://github.com/gpbacay/gpbacay_arcane.git
cd gpbacay_arcane
pip install -e .`} />
      </div>
    </div>
  );
}
