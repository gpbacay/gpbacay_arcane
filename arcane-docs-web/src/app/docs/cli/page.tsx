
export default function CLIPage() {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          CLI Commands
        </h1>
        <p className="text-xl text-zinc-400">
          Command-line utilities for model management and information.
        </p>
      </div>

      <div className="space-y-8 text-zinc-300 leading-7">
        <p>
          ARCANE comes with built-in CLI commands to help you manage your installation and inspect available components.
        </p>

        <div className="grid gap-4">
          <div className="rounded-none border border-zinc-800 bg-zinc-900 p-4">
            <h3 className="font-mono text-zinc-100 mb-2">gpbacay-arcane-about</h3>
            <p className="text-sm text-zinc-400">Show library information and current version.</p>
          </div>

          <div className="rounded-none border border-zinc-800 bg-zinc-900 p-4">
            <h3 className="font-mono text-zinc-100 mb-2">gpbacay-arcane-list-models</h3>
            <p className="text-sm text-zinc-400">List all available pre-built model architectures.</p>
          </div>

          <div className="rounded-none border border-zinc-800 bg-zinc-900 p-4">
            <h3 className="font-mono text-zinc-100 mb-2">gpbacay-arcane-list-layers</h3>
            <p className="text-sm text-zinc-400">List all valid biological neural layers available for custom architectures.</p>
          </div>

          <div className="rounded-none border border-zinc-800 bg-zinc-900 p-4">
            <h3 className="font-mono text-zinc-100 mb-2">gpbacay-arcane-version</h3>
            <p className="text-sm text-zinc-400">Quickly print the installed version number.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
