
import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-black text-white">
      <h1 className="text-6xl font-bold text-purple-500 mb-4">404</h1>
      <h2 className="text-2xl font-semibold mb-6">Page Not Found</h2>
      <p className="text-zinc-400 mb-8 max-w-md text-center">
        The page you are looking for does not exist. It might have been moved or deleted.
      </p>
      <Link
        href="/"
        className="rounded-full bg-white px-8 py-3 text-sm font-medium text-black transition-colors hover:bg-zinc-200"
      >
        Back to Home
      </Link>
    </div>
  );
}
