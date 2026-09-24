import { NextResponse } from "next/server";

function arc1ApiBase() {
  return (
    process.env.ARC1_API_URL ||
    (process.env.NODE_ENV !== "production" ? "http://127.0.0.1:8002" : undefined)
  );
}

export async function GET() {
  const base = arc1ApiBase();
  if (!base) {
    return NextResponse.json(
      {
        ready: false,
        status: "offline",
        error:
          "ARC 1 backend is not configured. From arcane-docs-web run npm run dev:with-arc1.",
      },
      { status: 503 }
    );
  }

  try {
    const res = await fetch(`${base.replace(/\/$/, "")}/health`, { cache: "no-store" });
    const data = await res.json().catch(() => ({}));
    return NextResponse.json(data, { status: res.ok ? 200 : res.status });
  } catch {
    return NextResponse.json(
      {
        ready: false,
        status: "offline",
        error: "Could not reach the ARC 1 API.",
      },
      { status: 503 }
    );
  }
}
