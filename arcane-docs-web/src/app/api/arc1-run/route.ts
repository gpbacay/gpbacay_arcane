import { NextRequest, NextResponse } from "next/server";

function arc1ApiBase() {
  return (
    process.env.ARC1_API_URL ||
    (process.env.NODE_ENV !== "production" ? "http://127.0.0.1:8002" : undefined)
  );
}

export const maxDuration = 120;

export async function POST(request: NextRequest) {
  const base = arc1ApiBase();
  if (!base) {
    return NextResponse.json(
      {
        error:
          "ARC 1 backend is not configured. From arcane-docs-web run npm run dev:with-arc1.",
      },
      { status: 503 }
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const res = await fetch(`${base.replace(/\/$/, "")}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(120_000),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail =
        typeof (data as { detail?: unknown }).detail === "string"
          ? (data as { detail: string }).detail
          : typeof (data as { error?: unknown }).error === "string"
            ? (data as { error: string }).error
            : res.statusText;
      return NextResponse.json({ error: detail }, { status: res.status });
    }
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "ARC 1 run failed or timed out." },
      { status: 504 }
    );
  }
}
