import { NextRequest, NextResponse } from "next/server";

function slmApiBase() {
  return (
    process.env.SLM_API_URL ||
    (process.env.NODE_ENV !== "production" ? "http://127.0.0.1:8001" : undefined)
  );
}

export const maxDuration = 120;

export async function POST(request: NextRequest) {
  const base = slmApiBase();
  if (!base) {
    return NextResponse.json(
      {
        error:
          "SLM backend is not configured. From arcane-docs-web run npm run dev:with-slm.",
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
    const res = await fetch(`${base.replace(/\/$/, "")}/chat`, {
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
      { error: "ARCANE SLM request failed or timed out." },
      { status: 504 }
    );
  }
}
