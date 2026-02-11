import { NextRequest, NextResponse } from "next/server";

/**
 * Proxies MNIST prediction to the backend configured by MNIST_API_URL.
 * Uses dynamic env so it works on Vercel (set MNIST_API_URL in project settings)
 * and locally (dev:with-mnist sets it, or add to .env.local).
 */
export async function POST(request: NextRequest) {
  const base = process.env.MNIST_API_URL;
  if (!base || typeof base !== "string") {
    return NextResponse.json(
      {
        error:
          "MNIST backend not configured. Set MNIST_API_URL to your MNIST API base (e.g. https://your-mnist-api.railway.app or http://127.0.0.1:8000 for local).",
      },
      { status: 503 }
    );
  }

  const url = `${base.replace(/\/$/, "")}/predict`;
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail ?? data.error ?? res.statusText;
      const errorMessage = typeof detail === "string" ? detail : JSON.stringify(detail);
      return NextResponse.json(
        { error: errorMessage },
        { status: res.status }
      );
    }
    return NextResponse.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Backend request failed";
    return NextResponse.json(
      { error: message },
      { status: 502 }
    );
  }
}
