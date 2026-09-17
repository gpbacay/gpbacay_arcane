import { NextResponse } from "next/server";

function slmApiBase() {
  return (
    process.env.SLM_API_URL ||
    (process.env.NODE_ENV !== "production" ? "http://127.0.0.1:8001" : undefined)
  );
}

export async function GET() {
  const base = slmApiBase();
  if (!base) {
    return NextResponse.json(
      {
        ready: false,
        status: "offline",
        error:
          "SLM backend is not configured. From arcane-docs-web run npm run dev:with-slm, or start python examples/serve_slm_api.py and set SLM_API_URL.",
      },
      { status: 503 }
    );
  }

  try {
    const res = await fetch(`${base.replace(/\/$/, "")}/health`, {
      cache: "no-store",
    });
    const data = await res.json().catch(() => ({}));
    return NextResponse.json(data, { status: res.ok ? 200 : res.status });
  } catch {
    return NextResponse.json(
      {
        ready: false,
        status: "offline",
        error: "Could not reach the ARCANE SLM API.",
      },
      { status: 503 }
    );
  }
}
