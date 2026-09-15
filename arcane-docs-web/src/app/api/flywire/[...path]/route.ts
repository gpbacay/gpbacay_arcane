import { NextRequest, NextResponse } from "next/server";
import circuitFallback from "@/data/flywire_escape_circuit.json";

function localCircuit() {
  return NextResponse.json(circuitFallback);
}

async function proxy(request: NextRequest, path: string[]) {
  const suffix = path.join("/");
  const base = process.env.FLYWIRE_API_URL;
  const isCircuit = request.method === "GET" && (suffix === "circuit" || suffix === "");

  if (base && typeof base === "string") {
    const url = `${base.replace(/\/$/, "")}/${suffix || "circuit"}`;
    const init: RequestInit = {
      method: request.method,
      headers: { "Content-Type": "application/json" },
    };
    if (request.method !== "GET" && request.method !== "HEAD") {
      try {
        init.body = JSON.stringify(await request.json());
      } catch {
        return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
      }
    }
    try {
      const res = await fetch(url, init);
      const data = await res.json().catch(() => ({}));
      if (res.ok) return NextResponse.json(data);
      if (isCircuit) return localCircuit();
      const detail =
        (data as { detail?: unknown; error?: unknown }).detail ??
        (data as { error?: unknown }).error ??
        res.statusText;
      const errorMessage = typeof detail === "string" ? detail : JSON.stringify(detail);
      return NextResponse.json({ error: errorMessage }, { status: res.status });
    } catch {
      if (isCircuit) return localCircuit();
      return NextResponse.json({ error: "Backend request failed" }, { status: 502 });
    }
  }

  if (isCircuit) return localCircuit();
  return NextResponse.json(
    {
      error:
        "FlyWire backend not configured. Set FLYWIRE_API_URL (e.g. https://arcane-flywire-api.onrender.com or http://127.0.0.1:8001).",
    },
    { status: 503 }
  );
}

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  const { path } = await context.params;
  return proxy(request, path);
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  const { path } = await context.params;
  return proxy(request, path);
}
