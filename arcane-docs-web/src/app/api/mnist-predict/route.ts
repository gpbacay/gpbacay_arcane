import { NextRequest, NextResponse } from "next/server";
import { predictMnistRsaa } from "@/lib/mnist-rsaa";

/**
 * Tries the Python ARCANE MNIST API (MNIST_API_URL), then falls back to
 * in-process RSAA so the fruit-fly page works without local weights.
 */
export async function POST(request: NextRequest) {
  let body: { image_base64?: string; pixels?: number[] };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const pixels = Array.isArray(body.pixels) ? body.pixels : null;
  const base = process.env.MNIST_API_URL;

  if (base && typeof base === "string") {
    try {
      const url = `${base.replace(/\/$/, "")}/predict`;
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: body.image_base64 }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && typeof (data as { digit?: unknown }).digit === "number") {
        return NextResponse.json({ ...data, source: "api" });
      }
    } catch {
      // fall through to local RSAA
    }
  }

  if (!pixels || pixels.length < 28 * 28) {
    return NextResponse.json(
      { error: "MNIST backend unavailable and no pixel payload was provided." },
      { status: 503 }
    );
  }

  return NextResponse.json(predictMnistRsaa(pixels.slice(0, 784)));
}
