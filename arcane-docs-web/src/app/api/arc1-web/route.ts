import { NextResponse } from "next/server";

const FILES = { model: "arc1.onnx", config: "arc1.json" } as const;

/** Static file in public/arc1. A function response cannot carry the 5.5MB model on Vercel. */
export function GET(req: Request) {
  const which = new URL(req.url).searchParams.get("file");
  const name = which === "model" || which === "config" ? FILES[which] : null;
  if (!name) return NextResponse.json({ error: "Not found" }, { status: 404 });
  return NextResponse.redirect(new URL(`/arc1/${name}`, req.url));
}
