import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";

const FILES = {
  model: { name: "arc1.onnx", type: "application/octet-stream" },
  config: { name: "arc1.json", type: "application/json" },
} as const;

/** Serves the stock ARC 1 web model that ships in node/web. No other files. */
export async function GET(req: Request) {
  const which = new URL(req.url).searchParams.get("file");
  const file = which === "model" || which === "config" ? FILES[which] : null;
  if (!file) return NextResponse.json({ error: "Not found" }, { status: 404 });
  try {
    const bytes = await readFile(path.join(process.cwd(), "..", "node", "web", file.name));
    return new NextResponse(bytes, {
      headers: {
        "content-type": file.type,
        "cache-control": "public, max-age=31536000, immutable",
      },
    });
  } catch {
    return NextResponse.json({ error: "ARC 1 model is not on this server." }, { status: 404 });
  }
}
