import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  let file: Buffer;
  try {
    file = await readFile(path.join(process.cwd(), "private", "models", "arc1-tiny.rcn"));
  } catch {
    return NextResponse.json({ error: "Model file is not on this server." }, { status: 404 });
  }
  return new NextResponse(new Uint8Array(file), {
    headers: {
      "Content-Type": "application/octet-stream",
      "Content-Disposition": 'attachment; filename="arc1-tiny.rcn"',
      "Content-Length": String(file.length),
      "Cache-Control": "private, no-store",
    },
  });
}
