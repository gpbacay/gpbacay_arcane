import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  const file = await readFile(path.join(process.cwd(), "private", "models", "arc1-tiny.rcn"));
  return new NextResponse(file, {
    headers: {
      "Content-Type": "application/octet-stream",
      "Content-Disposition": 'attachment; filename="arc1-tiny.rcn"',
      "Content-Length": String(file.length),
      "Cache-Control": "private, no-store",
    },
  });
}
