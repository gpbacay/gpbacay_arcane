import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";
import { GRANT_COOKIE, gateConfig, verifyGrant } from "@/lib/star-gate";

export const runtime = "nodejs";

export async function GET(request: NextRequest) {
  const cfg = gateConfig();
  if (!cfg) {
    return NextResponse.json(
      { error: "Downloads are not configured. Set GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET and DOWNLOAD_COOKIE_SECRET." },
      { status: 503 }
    );
  }
  if (!verifyGrant(cfg.cookie, request.cookies.get(GRANT_COOKIE)?.value)) {
    return NextResponse.redirect(new URL("/api/arc1-download/login", request.nextUrl.origin));
  }
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
