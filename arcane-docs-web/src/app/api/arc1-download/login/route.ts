import { randomBytes } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";
import { STATE_COOKIE, gateConfig } from "@/lib/star-gate";

export const runtime = "nodejs";

export async function GET(request: NextRequest) {
  const cfg = gateConfig();
  if (!cfg) return NextResponse.json({ error: "Downloads are not configured." }, { status: 503 });
  const state = randomBytes(16).toString("hex");
  const url = new URL("https://github.com/login/oauth/authorize");
  url.searchParams.set("client_id", cfg.id);
  url.searchParams.set("redirect_uri", `${request.nextUrl.origin}/api/arc1-download/callback`);
  url.searchParams.set("state", state);
  // No scope: public profile only. Reading whether you starred a public repo needs nothing more.
  const res = NextResponse.redirect(url);
  res.cookies.set(STATE_COOKIE, state, { httpOnly: true, sameSite: "lax", secure: request.nextUrl.protocol === "https:", path: "/api/arc1-download", maxAge: 600 });
  return res;
}
