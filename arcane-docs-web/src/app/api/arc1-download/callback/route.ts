import { NextRequest, NextResponse } from "next/server";
import { GRANT_COOKIE, GRANT_TTL_S, REPO, STATE_COOKIE, gateConfig, signGrant } from "@/lib/star-gate";

export const runtime = "nodejs";

const back = (request: NextRequest, status: string) => {
  const res = NextResponse.redirect(new URL(`/docs/arc-1?download=${status}#get-started`, request.nextUrl.origin));
  res.cookies.delete({ name: STATE_COOKIE, path: "/api/arc1-download" });
  return res;
};

export async function GET(request: NextRequest) {
  const cfg = gateConfig();
  if (!cfg) return NextResponse.json({ error: "Downloads are not configured." }, { status: 503 });

  const code = request.nextUrl.searchParams.get("code");
  const state = request.nextUrl.searchParams.get("state");
  if (request.nextUrl.searchParams.get("error") === "access_denied") return back(request, "denied");
  if (!code || !state || state !== request.cookies.get(STATE_COOKIE)?.value) return back(request, "error");

  try {
    const tokenRes = await fetch("https://github.com/login/oauth/access_token", {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify({ client_id: cfg.id, client_secret: cfg.secret, code }),
      signal: AbortSignal.timeout(10_000),
    });
    const { access_token: token } = (await tokenRes.json()) as { access_token?: string };
    if (!token) return back(request, "error");

    const gh = { Authorization: `Bearer ${token}`, Accept: "application/vnd.github+json", "User-Agent": "arcane-docs" };
    const [user, star] = await Promise.all([
      fetch("https://api.github.com/user", { headers: gh, signal: AbortSignal.timeout(10_000) }).then((r) => r.json() as Promise<{ login?: string }>),
      fetch(`https://api.github.com/user/starred/${REPO}`, { headers: gh, signal: AbortSignal.timeout(10_000) }),
    ]);
    if (star.status === 404) return back(request, "star"); // not starred
    if (star.status !== 204 || !user.login) return back(request, "error");

    // Grant issued: send them straight to the file.
    const res = NextResponse.redirect(new URL("/api/arc1-download", request.nextUrl.origin));
    res.cookies.set(GRANT_COOKIE, signGrant(cfg.cookie, user.login), {
      httpOnly: true, sameSite: "lax", secure: request.nextUrl.protocol === "https:", path: "/api/arc1-download", maxAge: GRANT_TTL_S,
    });
    res.cookies.delete({ name: STATE_COOKIE, path: "/api/arc1-download" });
    return res;
  } catch {
    return back(request, "error");
  }
}
