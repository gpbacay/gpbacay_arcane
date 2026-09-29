import { createHmac, timingSafeEqual } from "node:crypto";

// Downloads of the ARC 1 model require the visitor to have starred this repo.
// GitHub OAuth proves who they are; we check the star once, keep no token, and
// remember the result in a short-lived signed cookie.
export const REPO = "gpbacay/gpbacay_arcane";
export const GRANT_COOKIE = "arc1_dl";
export const STATE_COOKIE = "arc1_oauth_state";
export const GRANT_TTL_S = 24 * 60 * 60;

export function gateConfig() {
  const { GITHUB_CLIENT_ID: id, GITHUB_CLIENT_SECRET: secret, DOWNLOAD_COOKIE_SECRET: cookie } = process.env;
  return id && secret && cookie ? { id, secret, cookie } : null;
}

const mac = (secret: string, data: string) => createHmac("sha256", secret).update(data).digest("base64url");

export function signGrant(secret: string, login: string, now = Date.now()) {
  const body = `${Buffer.from(login).toString("base64url")}.${Math.floor(now / 1000) + GRANT_TTL_S}`;
  return `${body}.${mac(secret, body)}`;
}

/** The GitHub login the grant was issued to, or null if it is forged or expired. */
export function verifyGrant(secret: string, value: string | undefined, now = Date.now()): string | null {
  const parts = value?.split(".");
  if (!parts || parts.length !== 3) return null;
  const [login, exp, sig] = parts;
  const want = Buffer.from(mac(secret, `${login}.${exp}`));
  const got = Buffer.from(sig);
  if (want.length !== got.length || !timingSafeEqual(want, got)) return null;
  if (Number(exp) * 1000 < now) return null;
  return Buffer.from(login, "base64url").toString();
}
