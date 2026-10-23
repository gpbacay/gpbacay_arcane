import { copyFileSync, existsSync, mkdirSync, readFileSync, symlinkSync } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";

// node/web.mjs imports onnxruntime-web. Vercel only installs this app's node_modules.
const ortInApp = path.join(process.cwd(), "node_modules", "onnxruntime-web");
const ortForNode = path.join(process.cwd(), "..", "node", "node_modules", "onnxruntime-web");
if (existsSync(ortInApp) && !existsSync(ortForNode)) {
  mkdirSync(path.dirname(ortForNode), { recursive: true });
  symlinkSync(ortInApp, ortForNode, process.platform === "win32" ? "junction" : "dir");
}

const repo = path.join(process.cwd(), "..");
const src = path.join(repo, "node", "web");

// The chatbot must run the latest ARC 1: arc1.json records which trained weights it was built from
// (examples/publish_arc1.py). Same fingerprint as there: sha256 of each file's sha256, JSON line endings normalized.
const web = JSON.parse(readFileSync(path.join(src, "arc1.json"), "utf8"));
const digest = createHash("sha256");
for (const rel of web.source?.files ?? []) {
  let data = readFileSync(path.join(repo, rel));
  if (rel.endsWith(".json")) data = Buffer.from(data.toString("latin1").replaceAll("\r\n", "\n"), "latin1");
  digest.update(createHash("sha256").update(data).digest());
}
if (!web.source || digest.digest("hex") !== web.source.sha256) {
  const message = "node/web/arc1.onnx was not built from the current Models/arc1_arc1_tiny.* weights. " +
    "Run python examples/publish_arc1.py (the publish-arc1 GitHub workflow does this on push).";
  if (process.env.npm_lifecycle_event === "prebuild") throw new Error(message); // never deploy a stale chatbot model
  console.warn(`[arc1] ${message}`);
}

const dest = path.join(process.cwd(), "public", "arc1");
mkdirSync(dest, { recursive: true });
for (const name of ["arc1.onnx", "arc1.json"]) {
  const from = path.join(src, name);
  if (existsSync(from)) copyFileSync(from, path.join(dest, name));
}
