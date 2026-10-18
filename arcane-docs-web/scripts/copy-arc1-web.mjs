import { copyFileSync, existsSync, mkdirSync, symlinkSync } from "node:fs";
import path from "node:path";

// node/web.mjs imports onnxruntime-web. Vercel only installs this app's node_modules.
const ortInApp = path.join(process.cwd(), "node_modules", "onnxruntime-web");
const ortForNode = path.join(process.cwd(), "..", "node", "node_modules", "onnxruntime-web");
if (existsSync(ortInApp) && !existsSync(ortForNode)) {
  mkdirSync(path.dirname(ortForNode), { recursive: true });
  symlinkSync(ortInApp, ortForNode, process.platform === "win32" ? "junction" : "dir");
}

const src = path.join(process.cwd(), "..", "node", "web");
const dest = path.join(process.cwd(), "public", "arc1");
mkdirSync(dest, { recursive: true });
for (const name of ["arc1.onnx", "arc1.json"]) {
  const from = path.join(src, name);
  if (existsSync(from)) copyFileSync(from, path.join(dest, name));
}
