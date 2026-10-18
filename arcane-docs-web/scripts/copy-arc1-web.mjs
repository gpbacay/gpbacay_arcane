import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import path from "node:path";

const src = path.join(process.cwd(), "..", "node", "web");
const dest = path.join(process.cwd(), "public", "arc1");
mkdirSync(dest, { recursive: true });
for (const name of ["arc1.onnx", "arc1.json"]) {
  const from = path.join(src, name);
  if (existsSync(from)) copyFileSync(from, path.join(dest, name));
}
