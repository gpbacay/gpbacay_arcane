#!/usr/bin/env node
/**
 * Starts the Arcane docs site with the ARC 1 API (Python FastAPI).
 *
 * Usage: npm run dev:with-arc1   (from arcane-docs-web)
 */

const { spawn } = require("child_process");
const path = require("path");
const http = require("http");

const ARC1_PORT = 8002;
const ARC1_HEALTH_URL = `http://127.0.0.1:${ARC1_PORT}/health`;
const POLL_MS = 400;
// TensorFlow cold start on Windows often exceeds 3 minutes before /health binds.
const POLL_TIMEOUT_MS = 600000;

const repoRoot = path.resolve(__dirname, "..", "..");
const pythonScript = path.join(repoRoot, "examples", "serve_arc1_api.py");

function waitForHealth() {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + POLL_TIMEOUT_MS;
    function poll() {
      const req = http.get(ARC1_HEALTH_URL, (res) => {
        if (res.statusCode === 200) return resolve();
        if (Date.now() < deadline) setTimeout(poll, POLL_MS);
        else reject(new Error("ARC 1 API health check timed out"));
      });
      req.on("error", () => {
        if (Date.now() >= deadline) reject(new Error("ARC 1 API did not become ready"));
        else setTimeout(poll, POLL_MS);
      });
    }
    poll();
  });
}

function main() {
  const pythonCmd = process.env.ARC1_PYTHON_CMD || (process.platform === "win32" ? "python" : "python3");
  console.log("[arc1] Starting Python ARC 1 API at", pythonScript);
  const useShell = process.platform === "win32";
  const py = useShell
    ? spawn(`${pythonCmd} "${pythonScript.replace(/"/g, '""')}"`, {
        cwd: repoRoot,
        stdio: "inherit",
        shell: true,
        env: { ...process.env, PYTHONUNBUFFERED: "1", PORT: String(ARC1_PORT) },
      })
    : spawn(pythonCmd, [pythonScript], {
        cwd: repoRoot,
        stdio: "inherit",
        shell: false,
        env: { ...process.env, PYTHONUNBUFFERED: "1", PORT: String(ARC1_PORT) },
      });

  py.on("error", (err) => {
    console.error("[arc1] Failed to start Python:", err.message);
    process.exit(1);
  });

  py.on("exit", (code, signal) => {
    if (code !== null && code !== 0) {
      console.error("[arc1] Python API exited with code", code);
    }
    if (nextRef.current) nextRef.current.kill(signal || "SIGTERM");
    process.exit(code ?? 1);
  });

  const nextRef = { current: null };

  waitForHealth()
    .then(() => {
      console.log("[arc1] ARC 1 API responding at http://localhost:" + ARC1_PORT);
      console.log("[next] Starting Next.js dev server...");
      const next = spawn("npm", ["run", "dev"], {
        cwd: path.resolve(__dirname, ".."),
        stdio: "inherit",
        shell: true,
        env: { ...process.env, ARC1_API_URL: "http://127.0.0.1:" + ARC1_PORT },
      });
      nextRef.current = next;
      next.on("exit", (code, signal) => {
        py.kill(signal || "SIGTERM");
        process.exit(code ?? 0);
      });
    })
    .catch((err) => {
      console.error("[arc1]", err.message);
      py.kill("SIGTERM");
      process.exit(1);
    });

  process.on("SIGINT", () => {
    py.kill("SIGINT");
    if (nextRef.current) nextRef.current.kill("SIGINT");
    process.exit(0);
  });
  process.on("SIGTERM", () => {
    py.kill("SIGTERM");
    if (nextRef.current) nextRef.current.kill("SIGTERM");
    process.exit(0);
  });
}

main();
