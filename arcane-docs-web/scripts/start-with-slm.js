#!/usr/bin/env node
/**
 * Starts the Arcane docs site with the SLM chat API (Python FastAPI).
 *
 * Usage: npm run dev:with-slm   (from arcane-docs-web)
 */

const { spawn } = require("child_process");
const path = require("path");
const http = require("http");

const SLM_PORT = 8001;
const SLM_HEALTH_URL = `http://127.0.0.1:${SLM_PORT}/health`;
const POLL_MS = 400;
const POLL_TIMEOUT_MS = 180000;

const repoRoot = path.resolve(__dirname, "..", "..");
const pythonScript = path.join(repoRoot, "examples", "serve_slm_api.py");

function waitForHealth() {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + POLL_TIMEOUT_MS;
    function poll() {
      const req = http.get(SLM_HEALTH_URL, (res) => {
        if (res.statusCode === 200) return resolve();
        if (Date.now() < deadline) setTimeout(poll, POLL_MS);
        else reject(new Error("SLM API health check timed out"));
      });
      req.on("error", () => {
        if (Date.now() >= deadline) reject(new Error("SLM API did not become ready"));
        else setTimeout(poll, POLL_MS);
      });
    }
    poll();
  });
}

function main() {
  const pythonCmd = process.env.SLM_PYTHON_CMD || (process.platform === "win32" ? "python" : "python3");
  console.log("[slm] Starting Python SLM API at", pythonScript);
  const useShell = process.platform === "win32";
  const py = useShell
    ? spawn(`${pythonCmd} "${pythonScript.replace(/"/g, '""')}"`, {
        cwd: repoRoot,
        stdio: "inherit",
        shell: true,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      })
    : spawn(pythonCmd, [pythonScript], {
        cwd: repoRoot,
        stdio: "inherit",
        shell: false,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      });

  py.on("error", (err) => {
    console.error("[slm] Failed to start Python:", err.message);
    process.exit(1);
  });

  py.on("exit", (code, signal) => {
    if (code !== null && code !== 0) {
      console.error("[slm] Python API exited with code", code);
    }
    if (nextRef.current) nextRef.current.kill(signal || "SIGTERM");
    process.exit(code ?? 1);
  });

  const nextRef = { current: null };

  waitForHealth()
    .then(() => {
      console.log("[slm] SLM API responding at http://localhost:" + SLM_PORT);
      console.log("[next] Starting Next.js dev server...");
      const next = spawn("npm", ["run", "dev"], {
        cwd: path.resolve(__dirname, ".."),
        stdio: "inherit",
        shell: true,
        env: { ...process.env, SLM_API_URL: "http://127.0.0.1:" + SLM_PORT },
      });
      nextRef.current = next;
      next.on("exit", (code, signal) => {
        py.kill(signal || "SIGTERM");
        process.exit(code ?? 0);
      });
    })
    .catch((err) => {
      console.error("[slm]", err.message);
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
